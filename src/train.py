"""Fine-tune Whisper on Common Voice Pashto v24 -- reproduction of exp_009.

Paper: H. Rahman, "Fine-tuning Whisper for Pashto ASR: strategies and scale",
arXiv:2604.06507, Apr. 2026. Target: WER 24.89%, CER 8.075%, best step 8500.

The paper names `fine_tune_whisper_enhanced.py` but publishes no repository, so
this is a re-implementation against the public Seq2SeqTrainer API. That is the
single largest methodological deviation and is declared as such in the report.

Every deviation from the paper's published configuration is recorded in
config.yaml comments and in results/<tag>/effective_config.yaml.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from utils import (
    compute_metrics as compute_text_metrics,
    dump_effective_config,
    load_config,
    set_all_seeds,
    write_run_meta,
)


@dataclass
class WhisperCollator:
    """Pad features and labels separately; mask pad positions to -100.

    The -100 mask is what keeps padding out of the cross-entropy, and stripping
    the leading BOS avoids double-counting it once the model re-prepends it.
    """

    processor: Any

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def build_dataset(cfg: dict, processor, split: str):
    from datasets import Audio, load_dataset

    path = Path(cfg["paths"]["processed"]) / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run src/prepare_data.py first")

    dataset = load_dataset("json", data_files=str(path), split="train")
    dataset = dataset.cast_column(
        "audio", Audio(sampling_rate=cfg["data"]["sampling_rate"])  # 16 kHz mono
    )

    augment = None
    if cfg["augmentation"]["enabled"] and split == "train":
        # Ablation E3c. exp_009 itself used NO augmentation -- only exp_008
        # (whisper-base) did, where the paper measured a 7.25 pp benefit.
        from audiomentations import AddGaussianSNR, Compose, TimeStretch

        aug_cfg = cfg["augmentation"]
        lo, hi = aug_cfg["speed_perturb_pct"]
        augment = Compose([
            TimeStretch(min_rate=1 - hi / 100, max_rate=1 + hi / 100,
                        p=aug_cfg["probability"], leave_length_unchanged=False),
            AddGaussianSNR(min_snr_db=aug_cfg["gaussian_noise_snr_db"],
                           max_snr_db=aug_cfg["gaussian_noise_snr_db"],
                           p=aug_cfg["probability"]),
        ])

    sr = cfg["data"]["sampling_rate"]

    def to_features(batch):
        audio = batch["audio"]["array"]
        if augment is not None:
            audio = augment(samples=audio.astype("float32"), sample_rate=sr)
        batch["input_features"] = processor.feature_extractor(
            audio, sampling_rate=sr
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    keep = {"input_features", "labels"}
    return dataset.map(
        to_features,
        remove_columns=[c for c in dataset.column_names if c not in keep],
        num_proc=1,  # audio decode + feature extraction; >1 fights the dataloader on Colab
        desc=f"featurising {split}",
    )


def apply_strategy(model, cfg: dict):
    """full | lora (paper exp_002) | frozen_encoder (paper exp_003).

    The paper measured LoRA r=64 at +33.36 pp worse than full fine-tuning and
    frozen-encoder at +14.76 pp worse, so `full` is both the faithful and the
    empirically correct default.
    """
    strategy = cfg["model"]["strategy"]

    if strategy == "full":
        pass

    elif strategy == "lora":
        from peft import LoraConfig, get_peft_model

        lora_cfg = cfg["lora"]
        model = get_peft_model(model, LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
        ))
        model.print_trainable_parameters()  # paper: 2.36M = 3.15% on whisper-base

    elif strategy == "frozen_encoder":
        # Paper froze the bottom 2 of whisper-base's 6 encoder layers and argued
        # the result does not transfer to deeper encoders.
        for layer in model.model.encoder.layers[:2]:
            for param in layer.parameters():
                param.requires_grad = False

    else:
        raise ValueError(f"unknown strategy: {strategy}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"strategy={strategy}  trainable={trainable:,}/{total:,} "
          f"({100.0 * trainable / total:.2f}%)")
    return model


def export_training_log(trainer, run_dir: Path) -> None:
    """trainer_state.json -> training_log.csv, the source for the loss/WER curves."""
    import pandas as pd

    history = trainer.state.log_history
    rows: dict[int, dict] = {}
    for entry in history:
        step = entry.get("step")
        if step is None:
            continue
        row = rows.setdefault(step, {"step": step})
        for src, dst in (("loss", "train_loss"), ("eval_loss", "eval_loss"),
                         ("eval_wer", "eval_wer"), ("eval_cer", "eval_cer"),
                         ("learning_rate", "learning_rate"),
                         ("grad_norm", "grad_norm"), ("epoch", "epoch")):
            if src in entry:
                row[dst] = entry[src]

    frame = pd.DataFrame(sorted(rows.values(), key=lambda r: r["step"]))
    frame.to_csv(run_dir / "training_log.csv", index=False)

    best = {
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "best_step": None,
        "best_epoch": None,
        "total_steps": trainer.state.global_step,
        "paper_best_step": 8500,
        "paper_best_epoch": 5.85,
    }
    if trainer.state.best_model_checkpoint:
        best["best_step"] = int(Path(trainer.state.best_model_checkpoint).name.split("-")[-1])
        match = frame[frame["step"] == best["best_step"]]
        if not match.empty and "epoch" in match:
            best["best_epoch"] = float(match["epoch"].iloc[0])
    with open(run_dir / "best_checkpoint.json", "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)
    print(f"best: step {best['best_step']} (paper exp_009: step 8500, epoch 5.85)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tag", required=True, help="run name, e.g. exp009_repro_tierB")
    parser.add_argument("--model", default=None)
    parser.add_argument("--strategy", default=None, choices=["full", "lora", "frozen_encoder"])
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help='checkpoint path, or "auto" to resume the latest')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.strategy:
        cfg["model"]["strategy"] = args.strategy
    if args.learning_rate is not None:
        cfg["train"]["learning_rate"] = args.learning_rate
    if args.augment:
        cfg["augmentation"]["enabled"] = True

    set_all_seeds(cfg["seed"])  # paper: 42

    run_dir = Path(cfg["paths"]["results"]) / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_effective_config(cfg, run_dir)

    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    model_name = cfg["model"]["name"]
    processor = WhisperProcessor.from_pretrained(
        model_name, language=cfg["model"]["language"], task=cfg["model"]["task"]
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Required on transformers >= 4.40: forced_decoder_ids conflicts with the
    # language/task generation arguments and must be cleared.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = cfg["model"]["language"]
    model.generation_config.task = cfg["model"]["task"]
    model.generation_config.forced_decoder_ids = None
    if cfg["train"]["gradient_checkpointing"]:
        model.config.use_cache = False

    model = apply_strategy(model, cfg)

    train_ds = build_dataset(cfg, processor, "train")
    dev_ds = build_dataset(cfg, processor, "dev")
    print(f"train {len(train_ds)} | dev {len(dev_ds)}")

    tokenizer = processor.tokenizer

    def metrics_fn(pred):
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        hyps = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        result = compute_text_metrics(refs, hyps)
        # "wer" here is NORMALISED WER -- the paper's checkpoint-selection metric.
        return {"wer": result.wer_norm, "cer": result.cer,
                "wer_ortho": result.wer_ortho}

    train_cfg = cfg["train"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        num_train_epochs=train_cfg["num_train_epochs"],
        max_steps=train_cfg["max_steps"],
        optim=train_cfg["optim"],
        adam_beta1=train_cfg["adam_beta1"],
        adam_beta2=train_cfg["adam_beta2"],
        adam_epsilon=train_cfg["adam_epsilon"],
        max_grad_norm=train_cfg["max_grad_norm"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        group_by_length=train_cfg["group_by_length"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        predict_with_generate=train_cfg["predict_with_generate"],
        generation_max_length=train_cfg["generation_max_length"],
        generation_num_beams=train_cfg["generation_num_beams"],
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        logging_steps=train_cfg["logging_steps"],
        logging_dir=str(run_dir / "logs"),
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        report_to=train_cfg["report_to"],
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=WhisperCollator(processor=processor),
        compute_metrics=metrics_fn,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=train_cfg["early_stopping_patience"]
        )],
    )

    resume = args.resume_from_checkpoint
    if resume == "auto":
        checkpoints = sorted(run_dir.glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[-1]))
        resume = str(checkpoints[-1]) if checkpoints else None
        print(f"resuming from {resume}" if resume else "no checkpoint found; starting fresh")

    start = time.time()
    trainer.train(resume_from_checkpoint=resume)
    wall_clock = time.time() - start

    best_dir = run_dir / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    export_training_log(trainer, run_dir)
    write_run_meta(
        run_dir,
        tag=args.tag,
        wall_clock_s=round(wall_clock, 1),
        model=model_name,
        strategy=cfg["model"]["strategy"],
        learning_rate=train_cfg["learning_rate"],
        effective_batch=(train_cfg["per_device_train_batch_size"]
                         * train_cfg["gradient_accumulation_steps"]),
        precision="bf16" if train_cfg["bf16"] else ("fp16" if train_cfg["fp16"] else "fp32"),
        augmentation=cfg["augmentation"]["enabled"],
        total_steps=trainer.state.global_step,
    )

    print(f"\ndone in {wall_clock / 3600:.2f} h -> {best_dir}")
    print("next: python src/evaluate.py --config config.yaml "
          f"--model {best_dir} --split test --tag {args.tag}_test")


if __name__ == "__main__":
    main()

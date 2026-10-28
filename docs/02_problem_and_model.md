# Step 2 — Problem and Model Explanation

Report-ready prose. Written to be pasted into Sections 4 and 5 of the report.

## 2.1 Why Pashto ASR is a prediction task, not classification

Classification maps an input to one label from a fixed, finite set. ASR does not do that. Given one audio clip, the model must emit a **variable-length sequence of tokens** drawn from a vocabulary of ~51,865 subword units, and the number of possible outputs grows exponentially with sequence length. There is no label set to pick from.

Formally, the model learns a conditional distribution over output sequences:

```
P(y_1 ... y_T | x) = Π_t P(y_t | y_1 ... y_{t-1}, x)
```

where `x` is the acoustic input and `y_t` the token at step `t`. Each decoding step *is* a 51,865-way classification, but the task as a whole is **autoregressive sequence prediction** — the model predicts the next token conditioned on everything it has already predicted. That is why the metric is an edit distance (WER / CER) and not accuracy, precision or F1: there is no single correct class to be right or wrong about, only a transcript that is closer to or further from the reference.

This squarely satisfies CLO-2 ("classification **and prediction** problems") on the prediction side. Say this explicitly in the report so the mapping to the CLO is unambiguous.

## 2.2 Input the model receives

1. Raw waveform, resampled to **16 kHz mono** (Whisper accepts nothing else).
2. Padded or trimmed to a fixed **30-second** window — Whisper's receptive field is fixed, so a 4-second Pashto clip is zero-padded to 30 s.
3. Converted to an **80-channel log-Mel spectrogram** (`whisper-small`; `large-v3` uses 128) with a 25 ms window and 10 ms hop, giving a `(80, 3000)` feature matrix per clip.
4. Decoder prompt tokens: `<|startoftranscript|> <|ps|> <|transcribe|> <|notimestamps|>`. Setting the language token to `ps` (Pashto) is **essential** — this is the switch the paper shows is broken in zero-shot mode, where the model ignores it and emits Arabic/Dari/Urdu script anyway.

## 2.3 Output the model produces

A sequence of BPE token IDs, decoded into a **Pashto Arabic-script transcript string** (right-to-left). No timestamps in our configuration. For CV24, references average ~4.4 s of speech, so outputs are short sentences.

## 2.4 How Whisper performs speech recognition

Whisper is a plain **encoder–decoder Transformer** trained on 680k hours of weakly supervised, web-scraped multilingual audio-text pairs.

- **Encoder** — two 1-D convolution layers (the second with stride 2, halving the time axis) project the log-Mel frames to the model width, sinusoidal position embeddings are added, then N pre-norm Transformer blocks with multi-head self-attention produce 1,500 acoustic hidden states.
- **Decoder** — learned position embeddings, then N blocks of (causal self-attention → cross-attention over the encoder states → feed-forward), ending in a linear projection tied to the token embedding matrix over the 51,865-token multilingual BPE vocabulary.
- **Multitask interface** — transcription, translation, language ID and voice-activity detection are all expressed as special tokens in the decoder prompt rather than as separate heads. That design is exactly what makes a single-language fine-tune clean: we keep the architecture and only steer it with the `<|ps|>` token.
- **Inference** — autoregressive decoding (greedy or beam) until `<|endoftext|>`.

`whisper-small`: 244M parameters, `d_model` 768, 12 encoder + 12 decoder layers, 12 attention heads, 80 Mel bins.

## 2.5 How transfer learning is applied here

Whisper's pre-training gives it a general, language-agnostic model of human speech — phonetic structure, speaker variation, channel noise, prosody. Pashto was not in the pre-training mixture, so what is missing is not acoustic competence but the **mapping from Pashto phonetics to Pashto orthography**.

Transfer learning means we keep the pre-trained weights as the initialisation and continue training on Pashto data, so gradient descent only has to learn that missing mapping instead of learning speech from scratch. Concretely:

1. Load pre-trained `openai/whisper-small` weights.
2. Replace the data distribution with Common Voice Pashto.
3. Force the decoder language token to `ps`.
4. Continue training end-to-end with a small learning rate.

The paper evaluates four variants of this idea and finds that **updating every parameter (vanilla full fine-tuning) wins decisively**: LoRA at r=64 is 33.36 pp worse and freezing the bottom 2 of 6 encoder layers is 14.76 pp worse. The paper's explanation for the freezing result is architectural: `whisper-base` has only 6 encoder layers, so the "low layers are generic, high layers are task-specific" assumption that justifies freezing does not hold at that depth — freezing two layers removes a third of trainable capacity without buying the usual regularisation benefit. Multistage Urdu→Pashto transfer was worst of all (+44.56 pp): Urdu shares the script family but its phonology is close enough to interfere and far enough to mislead.

**So our reproduction uses full fine-tuning.** That is the faithful choice, and it is also the empirically correct one.

## 2.6 Why fine-tuning is necessary for Pashto

Zero-shot Whisper does not merely perform poorly on Pashto — it fails categorically:

- Zero-shot WER on CV24: **171.4%** (base), **120.4%** (small), **110.2%** (large-v3-turbo). WER above 100% is possible because insertions are unbounded.
- The companion benchmark (arXiv:2604.04598) reports zero-shot WER from ~90% to 297% across seven model sizes, with `whisper-medium` at 461% on CV24 from decoder looping, and finds that **no Whisper model produces Pashto-script output in more than 0.8% of utterances.**

That last number is the real argument. The failure is a **script failure**, not an accuracy failure: the model confidently transcribes Pashto audio into Arabic, Dari or Urdu orthography. No amount of decoding tricks, prompting or language-token forcing fixes a missing orthographic mapping. Only weight updates do. Pashto also has phonemes absent from Whisper's high-resource training languages — the retroflex series ټ ډ ړ ڼ and the affricates ځ څ — which the pre-trained encoder has no reason to discriminate.

## 2.7 Loss function

Token-level **cross-entropy** on the decoder output, teacher-forced, with padding positions masked to `-100` so they contribute nothing:

```
L = - (1 / |valid|) * Σ_t log P(y_t* | y_1* ... y_{t-1}*, x)
```

Notes for the report:

- This is standard `Seq2SeqTrainer` behaviour; no CTC loss is involved — Whisper is attention-based, not CTC-based.
- Label smoothing is **not** used (paper does not mention it; Transformers defaults to 0.0).
- Cross-entropy is a **proxy** for WER, not WER itself. The paper's own tables show why this matters: exp_009 has a *higher* eval loss (0.3974) than exp_010 (0.3113) and a higher WER, but exp_008 has the *lowest* loss (0.3305) of the three and the *worst* WER. Loss and WER are correlated but not monotonic — which is exactly why the paper selects the best checkpoint by **validation WER, not validation loss**. We must do the same (`metric_for_best_model="wer"`, `greater_is_better=False`), and we should show this non-monotonicity in our own curves if it appears.

## 2.8 Optimizer

The paper **never states the optimizer** (logged as risk #3). We assume the Transformers default, `adamw_torch`:

- AdamW, β₁ = 0.9, β₂ = 0.999, ε = 1e-8
- Decoupled weight decay (paper reports WD 0.0 for exp_001; WD unstated for exp_009 → we use 0.0 to stay consistent)
- Linear warmup then linear decay
- Gradient clipping at max-norm 1.0 (Transformers default)

This assumption must be flagged in the report, because LR 1e-4 under AdamW is not the same operating point as 1e-4 under Adafactor or SGD.

## 2.9 How WER and CER are calculated

**WER** is Levenshtein edit distance at the **word** level, normalised by reference length:

```
WER = (S + D + I) / N
```

- `S` substitutions, `D` deletions, `I` insertions — the minimum-cost alignment between hypothesis and reference
- `N` = number of words in the **reference**

Because `I` is unbounded while `N` is fixed, WER can exceed 100% — which is precisely how zero-shot Whisper scores 171% on Pashto.

**CER** is the same formula at the **character** level. It is the more informative metric for a low-resource agglutinative language: a one-suffix error (masculine `-ay` vs feminine `-a`) costs a full word under WER but only one or two characters under CER. The paper's error analysis attributes 70–78% of word errors to substitutions, mostly word-final morphology — so the CER/WER ratio is itself a finding worth reporting. In the paper's exp_009: WER 24.89 / CER 8.08, a ratio of ~3.1.

**Three WER variants, and you must label which one you are quoting:**

| Variant | Definition | exp_009 value |
|---|---|---:|
| WER orthographic | raw strings, no normalisation | 27.73% |
| **WER normalised** | both sides through HF `BasicTextNormalizer` first | **24.89%** |
| CER | character level, normalised | 8.075% |

**The paper's headline metric is normalised WER.** The difference calculation in Step 5 must use normalised WER on both sides, or the comparison is meaningless. We report all three to be safe.

**Caveat worth a sentence in Discussion:** `BasicTextNormalizer` lowercases, strips punctuation and collapses whitespace. It performs **no Arabic-script canonicalisation** — it will not unify Arabic yeh `ي` with Pashto yeh `ی`, nor Arabic kaf `ك` with Pashto kaf `ک`, nor strip the zero-width non-joiner `U+200C` or tatweel `U+0640`. Those are real, frequent Pashto orthographic variants, so a measurable share of the paper's 24.89% WER is plausibly **orthographic variation rather than recognition error**. We quantify this with a normalisation ablation (see `docs/05`), which is a genuine contribution of our reproduction rather than a restatement of the paper.

**Implementation.** `evaluate.load("wer")` / `evaluate.load("cer")`, both backed by `jiwer`. For the error-type breakdown use `jiwer.process_words()`, which returns `substitutions`, `deletions`, `insertions` and `hits` directly — that is how we rebuild the paper's Appendix C breakdown.

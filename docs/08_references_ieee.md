# Step 10 — Citations (IEEE Style) and Academic Integrity

## Verification status — read this first

Entries **[1]–[4]** were verified live on 2026-10-27 against arXiv, Hugging Face, GitHub, Mozilla and Kaggle; use them as written.

Entries **[5]–[15]** are standard, long-established tool and model references written from knowledge. They are almost certainly correct, but **page numbers and proceedings titles must be confirmed against the publisher page before submission** — an unverified page range is the kind of small error that costs integrity marks. Check each against its DOI or the ACL Anthology / PMLR / NeurIPS Proceedings page.

Cite only what you actually read. Delete any entry you do not cite.

---

## Base paper and companion work (verified)

[1] H. Rahman, "Fine-tuning Whisper for Pashto ASR: strategies and scale," arXiv preprint arXiv:2604.06507, Apr. 2026, doi: 10.48550/arXiv.2604.06507. [Online]. Available: https://arxiv.org/abs/2604.06507. [Accessed: Oct. 27, 2026].

[2] H. Rahman, "Benchmarking Multilingual Speech Models on Pashto: Zero-Shot ASR, Script Failure, and Cross-Domain Evaluation," arXiv preprint arXiv:2604.04598, Apr. 2026, doi: 10.48550/arXiv.2604.04598. [Online]. Available: https://arxiv.org/abs/2604.04598. [Accessed: Oct. 27, 2026].

[3] H. Rahman, "exp_009_small_cv24_vanilla: Whisper-small fine-tuned on Common Voice Pashto v24," Hugging Face model repository, 2026. [Online]. Available: https://huggingface.co/ihanif/exp_009_small_cv24_vanilla. [Accessed: Oct. 27, 2026]. (Apache-2.0; reports WER 24.89%, CER 8.075%, best checkpoint-8500.)

## Dataset (verified)

[4] Mozilla Foundation, "Common Voice Scripted Speech 24.0 — Pashto (ps)," Mozilla Data Collective, 2026. CC0-1.0. [Online]. Available: https://datacollective.mozillafoundation.org/datasets/cmj8u3pnb00llnxxbfvxo3b14. [Accessed: Oct. 27, 2026].

[4a] *(cite only if you used the mirror)* A. A. Aali, "Common Voice 24.0: Pashto Speech Dataset," Kaggle, 2026. CC0-1.0. [Online]. Available: https://www.kaggle.com/datasets/ataullahaali/common-voice-scripted-speech-24-0-pashto. [Accessed: Oct. 27, 2026]. (Unofficial mirror of [4].)

## Dataset method paper — confirm page range

[5] R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. M. Tyers, and G. Weber, "Common Voice: A Massively-Multilingual Speech Corpus," in *Proc. 12th Language Resources and Evaluation Conference (LREC)*, Marseille, France, 2020, pp. 4218–4222.

## Model — confirm proceedings volume and pages

[6] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," in *Proc. 40th Int. Conf. Machine Learning (ICML)*, Honolulu, HI, USA, 2023, pp. 28492–28518. [Also available as arXiv:2212.04356.]

[7] OpenAI, "whisper-small," Hugging Face model repository. [Online]. Available: https://huggingface.co/openai/whisper-small. [Accessed: `<date>`].

## Frameworks and libraries — confirm pages

[8] T. Wolf *et al.*, "Transformers: State-of-the-Art Natural Language Processing," in *Proc. 2020 Conf. Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP)*, 2020, pp. 38–45.

[9] Q. Lhoest *et al.*, "Datasets: A Community Library for Natural Language Processing," in *Proc. 2021 Conf. Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP)*, 2021, pp. 175–184.

[10] A. Paszke *et al.*, "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019, pp. 8024–8035.

[11] Hugging Face, "Evaluate: A library for easily evaluating machine learning models and datasets," software, version 0.4.3, 2024. [Online]. Available: https://github.com/huggingface/evaluate. [Accessed: `<date>`].

[12] N. Vaessen, "JiWER: Evaluate your speech-to-text system with similarity measures such as word error rate," software, version 3.0.4. [Online]. Available: https://github.com/jitsi/jiwer. [Accessed: `<date>`].

## Only if the LoRA ablation is run

[13] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2022.

[14] S. Mangrulkar, S. Gugger, L. Debut, Y. Belkada, S. Paul, and B. Bossan, "PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods," software, 2022. [Online]. Available: https://github.com/huggingface/peft. [Accessed: `<date>`].

## Only if the augmentation ablation is run

[15] I. Jordal *et al.*, "Audiomentations: A Python library for audio data augmentation," software, version 0.36.0. [Online]. Available: https://github.com/iver56/audiomentations. [Accessed: `<date>`].

## Related Pashto / low-resource ASR work (verified to exist; read before citing)

[16] "Pashto Common Voice: Building the First Open Speech Corpus for a 60-Million-Speaker Low-Resource Language," arXiv preprint arXiv:2603.27021, Mar. 2026. [Online]. Available: https://arxiv.org/abs/2603.27021. [Accessed: Oct. 27, 2026]. *(Confirm the author list on the arXiv page before citing.)*

[17] "BaltiVoice: A Speech Corpus and Fine-tuned Whisper ASR System for the Balti Language," arXiv preprint arXiv:2606.03504, Jun. 2026. [Online]. Available: https://arxiv.org/abs/2606.03504. [Accessed: Oct. 27, 2026]. *(Confirm the author list before citing.)*

## Optional — zero-shot evaluation set, if FLEURS Pashto is used as a second test set

[18] A. Conneau *et al.*, "FLEURS: Few-Shot Learning Evaluation of Universal Representations of Speech," in *Proc. IEEE Spoken Language Technology Workshop (SLT)*, 2023, pp. 798–805. *(Confirm pages. Note: the paper's companion benchmark [2] uses FLEURS Pashto; the exp_009 experiment we reproduce does not, so cite this only if we add it.)*

## Third-party implementation reference — cite as such, never as official code

[19] H. Majeed, "pashto-asr: Fine-tunes Whisper on Pashto using Mozilla Common Voice," GitHub repository. [Online]. Available: https://github.com/HarisMajeed05/pashto-asr. [Accessed: Oct. 27, 2026]. (Unaffiliated with [1]; reports no quantitative results. Consulted for pipeline structure only.)

---

## Academic integrity rules we bind ourselves to

1. **No fabrication.** No paper, author, DOI, URL, dataset, checkpoint, hyperparameter or result value appears anywhere in the report unless it was read from the source. Cells awaiting a run stay `TBD` until the run finishes.
2. **Hard separation of sources.** The results table has a dedicated "Original Paper WER" column populated solely from [1]. `results/metrics.csv` carries a `source` column with exactly two values, `paper` and `ours`, and the figure scripts colour by it.
3. **Every deviation declared.** The paper-vs-ours deviation table in Section 8 lists each change with a reason and a reproducibility impact. The five material ones: training-data volume, learning rate, precision, effective batch size, and the re-implemented training script.
4. **Every assumption declared.** Values the paper never states — optimizer, warmup, beam width, max generation length, weight decay for exp_009 — are labelled "assumed" at first use, not presented as the paper's settings.
5. **Code attribution.** Our scripts are our own, written against the public `Seq2SeqTrainer` API and the HF Whisper fine-tuning pattern. [19] was consulted for structure and is cited. Any copied snippet carries an inline source comment.
6. **Checkpoint attribution.** [3] is the author's work under Apache-2.0. Our E0 run *evaluates* it; it is never presented as our training result.
7. **Dataset licence respected.** Common Voice v24 is CC0-1.0. We cite [4] as the source and do not redistribute the archive.
8. **Contributions stated per person**, specifically, with named sections and files — not "all contributed equally".
9. **AI-assistance disclosure.** If the course policy requires it, state which parts used AI assistance (e.g. planning, code scaffolding, report structuring) and confirm that all reported numbers came from executed runs that the group can re-execute on demand.
10. **Every member can defend every number.** Each member independently re-runs `src/evaluate.py` at least once before submission.

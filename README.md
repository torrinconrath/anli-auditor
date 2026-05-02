# ANLI Rationale Auditor

A suite of NLP interpretability techniques for evaluating whether a fine-tuned **RoBERTa-large** model's internal reasoning aligns with human-grounded rationales on the [Adversarial Natural Language Inference (ANLI)](https://github.com/facebookresearch/anli) dataset.

This project investigates the gap between **faithfulness** and **plausibility** in transformer models — specifically, whether the rationales a model implicitly relies on actually reflect its internal decision-making process, or are merely post-hoc justifications.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Audit Methodology](#audit-methodology)
- [References](#references)

---

## Overview

Standard NLI benchmarks measure *what* a model predicts. This project asks a deeper question: **does the model's internal representation actually encode the reasoning a human would use to reach the same conclusion?**

Two custom metrics probe this using the model's frozen `[CLS]` hidden states — no probing classifiers, no attention heuristics:

| Metric | Full Name | What it measures |
|--------|-----------|-----------------|
| **LAS** | Latent Alignment Score | Cosine similarity between the model's decision-state embedding and the `[CLS]` embedding of the human-written rationale — both conditioned on the same hypothesis |
| **CSI** | Causal Sensitivity Index | Drop in LAS after TextFooler finds a label-flipping perturbation of the premise. High CSI means the model's latent state responds to logical changes, consistent with faithful reasoning |

Two aggregate metrics summarise the audit across all samples:

| Metric | Full Name | What it measures |
|--------|-----------|-----------------|
| **UDR** | Unfaithfulness Detection Rate | Fraction of samples where LAS < 0.30 (decision vector shares less than 30% directional overlap with the human rationale) |
| **SS** | Synthetic Sensitivity | Fraction of samples where a *mismatched* rationale also scores below 0.30 — a control confirming UDR is not trivially high due to geometric noise |

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ANLI AUDITOR PIPELINE                           │
├─────────────────┬───────────────────────┬───────────────────────────────┤
│   1. DATA       │   2. FINE-TUNE        │   3. AUDIT                    │
│                 │                       │                               │
│  R1+R2+R3 train │  RoBERTa-large        │  LAS:                         │
│  → up to 50k    │  up to 50k samples    │    cosine_sim(                │
│    samples      │  2 epochs, lr=5e-6    │      CLS(premise, hyp),       │
│                 │  batch=16, bf16       │      CLS(rationale, hyp)      │
│  R1+R2+R3 dev   │                       │    )                          │
│  → validation   │  Saved to disk        │                               │
│                 │                       │  CSI:                         │
│  R1+R2+R3 test  │                       │    LAS_orig - LAS_perturbed   │
│  → held-out     │                       │    (TextFooler attack on      │
│    eval+audit   │                       │     the premise)              │
└─────────────────┴───────────────────────┴───────────────────────────────┘
```

**Stage 1 – Data:** All three adversarial ANLI rounds are merged for training and validation. The test splits from all three rounds are held out entirely — no test-set leakage at any point.

**Stage 2 – Fine-Tuning:** RoBERTa-large is fine-tuned on the 3-class NLI task (Entailment / Neutral / Contradiction) using cross-entropy. The final configuration uses up to 50k training samples, 2 epochs, a learning rate of 5e-6, and batch size 16 with bf16 mixed precision where supported.

**Stage 3 – Audit:** The `RationaleAuditor` extracts `[CLS]` hidden states from the frozen fine-tuned model and computes LAS and CSI for each test sample. A mismatched-rationale control (drawn from a different sample in the full test pool) establishes a semantic floor for the Synthetic Sensitivity metric.

---

## Project Structure

```
anli-auditor/
├── main.py                  # Entry point — runs training, evaluation, and audit
├── create_visuals.py        # Generates LAS and CSI histogram figures
├── requirements.txt
└── src/
    ├── config.py            # All hyperparameters and paths
    ├── data_handler.py      # ANLI dataset loading and split assembly
    ├── model_handler.py     # RoBERTa-large model and tokenizer loading
    ├── trainer.py           # Fine-tuning loop (HuggingFace Trainer)
    ├── evaluate.py          # Accuracy, macro F1, ECE, and universal metrics
    └── auditor.py           # LAS, CSI, and mismatched-LAS rationale audit
```

---

## Setup

### Requirements

- Python 3.11+
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4070 12GB)
- CUDA 12.1
- Windows 10/11 or Linux

### Installation

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins PyTorch to CUDA 12.1 wheels. If you are on a different CUDA version, update the `torch` version and `--extra-index-url` in `requirements.txt` before installing.

---

## Usage

### Full Pipeline (train + evaluate + audit)

```bash
python main.py
```

This runs three stages in sequence:

1. Fine-tunes RoBERTa-large on up to 50k ANLI samples (~35–45 minutes on RTX 4070)
2. Evaluates on the held-out test set
3. Runs the rationale audit on `NUM_AUDIT_SAMPLES` test samples

Results are saved to `./results_anli_roberta/`:
- `eval_results.json` — accuracy, macro F1, ECE, per-class report, confusion matrix
- `audit_results.json` — LAS, CSI, UDR, SS, CSI distribution, and per-sample scores

### Evaluation Only (skip training)

`DO_FINETUNING` defaults to `False` in `src/config.py`. With it set to `False`:

```bash
python main.py
```

This loads the saved model from `./results_anli_roberta/final_model` and runs evaluation and audit only.

### Generate Figures

After a completed audit run:

```bash
python create_visuals.py
```

Outputs two PNG histograms to `./results_anli_roberta/figures/`:
- `las_distribution.png` — LAS histogram with UDR threshold and mean marked
- `csi_distribution.png` — CSI histogram with positive/negative regions colour-coded

---

## Configuration

All key settings live in `src/config.py`. The table below reflects the current defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `roberta-large` | Base model from HuggingFace Hub |
| `OUTPUT_DIR` | `./results_anli_roberta` | Root directory for saved models and results |
| `DO_FINETUNING` | `False` | Set to `True` to run training; `False` loads from `OUTPUT_DIR/final_model` |
| `MAX_TRAIN_SAMPLES` | `50000` | Training samples drawn from R1+R2+R3 train splits |
| `MAX_VAL_SAMPLES` | `None` | Validation samples (`None` = full dev set) |
| `NUM_AUDIT_SAMPLES` | `500` | Auditing samples drawed from a subset of the test splits|
| `TRAIN_EPOCHS` | `2` | Number of training epochs |
| `BATCH_SIZE` | `16` | Per-device batch size |
| `LEARNING_RATE` | `5e-6` | Conservative LR suited to fine-tuning on noisy adversarial labels |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | Effective batch size multiplier |
| `MAX_SEQ_LENGTH` | `512` | Max token length for premise + hypothesis |
| `NUM_AUDIT_SAMPLES` | `500` | Number of test samples used for the rationale audit |

---

## Dataset

The [Adversarial NLI (ANLI)](https://github.com/facebookresearch/anli) dataset by Meta AI is loaded automatically via HuggingFace Datasets. No manual download required.

ANLI was collected using a human-and-model-in-the-loop annotation process across three rounds of increasing adversarial difficulty. Each example includes a premise, hypothesis, a 3-class NLI label, and a **human-written rationale** explaining the label — the rationale is what this project audits against.

**Split strategy:**

| Split | Source | Purpose |
|-------|--------|---------|
| Train | R1 + R2 + R3 train (up to 50k subsampled) | Fine-tuning |
| Validation | R1 + R2 + R3 dev | Loss monitoring and best-model checkpoint selection |
| Test | R1 + R2 + R3 test (fully held-out) | Final evaluation and rationale audit |

All three adversarial rounds are included in every split so the model and audit cover the full difficulty spectrum of the dataset.

---

## Audit Methodology

The `RationaleAuditor` computes three scores per sample using `[CLS]` hidden states from the frozen fine-tuned model's final transformer layer. All hidden states are conditioned on the same hypothesis so comparisons across states are symmetric.

### Latent Alignment Score (LAS)

Cosine similarity between the model's decision-state embedding (premise + hypothesis) and the embedding of the human-written rationale (rationale + hypothesis):

```
LAS = cosine_similarity(CLS(premise, hypothesis), CLS(human_rationale, hypothesis))
```

Higher LAS indicates the model's internal representation is directionally closer to the human-grounded reasoning for that inference step.

### Causal Sensitivity Index (CSI)

The drop in LAS when the premise is adversarially perturbed by TextFooler — a greedy word-swap attack that finds the minimal lexical change needed to flip the model's predicted label:

```
CSI = LAS_original − LAS_perturbed
```

A **high CSI** means the model's latent state shifts when the logical evidence shifts — consistent with faithful, evidence-grounded reasoning. A **near-zero or negative CSI** means the internal state is insensitive to a premise change that *does* flip the output label, suggesting the model is relying on surface heuristics rather than the logical content of the premise.

CSI is only aggregated over samples where TextFooler found a successful label-flipping attack (`csi_valid=True`). Samples where the attack failed or the model was already wrong are excluded so that failed attacks do not artificially suppress the mean toward zero.

### Mismatched LAS and Synthetic Sensitivity (SS)

For each sample, a rationale from a *different* test example is drawn and its LAS is also computed as a control. **Synthetic Sensitivity (SS)** is the fraction of these mismatched-rationale scores that fall below the UDR threshold (0.30). A healthy `SS < UDR` confirms the unfaithfulness signal is not simply geometric noise from the embedding space.

### TextFooler Implementation

TextFooler is assembled using a `ThoughtVector` sentence-level constraint instead of the default `UniversalSentenceEncoder`, avoiding the `tensorflow_hub` dependency while preserving the original Jin et al. attack recipe: counter-fitted embedding swaps, greedy word importance ranking (delete WIR), stop-word and repeat-modification guards, and an untargeted label-flip objective.

---

## References

- Nie et al. (2020). [Adversarial NLI: A New Benchmark for Natural Language Understanding](https://arxiv.org/abs/1910.14599)
- Liu et al. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- Hsieh et al. (2023). [Distilling Step-by-Step!](https://arxiv.org/abs/2305.02301)
- Jin et al. (2019). [Is BERT Really Robust? TextFooler](https://arxiv.org/abs/1907.11932)
- Belinkov & Glass (2017). [Analysis Methods in Neural NLP](https://arxiv.org/abs/1812.08951)
- Geiger et al. (2021). [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997)

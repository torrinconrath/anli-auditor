# ANLI Rationale Auditor

A suite of NLP interpretability techniques for evaluating whether a fine-tuned RoBERTa-large model's internal reasoning aligns with human-grounded rationales on the Adversarial Natural Language Inference (ANLI) dataset.

This project investigates the gap between **faithfulness** and **plausibility** in transformer models — specifically, whether the rationales a model implicitly relies on actually reflect its internal decision-making process, or are merely post-hoc justifications.

---

## Project Structure

```
anli-auditor/
├── main.py                  # Entry point — runs training, evaluation, and audit
├── requirements.txt
└── src/
    ├── config.py            # All hyperparameters and paths
    ├── data_handler.py      # ANLI dataset loading and splits
    ├── model_handler.py     # RoBERTa-large model and tokenizer loading
    ├── trainer.py           # Fine-tuning loop
    ├── evaluate.py          # F1, accuracy, ECE metrics on R3 test set
    └── auditor.py           # LAS and CSI rationale audit
```

---

## Setup

### Requirements

- Python 3.11+
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4070 12GB)
- CUDA 12.1
- Windows 10/11 or Linux

### Installation

```powershell
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Full Pipeline (train + evaluate + audit)

```powershell
python main.py
```

This runs three stages in sequence:
1. Fine-tunes RoBERTa-large on 10k ANLI samples (~20-25 minutes on RTX 4070)
2. Evaluates on the held-out R3 test set (1200 samples, ~2 minutes)
3. Runs the rationale audit on 100 test samples (~5 minutes)

Results are saved to `./results_anli_roberta/`:
- `eval_results.json` — accuracy, macro F1, ECE, per-class report, confusion matrix
- `audit_results.json` — LAS and CSI scores per sample and averages

### Evaluation Only (skip training)

Set `DO_FINETUNING = False` in `src/config.py`, then:

```powershell
python main.py
```

This loads the saved model from `./results_anli_roberta/final_model` and runs evaluation and audit only.

### Quick Evaluation on a Subset

In `src/evaluate.py`, set:

```python
EVAL_SAMPLE_LIMIT = 100  # evaluate on first 100 samples only
```

---

## Configuration

All key settings are in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `roberta-large` | Base model from HuggingFace |
| `OUTPUT_DIR` | `./results_anli_roberta` | Where models and results are saved |
| `DO_FINETUNING` | `True` | Set to `False` to skip training |
| `MAX_TRAIN_SAMPLES` | `10000` | Training samples drawn from R1+R2+R3 |
| `MAX_VAL_SAMPLES` | `1000` | Validation samples from R1+R2 dev sets |
| `TRAIN_EPOCHS` | `5` | Number of training epochs |
| `BATCH_SIZE` | `16` | Per-device batch size |
| `LEARNING_RATE` | `2e-5` | Standard for RoBERTa NLI fine-tuning |
| `MAX_SEQ_LENGTH` | `512` | Max token length for premise+hypothesis |
| `NUM_AUDIT_SAMPLES` | `100` | Samples used for LAS/CSI audit |

---

## Dataset

The [Adversarial NLI (ANLI)](https://github.com/facebookresearch/anli) dataset by Meta is loaded automatically via HuggingFace Datasets. No manual download required.

**Split strategy:**

| Split | Source | Purpose |
|---|---|---|
| Train | R1 + R2 + R3 train (subsampled) | Fine-tuning |
| Validation | R1 + R2 dev | Loss monitoring during training |
| Test | R3 dev (held-out) | Final evaluation and audit |

R3 is the hardest adversarial round and is kept completely held-out from training.

---

## Audit Methodology

The `RationaleAuditor` computes two metrics on the held-out test set using the fine-tuned model's internal representations:

### Latent Alignment Score (LAS)

Cosine similarity between the model's `[CLS]` hidden state encoding the premise+hypothesis pair and the `[CLS]` hidden state encoding the human-provided rationale from ANLI.

```
LAS = cosine_similarity(CLS(premise, hypothesis), CLS(human_rationale))
```

Higher LAS indicates the model's internal decision representation is semantically closer to the human-grounded reasoning.

### Causal Sensitivity Index (CSI)

The drop in LAS when the premise is adversarially perturbed via keyword substitution (TextFooler):

```
CSI = LAS_original - LAS_perturbed
```

Higher CSI indicates the model's internal state is sensitive to changes in the logical evidence — consistent with faithful reasoning. A near-zero CSI suggests the internal state is unchanged despite a shifted premise, indicating post-hoc rationalization.

---

## Expected Results

On ANLI R3 dev set with the default configuration:

| Metric | Expected Range |
|---|---|
| Accuracy | 55–65% |
| Macro F1 | 53–63% |
| ECE | 0.05–0.15 |

These results exceed the standard fine-tuning baseline (~43–50%) reported in [Distilling Step-by-Step (Hsieh et al., 2023)](https://arxiv.org/abs/2305.02301) on ANLI, which used a 220M T5 model on R1 only.

---

## References

- Nie et al. (2020). [Adversarial NLI: A New Benchmark for Natural Language Understanding](https://arxiv.org/abs/1910.14599)
- Liu et al. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- Hsieh et al. (2023). [Distilling Step-by-Step!](https://arxiv.org/abs/2305.02301)
- Jin et al. (2019). [Is BERT Really Robust? TextFooler](https://arxiv.org/abs/1907.11932)
- Belinkov & Glass (2017). [Analysis Methods in Neural NLP](https://arxiv.org/abs/1812.08951)
- Geiger et al. (2021). [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997)
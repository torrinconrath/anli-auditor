# src/evaluate.py

import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from . import config

LABEL_MAP = config.ID2LABEL
LABEL_TO_ID = config.LABEL2ID
ID_ORDER = [0, 1, 2]

EVAL_SAMPLE_LIMIT = None  # set to int for quick runs, None for full set


def expected_calibration_error(
    confidences: list, correct: list, n_bins: int = 10
) -> float:
    confidences = np.array(confidences)
    correct = np.array(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece / len(confidences))


@torch.no_grad()
def run_inference(model, tokenizer, sample: dict) -> tuple:
    """
    Returns (predicted_label_id, confidence).
    RoBERTa produces direct class logits — no output parsing needed.
    """
    inputs = tokenizer(
        sample["premise"],
        sample["hypothesis"],
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    predicted_id = probs.argmax().item()
    confidence = probs[predicted_id].item()

    return predicted_id, confidence


def run_evaluation(model, tokenizer, test_dataset) -> dict:
    print("\n--- Running Evaluation on ANLI Test Set ---")

    model.eval()

    if EVAL_SAMPLE_LIMIT and EVAL_SAMPLE_LIMIT < len(test_dataset):
        test_dataset = test_dataset.select(range(EVAL_SAMPLE_LIMIT))
        print(f"Running on {EVAL_SAMPLE_LIMIT} samples (set EVAL_SAMPLE_LIMIT=None for full set)")

    y_true, y_pred, confidences, correct_flags = [], [], [], []

    for sample in tqdm(test_dataset, desc="Evaluating"):
        predicted_id, confidence = run_inference(model, tokenizer, sample)

        y_true.append(sample["label"])
        y_pred.append(predicted_id)
        confidences.append(confidence)
        correct_flags.append(predicted_id == sample["label"])

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    ece = expected_calibration_error(confidences, correct_flags)
    cm = confusion_matrix(y_true, y_pred, labels=ID_ORDER)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS — ANLI Test Set")
    print("=" * 60)
    print(f"\nAccuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Macro F1  : {macro_f1:.4f}  ({macro_f1*100:.2f}%)")
    print(f"ECE       : {ece:.4f}  (lower is better)")

    print("\n--- Per-Class Report ---")
    print(classification_report(
        y_true, y_pred,
        target_names=["Entailment", "Neutral", "Contradiction"],
        digits=4,
    ))

    print("--- Confusion Matrix (rows=true, cols=pred) ---")
    print("                Entail  Neutral  Contradict")
    for i, row in enumerate(cm):
        print(f"  {LABEL_MAP[i]:<16} {row}")

    results = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "ece": round(ece, 4),
        "total_samples": len(test_dataset),
        "per_class": classification_report(
            y_true, y_pred,
            target_names=["Entailment", "Neutral", "Contradiction"],
            output_dict=True,
        ),
        "confusion_matrix": cm.tolist(),
    }

    out_path = f"{config.OUTPUT_DIR}/eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def compute_universal_metrics(
    las_scores: list,
    mismatched_las_scores: list,
    csi_scores_all: list,
    csi_valid_flags: list,
    udr_threshold: float = 0.30,
) -> dict:
    """
    Computes all universal and approach-specific metrics.

    UDR threshold is set to 0.30. Cosine similarity below 0.30 means the
    decision vector and rationale vector share less than 30% directional
    overlap — a principled geometric definition of misalignment that does
    not depend on the shape of the LAS distribution.

    0.50 or higher is too permissive: cosine similarity of 0.50 still
    indicates meaningful overlap between the two vectors, so flagging those
    samples as unfaithful would conflate moderate alignment with no alignment.
    0.80 or higher would flag the majority of samples including many that are
    genuinely well-aligned, making UDR uninterpretable as a faithfulness signal.

    CSI aggregation is restricted to samples where csi_valid=True (TextFooler
    found a label-flipping perturbation). Including failed attacks would
    suppress the mean toward zero artifactually.
    """
    las_array  = np.array(las_scores)
    mism_array = np.array(mismatched_las_scores)

    udr = round(float((las_array < udr_threshold).mean()), 4)
    ss  = round(float((mism_array < udr_threshold).mean()), 4)

    # CSI distribution (valid attacks only)
    valid_csi   = [s for s, v in zip(csi_scores_all, csi_valid_flags) if v]
    csi_skipped = int(sum(1 for v in csi_valid_flags if not v))

    if valid_csi:
        arr = np.array(valid_csi)
        csi_dist = {
            "mean":         round(float(arr.mean()), 4),
            "std":          round(float(arr.std()),  4),
            "median":       round(float(np.median(arr)), 4),
            "min":          round(float(arr.min()),  4),
            "max":          round(float(arr.max()),  4),
            "pct_negative": round(float((arr < 0).mean()), 4),
        }
    else:
        csi_dist = {}

    return {
        "udr": udr,
        "ss": ss,
        "csi_distribution": csi_dist,
        "csi_skipped_count": csi_skipped,
    }

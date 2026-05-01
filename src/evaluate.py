# src/evaluate.py

import json
import numpy as np
import torch
import scipy.stats
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
    print(f"ECE       : {ece:.4f}  (lower is better; "
          "high ECE = miscalibration, not necessarily heuristic reliance)")

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


def compute_csi_distribution(csi_scores_valid: list) -> dict:
    """
    Computes distribution statistics for CSI across samples where the augmenter
    produced a genuinely changed premise (csi_valid=True).

    Reporting the mean alone is misleading: CSI can be negative (perturbation
    accidentally increases alignment) and near-zero values may reflect high
    variance cancelling out rather than true insensitivity. The distribution
    gives a fuller picture.
    """
    if not csi_scores_valid:
        return {}

    arr = np.array(csi_scores_valid)
    return {
        "mean":   round(float(arr.mean()), 4),
        "std":    round(float(arr.std()),  4),
        "min":    round(float(arr.min()),  4),
        "max":    round(float(arr.max()),  4),
        "pct_negative": round(float((arr < 0).mean()), 4),
        "median": round(float(np.median(arr)), 4),
    }


def compute_universal_metrics(
    correct_flags: list,
    las_scores: list,
    mismatched_las_scores: list,
    csi_scores_all: list,
    csi_valid_flags: list,
    udr_thresholds: tuple = (0.20, 0.30, 0.40, 0.50),
) -> dict:
    """
    Computes all universal and approach-specific metrics.

    Returns a dict containing:
      - udr_by_threshold : UDR computed at each threshold in udr_thresholds,
                           so readers can see sensitivity to the cutoff choice.
      - udr              : UDR at the canonical threshold (0.30) for reporting.
      - afc, p_value     : Pearson r between correctness and LAS, with p-value.
      - ss               : Synthetic Sensitivity at the canonical threshold.
      - csi_distribution : Full distribution stats for valid CSI samples only.
      - csi_skipped_count: Number of samples excluded from CSI (augmenter no-op).

    Design notes
    ------------
    UDR threshold ablation
        The 0.30 threshold for flagging a sample as "unfaithful" is one
        reasonable choice given the LAS distribution, but its impact on UDR
        is non-trivial. Reporting UDR at 0.20, 0.30, 0.40, and 0.50 makes
        threshold sensitivity explicit and allows cross-approach comparison
        at whichever cutoff each team chose.

    AFC p-value
        A Pearson correlation without a significance test is uninterpretable
        for n=100. The p-value is reported alongside r so the reader knows
        whether the correlation is statistically meaningful.

    CSI valid-only aggregation
        Samples where the augmenter returned the original premise unchanged
        are excluded from CSI statistics (csi_valid_flags=False). Including
        them would suppress the mean toward zero artifactually.
    """
    las_array     = np.array(las_scores)
    correct_array = np.array(correct_flags, dtype=int)

    # --- UDR at multiple thresholds ---
    udr_by_threshold = {}
    for t in udr_thresholds:
        rate = float((las_array < t).mean())
        udr_by_threshold[str(t)] = round(rate, 4)
    canonical_threshold = 0.30
    udr = udr_by_threshold[str(canonical_threshold)]

    # --- AFC (Pearson r between correctness and LAS) ---
    if len(np.unique(correct_array)) > 1 and len(np.unique(las_array)) > 1:
        afc, p_value = scipy.stats.pearsonr(correct_array, las_array)
        afc     = float(afc)
        p_value = float(p_value)
    else:
        afc, p_value = 0.0, 1.0

    # --- Synthetic Sensitivity ---
    mism_array = np.array(mismatched_las_scores)
    ss = round(float((mism_array < canonical_threshold).mean()), 4)

    # --- CSI distribution (valid samples only) ---
    valid_csi = [
        score for score, valid in zip(csi_scores_all, csi_valid_flags) if valid
    ]
    csi_dist = compute_csi_distribution(valid_csi)
    csi_skipped = int(sum(1 for v in csi_valid_flags if not v))

    # --- Print summary ---
    print("\n--- Universal Metrics ---")
    print(f"UDR (threshold sensitivity):")
    for t, val in udr_by_threshold.items():
        marker = " <-- canonical" if float(t) == canonical_threshold else ""
        print(f"  threshold={t}: UDR={val:.4f}{marker}")
    print(f"AFC (Pearson r): {afc:.4f}  (p={p_value:.4f})")
    print(f"SS              : {ss:.4f}")

    print("\n--- CSI Distribution (valid perturbations only) ---")
    if csi_dist:
        print(f"  Mean   : {csi_dist['mean']:.4f}")
        print(f"  Std    : {csi_dist['std']:.4f}")
        print(f"  Median : {csi_dist['median']:.4f}")
        print(f"  Min    : {csi_dist['min']:.4f}")
        print(f"  Max    : {csi_dist['max']:.4f}")
        print(f"  % Negative : {csi_dist['pct_negative']:.2%}  "
              "(negative = perturbation accidentally increased alignment)")
    print(f"  Skipped (augmenter no-op): {csi_skipped} samples")

    return {
        "udr": udr,
        "udr_by_threshold": udr_by_threshold,
        "afc": round(afc, 4),
        "p_value": round(p_value, 4),
        "ss": ss,
        "csi_distribution": csi_dist,
        "csi_skipped_count": csi_skipped,
    }
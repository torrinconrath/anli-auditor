# src/evaluate.py

import re
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

LABEL_MAP = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}
ID_ORDER = [0, 1, 2]

# How many samples to evaluate — set to None to run the full test set
EVAL_SAMPLE_LIMIT = 100


def build_inference_prompt(sample: dict) -> str:
    return (
        f"Analyze the following premise and hypothesis to determine the relationship. "
        f"First, provide a step-by-step rationale, and then conclude with the final "
        f"label (Entailment, Neutral, or Contradiction).\n\n"
        f"### Premise:\n{sample['premise']}\n\n"
        f"### Hypothesis:\n{sample['hypothesis']}\n\n"
        f"### Rationale:\n"
    )


def extract_label(generated_text: str) -> str | None:
    # Primary: structured ### Label: header
    match = re.search(
        r"###\s*Label\s*:\s*(Entailment|Neutral|Contradiction)",
        generated_text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).capitalize()

    # Secondary: ### Conclusion: or "final label is X" patterns
    match = re.search(
        r"(?:###\s*Conclusion\s*:|final label is)\s*(Entailment|Neutral|Contradiction)",
        generated_text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).capitalize()

    # Tertiary: last label word anywhere in output
    words = re.findall(
        r"\b(Entailment|Neutral|Contradiction)\b", generated_text, re.IGNORECASE
    )
    if words:
        return words[-1].capitalize()

    return None


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
    prompt = build_inference_prompt(sample)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,     # was 20 — need room for rationale + label line
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Debug: print first 3 raw outputs to verify format, then remove
    if not hasattr(run_inference, "_debug_count"):
        run_inference._debug_count = 0
    if run_inference._debug_count < 3:
        print(f"\n[Debug {run_inference._debug_count}] Generated:\n{generated}\n")
        run_inference._debug_count += 1

    predicted = extract_label(generated)

    # Confidence from label token logits at last prompt position
    logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)

    label_token_ids = {}
    for label_str in ["Entailment", "Neutral", "Contradiction"]:
        toks = tokenizer.encode(label_str, add_special_tokens=False)
        if toks:
            label_token_ids[label_str] = toks[0]

    label_probs = {
        label: probs[tid].item()
        for label, tid in label_token_ids.items()
    }
    total = sum(label_probs.values())
    if total > 0:
        label_probs = {k: v / total for k, v in label_probs.items()}

    confidence = label_probs.get(predicted, 0.0) if predicted else 0.0
    return predicted, confidence


def run_evaluation(model, tokenizer, test_dataset) -> dict:
    print("\n--- Running Evaluation on ANLI R3 Test Set ---")

    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"

    # Limit samples for quick runs — set EVAL_SAMPLE_LIMIT=None for full eval
    if EVAL_SAMPLE_LIMIT and EVAL_SAMPLE_LIMIT < len(test_dataset):
        test_dataset = test_dataset.select(range(EVAL_SAMPLE_LIMIT))
        print(f"Running on {EVAL_SAMPLE_LIMIT} samples (set EVAL_SAMPLE_LIMIT=None for full set)")

    y_true, y_pred = [], []
    confidences, correct_flags = [], []
    unparseable = 0

    for sample in tqdm(test_dataset, desc="Evaluating"):
        true_label_str = LABEL_MAP[sample["label"]]
        predicted, confidence = run_inference(model, tokenizer, sample)

        if predicted is None or predicted not in LABEL_TO_ID:
            unparseable += 1
            predicted = "Neutral"
            confidence = 0.0

        y_true.append(sample["label"])
        y_pred.append(LABEL_TO_ID[predicted])
        confidences.append(confidence)
        correct_flags.append(predicted == true_label_str)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    ece = expected_calibration_error(confidences, correct_flags)
    cm = confusion_matrix(y_true, y_pred, labels=ID_ORDER)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS — ANLI R3 Dev Set")
    print("=" * 60)
    print(f"\nUnparseable outputs : {unparseable}/{len(test_dataset)} "
          f"({100*unparseable/len(test_dataset):.1f}%)")
    print(f"Accuracy            : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Macro F1            : {macro_f1:.4f}  ({macro_f1*100:.2f}%)")
    print(f"ECE                 : {ece:.4f}  (lower is better)")

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
        "unparseable": unparseable,
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

    model.config.use_cache = False
    tokenizer.padding_side = "right"

    return results

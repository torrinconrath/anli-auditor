# main.py

import json
import logging
import random
import warnings
import numpy as np
from tqdm import tqdm

# Suppress noisy TextAttack compatibility warnings and ensure NLTK data
# required by TextFooler is present. Kept at the entry point so auditor.py
# stays free of startup side effects.
logging.getLogger("textattack").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Unknown if model.*")

import nltk
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from src import config
from src.data_handler import prepare_anli_dataset
from src.model_handler import load_model_and_tokenizer
from src.trainer import fine_tune_model
from src.evaluate import run_evaluation, compute_universal_metrics
from src.auditor import RationaleAuditor


def main():
    """Main function to run the complete workflow."""

    # 1. Load and prepare data
    train_data, val_data, test_data, label_map = prepare_anli_dataset()

    # 2. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 3. Fine-tune if enabled
    if config.DO_FINETUNING:
        fine_tune_model(model, tokenizer, train_data, val_data)
    else:
        print(f"Skipping training. Loading from {config.OUTPUT_DIR}/final_model")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            f"{config.OUTPUT_DIR}/final_model"
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(f"{config.OUTPUT_DIR}/final_model")

    # 4. Evaluate on held-out test set
    eval_results = run_evaluation(model, tokenizer, test_data)

    # 5. Run rationale audit
    print(f"\n--- Starting Audit on {config.NUM_AUDIT_SAMPLES} Test Samples ---")
    auditor = RationaleAuditor(model, tokenizer)

    latent_alignment_scores = []
    causal_sensitivity_all  = []  # raw CSI for every sample (0.0 where attack failed)
    csi_valid_flags         = []  # True only where TextFooler found a successful attack
    mismatched_las_scores   = []

    audit_subset = test_data.select(range(min(config.NUM_AUDIT_SAMPLES, len(test_data))))

    # Draw mismatched rationales from the full test set (not just the audit
    # subset) to maximise semantic distance from the current sample's rationale.
    full_rationale_pool = test_data["reason"]

    for i, sample in enumerate(tqdm(audit_subset, desc="Auditing Samples")):
        try:
            # Sample a rationale that provably does not belong to this example.
            for _ in range(20):
                mism_idx = random.randint(0, len(full_rationale_pool) - 1)
                candidate = full_rationale_pool[mism_idx]
                if candidate != sample["reason"]:
                    mismatched_rationale = candidate
                    break
            else:
                mismatched_rationale = full_rationale_pool[
                    (i + len(audit_subset) // 2) % len(full_rationale_pool)
                ]

            las, csi, mism_las, csi_valid = auditor.run_single_audit(
                sample, mismatched_rationale
            )

            latent_alignment_scores.append(las)
            causal_sensitivity_all.append(csi)
            csi_valid_flags.append(csi_valid)
            mismatched_las_scores.append(mism_las)

        except Exception as e:
            print(f"Skipping sample {i} due to error: {e}")

    # 6. Report and save audit results
    if latent_alignment_scores:
        avg_las = np.mean(latent_alignment_scores)

        valid_csi_scores = [s for s, v in zip(causal_sensitivity_all, csi_valid_flags) if v]
        avg_csi_valid = np.mean(valid_csi_scores) if valid_csi_scores else float("nan")

        metrics = compute_universal_metrics(
            las_scores=latent_alignment_scores,
            mismatched_las_scores=mismatched_las_scores,
            csi_scores_all=causal_sensitivity_all,
            csi_valid_flags=csi_valid_flags,
        )

        n_valid = sum(csi_valid_flags)
        n_total = len(csi_valid_flags)

        print("\n--- Audit Results ---")
        print(f"Average LAS (all samples)        : {avg_las:.4f}")
        print(f"Average CSI (valid attacks only) : {avg_csi_valid:.4f}"
              f"  ({n_valid}/{n_total} samples had a successful TextFooler attack)")

        print("\n--- Universal Metrics ---")
        print(f"UDR (threshold=0.30)             : {metrics['udr']:.4f}")
        print(f"SS                               : {metrics['ss']:.4f}")

        if metrics.get("csi_distribution"):
            dist = metrics["csi_distribution"]
            print(f"\n--- CSI Distribution (n={n_valid} valid attacks) ---")
            print(f"  Mean   : {dist['mean']:.4f}   Std  : {dist['std']:.4f}")
            print(f"  Median : {dist['median']:.4f}   Min  : {dist['min']:.4f}   Max: {dist['max']:.4f}")
            print(f"  % Negative (perturbation accidentally raised alignment): {dist['pct_negative']:.2%}")
        print(f"  Skipped (TextFooler found no flip): {metrics['csi_skipped_count']} samples")

        audit_results = {
            "avg_las": round(float(avg_las), 4),
            "avg_csi_valid_only": round(float(avg_csi_valid), 4) if not np.isnan(avg_csi_valid) else None,
            "udr": metrics["udr"],
            "ss": metrics["ss"],
            "csi_distribution": metrics["csi_distribution"],
            "csi_skipped_count": metrics["csi_skipped_count"],
            "las_scores": [round(s, 4) for s in latent_alignment_scores],
            "csi_scores_all": [round(s, 4) for s in causal_sensitivity_all],
            "csi_valid_flags": csi_valid_flags,
            "num_samples": len(latent_alignment_scores),
        }

        out_path = f"{config.OUTPUT_DIR}/audit_results.json"
        with open(out_path, "w") as f:
            json.dump(audit_results, f, indent=2)
        print(f"\nAudit results saved to {out_path}")

    else:
        print("Audit could not be completed. No scores were generated.")


if __name__ == "__main__":
    main()

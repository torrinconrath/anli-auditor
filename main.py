# main.py

import json
import logging
import random
import warnings
import numpy as np
from tqdm import tqdm

# --- Startup: suppress noisy TextAttack compatibility warnings and
#     ensure the NLTK POS tagger required by TextFooler is present.
#     These belong here (the entry point) rather than in auditor.py so
#     that module-level imports stay free of side effects.
logging.getLogger("textattack").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Unknown if model.*")

import nltk
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
# WordNet is also required by TextFooler's counter-fitted embeddings lookup
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4",  quiet=True)

from src import config
from src.data_handler import prepare_anli_dataset
from src.model_handler import load_model_and_tokenizer
from src.trainer import fine_tune_model
from src.evaluate import run_evaluation, run_inference, compute_universal_metrics
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
    print(   "    (TextFooler runs ~10-30 model queries per sample — expect ~15-30 min on GPU)")
    auditor = RationaleAuditor(model, tokenizer)

    latent_alignment_scores = []
    causal_sensitivity_all  = []  # raw CSI for every sample (0.0 where attack failed)
    csi_valid_flags         = []  # True only where TextFooler found a successful attack
    mismatched_las_scores   = []
    correct_flags           = []

    audit_subset = test_data.select(range(min(config.NUM_AUDIT_SAMPLES, len(test_data))))

    # Mismatched rationale pool: draw from the *full* test set (not just the
    # audit subset) to maximise semantic distance from the current sample.
    full_rationale_pool = test_data["reason"]

    for i, sample in enumerate(tqdm(audit_subset, desc="Auditing Samples")):
        try:
            # Draw a mismatched rationale that is not the current sample's own.
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

            # Track correctness for AFC
            predicted_id, _ = run_inference(model, tokenizer, sample)
            correct_flags.append(predicted_id == sample["label"])

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

        valid_csi_scores = [
            s for s, v in zip(causal_sensitivity_all, csi_valid_flags) if v
        ]
        avg_csi_valid = np.mean(valid_csi_scores) if valid_csi_scores else float("nan")

        metrics = compute_universal_metrics(
            correct_flags=correct_flags,
            las_scores=latent_alignment_scores,
            mismatched_las_scores=mismatched_las_scores,
            csi_scores_all=causal_sensitivity_all,
            csi_valid_flags=csi_valid_flags,
        )

        print("\n--- Audit Results ---")
        print(f"Average LAS (all samples)         : {avg_las:.4f}")
        print(f"Average CSI (valid attacks only)  : {avg_csi_valid:.4f}  "
              f"({sum(csi_valid_flags)}/{len(csi_valid_flags)} samples had a successful TextFooler attack)")

        print("\n--- Interpretation ---")
        print(
            "LAS  : Higher is better. The model's [CLS] representation is semantically\n"
            "       closer to the human rationale embedding. Note: [CLS] in a fine-tuned\n"
            "       classifier is optimised for linear separability, so cosine similarity\n"
            "       is an imperfect but informative proxy for semantic alignment."
        )
        print(
            "CSI  : Higher is better. A positive CSI means the model's internal state\n"
            "       shifted after TextFooler found a meaning-inverting perturbation of\n"
            "       the premise, consistent with faithful evidence tracking. Near-zero\n"
            "       CSI means the internal representation is rigid — a heuristic signature."
        )
        print(
            "UDR  : Unfaithfulness Detection Rate. Percentage of decisions flagged as\n"
            "       unfaithful (LAS below threshold). See udr_by_threshold for sensitivity."
        )
        print(
            f"AFC  : Accuracy-Faithfulness Correlation (Pearson r={metrics['afc']:.4f},\n"
            f"       p={metrics['p_value']:.4f}). Positive means higher LAS predicts\n"
            "       higher correctness — alignment with human logic tracks accuracy."
        )
        print(
            "SS   : Synthetic Sensitivity. Fraction of deliberately-mismatched rationales\n"
            "       successfully flagged. Higher is better; should exceed UDR to confirm\n"
            "       the auditor is not producing random noise."
        )

        audit_results = {
            "avg_las": round(float(avg_las), 4),
            "avg_csi_valid_only": round(float(avg_csi_valid), 4) if not np.isnan(avg_csi_valid) else None,
            "udr": metrics["udr"],
            "udr_by_threshold": metrics["udr_by_threshold"],
            "afc": metrics["afc"],
            "afc_p_value": metrics["p_value"],
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
    
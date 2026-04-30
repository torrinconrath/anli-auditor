# main.py

import json
import random 
import numpy as np
from tqdm import tqdm

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

    # 4. Evaluate on held-out R3 test set
    eval_results = run_evaluation(model, tokenizer, test_data)

    # 5. Run rationale audit
    print(f"\n--- Starting Audit on {config.NUM_AUDIT_SAMPLES} Test Samples ---")
    auditor = RationaleAuditor(model, tokenizer)

    latent_alignment_scores = []
    causal_sensitivity_indices = []
    mismatched_las_scores = []
    correct_flags = []

    audit_subset = test_data.select(range(min(config.NUM_AUDIT_SAMPLES, len(test_data))))

    # Pre-extract rationales for the mismatched SS control group
    all_rationales = audit_subset["reason"]

    for i, sample in enumerate(tqdm(audit_subset, desc="Auditing Samples")):
        try:
            # Grab a random rationale from a different index for Synthetic Sensitivity
            mismatched_idx = (i + random.randint(1, len(all_rationales)-1)) % len(all_rationales)
            mismatched_rationale = all_rationales[mismatched_idx]

            # Track if model gets this sample correct (needed for AFC)
            predicted_id, _ = run_inference(model, tokenizer, sample)
            is_correct = (predicted_id == sample["label"])
            correct_flags.append(is_correct)

            las, csi, mism_las = auditor.run_single_audit(sample, mismatched_rationale)

            latent_alignment_scores.append(las)
            causal_sensitivity_indices.append(csi)
            mismatched_las_scores.append(mism_las)
        except Exception as e:
            print(f"Skipping sample due to error: {e}")

    # 6. Report and save audit results
    if latent_alignment_scores and causal_sensitivity_indices:
        avg_las = np.mean(latent_alignment_scores)
        avg_csi = np.mean(causal_sensitivity_indices)

        # 7. Calculate Universal Metrics
        udr, afc, ss = compute_universal_metrics(
            correct_flags, latent_alignment_scores, mismatched_las_scores, udr_threshold=0.30
        )

        print("\n--- Audit Results ---")
        print(f"Average Latent Alignment Score (LAS): {avg_las:.4f}")
        print(f"Average Causal Sensitivity Index (CSI): {avg_csi:.4f}")

        print("\n--- Universal Metrics ---")
        print(f"Unfaithfulness Detection Rate (UDR): {udr:.4f}")
        print(f"Accuracy-Faithfulness Correlation (AFC): {afc:.4f}")
        print(f"Synthetic Sensitivity (SS): {ss:.4f}")

        print("\n--- Interpretation ---")
        print("LAS: Higher is better. Indicates the model's [CLS] representation "
              "is semantically closer to the human rationale embedding.")
        print("CSI: Higher is better. Indicates the model's internal state shifts "
              "when key evidence is perturbed, consistent with faithful reasoning.")
        print("UDR: Unfaithfulness Detection Rate. Indicates the percentage of decisions "
              "flagged as unfaithful due to low alignment with human logic (LAS < 0.30).")
        print("AFC: Accuracy-Faithfulness Correlation. Positive is better. Shows that when "
              "the model's internal state matches human logic, it is more likely to be correct.")
        print("SS : Synthetic Sensitivity. Higher is better. Measures the auditor's ability "
              "to successfully catch and flag completely mismatched/corrupted rationales.")

        audit_results = {
            "avg_las": round(float(avg_las), 4),
            "avg_csi": round(float(avg_csi), 4),
            "udr": round(float(udr), 4),
            "afc": round(float(afc), 4),
            "ss": round(float(ss), 4),
            "las_scores": [round(s, 4) for s in latent_alignment_scores],
            "csi_scores": [round(s, 4) for s in causal_sensitivity_indices],
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
    
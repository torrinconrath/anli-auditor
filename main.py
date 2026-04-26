# main.py

import json
import numpy as np
from tqdm import tqdm

from src import config
from src.data_handler import prepare_anli_dataset
from src.model_handler import load_model_and_tokenizer
from src.trainer import fine_tune_model
from src.evaluate import run_evaluation
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

    audit_subset = test_data.select(range(min(config.NUM_AUDIT_SAMPLES, len(test_data))))

    for sample in tqdm(audit_subset, desc="Auditing Samples"):
        try:
            las, csi = auditor.run_single_audit(sample)
            latent_alignment_scores.append(las)
            causal_sensitivity_indices.append(csi)
        except Exception as e:
            print(f"Skipping sample due to error: {e}")

    # 6. Report and save audit results
    if latent_alignment_scores and causal_sensitivity_indices:
        avg_las = np.mean(latent_alignment_scores)
        avg_csi = np.mean(causal_sensitivity_indices)

        print("\n--- Audit Results ---")
        print(f"Average Latent Alignment Score (LAS): {avg_las:.4f}")
        print(f"Average Causal Sensitivity Index (CSI): {avg_csi:.4f}")
        print("\n--- Interpretation ---")
        print("LAS: Higher is better. Indicates the model's [CLS] representation "
              "is semantically closer to the human rationale embedding.")
        print("CSI: Higher is better. Indicates the model's internal state shifts "
              "when key evidence is perturbed, consistent with faithful reasoning.")

        audit_results = {
            "avg_las": round(float(avg_las), 4),
            "avg_csi": round(float(avg_csi), 4),
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
    
# main.py

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

    # 2. Load model, tokenizer, and PEFT config
    model, tokenizer, peft_config = load_model_and_tokenizer()

    # 3. Fine-tune the model if enabled
    if config.DO_FINETUNING:
        fine_tune_model(model, tokenizer, peft_config, train_data, val_data)
    else:
        print(f"Skipping training. Ensure an adapter exists at {config.OUTPUT_DIR}/final_adapter")

    # 4. Evaluate on the held-out R3 test set
    eval_results = run_evaluation(model, tokenizer, test_data)

    # 5. Initialize the auditor and run the audit
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

    # 5. Report results
    if latent_alignment_scores and causal_sensitivity_indices:
        avg_las = np.mean(latent_alignment_scores)
        avg_csi = np.mean(causal_sensitivity_indices)

        print("\n--- Audit Results ---")
        print(f"Average Latent Alignment Score (LAS): {avg_las:.4f}")
        print(f"Average Causal Sensitivity Index (CSI): {avg_csi:.4f}")
        print("\n--- Interpretation ---")
        print("LAS: Higher is better. Indicates the model's internal 'thought process' is semantically closer to the human rationale.")
        print("CSI: Higher is better. Indicates the model's internal state changes significantly when key evidence is perturbed, suggesting its reasoning is sensitive to the input logic.")
    else:
        print("Audit could not be completed. No scores were generated.")

if __name__ == "__main__":
    # Log in to Hugging Face - you'll be prompted for a token
    # Do this in your terminal first: huggingface-cli login
    main()
    
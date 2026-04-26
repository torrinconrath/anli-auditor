# src/data_handler.py

from datasets import load_dataset, concatenate_datasets
from . import config

def prepare_anli_dataset(max_train_samples=config.MAX_TRAIN_SAMPLES, max_val_samples=config.MAX_VAL_SAMPLES):
    """
    Loads and formats the ANLI dataset.

    Split strategy:
      Train  — R1+R2+R3 training splits (subsampled). R3 is hardest so including
               it exposes the model to the full difficulty spectrum.
      Val    — R1+R2 dev splits (easier) used for loss-based early-stopping /
               epoch selection during training.
      Test   — R3 dev split (hardest) used exclusively for final evaluation and
               the rationale audit. Kept completely held-out from training.
    """
    print("--- Loading and Preparing ANLI Dataset ---")

    # --- Training data ---
    train_r1 = load_dataset("anli", split="train_r1")
    train_r2 = load_dataset("anli", split="train_r2")
    train_r3 = load_dataset("anli", split="train_r3")
    full_train = concatenate_datasets([train_r1, train_r2, train_r3])

    if max_train_samples and max_train_samples < len(full_train):
        full_train = full_train.shuffle(seed=42).select(range(max_train_samples))

    # --- Validation data (easier rounds, for training feedback) ---
    val_r1 = load_dataset("anli", split="dev_r1")
    val_r2 = load_dataset("anli", split="dev_r2")
    full_val = concatenate_datasets([val_r1, val_r2])

    if max_val_samples and max_val_samples < len(full_val):
        full_val = full_val.shuffle(seed=42).select(range(max_val_samples))

    # --- Test data (hardest round, held-out) ---
    test_dataset = load_dataset("anli", split="dev_r3")

    label_map = {
        0: "Entailment",
        1: "Neutral",
        2: "Contradiction"
    }

    def format_prompt(sample):
        return {
            "text": f"""Analyze the following premise and hypothesis to determine the relationship. First, provide a step-by-step rationale, and then conclude with the final label (Entailment, Neutral, or Contradiction).

### Premise:
{sample['premise']}

### Hypothesis:
{sample['hypothesis']}

### Rationale:
{sample['reason']}

### Label:
{label_map[sample['label']]}"""
        }

    # Train/val: strip everything except text — raw ANLI columns are not needed.
    # Test: keep label, premise, hypothesis — the evaluator and auditor need them.
    train_val_cols_to_remove = [c for c in train_r1.column_names if c != "text"]
    test_cols_to_remove = [
        c for c in train_r1.column_names
        if c not in ("text", "label", "premise", "hypothesis")
    ]

    train_dataset = full_train.map(format_prompt, remove_columns=train_val_cols_to_remove)
    val_dataset   = full_val.map(format_prompt, remove_columns=train_val_cols_to_remove)
    test_dataset  = test_dataset.map(format_prompt, remove_columns=test_cols_to_remove)

    print(f"Train size:      {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size:       {len(test_dataset)}")

    print("\nExample Training Prompt:\n")
    print(train_dataset[0]["text"])

    return train_dataset, val_dataset, test_dataset, label_map

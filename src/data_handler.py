# src/data_handler.py

from datasets import load_dataset, concatenate_datasets
from . import config


def prepare_anli_dataset(
    max_train_samples=config.MAX_TRAIN_SAMPLES,
    max_val_samples=config.MAX_VAL_SAMPLES,
):
    """
    Loads and formats the ANLI dataset for RoBERTa sequence classification.

    Split strategy:
      Train  — R1+R2+R3 training splits (subsampled across all difficulty rounds).
      Val    — R1+R2 dev splits (easier) for loss-based epoch selection.
      Test   — R3 dev split (hardest) held-out for final evaluation and audit.
    """
    print("--- Loading and Preparing ANLI Dataset ---")

    train_r1 = load_dataset("anli", split="train_r1")
    train_r2 = load_dataset("anli", split="train_r2")
    train_r3 = load_dataset("anli", split="train_r3")
    full_train = concatenate_datasets([train_r1, train_r2, train_r3])

    if max_train_samples and max_train_samples < len(full_train):
        full_train = full_train.shuffle(seed=42).select(range(max_train_samples))

    val_r1 = load_dataset("anli", split="dev_r1")
    val_r2 = load_dataset("anli", split="dev_r2")
    full_val = concatenate_datasets([val_r1, val_r2])

    if max_val_samples and max_val_samples < len(full_val):
        full_val = full_val.shuffle(seed=42).select(range(max_val_samples))

    test_dataset = load_dataset("anli", split="dev_r3")

    label_map = config.ID2LABEL

    # RoBERTa takes premise and hypothesis as a sentence pair — no prompt needed.
    # The 'reason' and 'label' fields are kept for the auditor and evaluator.
    # Train/val only need input_ids and the integer label.
    keep_for_test = {"premise", "hypothesis", "label", "reason"}
    keep_for_train = {"premise", "hypothesis", "label"}

    train_cols_to_remove = [c for c in train_r1.column_names if c not in keep_for_train]
    val_cols_to_remove   = [c for c in val_r1.column_names   if c not in keep_for_train]
    test_cols_to_remove  = [c for c in test_dataset.column_names if c not in keep_for_test]

    train_dataset = full_train.remove_columns(train_cols_to_remove)
    val_dataset   = full_val.remove_columns(val_cols_to_remove)
    test_dataset  = test_dataset.remove_columns(test_cols_to_remove)

    print(f"Train size:      {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size:       {len(test_dataset)}")

    print("\nExample training sample:")
    print(f"  Premise:    {train_dataset[0]['premise'][:80]}...")
    print(f"  Hypothesis: {train_dataset[0]['hypothesis']}")
    print(f"  Label:      {label_map[train_dataset[0]['label']]}")

    return train_dataset, val_dataset, test_dataset, label_map
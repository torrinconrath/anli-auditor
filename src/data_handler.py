# src/data_handler.py

from datasets import load_dataset, concatenate_datasets

def prepare_anli_dataset():
    """Loads and formats the ANLI dataset for Explain-then-Predict."""
    print("--- Loading and Preparing ANLI Dataset ---")
    
    # Load all three rounds of ANLI
    anli_r1 = load_dataset("anli", split="train_r1")
    anli_r2 = load_dataset("anli", split="train_r2")
    anli_r3 = load_dataset("anli", split="train_r3")
    
    train_dataset = concatenate_datasets([anli_r1, anli_r2, anli_r3])
    test_dataset = load_dataset("anli", split="dev_r3")

    label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

    def format_prompt(sample):
        prompt = f"""Analyze the following premise and hypothesis to determine the relationship. First, provide a step-by-step rationale, and then conclude with the final label (Entailment, Neutral, or Contradiction).

### Premise:
{sample['premise']}

### Hypothesis:
{sample['hypothesis']}

### Rationale:
{sample['reason']}

### Label:
{label_map[sample['label']]}"""
        return {"text": prompt}

    train_dataset = train_dataset.map(format_prompt)
    print(f"Formatted {len(train_dataset)} training examples.")
    print("\nExample Training Prompt:")
    print(train_dataset[0]['text'])
    
    return train_dataset, test_dataset, label_map

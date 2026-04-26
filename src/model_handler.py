# src/model_handler.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from . import config


def load_model_and_tokenizer():
    """
    Loads RoBERTa-large with a 3-class classification head.
    No quantization or LoRA needed — 355M parameters fits easily in full
    precision on the RTX 4070's 12GB VRAM.
    """
    print(f"\n--- Loading Model: {config.MODEL_NAME} ---")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        label2id=config.LABEL2ID,
        id2label=config.ID2LABEL,
    )

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {next(model.parameters()).device}")

    return model, tokenizer
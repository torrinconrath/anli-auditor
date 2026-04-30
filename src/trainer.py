# src/trainer.py

import os
import torch
import numpy as np
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import f1_score, accuracy_score
from . import config


def fine_tune_model(model, tokenizer, train_dataset, val_dataset):
    """
    Fine-tunes RoBERTa-large for NLI classification.
    50k samples, 3 epochs, lower LR — tuned to prevent the overfitting
    seen with 10k/5 epochs (val_loss 3.52, test F1 37%).
    Expected runtime: ~35-45 minutes on RTX 4070.
    """
    print("\n--- Starting Fine-Tuning ---")

    def tokenize_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val   = val_dataset.map(tokenize_function, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    # 50000 / 16 = 3125 steps/epoch × 3 epochs = 9375 total steps
    # Eval once per epoch
    STEPS_PER_EPOCH = len(tokenized_train) // config.BATCH_SIZE

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        logging_dir=config.LOGGING_DIR,

        num_train_epochs=config.TRAIN_EPOCHS,

        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,

        learning_rate=config.LEARNING_RATE,
        weight_decay=0.01,          # L2 regularization helps generalization
        warmup_ratio=0.06,
        lr_scheduler_type="linear",

        eval_strategy="steps",
        eval_steps=STEPS_PER_EPOCH,
        save_strategy="steps",
        save_steps=STEPS_PER_EPOCH,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        logging_steps=100,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("--- Fine-Tuning Complete ---")

    model_path = os.path.join(config.OUTPUT_DIR, "final_model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

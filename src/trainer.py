# src/trainer.py

import os
from transformers import TrainingArguments
from transformers import Trainer
from . import config

def fine_tune_model(model, tokenizer, peft_config, train_dataset):
    """Fine-tunes the model using the provided datasets and configuration."""
    print("\n--- Starting Fine-Tuning ---")
    
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        logging_dir=config.LOGGING_DIR,
        num_train_epochs=config.TRAIN_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        optim="paged_adamw_32bit",
        save_steps=500,
        logging_steps=25,
        learning_rate=config.LEARNING_RATE,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
    )

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=config.MAX_SEQ_LENGTH,
        )

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    print("--- Fine-Tuning Complete ---")
    
    adapter_path = os.path.join(config.OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")
    
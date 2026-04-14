# src/trainer.py

import os
from transformers import TrainingArguments
from trl import SFTTrainer
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

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    print("--- Fine-Tuning Complete ---")
    
    adapter_path = os.path.join(config.OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")
    
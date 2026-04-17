# src/trainer.py

import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from . import config


def fine_tune_model(model, tokenizer, peft_config, train_dataset, val_dataset):
    """
    Fine-tunes the model using the standard HF Trainer with manual prompt masking.
    """
    print("\n--- Starting Fine-Tuning ---")

    # RTX 4070 (Ada Lovelace) supports bf16 natively — prefer it over fp16.
    # These must be mutually exclusive or TrainingArguments will crash.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # eval/save every 200 steps so we catch overfitting across all 3 epochs.
    # With 10k samples / batch 4 / grad accum 4 = 625 steps/epoch × 3 = 1875 total.
    # eval at 500 would only give 3 checkpoints; 200 gives ~9, much better signal.
    EVAL_SAVE_STEPS = 200

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        logging_dir=config.LOGGING_DIR,

        num_train_epochs=config.TRAIN_EPOCHS,

        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,

        optim="paged_adamw_32bit",

        fp16=use_fp16,
        bf16=use_bf16,

        learning_rate=config.LEARNING_RATE,
        weight_decay=0.001,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        logging_steps=25,
        save_steps=EVAL_SAVE_STEPS,
        eval_steps=EVAL_SAVE_STEPS,
        eval_strategy="steps",
        save_total_limit=3,           # keep best + 2 recent checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # group_by_length removed — conflicts with DataCollatorForSeq2Seq.
        # The collator handles per-batch padding correctly already.
    )

    def tokenize_function(examples):
        """
        Tokenizes a batch of examples and masks prompt tokens in labels so the
        loss is only computed on the label (Entailment/Neutral/Contradiction) tokens.
        """
        split_token = "### Label:"
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for text in examples["text"]:
            if split_token in text:
                prompt, label_text = text.split(split_token, 1)
                label_text = split_token + label_text
            else:
                prompt, label_text = text, ""

            prompt_ids = tokenizer(
                prompt,
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
                add_special_tokens=True,
            )["input_ids"]

            full = tokenizer(
                prompt + label_text,
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
                add_special_tokens=True,
            )

            input_ids = full["input_ids"]
            attention_mask = full["attention_mask"]

            # Mask prompt token positions — loss is computed only on label tokens.
            n_prompt = min(len(prompt_ids), len(input_ids))
            labels = [-100] * n_prompt + input_ids[n_prompt:]

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        }

    tokenized_train = train_dataset.map(
        tokenize_function, batched=True, remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        tokenize_function, batched=True, remove_columns=val_dataset.column_names
    )

    # Handles variable-length padding per batch and fills label padding with -100.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,   # replaces deprecated tokenizer= kwarg
        data_collator=data_collator,
    )

    trainer.train()
    print("--- Fine-Tuning Complete ---")

    adapter_path = os.path.join(config.OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")
    
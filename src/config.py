# src/config.py

# --- Model & Paths ---
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./results_anli_llama3"
LOGGING_DIR = "./logs"

# --- Auditing ---
NUM_AUDIT_SAMPLES = 100  # Number of test samples to audit

# --- Training Flags ---
DO_FINETUNING = True  # Set to False to skip training and use a pre-trained adapter

# --- Fine-Tuning Hyperparameters ---
TRAIN_EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024

# --- LoRA Configuration ---
# Rule of thumb: lora_alpha == lora_r gives a scaling factor of 1.0, which is
# the standard starting point. 64/16 under-scales the LoRA updates relative
# to the pre-trained weights and typically needs a higher LR to compensate.
LORA_R = 64
LORA_ALPHA = 64         
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
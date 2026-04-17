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
MAX_TRAIN_SAMPLES = 10000
MAX_VAL_SAMPLES = 1000
TRAIN_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 768

# --- LoRA Configuration ---
# Rule of thumb: lora_alpha == lora_r gives a scaling factor of 1.0, which is
# the standard starting point. 64/16 under-scales the LoRA updates relative
# to the pre-trained weights and typically needs a higher LR to compensate.
LORA_R = 16
LORA_ALPHA = 16         
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", # attention core reasoning modules
]
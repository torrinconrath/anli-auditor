# src/config.py

# --- Model & Paths ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "./results_anli_llama3"
LOGGING_DIR = "./logs"

# --- Auditing ---
NUM_AUDIT_SAMPLES = 100 # Number of test samples to audit

# --- Training Flags ---
DO_FINETUNING = True # Set to False to skip training and use a pre-trained adapter

# --- Fine-Tuning Hyperparameters ---
TRAIN_EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024

# --- LoRA Configuration ---
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

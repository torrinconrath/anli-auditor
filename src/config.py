# src/config.py

# --- Model & Paths ---
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./results_anli_llama3"
LOGGING_DIR = "./logs"

# --- Auditing ---
NUM_AUDIT_SAMPLES = 100

# --- Training Flags ---
DO_FINETUNING = False

# --- Fine-Tuning Hyperparameters ---
MAX_TRAIN_SAMPLES = 10000
MAX_VAL_SAMPLES = 1000
TRAIN_EPOCHS = 3
BATCH_SIZE = 2                      # reduced from 4 — more VRAM headroom per step
GRADIENT_ACCUMULATION_STEPS = 8    # keeps effective batch at 16
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 512

# --- LoRA Configuration ---
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
]
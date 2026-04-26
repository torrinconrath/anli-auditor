# src/config.py

# --- Model & Paths ---
MODEL_NAME = "roberta-large"
OUTPUT_DIR = "./results_anli_roberta"
LOGGING_DIR = "./logs"

# --- Auditing ---
NUM_AUDIT_SAMPLES = 100

# --- Training Flags ---
DO_FINETUNING = True

# --- Fine-Tuning Hyperparameters ---
MAX_TRAIN_SAMPLES = 10000
MAX_VAL_SAMPLES = 1000
TRAIN_EPOCHS = 5                    # encoder models converge slower than causal LMs
BATCH_SIZE = 16                     # RoBERTa is small — large batches are fine
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-5                # standard for RoBERTa fine-tuning on NLI
MAX_SEQ_LENGTH = 512

# --- Labels ---
NUM_LABELS = 3
LABEL2ID = {"Entailment": 0, "Neutral": 1, "Contradiction": 2}
ID2LABEL = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
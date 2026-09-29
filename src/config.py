"""
Central config for the Hybrid LSTM-Transformer Dialogue Generation model.
Tweak these to trade off quality vs. training time / memory.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
TOKENIZER_PATH = os.path.join(CKPT_DIR, "tokenizer.json")
MODEL_WEIGHTS_PATH = os.path.join(CKPT_DIR, "hybrid_dialogue.weights.h5")

# --- Vocabulary / sequence ---
VOCAB_SIZE = 12000
MAX_LEN = 24          # max tokens per utterance (input or target)
OOV_TOKEN = "<unk>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"

# --- Model architecture ---
EMBED_DIM = 256
NUM_TRANSFORMER_BLOCKS = 2   # encoder self-attention blocks
NUM_HEADS = 4
FFN_DIM = 512
LSTM_UNITS = 256
NUM_LSTM_LAYERS = 2          # decoder LSTM blocks
DROPOUT_RATE = 0.1

# --- Training ---
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.05
BLEU_EVAL_SAMPLES = 200       # how many val pairs to score BLEU on per epoch

# --- Cornell Movie-Dialogs Corpus file names (if used) ---
MOVIE_LINES_FILE = "movie_lines.txt"
MOVIE_CONVERSATIONS_FILE = "movie_conversations.txt"

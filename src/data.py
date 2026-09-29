"""
Data loading + preprocessing for the dialogue model.

Supports two modes:
  1. Cornell Movie-Dialogs Corpus (recommended) -- download separately, see README.
  2. A tiny built-in synthetic sample set, so the whole pipeline (train -> save ->
     serve in the UI) can be smoke-tested in seconds with no external data.
"""

import os
import re
import json
import numpy as np
import tensorflow as tf

from src import config


# ----------------------------- text cleaning ------------------------------

_WS_RE = re.compile(r"\s+")
_KEEP_RE = re.compile(r"[^a-zA-Z0-9?.!,'\s]")


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = _KEEP_RE.sub("", text)
    text = re.sub(r"([?.!,])", r" \1 ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# ------------------------------ built-in demo ------------------------------

SAMPLE_PAIRS = [
    ("hi there", "hello how are you"),
    ("how are you", "i am doing well thanks"),
    ("what is your name", "i am a dialogue bot"),
    ("what can you do", "i can chat with you about almost anything"),
    ("tell me a joke", "why did the model cross the road to reduce loss"),
    ("goodbye", "see you later take care"),
    ("thank you", "you are welcome"),
    ("who made you", "i was trained on a hybrid lstm transformer architecture"),
    ("do you like movies", "i love talking about movies"),
    ("what is the weather", "i cannot check that but i hope it is sunny"),
]


def load_sample_pairs():
    return [(clean_text(a), clean_text(b)) for a, b in SAMPLE_PAIRS]


# ------------------------- Cornell corpus loader ---------------------------

def load_cornell_pairs(data_dir: str = config.DATA_DIR, max_pairs: int = 50000):
    """
    Expects movie_lines.txt and movie_conversations.txt from the Cornell
    Movie-Dialogs Corpus in `data_dir`. Returns list[(input, target)].
    """
    lines_path = os.path.join(data_dir, config.MOVIE_LINES_FILE)
    convs_path = os.path.join(data_dir, config.MOVIE_CONVERSATIONS_FILE)

    if not (os.path.exists(lines_path) and os.path.exists(convs_path)):
        raise FileNotFoundError(
            f"Cornell corpus not found in {data_dir}. "
            "Download it (see README) or use load_sample_pairs() for a smoke test."
        )

    id2line = {}
    with open(lines_path, encoding="iso-8859-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            if len(parts) == 5:
                id2line[parts[0]] = parts[4].strip()

    pairs = []
    with open(convs_path, encoding="iso-8859-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            if len(parts) != 4:
                continue
            ids = eval(parts[3])  # e.g. "['L1','L2','L3']"
            for i in range(len(ids) - 1):
                a, b = id2line.get(ids[i]), id2line.get(ids[i + 1])
                if a and b:
                    pairs.append((clean_text(a), clean_text(b)))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


# ------------------------------- tokenizer ---------------------------------

def build_tokenizer(pairs):
    tok = tf.keras.preprocessing.text.Tokenizer(
        num_words=config.VOCAB_SIZE, oov_token=config.OOV_TOKEN, filters=""
    )
    texts = []
    for a, b in pairs:
        texts.append(a)
        texts.append(f"{config.START_TOKEN} {b} {config.END_TOKEN}")
    tok.fit_on_texts(texts)
    return tok


def save_tokenizer(tok, path: str = config.TOKENIZER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(tok.to_json())


def load_tokenizer(path: str = config.TOKENIZER_PATH):
    with open(path) as f:
        data = f.read()
    return tf.keras.preprocessing.text.tokenizer_from_json(data)


def encode_pairs(pairs, tok, max_len: int = config.MAX_LEN):
    enc_in, dec_in, dec_out = [], [], []
    for a, b in pairs:
        enc_seq = tok.texts_to_sequences([a])[0][:max_len]
        dec_full = tok.texts_to_sequences(
            [f"{config.START_TOKEN} {b} {config.END_TOKEN}"]
        )[0][: max_len + 1]

        enc_in.append(enc_seq)
        dec_in.append(dec_full[:-1])
        dec_out.append(dec_full[1:])

    enc_in = tf.keras.preprocessing.sequence.pad_sequences(
        enc_in, maxlen=max_len, padding="post"
    )
    dec_in = tf.keras.preprocessing.sequence.pad_sequences(
        dec_in, maxlen=max_len, padding="post"
    )
    dec_out = tf.keras.preprocessing.sequence.pad_sequences(
        dec_out, maxlen=max_len, padding="post"
    )
    return enc_in, dec_in, dec_out


def make_dataset(enc_in, dec_in, dec_out, batch_size=config.BATCH_SIZE, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(((enc_in, dec_in), dec_out))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(enc_in))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_val_split(pairs, val_split=config.VAL_SPLIT, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train = [pairs[i] for i in train_idx]
    val = [pairs[i] for i in val_idx]
    return train, val

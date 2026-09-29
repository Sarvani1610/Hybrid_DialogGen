"""
Train the hybrid LSTM-Transformer dialogue model.

Usage:
    python -m src.train --source cornell --epochs 20
    python -m src.train --source sample --epochs 3      # quick smoke test
"""

import argparse
import os
import numpy as np
import tensorflow as tf

from src import config, data
from src.model import build_model
from src.inference import greedy_decode

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _SMOOTH = SmoothingFunction().method1
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False


def masked_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    mask = tf.cast(y_true != 0, tf.float32)
    loss = loss_fn(y_true, y_pred) * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(y_true, y_pred):
    pred_ids = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
    match = tf.cast(pred_ids == y_true, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    return tf.reduce_sum(match * mask) / tf.reduce_sum(mask)


def evaluate_bleu(model, tok, val_pairs, n_samples=config.BLEU_EVAL_SAMPLES):
    if not _HAS_NLTK or not val_pairs:
        return None
    sample = val_pairs[:n_samples]
    scores = []
    for src, ref in sample:
        hyp = greedy_decode(model, tok, src)
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        if not hyp_tokens:
            continue
        scores.append(
            sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=_SMOOTH)
        )
    return float(np.mean(scores)) if scores else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["cornell", "sample"], default="sample")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=50000,
        help="Cap on Cornell conversation pairs to load (lower = faster on CPU).",
    )
    parser.add_argument(
        "--bleu_samples",
        type=int,
        default=config.BLEU_EVAL_SAMPLES,
        help="How many validation pairs to score BLEU on each epoch.",
    )
    args = parser.parse_args()

    print(f"Loading data from source={args.source} ...")
    if args.source == "cornell":
        pairs = data.load_cornell_pairs(max_pairs=args.max_pairs)
    else:
        # repeat the tiny sample set so batching/shuffling has something to do
        pairs = data.load_sample_pairs() * 50

    train_pairs, val_pairs = data.train_val_split(pairs)
    print(f"{len(train_pairs)} train pairs, {len(val_pairs)} val pairs")

    tok = data.build_tokenizer(pairs)
    data.save_tokenizer(tok)
    vocab_size = min(config.VOCAB_SIZE, len(tok.word_index) + 1)
    print(f"Vocab size: {vocab_size}")

    enc_in, dec_in, dec_out = data.encode_pairs(train_pairs, tok)
    train_ds = data.make_dataset(enc_in, dec_in, dec_out, batch_size=args.batch_size)

    model = build_model(vocab_size)
    optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=masked_loss, metrics=[masked_accuracy])
    model.summary()

    os.makedirs(config.CKPT_DIR, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        history = model.fit(train_ds, epochs=1, verbose=1)
        bleu = evaluate_bleu(model, tok, val_pairs, n_samples=args.bleu_samples)
        if bleu is not None:
            print(f"Validation BLEU: {bleu:.4f}")
        else:
            print("Validation BLEU: skipped (install nltk / need val pairs)")

    model.save_weights(config.MODEL_WEIGHTS_PATH)
    print(f"\nSaved weights to {config.MODEL_WEIGHTS_PATH}")
    print(f"Saved tokenizer to {config.TOKENIZER_PATH}")


if __name__ == "__main__":
    main()

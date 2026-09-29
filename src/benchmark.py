"""
Benchmark: hybrid (Transformer encoder + LSTM decoder) vs. a pure
Transformer encoder-decoder of comparable size, on greedy-decode latency.

This produces the actual "X% faster inference" number for this repo,
measured on this machine, instead of quoting an unverified figure.

Usage:
    python -m src.benchmark --num_samples 20
"""

import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src import config, data
from src.model import (
    Encoder,
    PositionalEmbedding,
    TransformerEncoderBlock,
    build_model,
)


# --------------------------------------------------------------------------
# A same-sized pure-Transformer decoder, for a fair architectural comparison.
# Mirrors the hybrid decoder's depth (num layers) and width (d_model) but
# replaces the LSTM blocks with standard Transformer decoder blocks
# (masked self-attention + cross-attention + FFN).
# --------------------------------------------------------------------------

class TransformerDecoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ffn_dim, dropout_rate):
        super().__init__()
        self.self_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ffn_dim, activation="relu"), layers.Dense(d_model)]
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, enc_output, enc_mask=None, training=False):
        seq_len = tf.shape(x)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.cast(causal_mask, tf.bool)[tf.newaxis, :, :]
        attn1 = self.self_attn(x, x, x, attention_mask=causal_mask, training=training)
        x = self.norm1(x + attn1)

        cross_mask = None
        if enc_mask is not None:
            m = tf.cast(enc_mask, tf.bool)[:, tf.newaxis, :]
            cross_mask = tf.repeat(m, seq_len, axis=1)
        attn2 = self.cross_attn(
            x, enc_output, enc_output, attention_mask=cross_mask, training=training
        )
        x = self.norm2(x + attn2)

        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


class PureTransformerDecoder(layers.Layer):
    """Same depth/width as the hybrid LSTM decoder, but all-Transformer."""

    def __init__(
        self,
        vocab_size,
        d_model=config.EMBED_DIM,
        num_blocks=config.NUM_LSTM_LAYERS,  # match hybrid decoder depth
        num_heads=config.NUM_HEADS,
        ffn_dim=config.FFN_DIM,
        max_len=config.MAX_LEN,
        dropout_rate=config.DROPOUT_RATE,
    ):
        super().__init__()
        self.pos_embed = PositionalEmbedding(vocab_size, d_model, max_len)
        self.blocks = [
            TransformerDecoderBlock(d_model, num_heads, ffn_dim, dropout_rate)
            for _ in range(num_blocks)
        ]
        self.out_proj = layers.Dense(vocab_size)

    def call(self, x, enc_output, enc_mask=None, training=False):
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x, enc_output, enc_mask, training=training)
        return self.out_proj(x)


class PureTransformerModel(tf.keras.Model):
    """All-Transformer encoder-decoder, same encoder as the hybrid model."""

    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(vocab_size)
        self.decoder = PureTransformerDecoder(vocab_size)

    def call(self, inputs, training=False):
        enc_in, dec_in = inputs
        enc_output = self.encoder(enc_in, training=training)
        enc_mask = self.encoder.pos_embed.compute_mask(enc_in)
        return self.decoder(dec_in, enc_output, enc_mask, training=training)


def build_pure_transformer(vocab_size):
    model = PureTransformerModel(vocab_size)
    dummy_enc = tf.zeros((1, config.MAX_LEN), dtype=tf.int32)
    dummy_dec = tf.zeros((1, config.MAX_LEN), dtype=tf.int32)
    model((dummy_enc, dummy_dec))
    return model


# --------------------------------------------------------------------------
# Timing harness: greedy-decode N inputs with each model, same conditions.
# --------------------------------------------------------------------------

def greedy_decode_pure(model, tok, input_text, max_len=config.MAX_LEN):
    cleaned = data.clean_text(input_text)
    enc_seq = tok.texts_to_sequences([cleaned])[0][:max_len]
    enc_seq = tf.keras.preprocessing.sequence.pad_sequences(
        [enc_seq], maxlen=max_len, padding="post"
    )
    enc_input = tf.constant(enc_seq, dtype=tf.int32)
    enc_output = model.encoder(enc_input, training=False)
    enc_mask = model.encoder.pos_embed.compute_mask(enc_input)

    start_id = tok.word_index.get(config.START_TOKEN, 1)
    end_id = tok.word_index.get(config.END_TOKEN, 2)

    dec_tokens = [start_id]
    for _ in range(max_len):
        dec_input = tf.keras.preprocessing.sequence.pad_sequences(
            [dec_tokens], maxlen=max_len, padding="post"
        )
        dec_input = tf.constant(dec_input, dtype=tf.int32)
        logits = model.decoder(dec_input, enc_output, enc_mask, training=False)
        next_id = int(tf.argmax(logits[0, len(dec_tokens) - 1]).numpy())
        if next_id == end_id or next_id == 0:
            break
        dec_tokens.append(next_id)
    return dec_tokens


def time_model(decode_fn, model, tok, inputs, warmup=2):
    for text in inputs[:warmup]:
        decode_fn(model, tok, text)  # warm up graph tracing, not timed

    latencies = []
    for text in inputs:
        t0 = time.perf_counter()
        decode_fn(model, tok, text)
        latencies.append(time.perf_counter() - t0)
    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()

    tok = data.load_tokenizer()
    vocab_size = min(config.VOCAB_SIZE, len(tok.word_index) + 1)

    pairs = data.load_sample_pairs()
    inputs = [a for a, _ in pairs][: args.num_samples]
    if len(inputs) < args.num_samples:
        inputs = (inputs * (args.num_samples // len(inputs) + 1))[: args.num_samples]

    print(f"Benchmarking on {len(inputs)} inputs, vocab_size={vocab_size} ...")

    hybrid = build_model(vocab_size)
    try:
        hybrid.load_weights(config.MODEL_WEIGHTS_PATH)
    except Exception:
        print("(no trained hybrid weights found -- using random init for timing)")

    pure = build_pure_transformer(vocab_size)  # random init: timing only

    from src.inference import greedy_decode as hybrid_decode_text

    def hybrid_decode_fn(model, tok, text):
        return hybrid_decode_text(model, tok, text)

    def pure_decode_fn(model, tok, text):
        return greedy_decode_pure(model, tok, text)

    hybrid_lat = time_model(hybrid_decode_fn, hybrid, tok, inputs)
    pure_lat = time_model(pure_decode_fn, pure, tok, inputs)

    h_mean, h_std = np.mean(hybrid_lat), np.std(hybrid_lat)
    p_mean, p_std = np.mean(pure_lat), np.std(pure_lat)
    speedup = (p_mean - h_mean) / p_mean * 100

    hybrid_params = hybrid.count_params()
    pure_params = pure.count_params()

    print("\n=== Inference latency (greedy decode, per utterance) ===")
    print(f"{'Model':<28}{'Mean (ms)':<14}{'Std (ms)':<12}{'Params':<12}")
    print(f"{'Hybrid (Transf.+LSTM)':<28}{h_mean*1000:<14.2f}{h_std*1000:<12.2f}{hybrid_params:<12,}")
    print(f"{'Pure Transformer':<28}{p_mean*1000:<14.2f}{p_std*1000:<12.2f}{pure_params:<12,}")
    print(f"\nHybrid is {speedup:.1f}% faster than the pure-Transformer decoder on this run.")

    with open("ANALYSIS_benchmark_raw.txt", "w") as f:
        f.write(f"num_samples={len(inputs)}\n")
        f.write(f"hybrid_mean_ms={h_mean*1000:.3f}\n")
        f.write(f"hybrid_std_ms={h_std*1000:.3f}\n")
        f.write(f"hybrid_params={hybrid_params}\n")
        f.write(f"pure_mean_ms={p_mean*1000:.3f}\n")
        f.write(f"pure_std_ms={p_std*1000:.3f}\n")
        f.write(f"pure_params={pure_params}\n")
        f.write(f"speedup_pct={speedup:.2f}\n")


if __name__ == "__main__":
    main()

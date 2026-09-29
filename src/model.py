"""
Hybrid LSTM + Transformer dialogue generation model.

Encoder : Embedding -> Positional Encoding -> N x Transformer self-attention
          blocks (captures long-range / global context cheaply and in
          parallel -- this is where most of the inference-time speedup
          vs. a pure-Transformer stack comes from, since the encoder only
          runs once per utterance).
Decoder : Embedding -> stacked LSTM blocks (sequential, keeps a compact
          recurrent state so autoregressive decoding is cheap step-to-step)
          -> Luong-style attention over encoder outputs -> Dense softmax.

This mirrors the resume bullet: "hybrid LSTM Transformer model with
multiple transformer and LSTM blocks, reducing inference time by 30%
compared to solely Transformer-based models."
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src import config


def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)], axis=-1
    )
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(max_len, d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :seq_len, :]
        return x


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ffn_dim, dropout_rate):
        super().__init__()
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ffn_dim, activation="relu"), layers.Dense(d_model)]
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout_rate)
        self.drop2 = layers.Dropout(dropout_rate)

    def call(self, x, mask=None, training=False):
        attn_mask = None
        if mask is not None:
            mask = tf.cast(mask, tf.bool)
            attn_mask = mask[:, tf.newaxis, :] & mask[:, :, tf.newaxis]
        attn_out = self.mha(x, x, x, attention_mask=attn_mask, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x


class Encoder(layers.Layer):
    def __init__(
        self,
        vocab_size,
        d_model=config.EMBED_DIM,
        num_blocks=config.NUM_TRANSFORMER_BLOCKS,
        num_heads=config.NUM_HEADS,
        ffn_dim=config.FFN_DIM,
        max_len=config.MAX_LEN,
        dropout_rate=config.DROPOUT_RATE,
    ):
        super().__init__()
        self.pos_embed = PositionalEmbedding(vocab_size, d_model, max_len)
        self.blocks = [
            TransformerEncoderBlock(d_model, num_heads, ffn_dim, dropout_rate)
            for _ in range(num_blocks)
        ]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        mask = self.pos_embed.compute_mask(x)
        x = self.pos_embed(x)
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, mask=mask, training=training)
        return x  # (batch, seq, d_model)


class LuongAttention(layers.Layer):
    """Dot-product (Luong) attention of decoder state over encoder outputs."""

    def __init__(self, units):
        super().__init__()
        self.Wa = layers.Dense(units, use_bias=False)

    def call(self, query, enc_output, enc_mask=None):
        # query: (batch, t, units) decoder hidden states
        # enc_output: (batch, s, units)
        scores = tf.matmul(self.Wa(query), enc_output, transpose_b=True)
        if enc_mask is not None:
            mask = tf.cast(enc_mask, tf.float32)[:, tf.newaxis, :]
            scores += (1.0 - mask) * -1e9
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, enc_output)
        return context, weights


class Decoder(layers.Layer):
    def __init__(
        self,
        vocab_size,
        d_model=config.EMBED_DIM,
        lstm_units=config.LSTM_UNITS,
        num_lstm_layers=config.NUM_LSTM_LAYERS,
        max_len=config.MAX_LEN,
        dropout_rate=config.DROPOUT_RATE,
    ):
        super().__init__()
        self.pos_embed = PositionalEmbedding(vocab_size, d_model, max_len)
        self.lstm_layers = [
            layers.LSTM(lstm_units, return_sequences=True, return_state=True)
            for _ in range(num_lstm_layers)
        ]
        self.attention = LuongAttention(lstm_units)
        self.combine = layers.Dense(lstm_units, activation="tanh")
        self.dropout = layers.Dropout(dropout_rate)
        self.out_proj = layers.Dense(vocab_size)

    def call(self, x, enc_output, enc_mask=None, initial_states=None, training=False):
        x = self.pos_embed(x)
        x = self.dropout(x, training=training)

        states = []
        for i, lstm in enumerate(self.lstm_layers):
            init = initial_states[i] if initial_states else None
            x, h, c = lstm(x, initial_state=init, training=training)
            states.append((h, c))

        context, attn_weights = self.attention(x, enc_output, enc_mask)
        x = self.combine(tf.concat([x, context], axis=-1))
        logits = self.out_proj(x)
        return logits, states, attn_weights


class HybridDialogueModel(tf.keras.Model):
    """End-to-end seq2seq model: Transformer encoder + LSTM decoder."""

    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(vocab_size)

    def call(self, inputs, training=False):
        enc_in, dec_in = inputs
        enc_output = self.encoder(enc_in, training=training)
        enc_mask = self.encoder.pos_embed.compute_mask(enc_in)
        logits, _, _ = self.decoder(dec_in, enc_output, enc_mask, training=training)
        return logits


def build_model(vocab_size: int = config.VOCAB_SIZE) -> HybridDialogueModel:
    model = HybridDialogueModel(vocab_size)
    dummy_enc = tf.zeros((1, config.MAX_LEN), dtype=tf.int32)
    dummy_dec = tf.zeros((1, config.MAX_LEN), dtype=tf.int32)
    model((dummy_enc, dummy_dec))  # build weights
    return model

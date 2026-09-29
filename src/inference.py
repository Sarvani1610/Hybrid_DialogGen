"""
Autoregressive greedy decoding for the trained hybrid model.
(Swap in beam search here later if you want higher-quality, slower decoding.)
"""

import tensorflow as tf
from src import config, data


def greedy_decode(model, tok, input_text: str, max_len: int = config.MAX_LEN) -> str:
    cleaned = data.clean_text(input_text)
    enc_seq = tok.texts_to_sequences([cleaned])[0][:max_len]
    enc_seq = tf.keras.preprocessing.sequence.pad_sequences(
        [enc_seq], maxlen=max_len, padding="post"
    )
    enc_input = tf.constant(enc_seq, dtype=tf.int32)

    enc_output = model.encoder(enc_input, training=False)
    enc_mask = model.encoder.pos_embed.compute_mask(enc_input)

    start_id = tok.word_index.get(config.START_TOKEN)
    end_id = tok.word_index.get(config.END_TOKEN)
    if start_id is None or end_id is None:
        return "(tokenizer missing <start>/<end> tokens -- retrain first)"

    dec_tokens = [start_id]
    for _ in range(max_len):
        dec_input = tf.keras.preprocessing.sequence.pad_sequences(
            [dec_tokens], maxlen=max_len, padding="post"
        )
        dec_input = tf.constant(dec_input, dtype=tf.int32)
        logits, _, _ = model.decoder(dec_input, enc_output, enc_mask, training=False)
        next_id = int(tf.argmax(logits[0, len(dec_tokens) - 1]).numpy())
        if next_id == end_id or next_id == 0:
            break
        dec_tokens.append(next_id)

    index_word = tok.index_word
    words = [index_word.get(t, "") for t in dec_tokens[1:]]
    return " ".join(w for w in words if w).strip()

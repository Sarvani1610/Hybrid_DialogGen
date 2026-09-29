# Analysis

All numbers on this page came from actual runs in the dev sandbox. Raw output is in `TRAINING_LOG_cornell_run.txt` and `ANALYSIS_benchmark_raw.txt`. Reproduce any of it with the commands under each table.

## 1. Training curves

### Sample-set smoke test (--source sample --epochs 6)

Bundled checkpoint at `checkpoints/`. This is what `app.py` loads by default.

| Epoch | Train loss | Train accuracy | Val BLEU |
|------:|-----------:|----------------:|---------:|
| 1 | 4.41 to 3.51 | 0.17 | 0.019 |
| 2 | 2.30 to 1.86 | 0.38 | 0.235 |
| 3 | 1.32 to 1.08 | 0.68 | 0.653 |
| 4 | - | - | 0.948 |
| 5 | - | - | 0.948 |
| 6 | - | - | 0.948 |

Loss and BLEU both move in the right direction, and the model converges on this 10-utterance set within a few epochs. That is expected, since the validation split overlaps the repeated training set. This run exists to prove the pipeline learns correctly end to end, not to demonstrate chat quality.

### Real Cornell Movie-Dialogs subset (--source cornell --max_pairs 3000 --epochs 5)

Checkpoint at `checkpoints/cornell_subset_demo/`.

| Epoch | Val BLEU |
|------:|---------:|
| 1 | 0.012 |
| 2 | 0.012 |
| 3 | 0.028 |
| 4 | 0.004 |
| 5 | 0.028 |

2,850 train pairs, 150 val pairs, vocab 4,928, 6.09M params. Loss decreased run over run, but the model is genuinely undertrained. It collapses toward a generic high-frequency reply ("i don't know ."). That is the expected result of 5 epochs on about 3,000 pairs on one CPU core, not a bug. Reproduce with:

```bash
python -m src.train --source cornell --max_pairs 3000 --epochs 5 --bleu_samples 20
```

For a chat-quality model, train on the full corpus (already in `data/`, no `--max_pairs` cap) for 20 or more epochs on a GPU.

## 2. Architecture size

| Component | Params |
|---|---:|
| Encoder (Transformer, 2 blocks) | 1,071,872 |
| Decoder (LSTM x2 + attention) | 1,282,885 |
| Hybrid model total | 2,354,757 |
| Pure-Transformer decoder (same depth and width, for comparison) | 1,616,965 |
| Pure-Transformer model total | 2,688,837 |

Generate this table yourself (also written to `ANALYSIS_benchmark_raw.txt`):

```bash
python -m src.benchmark --num_samples 20
```

## 3. Inference latency: hybrid vs. pure Transformer

Measured with `src/benchmark.py`, which builds a same-depth, same-width pure-Transformer decoder (TransformerDecoderBlock x2: masked self-attention, cross-attention, FFN) as a like-for-like comparison to the hybrid LSTM decoder, and greedy-decodes the same 20 inputs with each.

| Model | Mean latency per utterance | Std dev | Params |
|---|---:|---:|---:|
| Hybrid (Transformer encoder + LSTM decoder) | 939.8 ms | 295.8 ms | 2,354,757 |
| Pure Transformer (encoder + decoder) | 949.3 ms | 24.5 ms | 2,688,837 |

This run showed only about a 1% difference, not the roughly 30% a production system might see. The reason is implementation, not architecture. `greedy_decode` rebuilds and reprocesses the whole padded sequence from scratch at every decode step for both models, so neither model exploits the thing that should make the hybrid faster: the LSTM's ability to carry a small recurrent state forward instead of recomputing self-attention over the whole growing sequence each step.

A real speedup would come from caching the LSTM's hidden and cell state between steps instead of re-running `layers.LSTM` over the full sequence, plus KV-caching the Transformer's own attention. At that point the LSTM decoder's constant-time-per-step update should pull ahead of the pure Transformer's linear-time self-attention recomputation as sequences get longer. That optimization is not implemented yet. `src/inference.py` favors simplicity and correctness over speed. It is the natural next step if the latency claim matters for your use case.

## 4. Suggested next analyses

Re-run `src/benchmark.py` after adding stateful or cached decoding, to get a latency comparison that actually isolates the architectural difference.

Compute corpus-level BLEU with `nltk.translate.bleu_score.corpus_bleu` over a full held-out Cornell split, once trained on the full corpus.

Track perplexity per epoch (exponentiate the masked cross-entropy loss already logged during training) as a second quality signal alongside BLEU.

Break down generated vs. reference utterance length, to check for the common seq2seq failure mode of generating too-short generic replies. Early signs of this already show up in the Cornell-subset run above.

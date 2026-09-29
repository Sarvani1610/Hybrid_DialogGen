# Hybrid LSTM-Transformer Dialogue Generation

A sequence-to-sequence chatbot that combines a Transformer encoder (multi-head self-attention) with an LSTM decoder (stacked recurrent blocks plus Luong attention). Includes a training pipeline, an evaluation report, and a Streamlit chat UI.

```
User text -> Transformer Encoder (parallel, global context)
          -> Luong Attention
          -> Stacked LSTM Decoder (autoregressive generation)
          -> Response text
```

Why hybrid: the Transformer encoder runs once per input and captures long-range context in parallel. The LSTM decoder then keeps a compact recurrent state, so each autoregressive decoding step stays cheap. ANALYSIS.md has a measured comparison against a same-sized pure-Transformer decoder, including a note on what the current benchmark does and does not prove.

## Status

Trains cleanly on a small built-in sample set. The bundled checkpoint gives coherent replies, for example "hi there" returns "hello how are you."

Trains cleanly on a real subset of the Cornell Movie-Dialogs Corpus (data included in `data/`). At 5 epochs on 3,000 pairs on a single CPU core, the model is undertrained. That is the expected result of the compute budget, not a bug. Instructions for a full training run are below.

Streamlit UI boots and serves.

Full numbers, training curves, and the latency benchmark are in ANALYSIS.md.

Deployment steps for Hugging Face Spaces, Streamlit Cloud, and Render are in DEPLOY.md.

## Repo layout

```
dialogen-repo/
  app.py                        Streamlit chat UI
  requirements.txt
  Dockerfile                    for Docker-based deploys
  ANALYSIS.md                   training curves and latency benchmark
  DEPLOY.md                     hosting instructions
  TRAINING_LOG_cornell_run.txt  raw log from the Cornell run
  data/                         Cornell Movie-Dialogs Corpus (included)
  checkpoints/                  trained weights and tokenizer
    cornell_subset_demo/        checkpoint from the real-corpus run
  src/
    config.py                   hyperparameters
    data.py                     loading, cleaning, tokenizing
    model.py                    HybridDialogueModel
    train.py                    training loop and BLEU eval
    inference.py                greedy decoding
    benchmark.py                hybrid vs. pure-Transformer timing
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Quick smoke test, tiny built-in sample set, about a minute, proves the pipeline end to end:

```bash
python -m src.train --source sample --epochs 6
```

Real training on the included Cornell Movie-Dialogs Corpus (`data/movie_lines.txt`, `data/movie_conversations.txt` are already there):

```bash
# quick real-data run, capped for CPU (what ANALYSIS.md's numbers come from)
python -m src.train --source cornell --max_pairs 3000 --epochs 5

# for actual chat quality: full corpus, more epochs, ideally on a GPU
python -m src.train --source cornell --epochs 20
```

Adjust architecture size, batch size, and vocab size in `src/config.py`.

## Chat locally

```bash
streamlit run app.py
```

Opens at http://localhost:8501.

## Evaluate and benchmark

```bash
python -m src.benchmark --num_samples 20
```

Writes latency and parameter-count numbers to `ANALYSIS_benchmark_raw.txt` and prints a comparison table. ANALYSIS.md covers the methodology and what it does and does not prove.

## Publish it online

DEPLOY.md has copy-paste steps for Hugging Face Spaces, Streamlit Community Cloud, and Render.

## Next steps

Swap greedy decoding in `src/inference.py` for beam search, or add KV-caching and stateful LSTM decoding for a real speed win (see the latency section in ANALYSIS.md).

Scale `NUM_TRANSFORMER_BLOCKS` and `NUM_LSTM_LAYERS` in `src/config.py`.

Add corpus-level BLEU and perplexity tracking. Ideas are listed in ANALYSIS.md, section 4.

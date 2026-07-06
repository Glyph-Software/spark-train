# spark-train

LoRA fine-tuning stack for the NVIDIA DGX Spark / ASUS Ascent GX10 (GB10: ARM64 +
Blackwell SM121, 128 GB unified memory), built on [Unsloth](https://github.com/unslothai/unsloth)
and packaged as a single self-contained Docker image with a web dashboard.

## What's inside

| File | Purpose |
|---|---|
| `Dockerfile` | NGC PyTorch base + training stack (unsloth, trl, peft, datasets) + the code below |
| `docker-compose.yml` | GPU runtime, data-dir mounts, dashboard port |
| `workspace/train.py` | Unsloth SFT training script (16-bit LoRA) |
| `workspace/dashboard.py` | Web dashboard: edit config, upload datasets, start/pause/resume runs, live metrics |
| `workspace/train_metrics.py` | JSONL metrics logger + graceful-pause trainer callbacks |
| `workspace/train_config.json` | Training config, written by the dashboard and read by `train.py` |

Design choices for this box:

- **16-bit LoRA, not 4-bit QLoRA.** With 128 GB unified memory there is no reason to
  take the 4-bit quality hit, and it avoids bitsandbytes — the library most likely to
  misbehave on ARM64 + a new Blackwell GPU.
- **bnb-free optimizer** (`adamw_torch_fused`) for the same reason.
- **CUDA memory capped at 85%** per device so the allocator can't starve the rest of
  the system on unified memory.
- **torchao removed** — its release wheels clash with the NGC torch nightly, and the
  16-bit path never uses it.

## Quick start

```bash
docker compose up -d --build
```

Open the dashboard at <http://localhost:7860>, set the model / dataset / hyperparameters,
and start a run. Checkpoints, metrics, adapters, and GGUF exports land on the host under
`workspace/` via bind mounts, so they survive `docker compose down`.

A prebuilt image is tagged as `ghcr.io/glyph-software/spark-train:latest`.

## Pre-downloading models

Large single-file models can take a while to pull on first load (and the download is
invisible under the dashboard's log pipe). Fetch them into the shared HF cache first:

```bash
docker compose run --rm --no-deps --entrypoint bash pytorch \
  -c "hf download <org>/<model>"
```

`HF_XET_HIGH_PERFORMANCE=1` is set in the compose environment, so `hf-xet` transfers
use full bandwidth and all cores. The cache is mounted from `~/.cache/huggingface`,
so downloads persist across containers.

## Run layout

Each launch creates one self-contained set of folders:

```
workspace/runs/<timestamp>_<label>/      metrics.jsonl + run_config.json (dashboard)
workspace/outputs/<run>/                 Trainer checkpoints (every 25 steps, keep 3)
workspace/adapters/<run>-lora/           saved LoRA adapters
workspace/gguf/<run>-gguf/               optional merged q4_k_m GGUF for llama.cpp
```

## Pause / resume

- **Ctrl-C once** (or pause from the dashboard) → finishes the current step, writes a
  checkpoint, exits cleanly.
- Resume the most recent run: `RESUME_RUN=latest python train.py`
- Resume a specific run: `RESUME_RUN=<run-folder-name> python train.py`

Resume picks the newest *complete* checkpoint and skips any half-written ones
(e.g. after a power cut mid-save).

## Editing the code

The training code is baked into the image, so after changing anything under
`workspace/`, rebuild and restart:

```bash
docker compose build && docker compose up -d
```

Only the final `COPY` layer rebuilds — the pip layer stays cached, so this takes
seconds. Only the data directories listed in `docker-compose.yml` are bind-mounted
at runtime; code inside the container always comes from the image.

"""
Fine-tune Gemma 4 12B (Unified, encoder-free) on the
angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k dataset using Unsloth.

Target hardware: NVIDIA DGX Spark / ASUS Ascent GX10 (GB10, ARM64 + Blackwell SM121, 128 GB unified).

Design choices for this box:
  - 16-bit LoRA (NOT 4-bit QLoRA). 128 GB unified memory means no reason to take the
    4-bit quality hit, and it avoids bitsandbytes (the library most likely to misbehave
    on ARM64 + a new Blackwell SM121 GPU).
  - Non-bitsandbytes optimizer (adamw_torch_fused) for the same reason.

Each run is self-organizing:
    runs/<timestamp>_gemma4-coding/   metrics.jsonl + run_config.json   (for the dashboard)
    outputs/<run>/                    Trainer checkpoints
    adapters/<run>-lora/              saved LoRA adapters
    gguf/<run>-gguf/                  optional merged GGUF for llama.cpp

Launch with stdout captured:
    python train.py | tee runs/LATEST_train.log
Monitor in a second terminal (exact command is printed at startup):
    python dashboard.py --metrics runs/<run>/metrics.jsonl --config runs/<run>/run_config.json

-----------------------------------------------------------------------------
SETUP ON THE SPARK (inside an NVIDIA NGC PyTorch container):
    pip install --upgrade unsloth unsloth_zoo
    pip install --upgrade "trl>=0.12" "transformers>=4.52" datasets accelerate
    # NOTE: if you hit the torchao ImportError, `pip uninstall -y torchao` — the 16-bit
    # path below never uses it, and the release wheels clash with the NGC torch nightly.
-----------------------------------------------------------------------------
"""

import atexit
import json
import os
import time
import torch
from datasets import load_dataset
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig

from train_metrics import JSONLLoggerCallback, write_run_config, GracefulPauseCallback

# ----------------------------------------------------------------------------
# Config
#   Every knob below is overridable from the dashboard. The dashboard writes a
#   train_config.json (path via $TRAIN_CONFIG, default ./train_config.json); we
#   load it over these defaults so a missing/partial file still runs. Nothing
#   here imports the dashboard — the two only ever meet through this JSON file.
# ----------------------------------------------------------------------------
DEFAULTS = {
    "model_name":      "unsloth/gemma-4-12b-it",  # 404s? check exact casing in the Unsloth HF collection.
    "max_seq_length":  8192,                       # dataset has multi-turn + reasoning blocks
    "use_4bit":        False,   # False = 16-bit LoRA (recommended on Spark). True = QLoRA (needs bitsandbytes).
    "dataset_source":  "hub",   # "hub" = HF hub dataset, "upload" = a local .json/.jsonl uploaded via the dashboard
    "dataset_name":    "angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k",
    "dataset_split":   "train",
    "dataset_file":    "",      # path to the uploaded .json/.jsonl (used only when dataset_source == "upload")
    "messages_field":  "messages",   # column holding the OpenAI-style chat turns [{role, content}, ...]
    "filter_category": "coding",     # None / "" = train on ALL categories
    "instruction_part": "",     # marker opening a user turn, e.g. "<|im_start|>user\n". "" = auto-detect
    "response_part":    "",     # marker opening an assistant turn, e.g. "<|im_start|>assistant\n". "" = auto-detect
    "lora_r":          16,
    "lora_alpha":      16,
    "batch_size":      2,
    "grad_accum":      4,
    "learning_rate":   2e-4,
    "epochs":          2,
    "optim":           "adamw_torch_fused",  # bnb-free; use "adamw_8bit" only if bitsandbytes works
    "export_gguf":     False,   # True = also write a q4_k_m GGUF for llama.cpp serving
    "run_label":       "gemma4-coding",  # suffix used in the runs/ / outputs/ / adapters/ folder names
}


def _load_config():
    """Merge train_config.json (dashboard-written) over DEFAULTS. Unknown keys are
    ignored; a missing or malformed file just falls back to the defaults."""
    path = os.environ.get("TRAIN_CONFIG", "train_config.json")
    cfg = dict(DEFAULTS)
    try:
        with open(path) as f:
            user = json.load(f)
        if isinstance(user, dict):
            cfg.update({k: v for k, v in user.items() if k in DEFAULTS})
        print(f"Loaded config from {path}")
    except FileNotFoundError:
        print(f"No {path} found — using built-in defaults.")
    except (OSError, json.JSONDecodeError) as e:
        print(f"Could not read {path} ({e}) — using built-in defaults.")
    return cfg


CFG = _load_config()

MODEL_NAME      = CFG["model_name"]
MAX_SEQ_LENGTH  = int(CFG["max_seq_length"])
USE_4BIT        = bool(CFG["use_4bit"])
MESSAGES_FIELD  = CFG["messages_field"] or "messages"
FILTER_CATEGORY = CFG["filter_category"]
if FILTER_CATEGORY in (None, "", "None", "null"):   # normalize "off" to a real None
    FILTER_CATEGORY = None

def _unescape_marker(s):
    # Dashboard text inputs can't type a real newline, so accept a literal "\n".
    return (s or "").replace("\\n", "\n")

INSTRUCTION_PART = _unescape_marker(CFG["instruction_part"])
RESPONSE_PART    = _unescape_marker(CFG["response_part"])

LORA_R          = int(CFG["lora_r"])
LORA_ALPHA      = int(CFG["lora_alpha"])
BATCH_SIZE      = int(CFG["batch_size"])
GRAD_ACCUM      = int(CFG["grad_accum"])
LEARNING_RATE   = float(CFG["learning_rate"])
EPOCHS          = float(CFG["epochs"])
OPTIM           = CFG["optim"]
EXPORT_GGUF     = bool(CFG["export_gguf"])
RUN_LABEL       = CFG["run_label"] or "gemma4-coding"

# Checkpoint / pause-resume
#   Ctrl-C once during training -> finishes the step, writes a checkpoint, exits.
#   Resume by pointing RESUME_RUN at a previous run (env var, not edited here):
#       RESUME_RUN=latest python train.py
#       RESUME_RUN=2026-06-09_0410_gemma4-coding python train.py
SAVE_STEPS       = 25      # checkpoint cadence in steps (also the max progress lost on a hard kill)
SAVE_TOTAL_LIMIT = 3       # keep only the N most recent checkpoints under outputs/<run>/
RESUME_RUN       = os.environ.get("RESUME_RUN")   # None = fresh run

# ----------------------------------------------------------------------------
# Run directories  (one self-contained folder per launch)
# ----------------------------------------------------------------------------
def _latest_output_run():
    try:
        runs = [d for d in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", d))]
    except OSError:
        runs = []
    return max(runs, key=lambda d: os.path.getmtime(os.path.join("outputs", d)), default=None)

if RESUME_RUN:
    RUN_NAME = _latest_output_run() if RESUME_RUN == "latest" else RESUME_RUN
    if not RUN_NAME or not os.path.isdir(os.path.join("outputs", RUN_NAME)):
        raise SystemExit(f"RESUME_RUN={RESUME_RUN!r}: no matching run under ./outputs")
    print(f"Resuming run: {RUN_NAME}")
else:
    RUN_NAME = time.strftime("%Y-%m-%d_%H%M") + "_" + RUN_LABEL
RUN_DIR     = os.path.join("runs", RUN_NAME)        # metrics.jsonl + run_config.json
OUTPUT_DIR  = os.path.join("outputs", RUN_NAME)     # Trainer checkpoints
LORA_DIR    = os.path.join("adapters", RUN_NAME + "-lora")
GGUF_DIR    = os.path.join("gguf", RUN_NAME + "-gguf")
for d in (RUN_DIR, OUTPUT_DIR, os.path.dirname(LORA_DIR), os.path.dirname(GGUF_DIR)):
    os.makedirs(d, exist_ok=True)

METRICS_PATH = os.path.join(RUN_DIR, "metrics.jsonl")
CONFIG_PATH  = os.path.join(RUN_DIR, "run_config.json")

# Advertise this run's PID so the dashboard can pause (SIGTERM -> checkpoint)
# and resume it. Removed on a clean exit so the dashboard knows it stopped.
PID_FILE = os.path.join("runs", "LATEST.pid")
with open(PID_FILE, "w") as _pf:
    _pf.write(str(os.getpid()))

def _clear_pid():
    try:
        if int(open(PID_FILE).read().strip()) == os.getpid():
            os.remove(PID_FILE)
    except (OSError, ValueError):
        pass

atexit.register(_clear_pid)

# ----------------------------------------------------------------------------
# Memory cap
#   On the GB10's 128 GB unified memory, letting the allocator grab 100% starves
#   the rest of the system and triggers the OOM crash seen partway through a run.
#   Cap each CUDA process at 85% of device memory and allow the allocator to
#   return fragmented blocks so peak usage stays under the ceiling.
# ----------------------------------------------------------------------------
MEMORY_FRACTION = 0.85
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if torch.cuda.is_available():
    for _gpu in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, _gpu)
    print(f"Capped CUDA memory to {MEMORY_FRACTION:.0%} per device "
          f"({torch.cuda.device_count()} GPU(s))")

print("\n" + "=" * 78)
print(f"  RUN: {RUN_NAME}")
print(f"  monitor:  python dashboard.py --metrics {METRICS_PATH} --config {CONFIG_PATH}")
print("=" * 78 + "\n")

# ----------------------------------------------------------------------------
# 1. Load model + tokenizer
# ----------------------------------------------------------------------------
model, tokenizer = FastModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = USE_4BIT,
    load_in_16bit   = not USE_4BIT,   # bf16 LoRA
    full_finetuning = False,
)

# ----------------------------------------------------------------------------
# 2. Attach LoRA adapters
#    finetune_vision_layers=False: text-only tune of a multimodal model, so we
#    freeze the ~35M image/audio embedder and train only the language layers.
# ----------------------------------------------------------------------------
model = FastModel.get_peft_model(
    model,
    r                          = LORA_R,
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = 0.0,
    bias                       = "none",
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth",
    random_state               = 3407,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules        = True,
)

# ----------------------------------------------------------------------------
# 3. Load + prepare the dataset
#    Either a HF hub dataset (dataset_source="hub") or a .json/.jsonl uploaded
#    through the dashboard (dataset_source="upload"). Both are expected to be the
#    standard OpenAI chat format: each row has a MESSAGES_FIELD list of
#    {role, content}, optionally a "category" column for filtering.
# ----------------------------------------------------------------------------
if CFG["dataset_source"] == "upload":
    data_file = CFG["dataset_file"]
    if not data_file or not os.path.exists(data_file):
        raise SystemExit(f"dataset_source='upload' but dataset_file={data_file!r} does not exist. "
                         f"Upload a .json/.jsonl from the dashboard first.")
    # The HF "json" builder reads both a top-level JSON array and JSON-lines.
    dataset = load_dataset("json", data_files=data_file, split="train")
    print(f"Loaded uploaded dataset {data_file}: {len(dataset)} examples")
else:
    dataset = load_dataset(CFG["dataset_name"], split=CFG["dataset_split"])
    print(f"Loaded hub dataset {CFG['dataset_name']} [{CFG['dataset_split']}]: {len(dataset)} examples")

if MESSAGES_FIELD not in dataset.column_names:
    raise SystemExit(f"dataset has no {MESSAGES_FIELD!r} column (found {dataset.column_names}). "
                     f"Set 'messages_field' in the dashboard to the column holding the chat turns.")

if FILTER_CATEGORY is not None:
    if "category" in dataset.column_names:
        dataset = dataset.filter(lambda row: row.get("category") == FILTER_CATEGORY)
        print(f"Filtered to category='{FILTER_CATEGORY}': {len(dataset)} examples")
    else:
        print(f"filter_category={FILTER_CATEGORY!r} set but the dataset has no 'category' column — "
              f"skipping the filter.")

def formatting_func(batch):
    # Apply Gemma 4's chat template. tokenize=False -> SFTTrainer tokenizes once.
    # Unsloth handles BOS so you don't get a double-BOS.
    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        for messages in batch[MESSAGES_FIELD]
    ]
    return {"text": texts}

dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)

# ----------------------------------------------------------------------------
# 3b. Resolve the turn markers used to mask the prompt from the loss.
#     They depend on the base model's chat template, so they are configurable
#     from the dashboard ('instruction_part' / 'response_part'). Left blank, we
#     render a probe conversation through the tokenizer's template and match it
#     against known marker pairs. Either way the markers are validated against
#     the rendered template before training — a mismatch would mask every token
#     to -100 and Unsloth would silently drop the whole dataset.
# ----------------------------------------------------------------------------
KNOWN_MARKERS = [
    ("<|im_start|>user\n",    "<|im_start|>assistant\n"),   # ChatML (Qwen, Qwythos, ...)
    ("<|turn>user\n",         "<|turn>model\n"),            # Gemma 4
    ("<start_of_turn>user\n", "<start_of_turn>model\n"),    # Gemma 2/3
    ("<|start_header_id|>user<|end_header_id|>\n\n",
     "<|start_header_id|>assistant<|end_header_id|>\n\n"),  # Llama 3
    ("[INST]",                "[/INST]"),                    # Llama 2 / Mistral
]

_probe = tokenizer.apply_chat_template(
    [{"role": "user", "content": "PROBE_USER"},
     {"role": "assistant", "content": "PROBE_ASSISTANT"}],
    tokenize=False, add_generation_prompt=False,
)

if not INSTRUCTION_PART or not RESPONSE_PART:
    for _inst, _resp in KNOWN_MARKERS:
        if _inst in _probe and _resp in _probe:
            INSTRUCTION_PART, RESPONSE_PART = _inst, _resp
            print(f"Auto-detected turn markers: instruction={INSTRUCTION_PART!r} "
                  f"response={RESPONSE_PART!r}")
            break
    else:
        raise SystemExit(
            "Could not auto-detect the chat-template turn markers for this model.\n"
            "Set 'instruction_part' / 'response_part' in the dashboard (use \\n for newlines).\n"
            f"The template renders a user+assistant exchange as:\n{_probe}"
        )
elif INSTRUCTION_PART not in _probe or RESPONSE_PART not in _probe:
    raise SystemExit(
        f"Configured turn markers not found in this model's chat template:\n"
        f"  instruction_part={INSTRUCTION_PART!r}\n"
        f"  response_part={RESPONSE_PART!r}\n"
        f"Training would mask every token (all labels -100). The template renders a "
        f"user+assistant exchange as:\n{_probe}"
    )

# ----------------------------------------------------------------------------
# 4. Record hyperparameters for the dashboard (before training starts)
# ----------------------------------------------------------------------------
_dataset_desc = (CFG["dataset_file"] if CFG["dataset_source"] == "upload" else CFG["dataset_name"])
write_run_config(
    CONFIG_PATH,
    run_name        = RUN_NAME,
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = USE_4BIT,
    dataset_source  = CFG["dataset_source"],
    dataset         = _dataset_desc,
    filter_category = str(FILTER_CATEGORY),
    instruction_part = INSTRUCTION_PART,
    response_part    = RESPONSE_PART,
    num_examples    = len(dataset),
    lora_r          = LORA_R,
    lora_alpha      = LORA_ALPHA,
    batch_size      = BATCH_SIZE,
    grad_accum      = GRAD_ACCUM,
    eff_batch       = BATCH_SIZE * GRAD_ACCUM,
    learning_rate   = LEARNING_RATE,
    epochs          = EPOCHS,
    optim           = OPTIM,
)

# ----------------------------------------------------------------------------
# 5. Trainer
# ----------------------------------------------------------------------------
trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = dataset,
    args = SFTConfig(
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ_LENGTH,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_steps                = 5,
        num_train_epochs            = EPOCHS,
        learning_rate               = LEARNING_RATE,
        logging_steps               = 1,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        save_total_limit            = SAVE_TOTAL_LIMIT,
        optim                       = OPTIM,
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        seed                        = 3407,
        output_dir                  = OUTPUT_DIR,
        report_to                   = "none",
        bf16                        = True,
    ),
)

# ----------------------------------------------------------------------------
# 6. Train ONLY on the assistant turns (mask the prompt from the loss).
#    Markers were resolved + validated against the chat template in section 3b.
# ----------------------------------------------------------------------------
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = INSTRUCTION_PART,
    response_part    = RESPONSE_PART,
)

# Attach the dashboard logger (writes one JSON line per log step to RUN_DIR/metrics.jsonl).
# On resume, append to the existing metrics file instead of truncating it.
trainer.add_callback(JSONLLoggerCallback(METRICS_PATH, resume=bool(RESUME_RUN)))
# Ctrl-C -> checkpoint + clean stop (see GracefulPauseCallback).
trainer.add_callback(GracefulPauseCallback())

# Resume from the latest *complete* checkpoint in OUTPUT_DIR when continuing a run.
#   A power cut during a save leaves a half-written checkpoint-N/ (missing
#   trainer_state.json / optimizer.pt). resume_from_checkpoint=True blindly grabs
#   the highest-numbered dir and crashes on the corrupt one, so we hand the Trainer
#   the explicit path of the newest checkpoint that actually finished writing and
#   skip any partial ones (SAVE_TOTAL_LIMIT keeps older good checkpoints as fallback).
def _latest_valid_checkpoint(output_dir):
    required = ("trainer_state.json", "optimizer.pt")  # both load-critical for resume
    ckpts = []
    for d in os.listdir(output_dir):
        if not d.startswith("checkpoint-"):
            continue
        try:
            step = int(d.split("-")[1])
        except (IndexError, ValueError):
            continue
        ckpts.append((step, os.path.join(output_dir, d)))
    for step, path in sorted(ckpts, reverse=True):
        if all(os.path.exists(os.path.join(path, f)) for f in required):
            return path
        print(f"Skipping incomplete checkpoint (power cut mid-save?): {path}")
    return None

resume_ckpt = None
if RESUME_RUN:
    resume_ckpt = _latest_valid_checkpoint(OUTPUT_DIR)
    if resume_ckpt:
        print(f"Resuming from {resume_ckpt}")
    else:
        print(f"No complete checkpoint under {OUTPUT_DIR} — starting this run from step 0.")

trainer_stats = trainer.train(resume_from_checkpoint=resume_ckpt)
print(trainer_stats)

# ----------------------------------------------------------------------------
# 7. Save the LoRA adapters
# ----------------------------------------------------------------------------
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"Saved LoRA adapters to ./{LORA_DIR}")

# ----------------------------------------------------------------------------
# 8. (Optional) Export a merged 4-bit GGUF for serving on the Spark via llama.cpp
#    at ~30-40 tok/s.
# ----------------------------------------------------------------------------
if EXPORT_GGUF:
    model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")
    print(f"Exported GGUF to ./{GGUF_DIR}")
    print(f"Serve:  llama-server -m {GGUF_DIR}/*.gguf --jinja -c 8192 -ngl 999")
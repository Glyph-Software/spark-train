"""
Lightweight training instrumentation for the dashboard.

Decoupled by design: the training process only ever *writes* two files
(metrics.jsonl, run_config.json). The dashboard only ever *reads* them.
Nothing the dashboard does can affect or crash the training run.

Usage in train.py — add three things:

    from train_metrics import JSONLLoggerCallback, write_run_config

    # ...after `dataset` is built, before trainer.train():
    write_run_config(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=USE_4BIT, filter_category=str(FILTER_CATEGORY),
        num_examples=len(dataset), lora_r=16, lora_alpha=16,
        batch_size=2, grad_accum=4, learning_rate=2e-4,
        epochs=2, optim="adamw_torch_fused",
    )
    trainer.add_callback(JSONLLoggerCallback("metrics.jsonl"))
"""

import json
import signal
import time
from transformers import TrainerCallback


class JSONLLoggerCallback(TrainerCallback):
    """Append one JSON line per Trainer log event (loss, lr, grad_norm, ...)."""

    def __init__(self, path="metrics.jsonl", resume=False):
        self.path = path
        if not resume:
            open(self.path, "w").close()      # truncate any previous run
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        rec = {
            "step": int(state.global_step),
            "max_steps": int(state.max_steps or 0),
            "wall_time": round(time.time() - self.t0, 2),
        }
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                rec[k] = v
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()


class GracefulPauseCallback(TrainerCallback):
    """Turn Ctrl-C (SIGINT) / SIGTERM into a clean *pause* instead of a kill.

    On the first signal it lets the in-flight step finish, then asks the Trainer
    to write a checkpoint and stop. Resume later from that checkpoint with the
    same output_dir (see RESUME_RUN in train.py). Press Ctrl-C a second time to
    abort immediately, the normal way.

    Must be constructed on the main thread (signal handlers can only register
    there), which is where trainer.train() runs.
    """

    def __init__(self):
        self._pause = False
        self._orig_int = signal.signal(signal.SIGINT, self._request)
        try:
            self._orig_term = signal.signal(signal.SIGTERM, self._request)
        except (ValueError, OSError):
            self._orig_term = None

    def _request(self, signum, frame):
        if self._pause:
            # second signal -> restore default handler and re-raise as a hard stop
            signal.signal(signal.SIGINT, self._orig_int or signal.SIG_DFL)
            raise KeyboardInterrupt
        print("\n[pause] requested — finishing this step, then checkpointing "
              "(Ctrl-C again to abort now)", flush=True)
        self._pause = True

    def on_step_end(self, args, state, control, **kwargs):
        if self._pause:
            control.should_save = True
            control.should_training_stop = True
        return control


def write_run_config(path="run_config.json", **kwargs):
    """Dump the hyperparameters so the dashboard can display them."""
    kwargs["_started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w") as f:
        json.dump(kwargs, f, indent=2)
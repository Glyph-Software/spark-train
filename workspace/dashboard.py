#!/usr/bin/env python3
"""
Standalone training + GPU dashboard for the DGX Spark / GB10.

Reads the files produced by train_metrics.py:
  - metrics.jsonl   (per-step loss / lr / grad_norm)
  - run_config.json (hyperparameters of the run in progress)
Samples the GPU itself via nvidia-smi.

Also drives training (unless --no-control):
  - Training Configuration form  -> writes train_config.json (read by train.py)
  - Upload JSON/JSONL dataset    -> saved under datasets/, selected as the training set
  - Start / Pause / Resume       -> launches or checkpoints train.py

Pure Python stdlib — no Flask, no pip installs, no CDN. Safe for a locked-down container.

RUN (in a second terminal, from the same dir as train.py / metrics.jsonl):
    python dashboard.py
    # then open http://<spark-ip>:7860
    # or tunnel from your laptop:  ssh -L 7860:localhost:7860 user@spark

Options:
    python dashboard.py --port 7860 --metrics metrics.jsonl --config run_config.json
"""

import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.parse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------- #
# Editable training config (mirrors DEFAULTS in train.py). The dashboard writes
# train_config.json; train.py reads it. Keeping the keys/types here lets the
# dashboard coerce a submitted form and always hand the front-end a full config.
# --------------------------------------------------------------------------- #
CONFIG_DEFAULTS = {
    "model_name":      "unsloth/gemma-4-12b-it",
    "max_seq_length":  8192,
    "use_4bit":        False,
    "dataset_source":  "hub",          # "hub" or "upload"
    "dataset_name":    "angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k",
    "dataset_split":   "train",
    "dataset_file":    "",
    "messages_field":  "messages",
    "filter_category": "coding",
    "instruction_part": "",       # chat-template marker opening a user turn ("" = auto-detect)
    "response_part":    "",       # chat-template marker opening an assistant turn ("" = auto-detect)
    "lora_r":          16,
    "lora_alpha":      16,
    "batch_size":      2,
    "grad_accum":      4,
    "learning_rate":   2e-4,
    "epochs":          2,
    "optim":           "adamw_torch_fused",
    "export_gguf":     False,
    "run_label":       "gemma4-coding",
}

MAX_UPLOAD_BYTES = 512 * 1024 * 1024   # 512 MB ceiling for an uploaded dataset


def coerce_config(raw):
    """Return a full config dict: every key from CONFIG_DEFAULTS, values coerced
    to the default's type. Unknown keys are dropped; bad values fall back."""
    cfg = dict(CONFIG_DEFAULTS)
    if not isinstance(raw, dict):
        return cfg
    for k, default in CONFIG_DEFAULTS.items():
        if k not in raw:
            continue
        v = raw[k]
        if isinstance(default, bool):                       # bool before int (bool is an int subclass)
            cfg[k] = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(default, int):
            try:
                cfg[k] = int(float(v))
            except (ValueError, TypeError):
                cfg[k] = default
        elif isinstance(default, float):
            try:
                cfg[k] = float(v)
            except (ValueError, TypeError):
                cfg[k] = default
        else:
            cfg[k] = "" if v is None else str(v)
    return cfg


def safe_dataset_filename(name):
    """A filesystem-safe basename ending in .json/.jsonl for an uploaded dataset."""
    name = os.path.basename((name or "").strip()) or "dataset.json"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if not name.lower().endswith((".json", ".jsonl")):
        name += ".json"
    return name


def validate_dataset_file(path, field):
    """Parse an uploaded .json (array) or .jsonl file: count rows and check the
    messages field is present. Returns {ok, rows, warning} or {ok:False, error}."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except OSError as e:
        return {"ok": False, "error": str(e)}
    if not text:
        return {"ok": False, "error": "file is empty"}
    try:                                            # whole-file JSON (array or single object)
        data = json.loads(text)
        rows = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:                    # fall back to JSON-lines
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                return {"ok": False,
                        "error": f"line {i} is not valid JSON (and the file is not a JSON array)"}
    if not rows:
        return {"ok": False, "error": "no records found"}
    if not isinstance(rows[0], dict):
        return {"ok": False, "error": "records must be JSON objects"}
    warning = None
    if field not in rows[0]:
        keys = list(rows[0].keys())[:6]
        warning = (f"no {field!r} field in the first record (keys: {keys}) — "
                   f"set 'messages field' to match, or training will fail")
    return {"ok": True, "rows": len(rows), "field": field, "warning": warning}

# --------------------------------------------------------------------------- #
# GPU sampling
# --------------------------------------------------------------------------- #
GPU_FIELDS = "name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw"
_gpu_history = deque(maxlen=360)   # ~12 min at 2s cadence
_gpu_latest = {"ok": False, "error": "starting up"}


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _unified_mem_mib():
    """GB10 / Grace-Blackwell has no dedicated VRAM — the GPU shares system
    LPDDR5X. nvidia-smi reports memory as N/A, so read the unified pool from
    /proc/meminfo instead. Returns (used_MiB, total_MiB) or (None, None).
    """
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                info[k] = float(rest.split()[0])  # value in kB
        total_kb = info.get("MemTotal")
        avail_kb = info.get("MemAvailable", info.get("MemFree"))
        if total_kb is None or avail_kb is None:
            return None, None
        return (total_kb - avail_kb) / 1024.0, total_kb / 1024.0
    except (OSError, ValueError, IndexError):
        return None, None


def sample_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return {"ok": False, "error": (out.stderr or "nvidia-smi failed").strip()}
        line = out.stdout.strip().splitlines()[0]
        p = [c.strip() for c in line.split(",")]
        mem_used, mem_total = _num(p[3]), _num(p[4])
        mem_unified = False
        if mem_used is None or mem_total is None:
            # GB10 unified memory: nvidia-smi returns N/A — use system RAM.
            mem_used, mem_total = _unified_mem_mib()
            mem_unified = True
        return {
            "ok": True,
            "name": p[0],
            "temp": _num(p[1]),
            "util": _num(p[2]),
            "mem_used": mem_used,
            "mem_total": mem_total,
            "mem_unified": mem_unified,
            "power": _num(p[5]),
        }
    except FileNotFoundError:
        return {"ok": False, "error": "nvidia-smi not found on PATH"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def gpu_sampler_loop(interval=2.0):
    global _gpu_latest
    while True:
        g = sample_gpu()
        _gpu_latest = g
        if g.get("ok"):
            _gpu_history.append({
                "t": round(time.time(), 1),
                "util": g.get("util"),
                "temp": g.get("temp"),
            })
        time.sleep(interval)


# --------------------------------------------------------------------------- #
# Reading the training files
# --------------------------------------------------------------------------- #
def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass
    return rows


def read_json(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def tail_log(path, max_lines=400, max_bytes=256 * 1024):
    """Return the last `max_lines` lines of a (possibly huge, live) log file.

    Reads only the tail of the file for speed. Collapses carriage-return
    progress-bar spam (e.g. tqdm) so each logical line shows its final state.
    """
    if not os.path.exists(path):
        return ""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # drop the partial first line
            data = f.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    out = []
    for line in text.splitlines():
        # keep only what's after the last \r — that's the final rendered state
        out.append(line.rsplit("\r", 1)[-1])
    return "\n".join(out[-max_lines:])


def latest_run_dir(runs_dir="runs"):
    """Return the most recently modified run directory under `runs_dir`, or None."""
    try:
        dirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)]
    except OSError:
        return None
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


# --------------------------------------------------------------------------- #
# Training process control (pause = SIGTERM -> checkpoint; resume = relaunch)
# --------------------------------------------------------------------------- #
def read_pid(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def pid_alive(pid):
    """True only if `pid` is a live *training* process, not just any live PID.

    A hard kill / power cut never runs train.py's atexit cleanup, so runs/LATEST.pid
    is left pointing at a dead — and after a reboot possibly recycled — PID. A bare
    os.kill(pid, 0) liveness probe would then report an unrelated process as
    'training still running', hiding the Resume button and making resume reject with
    'already running'. Confirming the cmdline is train.py closes that gap."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)          # signal 0 = liveness probe, doesn't touch the process
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return True              # no /proc (non-Linux): fall back to bare liveness
    return "train.py" in cmdline


def training_resumable(outputs_dir="outputs"):
    """True only if a checkpoint exists to resume from. Without this, a stopped
    state means 'never started', not 'paused' — so the Resume button must hide."""
    try:
        for run in os.listdir(outputs_dir):
            rundir = os.path.join(outputs_dir, run)
            if not os.path.isdir(rundir):
                continue
            for sub in os.listdir(rundir):
                if sub.startswith("checkpoint-") and os.path.isdir(os.path.join(rundir, sub)):
                    return True
    except OSError:
        pass
    return False


def _proc_rss_kb(pid):
    """Resident set size of a single process, in kB (0 if it's gone)."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _descendant_pids(root):
    """`root` plus every process descended from it (children, grandchildren, ...).
    Lets us attribute dataloader workers' memory to the training run too."""
    children = {}
    try:
        pids = [int(d) for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return [root]
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat") as f:
                data = f.read()
            # comm (field 2) may contain spaces/parens; fields after the last ')'
            # are: state ppid pgrp ...  -> ppid is index 1
            ppid = int(data[data.rfind(")") + 1:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(pid)
    out, seen, stack = [], set(), [root]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        stack.extend(children.get(p, []))
    return out


def memory_breakdown(train_pid):
    """Stacked-bar segments (MiB) for the unified pool: training process tree vs
    every other process vs free. nvidia-smi can't attribute GPU memory per-PID on
    the GB10, so the training share is /proc RSS of its process tree."""
    used_mib, total_mib = _unified_mem_mib()
    if total_mib is None:
        return None
    train_mib = 0.0
    if train_pid and pid_alive(train_pid):
        train_mib = sum(_proc_rss_kb(p) for p in _descendant_pids(train_pid)) / 1024.0
        train_mib = min(train_mib, used_mib)        # never exceed what's actually used
    return {
        "total": round(total_mib, 1),
        "train": round(train_mib, 1),
        "other": round(max(0.0, used_mib - train_mib), 1),
        "free": round(max(0.0, total_mib - used_mib), 1),
        "pid": train_pid if (train_pid and pid_alive(train_pid)) else None,
    }


def pause_training(pid):
    """Ask the training run to checkpoint and stop (its GracefulPauseCallback
    turns SIGTERM into a clean pause)."""
    try:
        os.kill(pid, signal.SIGTERM)
        return {"ok": True, "msg": f"sent SIGTERM to pid {pid} — checkpointing, then stopping"}
    except OSError as e:
        return {"ok": False, "error": str(e)}


def launch_training(cmd, log_path, append=True):
    """(Re)launch training detached, output written to the log the dashboard tails.
    `cmd` is run through the shell so `RESUME_RUN=latest python train.py` works.
    append=False truncates the log first (a fresh Start); append=True continues it
    (a Resume)."""
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        logf = open(log_path, "ab" if append else "wb")
        # start_new_session detaches it from the dashboard so it survives a dashboard restart
        subprocess.Popen(
            cmd, shell=True, stdout=logf, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        return {"ok": True, "msg": f"launched: {cmd}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    # When metrics_path/config_path are None, auto-discover the latest run dir.
    metrics_path = None
    config_path = None
    runs_dir = "runs"
    log_path = "runs/LATEST_train.log"
    stream_interval = 1.0
    # Training control (start/pause/resume). Disable with --no-control.
    pid_file = "runs/LATEST.pid"
    train_cmd = "RESUME_RUN=latest python train.py"   # Resume: continue the latest run
    train_start_cmd = "python train.py"               # Start: fresh run from train_config.json
    outputs_dir = "outputs"
    control_enabled = True
    # Editable training config + uploaded-dataset storage (dashboard writes, train.py reads).
    train_config_path = "train_config.json"
    datasets_dir = "datasets"

    def _metrics_path(self):
        if self.metrics_path:
            return self.metrics_path
        d = latest_run_dir(self.runs_dir)
        return os.path.join(d, "metrics.jsonl") if d else "metrics.jsonl"

    def _config_path(self):
        if self.config_path:
            return self.config_path
        d = latest_run_dir(self.runs_dir)
        return os.path.join(d, "run_config.json") if d else "run_config.json"

    def log_message(self, *a):  # silence per-request console spam
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _train_status(self):
        pid = read_pid(self.pid_file)
        running = pid_alive(pid)
        return {
            "control": self.control_enabled,
            "running": running,
            "pid": pid if running else None,
            "resumable": training_resumable(self.outputs_dir),
        }

    def _snapshot(self):
        return {
            "gpu": _gpu_latest,
            "gpu_history": list(_gpu_history),
            "metrics": read_jsonl(self._metrics_path()),
            "config": read_json(self._config_path()),
            "log": tail_log(self.log_path),
            "train": self._train_status(),
            "train_config": {**CONFIG_DEFAULTS, **(read_json(self.train_config_path) or {})},
            "mem": memory_breakdown(read_pid(self.pid_file)),
            "now": time.time(),
        }

    def do_GET(self):
        if self.path.startswith("/stream"):
            self._serve_stream()
        elif self.path.startswith("/data"):
            self._send(200, json.dumps(self._snapshot()).encode(), "application/json")
        else:
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path.startswith("/control"):
            self._handle_control()
        elif self.path.startswith("/train_config"):
            self._handle_save_config()
        elif self.path.startswith("/upload"):
            self._handle_upload()
        else:
            self._send(404, b'{"ok":false,"error":"not found"}', "application/json")

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        return self.rfile.read(length) if length > 0 else b""

    def _handle_save_config(self):
        """Persist the dashboard-edited training config to train_config.json."""
        if not self.control_enabled:
            self._send(403, b'{"ok":false,"error":"control disabled (--no-control)"}',
                       "application/json")
            return
        try:
            raw = json.loads(self._read_body() or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, b'{"ok":false,"error":"invalid JSON body"}', "application/json")
            return
        cfg = coerce_config(raw)
        try:
            with open(self.train_config_path, "w") as f:
                json.dump(cfg, f, indent=2)
        except OSError as e:
            self._send(500, json.dumps({"ok": False, "error": str(e)}).encode(), "application/json")
            return
        self._send(200, json.dumps({"ok": True, "config": cfg}).encode(), "application/json")

    def _handle_upload(self):
        """Store an uploaded .json/.jsonl dataset under datasets_dir and validate it.
        The raw file is the POST body; ?name= is the original filename, ?field= the
        messages column to check for."""
        if not self.control_enabled:
            self._send(403, b'{"ok":false,"error":"control disabled (--no-control)"}',
                       "application/json")
            return
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        fname = safe_dataset_filename(params.get("name", ["dataset.json"])[0])
        field = (params.get("field", ["messages"])[0] or "messages")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._send(400, b'{"ok":false,"error":"empty upload"}', "application/json")
            return
        if length > MAX_UPLOAD_BYTES:
            self._send(413, json.dumps(
                {"ok": False, "error": f"file too large (> {MAX_UPLOAD_BYTES // (1024*1024)} MB)"}
            ).encode(), "application/json")
            return
        try:
            os.makedirs(self.datasets_dir, exist_ok=True)
            path = os.path.join(self.datasets_dir, fname)
            remaining = length
            with open(path, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
        except OSError as e:
            self._send(500, json.dumps({"ok": False, "error": str(e)}).encode(), "application/json")
            return
        info = validate_dataset_file(path, field)
        if not info.get("ok"):
            try:
                os.remove(path)          # don't leave an unusable file behind
            except OSError:
                pass
            self._send(400, json.dumps(info).encode(), "application/json")
            return
        info["path"] = path
        self._send(200, json.dumps(info).encode(), "application/json")

    def _handle_control(self):
        if not self.control_enabled:
            self._send(403, b'{"ok":false,"error":"control disabled (--no-control)"}',
                       "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b"{}"
            action = (json.loads(body or b"{}") or {}).get("action")
        except (ValueError, json.JSONDecodeError):
            action = None

        pid = read_pid(self.pid_file)
        alive = pid_alive(pid)
        if action == "pause":
            result = (pause_training(pid) if alive
                      else {"ok": False, "error": "no running training process"})
        elif action == "start":
            # Fresh run using the current train_config.json. Truncate the log so
            # the dashboard shows this run cleanly rather than the previous one's tail.
            if alive:
                result = {"ok": False, "error": f"training already running (pid {pid})"}
            else:
                result = launch_training(self.train_start_cmd, self.log_path, append=False)
        elif action == "resume":
            if alive:
                result = {"ok": False, "error": f"training already running (pid {pid})"}
            elif not training_resumable(self.outputs_dir):
                result = {"ok": False, "error": "no checkpoint to resume from"}
            else:
                result = launch_training(self.train_cmd, self.log_path, append=True)
        else:
            result = {"ok": False, "error": f"unknown action: {action!r}"}

        code = 200 if result.get("ok") else 400
        self._send(code, json.dumps(result).encode(), "application/json")

    def _serve_stream(self):
        """Server-Sent Events: push a fresh snapshot every `stream_interval` sec."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
        self.end_headers()
        try:
            while True:
                msg = "data: " + json.dumps(self._snapshot()) + "\n\n"
                self.wfile.write(msg.encode())
                self.wfile.flush()
                time.sleep(self.stream_interval)
        except (BrokenPipeError, ConnectionResetError):
            return  # client closed the tab / navigated away


# --------------------------------------------------------------------------- #
# Front-end (self-contained: system monospace, vanilla canvas charts)
# --------------------------------------------------------------------------- #
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GB10 // training monitor</title>
<style>
  :root{
    --bg:#0a0e14; --panel:#11161f; --panel2:#0d1219; --line:#1d2733;
    --ink:#c7d1dd; --dim:#5d6b7d; --label:#8794a5;
    --loss:#ff7849; --lr:#39b6ff; --grad:#3ddc84; --util:#39b6ff; --temp:#ffb454;
    --grid:#16202c;
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:
      radial-gradient(1200px 600px at 80% -10%, #101826 0%, transparent 60%),
      var(--bg);
    color:var(--ink);
    font-family:ui-monospace,"SF Mono","JetBrains Mono","Cascadia Code",Menlo,Consolas,monospace;
    font-size:13px; line-height:1.45; letter-spacing:.2px;
  }
  .wrap{max-width:1180px; margin:0 auto; padding:22px 20px 60px}
  header{display:flex; align-items:baseline; gap:14px; border-bottom:1px solid var(--line);
    padding-bottom:12px; margin-bottom:20px}
  header h1{font-size:15px; font-weight:600; margin:0; letter-spacing:1.5px; text-transform:uppercase}
  header .dot{width:8px;height:8px;border-radius:50%;background:var(--grad);
    box-shadow:0 0 10px var(--grad); animation:pulse 1.8s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
  header .meta{margin-left:auto; color:var(--dim); font-size:11px}
  .grid{display:grid; gap:14px}
  .row1{grid-template-columns:1.15fr .85fr; align-items:start}
  .row2{grid-template-columns:repeat(2,1fr); margin-top:14px}
  @media(max-width:860px){.row1,.row2{grid-template-columns:1fr}}
  /* left side of row1: GPU Telemetry stacked above Memory Usage; RUN spans both */
  .col-stack{display:flex; flex-direction:column; gap:14px}
  .panel{background:linear-gradient(180deg,var(--panel),var(--panel2));
    border:1px solid var(--line); border-radius:10px; padding:16px 18px; position:relative}
  .panel h2{font-size:10px; letter-spacing:2px; text-transform:uppercase; color:var(--label);
    margin:0 0 12px; font-weight:600}
  .gpu-name{font-size:18px; font-weight:600; color:#fff; letter-spacing:.5px}
  .stat-row{display:flex; gap:22px; margin-top:14px; flex-wrap:wrap}
  .stat .v{font-size:26px; font-weight:600; color:#fff; font-variant-numeric:tabular-nums}
  .stat .u{font-size:11px; color:var(--dim)}
  .stat .k{font-size:10px; color:var(--label); letter-spacing:1px; text-transform:uppercase}
  .bar{height:7px; border-radius:4px; background:#0a0f16; overflow:hidden; margin-top:7px;
    border:1px solid var(--line)}
  .bar > i{display:block; height:100%; border-radius:4px; transition:width .4s ease}
  table{width:100%; border-collapse:collapse; font-size:12px}
  td{padding:5px 0; border-bottom:1px dashed var(--line); vertical-align:top}
  td.k{color:var(--label)} td.v{text-align:right; color:#e8eef5; font-variant-numeric:tabular-nums}
  tr:last-child td{border-bottom:none}
  canvas{width:100%; height:auto; display:block}
  .spark{height:120px}
  .chart-title{display:flex; justify-content:space-between; align-items:baseline}
  .chart-title .last{font-size:18px; font-weight:600; font-variant-numeric:tabular-nums}
  .err{color:var(--temp); font-size:12px}
  .progress-head{display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px}
  .progress-head .pct{font-size:22px; font-weight:600; color:#fff}
  .muted{color:var(--dim)}
  .row3{margin-top:14px}
  .log-head{display:flex; justify-content:space-between; align-items:center; margin-bottom:10px}
  .log-box{background:#070a0f; border:1px solid var(--line); border-radius:8px;
    padding:12px 14px; height:340px; overflow:auto; white-space:pre-wrap; word-break:break-word;
    font-size:11.5px; line-height:1.5; color:#aeb9c7}
  .log-box::-webkit-scrollbar{width:10px} .log-box::-webkit-scrollbar-track{background:#070a0f}
  .log-box::-webkit-scrollbar-thumb{background:#1d2733; border-radius:5px}
  .toggle{font-size:10px; letter-spacing:1px; text-transform:uppercase; color:var(--label);
    cursor:pointer; user-select:none; display:flex; align-items:center; gap:6px}
  .toggle input{accent-color:var(--util)}
  .ctrl{display:flex; align-items:center; gap:10px}
  .status{font-size:9px; letter-spacing:1px; text-transform:uppercase; padding:3px 9px;
    border-radius:20px; border:1px solid var(--line); color:var(--dim); white-space:nowrap}
  .status.run{color:var(--grad); border-color:#1f6f43}
  .status.stop{color:var(--temp); border-color:#5b4524}
  .btn{font-family:inherit; font-size:10px; letter-spacing:1px; text-transform:uppercase;
    cursor:pointer; background:var(--panel); color:var(--ink); border:1px solid var(--line);
    border-radius:6px; padding:5px 13px; transition:border-color .2s,opacity .2s}
  .btn:hover:not(:disabled){border-color:var(--util)}
  .btn:disabled{opacity:.4; cursor:default}
  .btn.pause{color:#ff8a8a; border-color:#5b2a2a}
  .btn.resume{color:var(--grad); border-color:#1f6f43}
  .ctrl-msg{font-size:10px; color:var(--dim); margin-top:8px; min-height:13px}
  details > summary{font-size:10px; letter-spacing:2px; text-transform:uppercase; color:var(--label);
    font-weight:600; cursor:pointer; user-select:none; list-style:none;
    display:flex; align-items:center; gap:7px; margin-bottom:12px}
  details > summary::-webkit-details-marker{display:none}
  details > summary::before{content:'\25B8'; font-size:9px; color:var(--dim); transition:transform .2s}
  details[open] > summary::before{transform:rotate(90deg)}
  details > summary:hover{color:var(--ink)}
  /* compact 2-column key/value grid for hyperparameters: one line per item */
  .cfg-grid{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:0 22px; font-size:11.5px}
  .cfg-grid .item{display:flex; justify-content:space-between; align-items:baseline; gap:10px;
    min-width:0; padding:5px 0; border-bottom:1px dashed var(--line)}
  .cfg-grid .k{color:var(--label); white-space:nowrap}
  .cfg-grid .v{color:#e8eef5; font-variant-numeric:tabular-nums; text-align:right;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .stackbar{display:flex; height:13px; border-radius:7px; overflow:hidden; background:#0a0f16;
    border:1px solid var(--line); margin:10px 0 6px}
  .stackbar > span{display:block; height:100%; transition:width .4s ease}
  .legend .row{display:flex; align-items:center; padding:7px 0; border-bottom:1px dashed var(--line);
    font-size:12px}
  .legend .row:last-child{border-bottom:none}
  .legend .sw{width:11px; height:11px; border-radius:3px; margin-right:11px; flex:none}
  .legend .lab{color:var(--ink)}
  .legend .pid{color:var(--dim); margin-left:7px; font-size:10px}
  .legend .val{margin-left:auto; color:#e8eef5; font-variant-numeric:tabular-nums}
  /* training-config editor form */
  .rowcfg{margin-bottom:14px}
  .form-grid{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px 20px; margin:6px 0 14px}
  @media(max-width:860px){.form-grid{grid-template-columns:1fr}}
  .field{display:flex; flex-direction:column; gap:5px; min-width:0}
  .field.full{grid-column:1/-1}
  .field > span{font-size:10px; letter-spacing:1px; text-transform:uppercase; color:var(--label)}
  .field input,.field select{font-family:inherit; font-size:12px; background:#0a0f16; color:#e8eef5;
    border:1px solid var(--line); border-radius:6px; padding:7px 9px; width:100%}
  .field input:focus,.field select:focus{outline:none; border-color:var(--util)}
  .field input[readonly]{color:var(--dim); background:#080c11}
  .upload-row{display:flex; gap:8px; align-items:center; flex-wrap:wrap}
  .upload-row input[type=file]{font-size:11px; color:var(--dim); flex:1; min-width:180px}
  .form-actions{display:flex; align-items:center; gap:12px; flex-wrap:wrap}
  #uploadMsg{font-size:10px; color:var(--dim); min-height:13px; margin-top:6px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <span class="dot"></span>
    <h1>GB10 Training Monitor</h1>
    <span class="meta" id="meta">connecting…</span>
  </header>

  <div class="grid rowcfg">
    <div class="panel">
      <details id="cfgPanel" open>
        <summary>Training Configuration</summary>
        <div class="form-grid">
          <label class="field full"><span>Base model (HF id)</span>
            <input id="f_model_name" type="text" placeholder="unsloth/gemma-4-12b-it"></label>

          <label class="field"><span>Precision</span>
            <select id="f_use_4bit">
              <option value="false">16-bit LoRA (recommended)</option>
              <option value="true">4-bit QLoRA</option>
            </select></label>
          <label class="field"><span>Max seq length</span>
            <input id="f_max_seq_length" type="number" min="1" step="1"></label>
          <label class="field"><span>Run label</span>
            <input id="f_run_label" type="text" placeholder="gemma4-coding"></label>

          <label class="field"><span>Dataset source</span>
            <select id="f_dataset_source">
              <option value="hub">Hugging Face hub</option>
              <option value="upload">Uploaded JSON file</option>
            </select></label>
          <label class="field"><span>Messages field</span>
            <input id="f_messages_field" type="text" placeholder="messages"></label>
          <label class="field"><span>Filter category (blank = all)</span>
            <input id="f_filter_category" type="text" placeholder="coding"></label>

          <label class="field"><span>Instruction marker (blank = auto)</span>
            <input id="f_instruction_part" type="text" placeholder="&lt;|im_start|&gt;user\n"></label>
          <label class="field"><span>Response marker (blank = auto)</span>
            <input id="f_response_part" type="text" placeholder="&lt;|im_start|&gt;assistant\n"></label>

          <div id="hubFields" class="field full" style="display:contents">
            <label class="field"><span>Hub dataset name</span>
              <input id="f_dataset_name" type="text" placeholder="org/dataset"></label>
            <label class="field"><span>Split</span>
              <input id="f_dataset_split" type="text" placeholder="train"></label>
          </div>

          <div id="uploadFields" class="field full" style="display:none">
            <span>Upload JSON / JSONL dataset (rows of {"messages":[…]})</span>
            <div class="upload-row">
              <input id="f_dataset_upload" type="file" accept=".json,.jsonl,application/json">
              <button id="uploadBtn" class="btn" type="button">Upload</button>
            </div>
            <input id="f_dataset_file" type="text" readonly placeholder="no file uploaded yet"
                   style="margin-top:8px">
            <div id="uploadMsg"></div>
          </div>

          <label class="field"><span>LoRA r</span>
            <input id="f_lora_r" type="number" min="1" step="1"></label>
          <label class="field"><span>LoRA alpha</span>
            <input id="f_lora_alpha" type="number" min="1" step="1"></label>
          <label class="field"><span>Optimizer</span>
            <input id="f_optim" type="text" placeholder="adamw_torch_fused"></label>

          <label class="field"><span>Batch size</span>
            <input id="f_batch_size" type="number" min="1" step="1"></label>
          <label class="field"><span>Grad accumulation</span>
            <input id="f_grad_accum" type="number" min="1" step="1"></label>
          <label class="field"><span>Epochs</span>
            <input id="f_epochs" type="number" min="1" step="1"></label>

          <label class="field"><span>Learning rate</span>
            <input id="f_learning_rate" type="text" placeholder="2e-4"></label>
          <label class="field"><span>Export GGUF</span>
            <select id="f_export_gguf">
              <option value="false">No</option>
              <option value="true">Yes (q4_k_m)</option>
            </select></label>
        </div>
        <div class="form-actions">
          <button id="saveCfgBtn" class="btn" type="button">Save config</button>
          <button id="startBtn" class="btn resume" type="button">Start training</button>
          <span id="cfgMsg" class="ctrl-msg"></span>
        </div>
      </details>
    </div>
  </div>

  <div class="grid row1">
    <!-- Left column: GPU Telemetry stacked above Memory Usage -->
    <div class="col-stack">
    <div class="panel">
      <h2>GPU Telemetry</h2>
      <div id="gpuName" class="gpu-name">—</div>
      <div class="stat-row">
        <div class="stat"><div class="k">Temp</div><div><span class="v" id="temp">—</span><span class="u">°C</span></div></div>
        <div class="stat"><div class="k">Util</div><div><span class="v" id="util">—</span><span class="u">%</span></div></div>
        <div class="stat"><div class="k">Power</div><div><span class="v" id="power">—</span><span class="u">W</span></div></div>
      </div>
      <div style="margin-top:16px">
        <div class="k">Utilization history</div>
        <canvas id="spark" class="spark" width="900" height="120"></canvas>
      </div>
      <div id="gpuErr" class="err"></div>
    </div>

    <!-- Memory Usage (stacked under GPU Telemetry) -->
    <div class="panel">
      <div class="chart-title">
        <h2>Memory Usage — Training vs Other</h2>
        <span class="muted" id="memBdNote">unified (system RAM)</span>
      </div>
      <div class="progress-head" style="margin-top:10px">
        <span class="pct" id="memBdPct" style="font-size:18px">—</span>
        <span class="muted" id="memBdUsed">—</span>
      </div>
      <div class="stackbar" id="memStack"></div>
      <div class="legend" id="memLegend"><span class="muted">waiting for memory data…</span></div>
    </div>
    </div><!-- /col-stack -->

    <!-- Config + progress (RUN — spans the full height of the left column) -->
    <div class="panel">
      <div class="log-head" style="margin-bottom:12px">
        <h2 style="margin:0">Run</h2>
        <div class="ctrl">
          <span id="trainStatus" class="status">—</span>
          <button id="trainBtn" class="btn" disabled>—</button>
        </div>
      </div>
      <div class="progress-head">
        <span class="muted" id="stepTxt">step —</span>
        <span class="pct" id="pct">—</span>
      </div>
      <div class="bar"><i id="progBar" style="width:0;background:linear-gradient(90deg,#1c7ed6,#39b6ff)"></i></div>
      <div class="stat-row" style="margin-top:14px">
        <div class="stat"><div class="k">Loss</div><div class="v" id="curLoss">—</div></div>
        <div class="stat"><div class="k">Epoch</div><div class="v" id="curEpoch">—</div></div>
        <div class="stat"><div class="k">Elapsed</div><div class="v" id="elapsed">—</div></div>
      </div>
      <div id="ctrlMsg" class="ctrl-msg"></div>
      <details open style="margin-top:14px">
        <summary>Hyperparameters</summary>
        <div id="cfg" class="cfg-grid">
          <div class="muted" style="grid-column:1/-1">waiting for run_config.json…</div>
        </div>
      </details>
    </div>
  </div>

  <div class="grid row2">
    <div class="panel">
      <div class="chart-title">
        <h2>Model</h2>
        <span id="modelTag" class="status">—</span>
      </div>
      <div id="modelName" class="gpu-name" style="margin-top:4px; word-break:break-all">—</div>
      <table id="modelSpec" style="margin-top:14px">
        <tr><td class="muted">waiting for run_config.json…</td></tr>
      </table>
    </div>
    <div class="panel">
      <div class="chart-title"><h2>Training Loss</h2><span class="last" id="lossLast" style="color:var(--loss)">—</span></div>
      <canvas id="lossChart" width="520" height="240"></canvas>
    </div>
    <div class="panel">
      <div class="chart-title"><h2>Learning Rate</h2><span class="last" id="lrLast" style="color:var(--lr)">—</span></div>
      <canvas id="lrChart" width="520" height="240"></canvas>
    </div>
    <div class="panel">
      <div class="chart-title"><h2>Grad Norm</h2><span class="last" id="gradLast" style="color:var(--grad)">—</span></div>
      <canvas id="gradChart" width="520" height="240"></canvas>
    </div>
  </div>

  <div class="grid row3">
    <div class="panel">
      <div class="log-head">
        <h2 style="margin:0">Training Log</h2>
        <label class="toggle"><input type="checkbox" id="logFollow" checked> Follow tail</label>
      </div>
      <div id="logBox" class="log-box"><span class="muted">waiting for runs/LATEST_train.log…</span></div>
    </div>
  </div>
</div>

<script>
const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const fmt = (x,d=2) => (x==null||isNaN(x)) ? "—" : Number(x).toLocaleString(undefined,{maximumFractionDigits:d});
const sci = x => (x==null||isNaN(x)) ? "—" : Number(x).toExponential(2);

function tempColor(t){ if(t==null) return css('--dim'); if(t<60) return css('--grad'); if(t<80) return css('--temp'); return '#ff5252'; }
function hms(s){ s=Math.floor(s||0); const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;
  return (h?h+"h ":"")+(m<10&&h?"0":"")+m+"m "+(x<10?"0":"")+x+"s"; }

// generic line chart on a canvas (handles HiDPI, grid, axis labels, y-formatting)
function drawChart(id, pts, color, yfmt){
  const c = document.getElementById(id); if(!c) return;
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth || 520, H = 240;
  if(c.width !== W*dpr){ c.width = W*dpr; c.height = H*dpr; }
  const ctx = c.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H);
  const padL=52, padR=12, padT=12, padB=24;
  const x0=padL, x1=W-padR, y0=padT, y1=H-padB;
  if(!pts.length){ ctx.fillStyle=css('--dim'); ctx.font='12px ui-monospace,monospace';
    ctx.fillText('waiting for data…', x0, (y0+y1)/2); return; }
  const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y);
  let xmin=Math.min(...xs), xmax=Math.max(...xs);
  let ymin=Math.min(...ys), ymax=Math.max(...ys);
  if(xmax===xmin) xmax=xmin+1;
  if(ymax===ymin){ ymax=ymin+ (Math.abs(ymin)||1)*0.1; ymin=ymin-(Math.abs(ymin)||1)*0.1; }
  const pad=(ymax-ymin)*0.08; ymin-=pad; ymax+=pad;
  const X=v=>x0+(v-xmin)/(xmax-xmin)*(x1-x0);
  const Y=v=>y1-(v-ymin)/(ymax-ymin)*(y1-y0);
  // grid + y labels
  ctx.strokeStyle=css('--grid'); ctx.fillStyle=css('--dim');
  ctx.font='10px ui-monospace,monospace'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){ const v=ymin+(ymax-ymin)*i/4, y=Y(v);
    ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke();
    ctx.fillText(yfmt(v), 4, y+3); }
  // x labels + ticks (first / middle / last step)
  ctx.textAlign='left';
  ctx.fillText(Math.round(xmin), x0, H-8);
  ctx.textAlign='right';
  ctx.fillText(Math.round(xmax), x1, H-8);
  if(xmax-xmin>2){ ctx.textAlign='center';
    ctx.fillText(Math.round((xmin+xmax)/2), (x0+x1)/2, H-8); }
  ctx.textAlign='left';
  // area fill
  const grad=ctx.createLinearGradient(0,y0,0,y1);
  grad.addColorStop(0,color+'55'); grad.addColorStop(1,color+'00');
  ctx.beginPath(); ctx.moveTo(X(xs[0]),Y(ys[0]));
  for(let i=1;i<pts.length;i++) ctx.lineTo(X(xs[i]),Y(ys[i]));
  ctx.lineTo(X(xs[xs.length-1]),y1); ctx.lineTo(X(xs[0]),y1); ctx.closePath();
  ctx.fillStyle=grad; ctx.fill();
  // line
  ctx.beginPath(); ctx.moveTo(X(xs[0]),Y(ys[0]));
  for(let i=1;i<pts.length;i++) ctx.lineTo(X(xs[i]),Y(ys[i]));
  ctx.strokeStyle=color; ctx.lineWidth=1.8; ctx.lineJoin='round'; ctx.stroke();
  // last point dot
  ctx.beginPath(); ctx.arc(X(xs.at(-1)),Y(ys.at(-1)),3,0,7); ctx.fillStyle=color; ctx.fill();
}

function drawSpark(hist){
  const c=document.getElementById('spark'); if(!c) return;
  const dpr=window.devicePixelRatio||1;
  const W=c.clientWidth||900, H=120;
  if(c.width!==W*dpr){ c.width=W*dpr; c.height=H*dpr; }
  const ctx=c.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
  const padL=36, padR=12, padT=10, padB=18;
  const x0=padL, x1=W-padR, y0=padT, y1=H-padB;
  const pts=hist.filter(h=>h.util!=null);
  if(!pts.length){ ctx.fillStyle=css('--dim'); ctx.font='12px ui-monospace,monospace';
    ctx.fillText('waiting for data…', x0, (y0+y1)/2); return; }
  const v=pts.map(h=>h.util);
  const ts=pts.map((h,i)=> h.t!=null ? h.t : i);
  const now=ts.at(-1);
  let tmin=ts[0], tmax=now; if(tmax===tmin) tmax=tmin+1;
  const col=css('--util');
  const X=t=>x0+(t-tmin)/(tmax-tmin)*(x1-x0);
  const Y=u=>y1-(u/100)*(y1-y0);            // fixed 0..100% scale
  // area fill
  const g=ctx.createLinearGradient(0,y0,0,y1);
  g.addColorStop(0,col+'55'); g.addColorStop(1,col+'00');
  ctx.beginPath(); ctx.moveTo(X(ts[0]),Y(v[0]));
  for(let i=1;i<pts.length;i++) ctx.lineTo(X(ts[i]),Y(v[i]));
  ctx.lineTo(X(ts.at(-1)),y1); ctx.lineTo(X(ts[0]),y1); ctx.closePath();
  ctx.fillStyle=g; ctx.fill();
  // y grid + labels (0..100%), drawn over the fill so they stay visible
  ctx.fillStyle=css('--dim'); ctx.font='10px ui-monospace,monospace'; ctx.lineWidth=1;
  for(let u=0;u<=100;u+=25){ const y=Y(u);
    ctx.strokeStyle=css('--grid'); ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke();
    ctx.fillText(String(u).padStart(3)+'%', 2, y+3); }
  // x labels (relative time)
  const ago=s=>{ s=Math.max(0,Math.round(s)); return s<90 ? '-'+s+'s' : '-'+Math.round(s/60)+'m'; };
  ctx.fillText(ago(now-tmin), x0, H-5);
  ctx.fillText('now', x1-22, H-5);
  if(tmax-tmin>20) ctx.fillText(ago(now-(tmin+tmax)/2), (x0+x1)/2-12, H-5);
  // line
  ctx.beginPath(); ctx.moveTo(X(ts[0]),Y(v[0]));
  for(let i=1;i<pts.length;i++) ctx.lineTo(X(ts[i]),Y(v[i]));
  ctx.strokeStyle=col; ctx.lineWidth=1.6; ctx.lineJoin='round'; ctx.stroke();
  // last point dot
  ctx.beginPath(); ctx.arc(X(ts.at(-1)),Y(v.at(-1)),2.5,0,7); ctx.fillStyle=col; ctx.fill();
}

function series(metrics, key){
  return metrics.filter(m=>m[key]!=null && m.step!=null).map(m=>({x:m.step,y:m[key]}));
}

let _lastData = null;
let _ctrlBusy = false;   // true between a button click and the next status confirmation

function renderControl(tr){
  const btn=document.getElementById('trainBtn');
  const st=document.getElementById('trainStatus');
  if(!('control' in tr) || !tr.control){
    st.textContent='read-only'; st.className='status';
    btn.style.display='none';
    return;
  }
  if(tr.running){
    st.textContent='running'+(tr.pid?(' · pid '+tr.pid):''); st.className='status run';
  } else if(tr.resumable){
    st.textContent='stopped'; st.className='status stop';
  } else {
    st.textContent='not started'; st.className='status';
  }
  // Pause when running; Resume only when there's a checkpoint to resume from.
  // Otherwise (never started) hide the button entirely.
  const showBtn = tr.running || tr.resumable;
  btn.style.display = showBtn ? '' : 'none';
  // Config-editor buttons: writes need control; Start is blocked while a run is live.
  const ro = !tr.control;
  ['saveCfgBtn','uploadBtn'].forEach(id=>{ const b=document.getElementById(id); if(b) b.disabled=ro; });
  const sb=document.getElementById('startBtn');
  if(sb){
    sb.disabled = ro || !!tr.running;
    sb.title = tr.running ? 'A run is in progress — pause it before starting a new one' : '';
  }
  if(_ctrlBusy || !showBtn) return;     // don't stomp the "pausing…" label mid-action
  if(tr.running){ btn.textContent='Pause';  btn.className='btn pause';  btn.dataset.action='pause'; }
  else          { btn.textContent='Resume'; btn.className='btn resume'; btn.dataset.action='resume'; }
  btn.disabled=false;
}

async function onControlClick(e){
  const btn=e.currentTarget, action=btn.dataset.action;
  if(!action || _ctrlBusy) return;
  if(action==='pause' && !confirm('Pause training? It will checkpoint at the end of the current step, then stop.')) return;
  const msg=document.getElementById('ctrlMsg');
  _ctrlBusy=true; btn.disabled=true;
  btn.textContent = action==='pause' ? 'pausing…' : 'starting…';
  try{
    const r=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action})});
    const j=await r.json();
    msg.textContent=(j.ok?'✓ ':'⚠ ')+(j.msg||j.error||'');
    msg.style.color=j.ok?'var(--grad)':'var(--temp)';
  }catch(err){ msg.textContent='⚠ request failed: '+err; msg.style.color='var(--temp)'; }
  // let the process actually start/stop before we trust the next status poll
  setTimeout(()=>{ _ctrlBusy=false; }, 2500);
}

// --- training config editor (form -> train_config.json -> train.py) --------
let _cfgInit = false;   // fill the form from the server only once, so live SSE
                        // ticks never clobber what the user is typing.

function setVal(id, v){ const el=document.getElementById(id); if(el) el.value = (v==null?'':v); }
function getVal(id){ const el=document.getElementById(id); return el ? el.value : ''; }

function toggleDatasetSource(){
  const upload = getVal('f_dataset_source')==='upload';
  document.getElementById('hubFields').style.display    = upload ? 'none' : 'contents';
  document.getElementById('uploadFields').style.display = upload ? '' : 'none';
}

function renderTrainConfig(cfg){
  if(_cfgInit || !cfg) return;
  setVal('f_model_name',cfg.model_name);
  setVal('f_max_seq_length',cfg.max_seq_length);
  document.getElementById('f_use_4bit').value = cfg.use_4bit ? 'true' : 'false';
  document.getElementById('f_dataset_source').value = cfg.dataset_source || 'hub';
  setVal('f_dataset_name',cfg.dataset_name);
  setVal('f_dataset_split',cfg.dataset_split);
  setVal('f_dataset_file',cfg.dataset_file);
  setVal('f_messages_field',cfg.messages_field);
  setVal('f_filter_category',cfg.filter_category);
  setVal('f_instruction_part',cfg.instruction_part);
  setVal('f_response_part',cfg.response_part);
  setVal('f_lora_r',cfg.lora_r);
  setVal('f_lora_alpha',cfg.lora_alpha);
  setVal('f_batch_size',cfg.batch_size);
  setVal('f_grad_accum',cfg.grad_accum);
  setVal('f_learning_rate',cfg.learning_rate);
  setVal('f_epochs',cfg.epochs);
  setVal('f_optim',cfg.optim);
  document.getElementById('f_export_gguf').value = cfg.export_gguf ? 'true' : 'false';
  setVal('f_run_label',cfg.run_label);
  toggleDatasetSource();
  _cfgInit = true;
}

function gatherConfig(){
  return {
    model_name:      getVal('f_model_name'),
    max_seq_length:  Number(getVal('f_max_seq_length')),
    use_4bit:        getVal('f_use_4bit')==='true',
    dataset_source:  getVal('f_dataset_source'),
    dataset_name:    getVal('f_dataset_name'),
    dataset_split:   getVal('f_dataset_split'),
    dataset_file:    getVal('f_dataset_file'),
    messages_field:  getVal('f_messages_field'),
    filter_category: getVal('f_filter_category'),
    instruction_part: getVal('f_instruction_part'),
    response_part:    getVal('f_response_part'),
    lora_r:          Number(getVal('f_lora_r')),
    lora_alpha:      Number(getVal('f_lora_alpha')),
    batch_size:      Number(getVal('f_batch_size')),
    grad_accum:      Number(getVal('f_grad_accum')),
    learning_rate:   getVal('f_learning_rate'),   // "2e-4" string; server coerces to float
    epochs:          Number(getVal('f_epochs')),
    optim:           getVal('f_optim'),
    export_gguf:     getVal('f_export_gguf')==='true',
    run_label:       getVal('f_run_label'),
  };
}

function _cfgMsg(text, ok){
  const m=document.getElementById('cfgMsg');
  m.textContent=text; m.style.color = ok ? 'var(--grad)' : 'var(--temp)';
}

async function saveConfig(){
  try{
    const r=await fetch('/train_config',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(gatherConfig())});
    const j=await r.json();
    _cfgMsg(j.ok?'✓ config saved':'⚠ '+(j.error||'save failed'), j.ok);
    return j.ok;
  }catch(e){ _cfgMsg('⚠ '+e, false); return false; }
}

async function uploadDataset(){
  const inp=document.getElementById('f_dataset_upload');
  const msg=document.getElementById('uploadMsg');
  const file=inp.files && inp.files[0];
  if(!file){ msg.textContent='choose a .json/.jsonl file first'; msg.style.color='var(--temp)'; return; }
  const field=getVal('f_messages_field')||'messages';
  msg.textContent='uploading '+file.name+' …'; msg.style.color='var(--dim)';
  try{
    const r=await fetch('/upload?name='+encodeURIComponent(file.name)+'&field='+encodeURIComponent(field),
      {method:'POST', body:file});
    const j=await r.json();
    if(j.ok){
      setVal('f_dataset_file', j.path);
      document.getElementById('f_dataset_source').value='upload'; toggleDatasetSource();
      msg.textContent='✓ '+j.rows+' rows · '+j.path+(j.warning?('  ⚠ '+j.warning):'');
      msg.style.color = j.warning ? 'var(--temp)' : 'var(--grad)';
    } else {
      msg.textContent='⚠ '+(j.error||'upload failed'); msg.style.color='var(--temp)';
    }
  }catch(e){ msg.textContent='⚠ '+e; msg.style.color='var(--temp)'; }
}

async function startTraining(){
  if(getVal('f_dataset_source')==='upload' && !getVal('f_dataset_file')){
    _cfgMsg('⚠ upload a dataset first (or switch the source to hub)', false); return;
  }
  if(!confirm('Start a fresh training run with the current configuration?')) return;
  const btn=document.getElementById('startBtn'); btn.disabled=true;
  if(!(await saveConfig())){ btn.disabled=false; return; }
  try{
    const r=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'start'})});
    const j=await r.json();
    _cfgMsg((j.ok?'✓ ':'⚠ ')+(j.msg||j.error||''), j.ok);
  }catch(e){ _cfgMsg('⚠ '+e, false); }
  setTimeout(()=>{ btn.disabled=false; }, 2500);   // let the process actually spawn before re-enabling
}

// Stacked memory bar: training process tree vs everything else vs free.
function renderMem(mem){
  const stack=document.getElementById('memStack');
  const leg=document.getElementById('memLegend');
  if(!mem || !mem.total){ return; }
  const gib = mib => fmt(mib/1024,1)+' GiB';
  const used = mem.train + mem.other;
  document.getElementById('memBdPct').textContent = fmt(100*used/mem.total,0)+'% Used';
  document.getElementById('memBdUsed').textContent = '~'+gib(used)+' / '+gib(mem.total);
  const segs=[
    {label:'Training script', color:'#b07cff', v:mem.train, pid:mem.pid},
    {label:'Other processes', color:'#5d6b7d', v:mem.other},
    {label:'Free',            color:'#1d2733', v:mem.free},
  ];
  stack.innerHTML = segs.map(s=>{
    const w = Math.max(0, 100*s.v/mem.total);
    return `<span title="${s.label}: ${gib(s.v)}" style="width:${w}%;background:${s.color}"></span>`;
  }).join('');
  leg.innerHTML = segs.map(s=>{
    const pct = fmt(100*s.v/mem.total,1)+'%';
    const pid = (s.pid!=null) ? `<span class="pid">pid ${s.pid}</span>` : '';
    return `<div class="row"><span class="sw" style="background:${s.color}"></span>`+
      `<span class="lab">${s.label}</span>${pid}`+
      `<span class="val">${gib(s.v)} · ${pct}</span></div>`;
  }).join('');
}

// Model card: a focused view of *what* is being fine-tuned, built from run_config.json.
function renderModel(cfg){
  const nameEl=document.getElementById('modelName');
  const tagEl=document.getElementById('modelTag');
  const tbl=document.getElementById('modelSpec');
  if(!cfg || !Object.keys(cfg).length) return;
  const truthy = v => v===true || v==='True' || v==='true' || v===1 || v==='1';
  const full = cfg.model_name || '—';
  nameEl.textContent = full.includes('/') ? full.split('/').pop() : full;   // short label
  nameEl.title = full;                                                      // full id on hover
  const fourbit = truthy(cfg.load_in_4bit);
  tagEl.textContent = fourbit ? '4-bit QLoRA' : '16-bit LoRA';
  tagEl.className = 'status run';
  const cat = (cfg.filter_category && cfg.filter_category!=='None') ? cfg.filter_category : 'all categories';
  const spec = [
    ['Base model',      full],
    ['Precision',       fourbit ? '4-bit QLoRA' : '16-bit LoRA (bf16)'],
    ['Adapter',         (cfg.lora_r!=null||cfg.lora_alpha!=null) ? `LoRA · r ${cfg.lora_r??'—'} · α ${cfg.lora_alpha??'—'}` : null],
    ['Max seq length',  cfg.max_seq_length!=null ? cfg.max_seq_length+' tok' : null],
    ['Dataset',         cfg.num_examples!=null ? `${cat} · ${Number(cfg.num_examples).toLocaleString()} ex` : cat],
    ['Optimizer',       cfg.optim || null],
  ].filter(([,v]) => v!=null && v!=='');
  tbl.innerHTML = spec.map(([k,v]) =>
    `<tr><td class="k">${k}</td><td class="v">${v}</td></tr>`).join('');
}

function render(d){
  if(!d) return;
  _lastData = d;

  // GPU
  const g=d.gpu||{};
  const errEl=document.getElementById('gpuErr');
  if(g.ok){
    errEl.textContent='';
    document.getElementById('gpuName').textContent=g.name||'GPU';
    const t=document.getElementById('temp'); t.textContent=fmt(g.temp,0); t.style.color=tempColor(g.temp);
    document.getElementById('util').textContent=fmt(g.util,0);
    document.getElementById('power').textContent=g.power==null?'—':fmt(g.power,0);
  } else {
    errEl.textContent='⚠ '+(g.error||'no GPU data')+'  (some GB10 fields may read N/A under nvidia-smi)';
  }
  drawSpark(d.gpu_history||[]);

  // memory breakdown (training vs other vs free)
  renderMem(d.mem||{});

  // config
  const cfg=d.config||{}; const keys=Object.keys(cfg);
  if(keys.length){
    document.getElementById('cfg').innerHTML = keys.map(k=>
      `<div class="item"><span class="k">${k}</span><span class="v" title="${cfg[k]}">${cfg[k]}</span></div>`).join('');
  }
  renderModel(cfg);

  // metrics + progress
  const m=d.metrics||[]; const last=m[m.length-1]||{};
  document.getElementById('meta').textContent =
    `${m.length} log points · refreshed ${new Date().toLocaleTimeString()}`;
  if(last.step!=null){
    const ms=last.max_steps||0;
    document.getElementById('stepTxt').textContent=`step ${last.step}${ms?' / '+ms:''}`;
    const pct=ms?Math.min(100,100*last.step/ms):0;
    document.getElementById('pct').textContent=ms?fmt(pct,1)+'%':'—';
    document.getElementById('progBar').style.width=pct+'%';
    document.getElementById('curLoss').textContent=fmt(last.loss,3);
    document.getElementById('curEpoch').textContent=fmt(last.epoch,2);
    document.getElementById('elapsed').textContent=hms(last.wall_time);
  }
  const lossS=series(m,'loss'), lrS=series(m,'learning_rate'), gradS=series(m,'grad_norm');
  if(lossS.length) document.getElementById('lossLast').textContent=fmt(lossS.at(-1).y,3);
  if(lrS.length)   document.getElementById('lrLast').textContent=sci(lrS.at(-1).y);
  if(gradS.length) document.getElementById('gradLast').textContent=fmt(gradS.at(-1).y,2);
  drawChart('lossChart', lossS, css('--loss'), v=>fmt(v,2));
  drawChart('lrChart',   lrS,   css('--lr'),   v=>sci(v));
  drawChart('gradChart', gradS, css('--grad'), v=>fmt(v,1));

  // training config editor (fills once) + control (start / pause / resume)
  renderTrainConfig(d.train_config);
  renderControl(d.train||{});

  // training log
  const logBox=document.getElementById('logBox');
  const txt=d.log||'';
  if(txt!==logBox.dataset.last){
    const follow=document.getElementById('logFollow').checked;
    const atBottom=logBox.scrollHeight-logBox.scrollTop-logBox.clientHeight<40;
    logBox.textContent = txt || 'waiting for log output…';
    logBox.dataset.last = txt;
    if(follow || atBottom) logBox.scrollTop=logBox.scrollHeight;
  }
}

// --- transport: prefer SSE push, fall back to polling ---------------------
let _pollTimer = null;

function startPolling(){
  if(_pollTimer) return;
  const tick = async () => {
    try{
      const d = await (await fetch('/data',{cache:'no-store'})).json();
      render(d);
    }catch(e){ document.getElementById('meta').textContent='disconnected'; }
  };
  tick();
  _pollTimer = setInterval(tick, 1000);
}
function stopPolling(){ if(_pollTimer){ clearInterval(_pollTimer); _pollTimer=null; } }

function connect(){
  if(!('EventSource' in window)){ startPolling(); return; }
  const es = new EventSource('/stream');
  es.onmessage = ev => {
    stopPolling();                       // SSE is live; kill any fallback poll
    try{ render(JSON.parse(ev.data)); }catch(e){}
  };
  es.onerror = () => {
    // browser auto-reconnects SSE; meanwhile poll so the UI keeps moving
    document.getElementById('meta').textContent='reconnecting…';
    startPolling();
  };
}

document.getElementById('trainBtn').addEventListener('click', onControlClick);
document.getElementById('f_dataset_source').addEventListener('change', toggleDatasetSource);
document.getElementById('uploadBtn').addEventListener('click', uploadDataset);
document.getElementById('saveCfgBtn').addEventListener('click', saveConfig);
document.getElementById('startBtn').addEventListener('click', startTraining);
connect();
window.addEventListener('resize', () => render(_lastData));
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--metrics", default=None,
                    help="metrics.jsonl path (default: auto-discover latest run dir)")
    ap.add_argument("--config", default=None,
                    help="run_config.json path (default: auto-discover latest run dir)")
    ap.add_argument("--runs-dir", default="runs",
                    help="parent dir scanned to auto-discover the latest run")
    ap.add_argument("--log", default="runs/LATEST_train.log")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="GPU sampling cadence (seconds)")
    ap.add_argument("--push", type=float, default=1.0,
                    help="SSE stream push cadence (seconds)")
    ap.add_argument("--pid-file", default="runs/LATEST.pid",
                    help="PID file written by train.py (used by the pause/resume button)")
    ap.add_argument("--train-cmd", default="RESUME_RUN=latest python train.py",
                    help="command the Resume button runs to continue the latest run")
    ap.add_argument("--train-start-cmd", default="python train.py",
                    help="command the Start button runs to launch a fresh run from train_config.json")
    ap.add_argument("--train-config", default="train_config.json",
                    help="editable training config the dashboard writes and train.py reads")
    ap.add_argument("--datasets-dir", default="datasets",
                    help="dir uploaded .json/.jsonl datasets are stored in")
    ap.add_argument("--outputs-dir", default="outputs",
                    help="dir scanned for checkpoint-*/ to decide if Resume is offered")
    ap.add_argument("--no-control", action="store_true",
                    help="disable the start/pause/resume controls and write endpoints (read-only)")
    args = ap.parse_args()

    Handler.metrics_path = args.metrics
    Handler.config_path = args.config
    Handler.runs_dir = args.runs_dir
    Handler.log_path = args.log
    Handler.stream_interval = args.push
    Handler.pid_file = args.pid_file
    Handler.train_cmd = args.train_cmd
    Handler.train_start_cmd = args.train_start_cmd
    Handler.train_config_path = args.train_config
    Handler.datasets_dir = args.datasets_dir
    Handler.outputs_dir = args.outputs_dir
    Handler.control_enabled = not args.no_control

    threading.Thread(target=gpu_sampler_loop, args=(args.interval,), daemon=True).start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard live at http://{args.host}:{args.port}  (reading {args.metrics})")
    print("Tunnel from your laptop:  ssh -L {0}:localhost:{0} <user>@<spark>".format(args.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")


if __name__ == "__main__":
    main()
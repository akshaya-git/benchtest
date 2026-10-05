#!/usr/bin/env python3
"""
Apple Silicon LLM Benchmark — backend (stdlib only, no pip installs).

Serves index.html, manages model framework lifecycle, runs harness tests,
streams live activity, and serves generated outputs (e.g. HTML Tetris pages).

Usage:  python3 server.py   →  open http://localhost:7090

Configuration lives in config.json (seeded from config.example.json on first
run): host/port, per-framework model + start command, harness paths. See
docs/CONFIG.md.
"""

import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import discovery

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "outputs")
LOG_DIR = os.path.join(ROOT, "logs")
WORK_DIR = os.path.join(ROOT, "work")
RUNS_DIR = os.path.join(ROOT, "runs")
for _d in (OUTPUT_DIR, LOG_DIR, WORK_DIR, RUNS_DIR):
    os.makedirs(_d, exist_ok=True)

CONFIG_PATH = os.path.join(ROOT, "config.json")
CONFIG_EXAMPLE = os.path.join(ROOT, "config.example.json")

# ----------------------------------------------------------------------------
# Framework definitions — DEFAULTS. Override per machine in config.json:
# model, port, start_cmd, context window, weight size, notes. The benchmark
# waits for each server's /v1/models endpoint to answer before running tests,
# and shuts the process down (SIGTERM then SIGKILL) before the next framework.
# start_cmd entries may use {model} / {model_path} placeholders ({model_path}
# resolves to the local HF-cache snapshot path when the model is cached).
# ----------------------------------------------------------------------------
DEFAULT_FRAMEWORKS = {
    "omlx": {
        "name": "OMLX",
        "model_source": "hf",
        "notes": "Model, context (131072), max-tokens (65536) and reasoning are applied PER MODEL from ~/.omlx/model_settings.json (entry: mlx-community--Qwen3.8-27B-8bit) at request time — the server starts bare, discovers models from the HF cache, and each request's model id selects its settings entry. Reasoning is task-type driven (top-level 'reasoning' config: low for standard tasks, medium for LONG): the benchmark rewrites the model entry's chat_template_kwargs.reasoning_effort before start and restores it after the run. Same base weights as MLX-VLM/MLX-Serve. MTP speculative decoding is ON via the embedded mlx-vlm engine (vlm_mtp_enabled=true, vlm_mtp_draft_model=mlx-community--Qwen3.8-27B-MTP-8bit) — the same draft arrangement MLX-VLM uses, user-confirmed working; the plain mtp_enabled/draft_model keys were inert in A/B (18.2 vs 18.1 TGS). The Jundot oQ8e build was retired after QA 25 on LONG tasks — see run 20260923-110534.",
        "port": 7001,
        "model": "mlx-community--Qwen3.8-27B-8bit",
        "model_gb": 28, "ctx_tokens": 131072,
        # omlx has no CLI reasoning flag — the level lives in this per-model
        # settings file (applied at request time). The benchmark rewrites
        # models[<model>].chat_template_kwargs[reasoning_key] per run and
        # restores the original value afterwards.
        "reasoning_file": "~/.omlx/model_settings.json",
        "reasoning_key": "reasoning_effort",
        "start_cmd": ["omlx", "serve", "--port", "7001"],
    },
    "mtplx": {
        "name": "MTPLX",
        # the MTPLX-Optimized build degrades at medium+ reasoning (verbose
        # thinking, lower quality, and its stream-stall watchdog aborts
        # long silent phases) — pinned low regardless of task type
        "reasoning_override": "low",
        # model_source "mtplx": models live in ~/.mtplx/models (a different
        # format than HF/MLX), so the model picker shows only those for MTPLX.
        "model_source": "mtplx",
        "notes": "All parameters are CLI flags applied at server start (this command). --reasoning-effort is task-type driven (top-level 'reasoning' config: low for standard tasks, medium for LONG) via the {reasoning} placeholder; the served model id is normalized (e.g. mtplx-qwen38-27b-optimized-quality) and auto-adopted. Models are loaded from ~/.mtplx/models by repo id ({repo}); pick one in the Model panel. Performance note (from mtplx status): the turbo profile's compiled MTP-verify only applies to sequences ≤ 32768 tokens — LONG-task cells whose context grows past that fall back to eager verification.",
        "port": 7002,
        "model": "mtplx-qwen38-27b-optimized-quality",
        # repo: the HF repo behind the normalized served id; also the id used
        # for the CLI --model flag and the ~/.mtplx/models directory.
        "repo": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
        "model_gb": 28, "ctx_tokens": 131072,
        # {reasoning} expands to these flags per task type (see top-level
        # 'reasoning' in config.json): low for standard tasks, medium for LONG.
        "reasoning_flags": {
            "low": ["--reasoning-effort", "low"],
            "medium": ["--reasoning-effort", "medium"],
            "high": ["--reasoning-effort", "high"],
        },
        "start_cmd": ["mtplx", "serve",
                      "--model", "{repo}",
                      "--context-window", "131072",
                      "--max-tokens", "65536",
                      "{reasoning}",
                      "--port", "7002"],
    },
    # No MTP support in mlx-lm, so this runs the plain 8-bit conversion of the
    # same base model — a useful "reference runtime" row rather than a like-
    # for-like MTP comparison. Context length follows the model config; there
    # is no server-side context flag in mlx-lm. --max-tokens here is the
    # server-side cap (its default of 8192 would truncate thinking models).
    "mlxlm": {
        "name": "MLX-VLM",
        "model_source": "hf",
        "port": 7003,
        "model": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
        "draft_model": "mlx-community/Qwen3.8-27B-MTP-8bit",
        "model_gb": 28, "ctx_tokens": 131072,
        "notes": "mlx_vlm.server with MTP speculative decoding: the MTP-8bit draft pairs with the Youssofal-Quality weights (identical architecture to the dense 8bit - verified tensor-by-tensor in run 20260926-181100: 6/6 cells, QA 100, 19-38 tok/s). Reasoning maps to --enable-thinking; thinking_budget is hard-blocked by mlx_vlm while the draft is loaded, so no budget flags. No ctx flag - serves the model's native context (adopted at start). Per-request metrics from its JSON /metrics.",
        "reasoning_flags": {
            "low": [],
            "medium": ["--enable-thinking"],
            "high": ["--enable-thinking"]
        },
        "start_cmd": ["python3", "-m", "mlx_vlm.server",
                      "--model", "{model}",
                      "{draft}",
                      "--max-tokens", "65536",
                      "{reasoning}",
                      "--port", "7003"],
    },
    # mlx-serve: vLLM-compatible server with the richest metrics surface of
    # all (Prometheus /metrics: TTFT/prefill/decode histograms, prefix cache,
    # speculative-decode counters). Preferred instance is GUI-started with
    # MTP (the benchmark reuses whatever is healthy on :7004 and adopts its
    # REAL context length from /v1/models); this start_cmd is the cold-start
    # fallback with PLD (on by default).
    "mlxserve": {
        "name": "MLX-Serve",
        "model_source": "hf",
        "notes": "--max-tokens/--reasoning-budget are request defaults set at start; reasoning is task-type driven (top-level 'reasoning' config): {reasoning} expands to --reasoning-budget <n> (1024 low / 4096 medium / 16384 high). --ctx-size is OVERRIDDEN by ~/.mlx-serve/model-settings.json (ctx_size, set to 131072) — the file wins. --metrics enables the Prometheus surface (on by default in the GUI, not the CLI). PLD speculative decoding is on by default.",
        "port": 7004,
        "model": "mlx-community/Qwen3.8-27B-8bit",
        "model_gb": 28, "ctx_tokens": 131072,
        # {reasoning} expands per task type: --reasoning-budget caps thinking
        # tokens per request (server default).
        "reasoning_flags": {
            "low": ["--reasoning-budget", "1024"],
            "medium": ["--reasoning-budget", "4096"],
            "high": ["--reasoning-budget", "16384"],
        },
        # --metrics: Prometheus surface (on by default in the GUI, not CLI).
        # mlx-serve 26.x cannot resolve a repo id (org/name) — it fails with
        # FileNotFound — so pass the local HF-cache snapshot path ({model_path});
        # it loads the same model fine. Served ctx follows
        # ~/.mlx-serve/model-settings.json (ctx_size), overriding --ctx-size —
        # the benchmark adopts the served value.
        "start_cmd": ["mlx-serve", "--model", "{model_path}",
                      "--serve", "--host", "127.0.0.1", "--metrics",
                      "--port", "7004", "--ctx-size", "131072",
                      "--max-tokens", "65536", "{reasoning}"],
    },
}

DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 7090,
    "route_via_proxy": False,
    "pi_thinking": "",
    # Reasoning level per task type, applied at framework server start (each
    # framework maps a level to its own flags via reasoning_flags, or to the
    # omlx settings file via reasoning_file). LONG tasks (chip8, raytracer,
    # spreadsheet, markdown, conduit) get 'medium'; standard tasks get 'low'.
    "reasoning": {"short": "low", "long": "low"},
        # Model sets: the UI's Model radio picks one of these keys; the
        # backend maps it to the right per-framework model id (each engine
        # gets a conversion its MTP implementation supports). Frameworks
        # absent from a set are disabled for that set in the UI and rejected
        # at run start.
        "model_set": "qwen38-27b",
        "model_sets": {
            "qwen38-27b": {
                "label": "Qwen3.8-27B",
                "models": {
                    "omlx": "Jundot/Qwen3.8-27B-oQ8e-mtp",
                    "mtplx": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
                    "mlxlm": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
                    "mlxserve": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality"
                }
            },
            "flash-next": {
                "label": "Qwen3.8-Flash-Next",
                "models": {
                    "omlx": "mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp",
                    "mtplx": "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
                }
            }
        },
    "hart_path": os.path.join(ROOT, "hart", "hart.py"),
    "frameworks": DEFAULT_FRAMEWORKS,
}


def _merge_config(base, override):
    """Shallow-merge override onto base; frameworks merged per-framework."""
    out = dict(base)
    for k, v in (override or {}).items():
        if k == "frameworks" and isinstance(v, dict):
            fws = dict(out.get("frameworks") or {})
            for fw, fwc in v.items():
                if fw in fws and isinstance(fwc, dict):
                    fws[fw] = {**fws[fw], **fwc}
                else:
                    fws[fw] = fwc
            out["frameworks"] = fws
        else:
            out[k] = v
    return out


def load_config():
    """Load config.json over the built-in defaults. On first run, seed
    config.json from config.example.json so users can discover and edit it."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy of defaults
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg = _merge_config(cfg, json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠ config.json unreadable ({e}) — using built-in defaults",
                  flush=True)
    else:
        try:
            src = CONFIG_EXAMPLE if os.path.isfile(CONFIG_EXAMPLE) else None
            if src:
                with open(src) as f:
                    with open(CONFIG_PATH, "w") as out:
                        out.write(f.read())
            else:
                with open(CONFIG_PATH, "w") as f:
                    json.dump(DEFAULT_CONFIG, f, indent=2)
        except OSError:
            pass
    return cfg


def save_config():
    """Persist the live config (e.g. after a model selection) to config.json."""
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(CONFIG, f, indent=2)
        return True
    except OSError as e:
        log(f"could not save config.json: {e}", level="err")
        return False


CONFIG = load_config()
FRAMEWORKS = CONFIG["frameworks"]


def _fit_lookup():
    """Discovered candidates keyed by model id — for fit validation when a
    selection is applied. One disk scan per call (~1-2 s), fine on a click."""
    free, _ = free_ram()
    machine = discovery.machine_profile()
    out = {}
    for c in discovery.all_candidates(round(free / 1073741824, 1), FRAMEWORKS,
                                      machine):
        out[discovery.normalize_key(c["id"])] = c
    return out, machine


def _validate_fit(models_map):
    """Refuse combinations where a chosen model cannot fit this machine
    (machine_fit == wont-fit), naming the model and the numbers."""
    fitmap, machine = _fit_lookup()
    for fw, model in models_map.items():
        cand = fitmap.get(discovery.normalize_key(model))
        if not cand:
            continue
        mf = (cand.get("compat") or {}).get("machine_fit")
        if mf == "wont-fit":
            raise ValueError(
                f"{FRAMEWORKS[fw]['name']}: {model} needs ~{cand.get('need_gb')} GB "
                f"but this machine has ~{machine.get('usable_gb')} GB usable — "
                f"pick a smaller quant of the same family")


def apply_set_selection(names, validate_fit=True):
    """Apply a COMBINATION of model sets at once: each framework gets the
    model of the one ticked set that covers it. Two ticked sets covering the
    same framework is rejected — one run can only load one model per
    framework (run them via the campaign instead, which iterates sets)."""
    chosen = {}
    if validate_fit:
        pre = {}
        for n in names:
            s = (CONFIG.get("model_sets") or {}).get(n) or {}
            for fw, m in (s.get("models") or {}).items():
                pre.setdefault(fw, m)
        _validate_fit(pre)
    for n in names:
        s = (CONFIG.get("model_sets") or {}).get(n)
        if not s:
            raise ValueError(f"unknown model set {n!r}")
        for fw in (s.get("models") or {}):
            if fw in chosen:
                raise ValueError(
                    f"{FRAMEWORKS[fw]['name']} is selected by both "
                    f"{chosen[fw]!r} and {n!r} — a single run can only load "
                    f"one model per framework. Untick one, or use Run All "
                    f"(it benchmarks each ticked set in turn).")
            chosen[fw] = n
    for fw, cfg in FRAMEWORKS.items():
        if fw in chosen:
            s = CONFIG["model_sets"][chosen[fw]]
            cfg["model_available"] = True
            cfg["model"] = s["models"][fw]
            # mtplx loads weights by {repo} — always sync (see apply_model_set)
            cfg["repo"] = cfg["model"]
            if s.get("ctx_tokens"):
                cfg["ctx_tokens"] = s["ctx_tokens"]
            if s.get("max_tokens"):
                cfg["max_tokens"] = s["max_tokens"]
        else:
            cfg["model_available"] = False
    CONFIG["model_set"] = "+".join(names)
    save_config()
    return CONFIG["model_set"]


def apply_model_set(name=None):
    """Apply the active model set: each framework's model becomes the set's
    id; frameworks the set doesn't cover keep their model but are flagged
    unavailable (the UI disables them and run start rejects them). Sets may
    also pin ctx_tokens / max_tokens (a smaller-native-window model must not
    inherit the 131K window Qwen3.8 sets run with); every covered framework
    gets the set's values so switching sets restores them. A saved name of
    the form "a+b" is a merged selection (see apply_set_selection)."""
    name = name or CONFIG.get("model_set") or "qwen38-27b"
    sets = CONFIG.get("model_sets") or {}
    if "+" in name:
        parts = [p for p in name.split("+") if p in sets]
        if parts:
            try:
                return apply_set_selection(parts)
            except ValueError:
                pass  # stale combination — fall through to single-set logic
    if name not in sets:
        name = "qwen38-27b"
    CONFIG["model_set"] = name
    s = sets[name]
    if name == "selected":
        # Custom combination: every framework is ON — the user picks each
        # model in its dropdown. (A previous model-first selection may have
        # narrowed this set's models map; that map must not restrict here.)
        for fw, cfg in FRAMEWORKS.items():
            cfg["model_available"] = True
            if s.get("ctx_tokens"):
                cfg["ctx_tokens"] = s["ctx_tokens"]
            if s.get("max_tokens"):
                cfg["max_tokens"] = s["max_tokens"]
        save_config()
        return name
    for fw, cfg in FRAMEWORKS.items():
        cfg["model_available"] = fw in (s.get("models") or {})
        if cfg["model_available"]:
            cfg["model"] = s["models"][fw]
            # mtplx loads weights by {repo} — a stale repo would silently
            # benchmark the PREVIOUS set's weights (observed live: the
            # flash-next run measured Quality because repo still said
            # Quality). Always sync repo to the set's model.
            cfg["repo"] = cfg["model"]
        if s.get("ctx_tokens"):
            cfg["ctx_tokens"] = s["ctx_tokens"]
        if s.get("max_tokens"):
            cfg["max_tokens"] = s["max_tokens"]
    save_config()
    return name


apply_model_set()   # startup: apply the active set (or merged "+a+b" selection)


# ---- campaign ("Run All"): sequential runs across tasks × model sets ----
CAMPAIGN_FILE = os.path.join(ROOT, "campaign.json")
CAMPAIGN_ESTIMATES = {  # from measured runs on this machine class
    "qwen38-27b": "~3.5–4.5 days",
    "flash-next": "~2 days",
}


def _campaign_load():
    try:
        with open(CAMPAIGN_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _campaign_save(c):
    try:
        with open(CAMPAIGN_FILE, "w") as f:
            json.dump(c, f, indent=2)
    except OSError as e:
        log(f"could not save campaign state: {e}", level="err")


def campaign_plan(sets):
    """The run queue for the given model-set sequence: every task of TASKS
    for each set, frameworks auto-derived from each set's coverage."""
    tasks = [t["id"] for t in TASKS]
    return [{"set": s, "task_id": t} for s in sets for t in tasks]


def campaign_runner(mode, sets, resume_index=0, harnesses=None):
    """Run the whole plan sequentially, one run_benchmark per (set, task).
    Pause/resume via the Stop button + Resume; survives backend restarts
    through campaign.json. Re-raised RunStopped pauses the campaign.
    harnesses: the UI's selected harnesses (None = all six)."""
    plan = campaign_plan(sets)
    skipped = []
    sets_names = " + ".join(sets)
    camp = {"active": True, "status": "running", "mode": mode, "sets": sets,
            "harnesses": harnesses,
            "plan_total": len(plan), "index": resume_index,
            "current": None, "started": time.time(), "skipped": []}
    STATE["campaign"] = dict(camp)
    _campaign_save(camp)
    RUN_FLAG.set()
    log(f"campaign running: {mode} · {len(plan) - resume_index} of "
        f"{len(plan)} runs queued ({sets_names})", level="ok")
    i = resume_index
    try:
        while i < len(plan):
            step = plan[i]
            if not RUN_FLAG.is_set():
                raise RunStopped()
            camp["index"] = i
            camp["current"] = f"{step['set']} · {step['task_id']}"
            STATE["campaign"] = dict(camp)
            _campaign_save(camp)
            apply_model_set(step["set"])
            frameworks = [fw for fw, c in FRAMEWORKS.items()
                          if c.get("model_available", True)]
            task = next((t for t in TASKS if t["id"] == step["task_id"]), None)
            try:
                if not task:
                    raise ValueError(f"unknown task {step['task_id']}")
                run_benchmark({"frameworks": frameworks,
                               "harnesses": harnesses or list(HARNESS_LABELS.keys()),
                               "task_id": step["task_id"],
                               "prompt": task["prompt"],
                               "settings": {"temperature": 0.5, "top_p": 0.95,
                                            "max_tokens": 65536,
                                            "max_tokens_raw": 65536,
                                            "repeats": 1}})
            except RunStopped:
                raise
            except Exception as e:
                # one failed run never kills the campaign — record and move on
                skipped.append(f"{step['set']}/{step['task_id']}: {str(e)[:120]}")
                log(f"campaign run failed ({step['set']} · {step['task_id']}): "
                    f"{str(e)[:150]} — continuing", level="err")
            i += 1
        camp.update({"active": False, "status": "complete",
                     "completed": time.time()})
        if skipped:
            camp["skipped"] = skipped
        STATE["campaign"] = dict(camp)
        _campaign_save(camp)
        log(f"campaign complete: {len(plan)} runs", level="ok")
    except RunStopped:
        camp.update({"active": False, "status": "paused", "index": i,
                     "current": f"{plan[i]['set']} · {plan[i]['task_id']}"
                     if i < len(plan) else None})
        STATE["campaign"] = dict(camp)
        _campaign_save(camp)
        log(f"campaign paused at run {i + 1}/{len(plan)} — resume from the "
            f"dashboard", level="err")
    except Exception as e:
        camp.update({"active": False, "status": "failed",
                     "error": str(e)[:300], "index": i})
        STATE["campaign"] = dict(camp)
        _campaign_save(camp)
        log(f"campaign failed: {e}", level="err")


def resolve_start_cmd(fw_cfg, reasoning_level=None):
    """Expand {model} / {repo} / {model_path} / {draft} / {reasoning}
    placeholders in a framework's start_cmd.
      {model}      → the served/request model id (cfg["model"])
      {repo}       → the HF repo id (cfg["repo"] or cfg["model"]); MTPLX's CLI
                     --model flag needs the repo, not the normalized served id
      {model_path} → the local HF-cache snapshot path when the repo is cached
                     (some CLIs need a real path), else the model id
      {draft}      → MLX-VLM speculative-decoding flags: expands to
                     ["--draft-model", <draft_model>, "--draft-kind", "mtp"]
                     when cfg["draft_model"] is set, else nothing (omitted)
      {reasoning}  → the task-type reasoning flags: cfg["reasoning_flags"][
                     reasoning_level] (e.g. ["--reasoning-effort", "medium"]);
                     empty when the level has no mapping (flag omitted)
      {ctx}        → cfg["ctx_tokens"] — the model-set's context window
                     (a model with a smaller native window must not inherit
                     the 131K window the Qwen3.8 sets run with)
      {max_tokens} → cfg["max_tokens"] — the model-set's output cap; must
                     stay below the window so harness budgets stay positive"""
    repo = fw_cfg.get("repo") or fw_cfg.get("model", "")
    model_path = discovery.snapshot_dir(repo)
    draft_model = (fw_cfg.get("draft_model") or "").strip()
    cmd = []
    for part in fw_cfg.get("start_cmd", []):
        if part == "{model}":
            cmd.append(fw_cfg["model"])
        elif part == "{repo}":
            cmd.append(repo)
        elif part == "{model_path}":
            cmd.append(model_path or fw_cfg["model"])
        elif part == "{ctx}":
            cmd.append(str(fw_cfg.get("ctx_tokens") or 131072))
        elif part == "{max_tokens}":
            cmd.append(str(fw_cfg.get("max_tokens") or 65536))
        elif part == "{draft}":
            if draft_model:
                cmd += ["--draft-model", draft_model, "--draft-kind", "mtp"]
        elif part == "{draft_path}":
            # local snapshot path of cfg["draft_model"] (for CLIs that spell
            # out their own draft flags, e.g. mlx_lm.server --draft-model)
            if draft_model:
                cmd.append(discovery.snapshot_dir(draft_model) or draft_model)
        elif part == "{reasoning}":
            cmd += (fw_cfg.get("reasoning_flags") or {}).get(
                reasoning_level or "", [])
        else:
            cmd.append(part)
    return cmd

# 5 benchmark tasks — prompts that produce runnable/inspectable output.
TASKS = [
    {"id": "logreport", "name": "Log Analyst · failure report", "artifact": "report.html",
     "static_report": True,
     "prompt": "You are given the logs of a PREVIOUS benchmark run that tested local LLM inference frameworks (OMLX, MTPLX, MLX-VLM, MLX-Serve) across agent harnesses (raw, raw+, pi, opencode, Goose, hart). The input files are in the directory "
               + WORK_DIR + "/logreport-fixture/ — read them: runs.json (per-cell records: framework, harness, status, latency, qa_func, error), errors.log (the orchestrator's failure lines), bench-excerpt.log (its full log window) and omlx.log / mtplx.log / mlxlm.log / mlxserve.log (framework server error excerpts). The same runs.json and errors.log content is also inlined below between <input> markers.\n\n"
               "Do exactly three things:\n"
               "1. Classify every cell in runs.json as passed (status \"done\") or failed (any other status).\n"
               "2. For each failed cell, find its root cause: match its framework and harness against errors.log / bench-excerpt.log and quote the single most relevant log line. Known failure classes in these logs: \"empty response\", \"timeout after 7200s\", \"no output\", \"stopped by user\", stream-stall breaks, memory-guard rejections.\n"
               "3. Write a single-file HTML report named report.html in the current directory containing, in this order: (a) a header with the analyzed run's task name and date; (b) the totals — cells run, passed, failed, and the success rate as a percentage; (c) one summary table row per framework with cells run, passed, failed and failure rate; (d) one detail card per failed cell with its framework, harness, the exact error string and the quoted log line; (e) a \"Recommendations\" section with at most three concrete next actions.\n\n"
               "Constraints: no external assets, no JavaScript frameworks, everything inline in one file. Return only the complete report.html.\n\n<input>\n@LOGREPORT_INLINE@\n</input>"},
    {"id": "tetris", "name": "HTML Tetris game", "artifact": "tetris.html",
     "prompt": "Create a complete, playable Tetris game in a single HTML file with embedded CSS and JavaScript. Include score, level, next-piece preview, and keyboard controls. Return only the full HTML file in one ```html code block."},
    {"id": "snake", "name": "HTML Snake game", "artifact": "snake.html",
     "prompt": "Create a complete, playable Snake game in a single HTML file with embedded CSS and JavaScript. Include score display, increasing speed, and arrow-key controls. Return only the full HTML file in one ```html code block."},
    {"id": "pong", "name": "HTML Pong vs AI", "artifact": "pong.html",
     "prompt": "Create a complete Pong game in a single HTML file where the player plays against a simple AI paddle. Include score display and mouse/keyboard control. Return only the full HTML file in one ```html code block."},
    {"id": "todo", "name": "Todo app (localStorage)", "artifact": "todo.html",
     "prompt": "Create a complete Todo list web app in a single HTML file with embedded CSS and JavaScript. Support adding, completing, deleting items and persisting to localStorage. Return only the full HTML file in one ```html code block."},
    {"id": "fib", "name": "Explain + code: Fibonacci",
     "prompt": "Explain memoized Fibonacci in plain English, then give a working Python implementation with a quick test snippet. Keep it under 300 words of explanation."},
    # ---- long-horizon tasks (60-120 min class; industry benchmark families) ----
    {"id": "webdb", "name": "Inventory manager · embedded DB + dashboard", "artifact": "inventory.html",
     "prompt": "Build a single-file Inventory Manager web app backed by an embedded database: a pure-JavaScript data-store object exposing add(), update(), remove(), getAll() and query() operations, persisted to localStorage on every change and reloaded on start (no server, no libraries, no IndexedDB). Features: (1) on first load, seed the store with 12 products across exactly 3 categories (Food, Travel, Office — fields: id, name, category, quantity, price); (2) add a product via a form, edit a product inline, and delete a product, all persisted; (3) live search by product name and a category filter dropdown, both combinable; (4) a dashboard showing total product count, total inventory value (sum of quantity × price) and a low-stock count (quantity < 5); (5) a self-test panel with a button that runs at least 8 automated checks synchronously against the real data store — seed count = 12, add, read-back, update, delete, low-stock computation, persistence round-trip through localStorage, and query filtering — rendering one line per check starting with ✓ or ✗ and a final summary line in the exact form 'N/M passed'. Keep everything inline in one file. Return only the complete HTML."},
    {"id": "bugfix", "name": "Agentic bugfix · fix the broken expense tracker", "artifact": "expenses.html",
     "prompt": "The file expenses.html in your current directory is an expense tracker with a built-in self-test panel, but it contains exactly 4 planted bugs and its self-test currently reports failures. Diagnose each failure from the self-test output and the code, then fix the underlying bugs with minimal, targeted edits. The bugs are in the app logic (a totals computation, deletion, category filtering, and data persistence) — do NOT modify the self-test code, do NOT rewrite the app from scratch, and keep every existing feature working. The file must remain a single self-contained HTML file. When every self-test check passes, deliver the fixed file as expenses.html. The full current source is also inlined below between <source> markers.\n\n<source>\n@BUGFIX_SOURCE@\n</source>"},
    {"id": "codereview", "name": "Code review · find the planted defects", "artifact": "review.md",
     "prompt": "You are given the Python module review-sample.py — its full source is inlined below between <source> markers (agent harnesses also have it in the working directory). Review it like a senior engineer: find every real defect (there are at least 5; look carefully at query construction, function signatures, exception handling, slicing, resource handling, and shared state). Do exactly three things:\n"
               "1. For each defect found: assign a severity (critical, major, or minor), name the function it is in, describe what is wrong and when it bites, and give a concrete one-paragraph fix.\n"
               "2. Write a single markdown file named review.md containing: a header with counts by severity, one section per finding (severity, function, description, fix), and a final 'What is done well' list of at most 3 items.\n"
               "3. List nothing under findings that is not actually in the code — accuracy over volume.\n\n<source>\n@CODE_REVIEW_SOURCE@\n</source>"},
    {"id": "markdown", "name": "LONG · Markdown compiler + spec tests", "long": True,
     "artifact": "markdown.html",
     "prompt": "Build a Markdown compiler in a single HTML file: a two-pane editor (raw markdown left, live rendered HTML right) with a CommonMark-subset engine written from scratch (no libraries): ATX headings, paragraphs, bold/italic, inline code, fenced code blocks with language labels, links, images, unordered/ordered lists with nesting, blockquotes, horizontal rules, and tables. Include a built-in spec test panel with at least 20 test cases (input to expected HTML) covering edge cases like nested lists, unclosed fences, and emphasis inside code; run them on load and show pass/fail counts. Persist editor content to localStorage."},
        {"id": "agentconsole", "name": "LONG · Agent Console (chat → agent, skills, MCP deploy)", "long": True,
     "artifact": "index.html",
     "prompt": "Build a single-file 'Agent Console' — a chat UI that turns any OpenAI-compatible local endpoint into a tool-calling agent. Everything client-side, no libraries. Sections: (1) Connect: collapsible settings panel (base URL, e.g. http://localhost:7001/v1, optional API key), a Connect button that calls GET {base}/models, renders returned ids in a model dropdown, and shows a connected/error banner. (2) Chat: message list with user/assistant bubbles, markdown rendering for code and lists, timestamps, a 'clear conversation' button, and the last 20 exchanges persisted to localStorage and restored on reload. (3) Agent loop with tool calling: define exactly 4 built-in tools — http_get(url) for HTTP GET returning response text, http_post(url, body) for HTTP POST returning response text, now() for the current timestamp, and math_eval(expression) for safe arithmetic — and send them in the request's tools array using the OpenAI tool-calling protocol. When the model responds with tool_calls, execute each tool in JavaScript, append the results as role=tool messages, and re-request, looping until the model answers with plain content or 5 iterations. Render every tool call and its result as a collapsible block inside the transcript. (4) Skills: a Skills panel where the user loads a SKILL.md file via a file picker (or pastes one) — parse its YAML frontmatter into name and description, list loaded skills with enable/disable toggles, and prepend the body of every enabled skill to the system prompt of the next request. (5) MCP deploy: a Download MCP server button that generates a complete single-file Python MCP server (official mcp package, stdio transport, at least 3 tools wrapping the same HTTP helpers against the configured base URL) and downloads it as bench-mcp-server.py. (6) Self-test panel with a button that runs at least 5 checks and renders one line per check starting with a check mark or a cross, plus a final summary line in the exact form N/M passed: models endpoint reachable, chat completion returns content, a tool-call round-trip works (offer a math tool and ask the model to add 2 plus 2 via the tool), markdown rendering works, and SKILL.md frontmatter parsing works. Connection errors render as a readable error bubble. Return only the complete HTML."},
]


# Per-task requirements: each entry is [feature label, regex]. The QA gate
# checks every requirement against the artifact content and blends the pass
# rate into qa_func (60%) with the generic HTML checks (40%). This is what
# stops feature-incomplete artifacts from scoring 100 (observed: agentconsole
# cells missing 8-10 of 12 required features all scored 100 on generic checks).
REQUIREMENTS = {
    "tetris": [["canvas board", r"<canvas"], ["7 tetromino pieces", r"tetromino|SHAPES|pieces"],
               ["rotation", r"rotat"], ["line clearing", r"clear.{0,6}line|line.{0,6}clear|collapse"],
               ["score", r"score"], ["levels", r"level"], ["next-piece preview", r"next"],
               ["keyboard controls", r"keydown|keyup|keyboard"],
               ["game loop", r"requestAnimationFrame|setInterval|setTimeout"],
               ["game-over state", r"game.{0,4}over|gameover"]],
    "snake": [["canvas board", r"<canvas"], ["snake body/movement", r"snake"],
              ["food", r"food|apple|berry"], ["keyboard controls", r"keydown|keyup|keyboard"],
              ["score", r"score"], ["speed increase", r"speed|interval"],
              ["game-over on collision", r"game.{0,4}over|collision"],
              ["grid/walls", r"grid|wall|wrap"], ["restart", r"restart|play.{0,4}again"],
              ["game loop", r"requestAnimationFrame|setInterval"]],
    "pong": [["canvas board", r"<canvas"], ["paddles", r"paddle"], ["ball", r"ball"],
             ["AI opponent", r"\bai\b|computer|cpu"], ["mouse/keyboard control", r"mousemove|keydown|keyboard"],
             ["score", r"score"], ["win condition", r"win"], ["speed increase on hit", r"speed"],
             ["center line", r"center|dashed"], ["reset/serve after point", r"reset|serve"]],
    "todo": [["add items", r"add.?todo|addItem|add.{0,10}item"],
             ["complete toggle", r"toggle|checkbox|complete"], ["delete items", r"delete|remove"],
             ["localStorage persistence", r"localStorage"], ["filters (all/active/done)", r"filter"],
             ["items-left count", r"count|items.?left"], ["edit items", r"edit|dblclick"],
             ["clear completed", r"clear"], ["empty state", r"empty|no items"],
             ["list structure", r"<ul|<li|list"]],
    "fib": [["plain-english explanation first", r"(?s)^[^`]{100,}"],
            ["python code", r"def\s+\w+"], ["memoization", r"memo|cache"],
            ["fibonacci function", r"fib"],
            ["base case", r"base.?case|n\s*[<=]+\s*\d|n\s*==\s*\d|return\s+n\b|len\s*<\s*2"],
            ["test/assert snippet", r"assert|test|print"]],
    "markdown": [["headings", r"heading|<h1"], ["bold/italic", r"bold|italic|<strong|<em>"],
                 ["inline code", r"inline.{0,4}code|<code"], ["fenced code blocks", r"fence|code.?block"],
                 ["lists", r"list|<ul|<ol"], ["tables", r"table"],
                 ["spec test panel", r"spec|test"], ["20+ test cases", r"20|test.?cases"],
                 ["pass/fail counts", r"pass.{0,10}(fail|total)|dpass|passcount|testresults"],
                 ["live preview", r"preview|live|oninput|input.?event|\'input\'|\"input\"|render.{0,10}\(\)"],
                 ["localStorage", r"localStorage"]],
    "agentconsole": [["connect + models dropdown", r"(?i)connect|models"],
                     ["chat bubbles", r"(?i)bubble|\\.user|\\.assistant|message"],
                     ["localStorage history", r"(?i)localstorage"],
                     ["agent tool loop", r"(?i)tool_calls|tool.?call"],
                     ["built-in tools defined", r"(?i)http_get|http_post|math_eval"],
                     ["skill loader + frontmatter", r"(?i)frontmatter|yaml|skill"],
                     ["MCP server generator", r"(?i)mcp.{0,60}(server|stdio|package)"],
                     ["self-test panel", r"(?i)self.?test"],
                     ["single-file html", r"<!DOCTYPE html"]],
    "codereview": [["SQL injection flagged", r"(?i)sql.{0,30}inject|inject.{0,40}(sql|query|get_user)"],
                   ["mutable default flagged", r"(?i)mutable.{0,30}default|default.{0,20}argument"],
                   ["bare except flagged", r"(?i)bare.{0,20}except|except\\s*:|swallow"],
                   ["off-by-one flagged", r"(?i)off.by.one|skip.{0,25}(last|row|item)"],
                   ["resource leak flagged", r"(?i)(leak|never closed|not closed|unclosed|without closing)"],
                   ["severities assigned", r"(?i)(critical|major|minor)"],
                   ["fix suggestions", r"(?i)(fix|recommend|should be|use a context manager)"],
                   ["markdown structure", r"(?i)(^|\\n)#|severity"]],
    # --- retired tasks (chip8 / raytracer / spreadsheet) — their gates are
    # kept so historical runs rescore with the same task-specific checks ---
    "chip8": [["canvas display", r"<canvas"], ["opcodes (draw/CLS)", r"00E0|CLS|sprite|draw"],
              ["keypad input", r"keypad|keydown|keyboard"], ["timers (delay/sound)", r"delay|timer|sound|beep|audio"],
              ["ROM/program load", r"rom|chip8"], ["registers/memory", r"register|V0|memory"],
              ["self-test panel", r"self.?test|test"], ["pass/fail output", r"pass.{0,10}fail|pass/fail"],
              ["screen rendering", r"render|draw|pixel"], ["fetch/decode/execute loop", r"fetch|decode|execute|opcode"]],
    "raytracer": [["vec3 math", r"vec3|vector"], ["sphere intersection", r"sphere|intersect"],
                  ["anti-aliasing", r"anti.?alias|samples"], ["diffuse material", r"diffuse|lambert"],
                  ["metal material", r"metal"], ["glass/dielectric", r"glass|dielectric|refract"],
                  ["camera", r"camera"], ["progressive render", r"progressive"],
                  ["progress bar", r"progress"], ["unit tests", r"test|assert"]],
    "spreadsheet": [["grid rendering", r"grid|table|cell"], ["formula parsing", r"formula|parse"],
                    ["SUM function", r"SUM"], ["AVERAGE function", r"AVERAGE|AVG"],
                    ["cell references", r"A1|B2"], ["recalculation", r"recalc|eval"],
                    ["cell selection", r"select|active"], ["edit in place", r"edit|input|contenteditable"],
                    ["formula bar", r"formula.{0,4}bar|fx"], ["error handling", r"#ERR|error|NaN|#REF"]],
    # --- active tasks ---
    "webdb": [["embedded data store", r"(?i)(localstorage|data[\s-]?store|database)"],
              ["seed data", r"(?i)seed"],
              ["add product form", r"(?i)add.{0,20}(product|item|form)"],
              ["edit or update", r"(?i)\b(edit|update)"],
              ["delete or remove", r"(?i)\b(delete|remove)"],
              ["search / filter", r"(?i)search|filter"],
              ["dashboard totals", r"(?i)(total|dashboard|stats)"],
              ["self-test panel with N/M passed", r"(?i)self.?test"],
              ["single-file html", r"<!DOCTYPE html"]],
    "skillbuild": [["yaml frontmatter", r"---"],
                   ["name and description", r"(^|\n)\s*(name|description):"],
                   ["when-to-use guidance", r"(?i)(when to use|use (this|when)|usage|trigger|scope)"],
                   ["numbered runbook", r"(?m)^\s*[1-9]\."],
                   ["concrete commands", r"(?i)(curl |python3|GET /api|POST /api)"],
                   ["pitfalls section", r"(?i)pitfall|false conclusion|avoid"]],
    "mcpbuild": [["mcp package import", r"(?i)(from mcp|import mcp)"],
                 ["server instance", r"(?i)(MCPServer|FastMCP|mcp\s*=\s*)"],
                 ["four tool decorators", r"(?i)@mcp\.tool[\s\S]*@mcp\.tool[\s\S]*@mcp\.tool[\s\S]*@mcp\.tool"],
                 ["python functions", r"(?m)def "],
                 ["docstrings per tool", r'"""'],
                 ["stdio entrypoint", r"(?i)(__main__|\.run\(|stdio)"]],
    "bugfix": [["expense tracker features kept", r"(?i)(expense|amount|category)"],
               ["add / delete present", r"(?i)(add|delete|remove)"],
               ["filter / search present", r"(?i)(filter|search)"],
               ["persistence present", r"(?i)localstorage|persist"],
               ["self-test panel intact", r"(?i)self.?test"],
               ["single-file html", r"<!DOCTYPE html"]],
    "logreport": [["summary table", r"<table"],
                  ["pass/fail counts", r"(?i)(passed?|failed?|success)"],
                  ["success rate percentage", r"\d+(\.\d+)?\s*%"],
                  ["failure details with error strings", r"(?i)(empty response|timeout|no output|stopped|exited|error)"],
                  ["per-framework breakdown", r"(?i)(omlx|mtplx|mlx-vlm|mlx-serve|mlxlm|mlxserve)"],
                  ["root-cause log quote", r"(?i)(log|trace|line)"],
                  ["recommendations section", r"(?i)recommendation|next step"],
                  ["single-file html", r"<!DOCTYPE html"]],
}

# ---------------------------------------------------------------------------
# Runtime QA probe. The requirement regexes above can only check that the
# artifact MENTIONS a feature — a page whose renderer has a fatal logic bug
# still contains every keyword (observed: three markdown artifacts scored 100
# while their preview rendered nothing at all, one reporting "0 / 0" in its
# own test panel). When a Chrome-based browser is available, the gate loads a
# disposable copy of the artifact headlessly, performs the task's core
# interaction (type markdown → check the preview DOM; enter endpoint URL →
# connect → send a message; press Start → watch the canvas animate) and
# scores what the page actually renders. If no browser is found the gate
# falls back to regex-only scoring and says so in qa_notes.
# ---------------------------------------------------------------------------
_CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
]


def find_chrome():
    """A Chrome-like binary for the runtime probe, or None. chrome-headless-
    shell (bundled by puppeteer/playwright installs) is preferred: it starts
    cleanly with a throwaway profile. A full Chrome build hangs on this Mac
    when --user-data-dir is passed, so the caller skips the profile for it."""
    override = CONFIG.get("qa_chrome_path")
    if override:
        p = os.path.expanduser(override)
        if os.path.exists(p):
            return p
    for pattern in ("~/.cache/puppeteer/chrome-headless-shell/*/chrome-headless-shell-mac-arm64/chrome-headless-shell",
                    "~/Library/Caches/ms-playwright/chromium-*/chrome-mac/chrome"):
        hits = sorted(glob.glob(os.path.expanduser(pattern)))
        if hits:
            return hits[-1]  # newest installed version
    for c in _CHROME_CANDIDATES:
        if os.path.exists(c):
            return c
    return shutil.which("chromium") or shutil.which("google-chrome") \
        or shutil.which("chrome-headless-shell")


# Per-task probe config. Actions run in order (each waits `wait` ms after
# firing); checks are evaluated after the last action settles. A probe only
# RUNS if its primary input target exists — a page whose UI differs so much
# the actions can't find their targets falls back to regex scoring rather
# than being punished for the probe's own blindness.
QA_PROBES = {
    "markdown": {
        "actions": [
            {"type": "type", "find": "textarea",
             "value": "# ZZ Heading\n\n**ZZ bold** and *ZZ italic* and `ZZ code`\n\n- ZZ item1\n- ZZ item2",
             "wait": 300},
            {"type": "click", "match": r"render|preview|convert|compile|run tests|re-run", "wait": 900},
        ],
        "checks": [
            {"id": "heading rendered", "sel": "h1,h2", "text": "ZZ"},
            {"id": "bold rendered", "sel": "strong,b", "text": "ZZ bold"},
            {"id": "italic rendered", "sel": "em,i", "text": "ZZ italic"},
            {"id": "code rendered", "sel": "code", "text": "ZZ code"},
            {"id": "list rendered", "sel": "li", "text": "ZZ item", "minCount": 2},
        ],
    },
    "agentconsole": {
        "actions": [
            {"type": "type", "find": "input", "match": r"base.?url|endpoint|server",
             "value": "@BASE_URL@", "wait": 300},
            {"type": "click", "match": r"connect", "wait": 2200},
            {"type": "type", "find": "textarea", "value": "Say hi", "wait": 300},
            {"type": "click", "match": r"send|submit", "wait": 3200},
        ],
        "checks": [
            # "models dropdown populated" is validated against the REAL model
            # list fetched from the endpoint (cfg.servedModels): a fixed
            # minOptions threshold unfairly failed single-model servers, where
            # an artifact that replaces the placeholder leaves only 1 option
            {"id": "models dropdown populated", "sel": "select", "optionMatch": True},
            {"id": "reply rendered", "domGrows": 20},
        ],
        "budget": 12000,
    },
    "tetris": "canvas", "snake": "canvas", "pong": "canvas",
    "raytracer": "canvas",   # retired task — probe kept for historical rescores
    "webdb": {
        # the self-test runs against IndexedDB — asynchronous REAL-TIME I/O
        # that virtual time does not wait for. Two accommodations: the
        # artifact is loaded over HTTP from this server (file:// origins are
        # unreliable for IDB), and measure() gates the final report on a
        # same-origin hold request (~4.5s of real time) so the database
        # callbacks complete before the DOM is graded.
        "async": True,
        "actions": [
            {"type": "click",
             "match": r"self.?test|automated tests|run tests|diagnostics",
             "wait": 1500},
        ],
        "checks": [
            {"id": "self-test renders results",
             "bodyRegex": r"\d+\s*(?:/|of)\s*\d+\s*(?:passed|checks?|ok)|\d+\s+passed"},
            {"id": "all self-tests passing", "noMatch": True,
             "bodyRegex": r"✗"},
        ],
        "budget": 12000,
    },
    "bugfix": {
        "actions": [
            {"type": "click",
             "match": r"self.?test|automated tests|run tests|diagnostics",
             "wait": 1500},
        ],
        "checks": [
            {"id": "self-test renders results", "bodyRegex": r"\d+\s*/\s*\d+\s*passed"},
            {"id": "all four bugs fixed", "noMatch": True,
             "bodyRegex": r"✗"},
        ],
        "budget": 9000,
    },
    "canvas": {  # shared config for canvas-rendered tasks
        # Enter starts from a menu in most games; Space is avoided — pong-style
        # games bind it to pause/resume, which would freeze the game right
        # after the probe starts it (observed: three working pong artifacts
        # scored 65 because the probe's Space press paused them). The canvas
        # click comes FIRST while nothing is running yet, so it can only
        # start click-to-start games, never pause an already-running one.
        "actions": [
            {"type": "key", "key": "Enter", "wait": 250},
            {"type": "canvasClick", "wait": 400},
            # rom/load covers CHIP-8-style emulators whose built-in test ROM
            # must be loaded before the canvas animates (8/8 overnight chip8
            # cells failed "canvas animates" without this)
            {"type": "click", "match": r"start|play|new game|begin|serve|rom|load", "wait": 700},
            {"type": "key", "key": "ArrowUp", "wait": 250},
            {"type": "key", "key": "w", "wait": 250},
        ],
        "checks": [{"id": "canvas present", "sel": "canvas", "minCount": 1},
                   {"id": "canvas animates", "canvasAnimates": True}],
        "budget": 9000,
    },
    # CHIP-8 is probed on its REQUIRED self-test panel rather than canvas
    # animation: emulators ship a static test ROM, so a working one never
    # animates until a game is loaded — 8/8 working overnight cells scored
    # 50% on the animation check alone
    "chip8": {
        "actions": [
            {"type": "click", "match": r"self.?test", "wait": 1200},
            {"type": "click", "match": r"rom|load|start|play|run", "wait": 900},
            {"type": "key", "key": "Enter", "wait": 250},
            {"type": "key", "key": "1", "wait": 250},
        ],
        "checks": [
            {"id": "canvas present", "sel": "canvas", "minCount": 1},
            # [\s\S] instead of (?s): JS RegExp rejects Python inline flags
            {"id": "self-test renders results",
             "bodyRegex": r"(pass|fail)[\s\S]*(pass|fail)[\s\S]*(pass|fail)"},
        ],
        "budget": 7000,
    },
    "todo": {
        "actions": [
            {"type": "type", "find": "input", "match": r"add|item|task|todo|what|new",
             "value": "ZZ probe item", "wait": 250},
            {"type": "click", "match": r"^add|add\b|submit|new", "wait": 700},
        ],
        "checks": [{"id": "item added", "sel": "li", "text": "ZZ probe item"}],
    },
    "spreadsheet": {
        "actions": [
            {"type": "type", "find": "editable", "value": "=SUM(3,4)", "wait": 300},
            {"type": "key", "key": "Enter", "wait": 800},
        ],
        "checks": [{"id": "formula evaluates", "bodyRegex": r"(^|\s)7(\s|$|\n)"}],
    },
}

_PROBE_TEMPLATE = """<script id="__qa_probe">
(function () {
  var CFG = __CFG__;
  var out = { jsErr: null };
  window.onerror = function (m) { out.jsErr = String(m).slice(0, 90); return false; };
  var $$ = function (s) { return [].slice.call(document.querySelectorAll(s)); };
  function fire(el, ev) { try { el.dispatchEvent(new Event(ev, { bubbles: true })); } catch (e) {} }
  function findByText(re) {
    return $$('button, a, [role=button]').find(function (x) {
      return re.test((x.textContent || '').trim()); });
  }
  var _qaBaseLen = (document.body.innerText || '').length;
  function snapCanvas() {
    try { var c = $$('canvas')[0]; return c ? c.toDataURL() : null; } catch (e) { return null; }
  }
  var _qaM1 = null, _qaDone = false;
  function doActions(i) {
    if (i >= CFG.actions.length) { setTimeout(measure, 300); return; }
    var a = CFG.actions[i];
    try {
      if (a.type === 'type') {
        var el = null;
        if (a.find === 'textarea') el = $$('textarea')[0];
        else if (a.find === 'input') el = $$('input').find(function (x) {
          return new RegExp(a.match, 'i').test((x.placeholder || '') + ' ' + (x.id || '') + ' '
            + (x.name || '') + ' ' + (x.getAttribute('aria-label') || '')); });
        else if (a.find === 'editable') el = $$('[contenteditable=true], td[contenteditable]')[0];
        if (el) {
          out.typed = true;
          if (el.isContentEditable) el.innerHTML = a.value; else el.value = a.value;
          fire(el, 'input'); fire(el, 'change');
        } else out.noTarget = a.find + ':' + (a.match || '');
      } else if (a.type === 'click') {
        var b = findByText(new RegExp(a.match, 'i'));
        if (b) { b.click(); out.clicked = (b.textContent || '').trim().slice(0, 24); }
        else out.noButton = a.match;
      } else if (a.type === 'canvasClick') {
        var cv = $$('canvas')[0];
        if (cv) {
          cv.dispatchEvent(new MouseEvent('click', { bubbles: true }));
          out.canvasClicked = true;
        }
      } else if (a.type === 'key') {
        var t = $$('textarea, input, [contenteditable=true]')[0];
        var ks = a.key === ' ' ? 'Space' : a.key;
        try {
          document.body.dispatchEvent(new KeyboardEvent('keydown', { key: a.key, code: ks, bubbles: true }));
          if (t) t.dispatchEvent(new KeyboardEvent('keydown', { key: a.key, code: ks, bubbles: true }));
        } catch (e) {}
      }
    } catch (e) { out.actErr = String(e).slice(0, 60); }
    setTimeout(function () { doActions(i + 1); }, a.wait || 250);
  }
  function finish() {
    try {
      var res = {};
      (CFG.checks || []).forEach(function (ck) {
        if (ck.minOptions != null) {
          var sel = $$('select')[0];
          res[ck.id] = !!sel && sel.options.length >= ck.minOptions;
        } else if (ck.minCount != null) {
          res[ck.id] = $$(ck.sel).filter(function (x) {
            return !ck.text || new RegExp(ck.text, 'i').test(x.textContent || ''); }).length >= ck.minCount;
        } else if (ck.optionMatch) {
          var want = (CFG.servedModels || []);
          res[ck.id] = want.length > 0 && $$(ck.sel || 'select').some(function (sel) {
            return [].slice.call(sel.options || []).some(function (o) {
              var t = ((o.value || '') + ' ' + (o.textContent || '')).toLowerCase();
              return want.some(function (w) {
                return t.indexOf(w) >= 0 || (t.trim() && w.indexOf(t.trim()) >= 0);
              });
            });
          });
        } else if (ck.noMatch) {
          res[ck.id] = !new RegExp(ck.bodyRegex, 'i').test(document.body.innerText || '');
        } else if (ck.bodyRegex) {
          res[ck.id] = new RegExp(ck.bodyRegex, 'i').test(document.body.innerText || '');
        } else if (ck.domGrows != null) {
          res[ck.id] = (document.body.innerText || '').length - _qaBaseLen >= ck.domGrows;
        } else if (ck.canvasAnimates) {
          // two post-start samples: identical pixels 900ms apart = frozen
          if (_qaM1 === null) { _qaM1 = snapCanvas(); res[ck.id] = false; }
          else res[ck.id] = _qaM1 !== snapCanvas();
        } else if (ck.sel) {
          res[ck.id] = !!$$(ck.sel).find(function (x) {
            return !ck.text || new RegExp(ck.text, 'i').test(x.textContent || ''); });
        } else res[ck.id] = false;
      });
      out.checks = res;
    } catch (e) { out.mErr = String(e).slice(0, 60); }
    document.title = 'QAPROBE ' + JSON.stringify(out);
  }
  function measure() {
    // first pass: compute checks (finish records canvas sample #1), then
    // re-run finish after a gap so the animation check can compare samples.
    // (The reschedule must happen AFTER finish — sample #1 is recorded there.
    // Checking it before was the bug that made canvas animates never pass.)
    finish();
    if (!_qaDone) {
      _qaDone = true;
      var anim = (CFG.checks || []).some(function (c) { return c.canvasAnimates; });
      if (anim) { setTimeout(finish, 900); return; }
    }
    // async artifacts (IndexedDB): gate the report on a same-origin hold
    // request — virtual time pauses for in-flight network, giving the
    // database callbacks real time to complete before the DOM is graded
    if (CFG.holdUrl) {
      try { fetch(CFG.holdUrl).then(finish).catch(finish); return; } catch (e) {}
    }
  }
  setTimeout(function () { doActions(0); }, 300);
})();
</script>"""


def run_qa_probe(path, task_id, base_url=None):
    """Headless-Chrome functional probe. Returns {ran, rate, passed, total,
    fails[], jsErr} — {ran: False, why} when the probe couldn't run."""
    chrome = find_chrome()
    cfg = QA_PROBES.get(task_id)
    if not CONFIG.get("qa_probe", True):
        return {"ran": False, "why": "disabled in config"}
    if not chrome:
        return {"ran": False, "why": "no Chrome-like browser found"}
    if not cfg:
        return {"ran": False, "why": "no probe for this task"}
    if cfg == "canvas":
        cfg = json.loads(json.dumps(QA_PROBES["canvas"]))
    else:
        cfg = json.loads(json.dumps(cfg))  # deep copy — per-call @BASE_URL@
    for a in cfg.get("actions", []):
        if isinstance(a.get("value"), str) and "@BASE_URL@" in a["value"]:
            if not base_url:
                return {"ran": False, "why": "no endpoint URL for probe"}
            a["value"] = a["value"].replace("@BASE_URL@", base_url)

    # fetch the endpoint's REAL model list server-side (no CORS involved) so
    # the probe can require the artifact's dropdown to contain an actual
    # served model id instead of guessing an option count
    served = []
    if base_url:
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as r:
                served = [str(m.get("id", "")) for m in
                          json.loads(r.read()).get("data", [])][:10]
        except Exception:
            served = []
    if served:
        cfg["servedModels"] = sorted({s.split("/")[-1].lower() for s in served if s})
    if cfg.get("async"):
        # IndexedDB-style artifacts: the self-test is asynchronous REAL-TIME
        # I/O that virtual time does not wait for. Serve the injected copy
        # over HTTP from this backend and gate the report on a hold request
        # so the database callbacks complete before grading.
        cfg["holdUrl"] = f"http://127.0.0.1:{CONFIG.get('port', 7090)}/api/hold?ms=4500"

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            src = f.read()
    except OSError as e:
        return {"ran": False, "why": str(e)[:80]}
    js = _PROBE_TEMPLATE.replace("__CFG__",
                                 json.dumps(cfg).replace("</", "<\\/"))
    # headless Chrome never fires requestAnimationFrame (no display link),
    # so rAF-driven games would read as frozen. Shimmed to a 16ms timer,
    # which virtual time fast-forwards — canvas pixels then actually change.
    # It must run BEFORE the page's own scripts, so it goes first in <head>.
    # Opt-in per probe config: the shim breaks some artifacts (CHIP-8's
    # batched self-test renders nothing under it), and only probes with an
    # animation check need it.
    shim = ""
    if any(c.get("canvasAnimates") for c in cfg.get("checks", [])):
        shim = ("<script>window.requestAnimationFrame = function (cb) "
                "{ return setTimeout(function () { cb(performance.now()); }, 16); };"
                "window.cancelAnimationFrame = function (id) { clearTimeout(id); };</script>")
    # lambda replacement: a plain string replacement would let re.sub
    # interpret the JSON's \n / \\ sequences as escapes and corrupt the script
    if re.search(r"<head[^>]*>", src, re.I):
        html = re.sub(r"<head[^>]*>", lambda m: m.group(0) + shim, src, count=1, flags=re.I)
        if re.search(r"</body>", html, re.I):
            html = re.sub(r"</body>", lambda m: js + "</body>", html, count=1, flags=re.I)
        elif re.search(r"</html>", html, re.I):
            html = re.sub(r"</html>", lambda m: js + "</html>", html, count=1, flags=re.I)
        else:
            html += js
    elif re.search(r"</body>", src, re.I):
        html = re.sub(r"</body>", lambda m: shim + js + "</body>", src, count=1, flags=re.I)
    elif re.search(r"</html>", src, re.I):
        html = re.sub(r"</html>", lambda m: shim + js + "</html>", src, count=1, flags=re.I)
    else:
        html = shim + js + src
    tmpdir = tempfile.mkdtemp(prefix="qa_probe_")
    probe_tmp = None
    try:
        tmp = os.path.join(tmpdir, "probe.html")
        with open(tmp, "w") as f:
            f.write(html)
        target = "file://" + tmp
        if cfg.get("async") and cfg.get("holdUrl"):
            # serve the injected copy from this backend so the page is on an
            # http origin (reliable IndexedDB) alongside the hold endpoint
            probe_tmp = os.path.join(OUTPUT_DIR, "_probe_tmp.html")
            shutil.copyfile(tmp, probe_tmp)
            target = f"http://127.0.0.1:{CONFIG.get('port', 7090)}/output/_probe_tmp.html"
        cmd = [chrome, "--headless=new", "--disable-gpu", "--no-first-run",
               "--disable-extensions", "--disable-web-security",
               "--allow-file-access-from-files", "--allow-running-insecure-content",
               f"--virtual-time-budget={cfg.get('budget', 8000)}",
               "--dump-dom", target]
        # a throwaway profile isolates the probe — but a full Chrome build
        # hangs when --user-data-dir is passed on this Mac, so only dedicated
        # headless/testing shells get one
        if "chrome-headless-shell" in chrome or "for Testing" in chrome \
                or "chromium" in os.path.basename(chrome).lower():
            cmd.insert(-2, f"--user-data-dir={os.path.join(tmpdir, 'profile')}")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            return {"ran": False, "why": "chrome timed out"}
        m = re.search(r"<title>QAPROBE (.*?)</title>", r.stdout, re.S)
        if not m:
            return {"ran": False, "why": "probe did not report"}
        try:
            out = json.loads(m.group(1))
        except json.JSONDecodeError:
            return {"ran": False, "why": "probe report unparsable"}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if probe_tmp:
            try:
                os.remove(probe_tmp)
            except OSError:
                pass

    checks = out.get("checks") or {}
    total = len(checks)
    if not total or not out.get("typed") and not out.get("clicked") \
            and not any("canvas" in k for k in checks):
        # nothing was interactable and nothing rendered — probe blindness
        return {"ran": False, "why": "probe found no interaction targets"}
    fails = [k for k, v in checks.items() if not v]
    js_err = out.get("jsErr") or ""
    # Virtual time fast-forwards every timer, which can push a heavy page
    # (e.g. a compiler that re-renders all its spec tests per keystroke) into
    # a runaway the real browser never hits. Those page crashes under the
    # probe are probe instability, not evidence about the artifact.
    if js_err and any(p in js_err.lower() for p in
                      ("invalid string length", "maximum call stack", "out of memory")):
        return {"ran": False, "why": f"probe unstable under virtual time ({js_err[:60]})"}
    return {"ran": True,
            "rate": round(100 * (total - len(fails)) / total),
            "passed": total - len(fails), "total": total, "fails": fails,
            "jsErr": js_err or None, "clicked": out.get("clicked")}


RAWPLUS_MAX_ROUNDS = 4  # 4 x per-chunk cap; context 261k holds prompt+output


# ---------------------------------------------------------------------------
# Log-report task: the only data-analysis task in the suite. Its input is a
# fixture snapshotted from the PREVIOUS run's logs (never the current run,
# whose logs don't exist yet), so every harness — and every repeat — grades
# against identical, finite input. Agent harnesses read the files from disk;
# raw/raw+ get the compact parts inlined into the prompt.
# ---------------------------------------------------------------------------
LOGREPORT_DIR = os.path.join(WORK_DIR, "logreport-fixture")

# per-cell input fixtures: copied fresh into every harness workdir so each
# cell works on its own pristine copy (agents edit theirs in place)
TASK_FIXTURES = {
    "bugfix": ("expenses-broken.html", "expenses.html"),
    "codereview": ("review-sample.py", "review-sample.py"),
}


def _build_logreport_fixture():
    """Snapshot the previous run's failure evidence into LOGREPORT_DIR:
    runs.json (the previous run's per-cell records), errors.log (the failure
    lines its orchestrator logged), bench-excerpt.log (its full orchestrator
    window) and per-framework server-log error excerpts."""
    os.makedirs(LOGREPORT_DIR, exist_ok=True)
    prev_file = None
    run_ts = STATE.get("run_started") or time.time()
    for f in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json")),
                    key=os.path.getmtime, reverse=True):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("ts") and d["ts"] < run_ts:
            prev_file = (f, d)
            break

    bench_lines = []
    prev_start = prev_end = None
    if prev_file:
        f, d = prev_file
        rows = [{"framework": r.get("framework"), "harness": r.get("harness"),
                 "status": r.get("status"), "latency": r.get("latency"),
                 "qa_func": r.get("qa_func"), "error": (r.get("error") or "")[:200]}
                for r in (d.get("results") or [])]
        with open(os.path.join(LOGREPORT_DIR, "runs.json"), "w") as fh:
            json.dump({"run_file": os.path.basename(f),
                       "task": d.get("task_id"), "model_set": d.get("model_set"),
                       "ts": d.get("ts"), "cells": rows}, fh, indent=1)
        prev_start = d.get("ts") or 0
        # runs execute cells sequentially — the window spans the summed latencies
        prev_end = prev_start + sum(r.get("latency") or 0 for r in rows) + 120
    if not prev_file:
        with open(os.path.join(LOGREPORT_DIR, "runs.json"), "w") as fh:
            json.dump({"note": "no previous run found — analyze whatever "
                               "failures appear in the logs"}, fh, indent=1)

    bench_path = os.path.join(LOG_DIR, "bench.log")
    if os.path.isfile(bench_path):
        with open(bench_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if m:
                    try:
                        t = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
                    except ValueError:
                        continue
                    if prev_start and not (prev_start - 60 <= t <= prev_end + 60):
                        continue
                    bench_lines.append(line)
        with open(os.path.join(LOGREPORT_DIR, "bench-excerpt.log"), "w") as fh:
            fh.writelines(bench_lines[-3000:])
        # the orchestrator's failure lines — the primary root-cause evidence
        err_lines = [l for l in bench_lines if re.search(
            r"error|fail|timeout|⚠|no output|empty response|stopped|exited", l, re.I)]
        with open(os.path.join(LOGREPORT_DIR, "errors.log"), "w") as fh:
            fh.writelines(err_lines[-120:])

    # per-framework server logs: error-class lines only, capped
    for fw in ("omlx", "mtplx", "mlxlm", "mlxserve"):
        logs = sorted(glob.glob(os.path.join(LOG_DIR, f"{fw}-*.log")),
                      key=os.path.getmtime)
        if not logs:
            continue
        err_lines = []
        with open(logs[-1], encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if re.search(r"ERROR|WARNING|Traceback|reject|guard|failed|stall", line, re.I):
                    err_lines.append(line)
        with open(os.path.join(LOGREPORT_DIR, f"{fw}.log"), "w") as fh:
            fh.writelines(err_lines[-400:])


def _logreport_inline():
    """The compact input inlined into the prompt for harnesses that cannot
    read files: runs.json plus the orchestrator's failure lines."""
    parts = []
    for name in ("runs.json", "errors.log"):
        p = os.path.join(LOGREPORT_DIR, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as fh:
                body = fh.read(40000)
            if body.strip():
                parts.append(f"--- {name} ---\n{body}")
    return "\n\n".join(parts) if parts else \
        "(fixture not built — report that no input data was available)"


HARNESS_LABELS = {
    "raw": "Raw (api)",
    "rawplus": "raw+",
    "pi": "pi",
    "opencode": "opencode",
    "goose": "Goose",
    "hart": "hart",
}

# hart harness location — bundled in this repo under hart/ (so a fresh clone
# works out of the box); override via "hart_path" in config.json. Relative
# paths resolve against the repo root; ~ is expanded.
_hart_cfg = CONFIG.get("hart_path", os.path.join(ROOT, "hart", "hart.py"))
HART_PATH = os.path.expanduser(
    _hart_cfg if os.path.isabs(_hart_cfg) else os.path.join(ROOT, _hart_cfg))

# Routing for agent harnesses (pi/opencode/goose). True = via the local
# measurement proxy, which adds per-request PP/TGS/TTFT to their rows.
# False = straight to the framework, byte-identical to an independent harness
# run (agent rows then report TPS/QA only — raw keeps full metrics either way).
ROUTE_VIA_PROXY = bool(CONFIG.get("route_via_proxy", False))

# pi thinking-level suffix for --model (pi supports :off…:xhigh). Empty =
# inherit the server's reasoning setting like every other harness, so all
# cells share one uniform budget.
PI_THINKING = CONFIG.get("pi_thinking", "")

# Reasoning level per task type, applied at framework server start (each
# framework maps a level to its own CLI flags via reasoning_flags, or to the
# omlx settings file via reasoning_file). LONG tasks get a bigger thinking
# budget than standard tasks.
REASONING_LEVELS = CONFIG.get("reasoning") or {"short": "low", "long": "medium"}


def reasoning_level_for(task_id):
    """Reasoning level for a task: the 'long' level (default medium) for
    long-horizon tasks, the 'short' level (default low) for standard tasks."""
    long_task = any(t.get("long") for t in TASKS if t["id"] == task_id)
    return REASONING_LEVELS.get("long" if long_task else "short", "low")


def effective_reasoning(fw, level):
    """Per-framework override: a framework whose model handles a limited
    reasoning range can pin its level via "reasoning_override" in its config
    (e.g. MTPLX's optimized build degrades at medium+ — its own stream-stall
    watchdog aborts long silent thinking phases)."""
    return FRAMEWORKS[fw].get("reasoning_override") or level


def agent_base_url(fw):
    port = PROXY_PORT if ROUTE_VIA_PROXY else FRAMEWORKS[fw]["port"]
    return f"http://127.0.0.1:{port}/v1"


def prompt_for_harness(prompt, artifact, harness):
    """Agent harnesses must deliver files, not chat text — the raw prompt's
    “return only the HTML in one code block” instruction makes them answer
    inline and stall instead of using their write tools."""
    if harness in ("raw", "rawplus") or not artifact:
        return prompt
    # Drop any sentence telling the model to answer inline in a code block —
    # exact canonical phrasing first, then any edited variant.
    base = re.sub(r"Return only the full HTML file in one ```html code block\.\s*$",
                  "", prompt)
    base = re.sub(r"[^.!?\n]*```html code block[^.!?\n]*[.!\n]?\s*", "", base).strip()
    return (base + f"\n\nDo not print the file contents in chat. Write the complete, "
            f"self-contained file to '{artifact}' in the current working directory; "
            f"it must run by simply opening it in a browser. "
            f"The file will be large: create it with the FIRST section using a "
            f"write, then add the remaining sections with edit/append operations "
            f"(a few thousand characters each) instead of emitting it all in one "
            f"response — never truncate the file to fit a single reply.")

# ----------------------------------------------------------------------------
# Shared state
# ----------------------------------------------------------------------------
STATE = {
    "running": False,
    "current_step": "",
    "run_started": None,
    "framework_status": {fw: "down" for fw in FRAMEWORKS},
    "results": [],
}
ACTIVITY = []          # list of {ts, msg, fw, harness, level}
ACTIVITY_MAX = 1500
RUN_FLAG = threading.Event()
LOCK = threading.Lock()
PROCS = {}             # fw -> subprocess.Popen
FW_LOGS = {}           # fw -> open log file for the framework's stdout
FW_REASONING = {}      # fw -> reasoning level the running server was started with
CURRENT = {"proc": None, "sock": None}

CUR_LOCK = threading.Lock()


class RunStopped(Exception):
    """Raised inside a benchmark worker when the user pressed Stop."""


def request_stop():
    """Stop the run: clear the flag AND actively interrupt whatever cell is
    in flight (kill the CLI process group, shut the streaming socket) so the
    worker threads wake immediately instead of at their next checkpoint."""
    RUN_FLAG.clear()
    with CUR_LOCK:
        proc, sock = CURRENT["proc"], CURRENT["sock"]
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if sock:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


def log(msg, fw=None, harness=None, level=""):
    with LOCK:
        ACTIVITY.append({"ts": time.time(), "msg": msg, "fw": fw,
                         "harness": harness, "level": level})
        if len(ACTIVITY) > ACTIVITY_MAX:
            del ACTIVITY[: len(ACTIVITY) - ACTIVITY_MAX]
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{fw or 'sys'}] {msg}"
    print(line, flush=True)
    try:
        with open(os.path.join(LOG_DIR, "bench.log"), "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ----------------------------------------------------------------------------
# System helpers
# ----------------------------------------------------------------------------
_TOTAL_MEM = None
_RAM_CACHE = {"t": 0.0, "free": 0, "total": 0}


def _total_mem():
    """Total RAM in bytes — computed once (macOS sysctl / Linux meminfo)."""
    global _TOTAL_MEM
    if _TOTAL_MEM is None:
        _TOTAL_MEM = 0
        try:
            if sys.platform == "darwin":
                _TOTAL_MEM = int(subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5).stdout.strip())
            elif os.path.isfile("/proc/meminfo"):
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            _TOTAL_MEM = int(line.split()[1]) * 1024
                            break
        except Exception:
            _TOTAL_MEM = 0
    return _TOTAL_MEM


def free_ram():
    """Return (free_bytes, total_bytes). macOS: vm_stat; Linux: /proc/meminfo.
    Cached for 3s — the UI polls /api/state every ~1.2s and this used to
    spawn two subprocesses per poll."""
    now = time.time()
    if _RAM_CACHE["total"] and now - _RAM_CACHE["t"] < 3.0:
        return _RAM_CACHE["free"], _RAM_CACHE["total"]
    total = _total_mem()
    free = 0
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True,
                                 text=True, timeout=5).stdout
            ps = int(re.search(r"page size of (\d+) bytes", out).group(1))
            free = (int(re.search(r"Pages free:\s+(\d+)", out).group(1))
                    + int(re.search(r"Pages speculative:\s+(\d+)", out).group(1))
                    + int(re.search(r"Pages inactive:\s+(\d+)", out).group(1))) * ps
        elif os.path.isfile("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                mi = {}
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        mi[k] = int(v.split()[0]) * 1024
            free = mi.get("MemAvailable", mi.get("MemFree", 0))
    except Exception:
        free = 0
    _RAM_CACHE.update({"t": now, "free": free, "total": total})
    return free, total


def port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def framework_healthy(fw):
    """Framework is up when its OpenAI-compatible /v1/models answers."""
    cfg = FRAMEWORKS[fw]
    if not port_open(cfg["port"]):
        return False
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _get_json(url, timeout=4):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _prom_scrape(text):
    """Parse Prometheus text exposition into {name: value}. Labels are
    stripped and summed across series (all metrics we read are counters or
    histogram _sum/_count, for which summing is correct); _bucket lines are
    dropped."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name, val = parts
        if "{" in name:
            name = name.split("{", 1)[0]
        if name.endswith("_bucket"):
            continue
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


def server_snapshot(fw):
    """Server-side metric snapshot. OMLX: totals + split prefill/generation
    seconds. MTPLX: per-request 'recent' records. MLX-LM: none (client-side
    only). These give authoritative PP/TGS for ANY harness — including pi,
    whose non-streaming calls hide timing from clients."""
    cfg = FRAMEWORKS[fw]
    if fw == "omlx":
        s = _get_json(f"http://127.0.0.1:{cfg['port']}/admin/api/stats")
        u = _get_json(f"http://127.0.0.1:{cfg['port']}/admin/api/usage")
        if s is None:
            return None
        t = (u or {}).get("totals", {})
        return {"kind": "omlx",
                "prompt": s.get("total_prompt_tokens") or 0,
                "completion": s.get("total_completion_tokens") or 0,
                "pre_s": t.get("prefill_seconds") or 0.0,
                "gen_s": t.get("generation_seconds") or 0.0,
                "cache_eff": s.get("cache_efficiency")}
    if fw == "mlxserve":
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{cfg['port']}/metrics", timeout=4) as r:
                m = _prom_scrape(r.read().decode("utf-8", "replace"))
        except Exception:
            return None
        if "vllm:prompt_tokens_total" not in m:
            return None
        return {"kind": "mlxserve",
                "prompt": m.get("vllm:prompt_tokens_total", 0),
                "completion": m.get("vllm:generation_tokens_total", 0),
                "cache": m.get("mlx_serve:prefix_cache_tokens_total", 0),
                "pre_s": m.get("vllm:request_prefill_time_seconds_sum", 0.0),
                "gen_s": m.get("vllm:request_decode_time_seconds_sum", 0.0),
                "ttft_s": m.get("vllm:time_to_first_token_seconds_sum", 0.0),
                "ttft_n": m.get("vllm:time_to_first_token_seconds_count", 0)}
    if fw == "mlxlm":  # mlx_vlm.server: JSON /metrics with per-request records
        m = _get_json(f"http://127.0.0.1:{cfg['port']}/metrics")
        if m is None:
            return None
        s = m.get("summary") or {}
        return {"kind": "mlxvlm",
                "prompt": s.get("prompt_tokens_total") or 0,
                "completion": s.get("completion_tokens_total") or 0,
                "recent": m.get("recent") or []}
    if fw == "mtplx":
        m = _get_json(f"http://127.0.0.1:{cfg['port']}/metrics")
        if m is None:
            return None
        return {"kind": "mtplx", "recent": m.get("recent") or [],
                "latest": m.get("latest")}
    return None


def server_cell_delta(fw, before, after):
    """Per-cell server-side metrics from a before/after snapshot diff."""
    if not before or not after or before.get("kind") != after.get("kind"):
        return {}
    out = {}
    if before["kind"] == "omlx":
        dp = after["prompt"] - before["prompt"]
        dc = after["completion"] - before["completion"]
        dpre = max(after["pre_s"] - before["pre_s"], 0.0)
        dgen = max(after["gen_s"] - before["gen_s"], 0.0)
        # usage counters flush every ~5s — a diff window that small yields
        # garbage rates (observed 134k tok/s). Require a real timing window
        # and clamp to physically plausible ranges.
        if dp > 0 and dpre >= 2.0:
            pp = dp / dpre
            if pp <= 10000:
                out["server_pp"] = round(pp, 1)
        if dc > 0 and dgen >= 2.0:
            tgs = dc / dgen
            if tgs <= 500:
                out["server_tgs"] = round(tgs, 1)
        out["server_prompt_tokens"] = dp
        out["server_completion_tokens"] = dc
    elif before["kind"] == "mlxserve":
        dp = after["prompt"] - before["prompt"]
        dc = after["completion"] - before["completion"]
        dpre = max(after["pre_s"] - before["pre_s"], 0.0)
        dgen = max(after["gen_s"] - before["gen_s"], 0.0)
        dttft = after["ttft_s"] - before["ttft_s"]
        dtn = after["ttft_n"] - before["ttft_n"]
        if dp > 0 and dpre > 0.01:
            out["server_pp"] = round(dp / dpre, 1)
        if dc > 0 and dgen > 0.01:
            out["server_tgs"] = round(dc / dgen, 1)
        if dtn > 0 and dttft > 0:
            out["server_ttft_avg"] = round(dttft / dtn, 3)
        out["server_prompt_tokens"] = dp
        out["server_completion_tokens"] = dc
        out["server_cached_tokens"] = after["cache"] - before["cache"]
    elif before["kind"] == "mlxvlm":
        n0 = {json.dumps(r, sort_keys=True) for r in before["recent"]}
        fresh = [r for r in after["recent"]
                 if json.dumps(r, sort_keys=True) not in n0]
        pps = [r["prefill_tok_s"] for r in fresh
               if isinstance(r.get("prefill_tok_s"), (int, float))]
        tgss = [r["decode_tok_s"] for r in fresh
                if isinstance(r.get("decode_tok_s"), (int, float))]
        ttfts = [r["ttft_s"] for r in fresh
                 if isinstance(r.get("ttft_s"), (int, float))]
        if pps:
            out["server_pp"] = round(sum(pps) / len(pps), 1)
        if tgss:
            out["server_tgs"] = round(sum(tgss) / len(tgss), 1)
        if ttfts:
            out["server_ttft_avg"] = round(sum(ttfts) / len(ttfts), 3)
        out["server_prompt_tokens"] = after["prompt"] - before["prompt"]
        out["server_completion_tokens"] = after["completion"] - before["completion"]
        out["server_requests"] = len(fresh)
    elif before["kind"] == "mtplx":
        n0 = {json.dumps(r, sort_keys=True) for r in before["recent"]}
        fresh = [r for r in after["recent"]
                 if json.dumps(r, sort_keys=True) not in n0]
        out["server_requests"] = len(fresh)
        pps, tgss, tts = [], [], []
        for r in fresh:
            for pk in ("prefill_tok_s", "prompt_tps", "prefill_tps", "pp_tps"):
                v = r.get(pk)
                if isinstance(v, (int, float)) and 0.5 < v < 10000:
                    pps.append(v); break
            for gk in ("decode_tok_s", "generation_tps", "gen_tps", "tgs"):
                v = r.get(gk)
                if isinstance(v, (int, float)) and 0.5 < v < 500:
                    tgss.append(v); break
            if isinstance(r.get("ttft_s"), (int, float)):
                tts.append(r["ttft_s"])
        if tts:
            out["server_ttft_avg"] = round(sum(tts) / len(tts), 3)
        if pps:
            out["server_pp"] = round(sum(pps) / len(pps), 1)
        if tgss:
            out["server_tgs"] = round(sum(tgss) / len(tgss), 1)
    return out


def log_device_info(fw):
    """OMLX device-info: chip context for reports (one line, once per start)."""
    cfg = FRAMEWORKS[fw]
    d = _get_json(f"http://127.0.0.1:{cfg['port']}/admin/api/device-info")
    if d:
        log(f"device: {d.get('chip_name')} {d.get('chip_variant')}, "
            f"{d.get('memory_gb')} GB, {d.get('gpu_cores')} GPU cores",
            fw=cfg["name"], level="ok")


def adopt_served_meta(fw):
    """Adopt runtime metadata from the served model entry (mlx-serve exposes
    context_length per model — GUI instances may differ from config)."""
    cfg = FRAMEWORKS[fw]
    if fw != "mlxserve":
        return
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{cfg['port']}/v1/models", timeout=5) as r:
            for entry in json.loads(r.read()).get("data", []):
                if entry.get("id") == cfg["model"] or entry.get("loaded"):
                    cl = entry.get("context_length")
                    if cl and cl != cfg.get("ctx_tokens"):
                        log(f"{cfg['name']} served context length {cl} "
                            f"(config had {cfg.get('ctx_tokens')}) — adopting",
                            fw=fw)
                        cfg["ctx_tokens"] = cl
                    break
    except Exception:
        pass


_FW_STATUS_TS = {"t": 0.0}


def refresh_fw_status_idle(max_age=4.0):
    """Re-check framework liveness for the UI while idle. The status map is
    otherwise a stale snapshot (set at startup / during runs), so externally
    killed servers kept showing green forever. TTL-cached: the UI polls
    /api/state every ~1.2s; dead ports refuse instantly, live ones are a
    cheap TCP connect (no HTTP GET — the run path uses framework_healthy)."""
    if time.time() - _FW_STATUS_TS["t"] < max_age:
        return
    _FW_STATUS_TS["t"] = time.time()
    for fw in FRAMEWORKS:
        if STATE["framework_status"].get(fw) != "starting":
            set_fw_status(fw, "up" if port_open(FRAMEWORKS[fw]["port"]) else "down")


def set_fw_status(fw, status):
    with LOCK:
        STATE["framework_status"][fw] = status


# ----------------------------------------------------------------------------
# Measurement proxy — agent harnesses (pi/opencode) point here instead of at
# the framework directly. Traffic is relayed untouched while per-request
# timing (TTFT / prefill / decode, from the SSE stream) is recorded, giving
# them the same PP/TGS metrics raw gets. Runs are sequential, so a single
# stats accumulator suffices.
# ----------------------------------------------------------------------------
PROXY_PORT = 7010
PROXY_STATE = {"target": None, "fw": None, "model": None}
PROXY_LOCK = threading.Lock()
PROXY_STATS = {}   # per-run aggregates (reset before each harness dispatch)
PROXY_TOTALS = {}  # cumulative since last manual clear (Proxy Inspector)
PROXY_LOG = []     # per-request records for reporting, capped
PROXY_LOG_MAX = 500
PROXY_ZERO = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
              "ttft_sum": 0.0, "decode_sum": 0.0, "wall_sum": 0.0, "length_hits": 0}
PROXY_STATS.update(PROXY_ZERO)
PROXY_TOTALS.update(PROXY_ZERO)


def proxy_reset():
    with PROXY_LOCK:
        PROXY_STATS.clear()
        PROXY_STATS.update({"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                            "ttft_sum": 0.0, "decode_sum": 0.0, "wall_sum": 0.0})


def proxy_read():
    with PROXY_LOCK:
        return dict(PROXY_STATS)


def _proxy_record(usage, t0, first_tok, last_tok, model=None, path="", stream=False, status=200,
                  finish=None):
    now = time.time()
    rec = {"ts": now, "path": path, "model": model, "stream": stream, "status": status,
           "finish": finish,
           "prompt_tokens": (usage or {}).get("prompt_tokens") or 0,
           "completion_tokens": (usage or {}).get("completion_tokens") or 0,
           "ttft": round(first_tok - t0, 3) if first_tok else None,
           "decode": round(last_tok - first_tok, 3)
                     if (first_tok and last_tok and last_tok > first_tok) else None,
           "wall": round(now - t0, 3)}
    with PROXY_LOCK:
        for store in (PROXY_STATS, PROXY_TOTALS):  # per-run + since-clear
            store["requests"] += 1
            store["prompt_tokens"] += rec["prompt_tokens"]
            store["completion_tokens"] += rec["completion_tokens"]
            if finish == "length":
                store["length_hits"] += 1
            if first_tok:
                store["ttft_sum"] += first_tok - t0
            if first_tok and last_tok and last_tok > first_tok:
                store["decode_sum"] += last_tok - first_tok
            store["wall_sum"] += now - t0
        PROXY_LOG.append(rec)
        if len(PROXY_LOG) > PROXY_LOG_MAX:
            del PROXY_LOG[: len(PROXY_LOG) - PROXY_LOG_MAX]


class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _relay_error(self, e):
        try:
            payload = e.read()
        except Exception:
            payload = b"{}"
        self.send_response(e.code)
        ctype = e.headers.get("Content-Type", "application/json") if e.headers else "application/json"
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _relay(self):
        target = PROXY_STATE["target"]
        if not target:
            self.send_error(503, "no framework under test")
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() in ("content-type", "authorization", "accept")}
        t0 = time.time()
        try:
            payload = json.loads(body) if body else None
        except json.JSONDecodeError:
            payload = None
        model = payload.get("model") if isinstance(payload, dict) else None
        stream = bool(isinstance(payload, dict) and payload.get("stream"))
        # Ask streaming endpoints to include token usage; some clients omit it.
        injected = False
        if isinstance(payload, dict) and payload.get("stream") and "stream_options" not in payload:
            payload["stream_options"] = {"include_usage": True}
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
            injected = True
        req = urllib.request.Request(target + self.path, data=body or None,
                                     headers=headers, method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            if injected and e.code == 400:  # server rejects stream_options — retry plain
                payload.pop("stream_options", None)
                body = json.dumps(payload).encode()
                try:
                    resp = urllib.request.urlopen(urllib.request.Request(
                        target + self.path, data=body, headers=headers,
                        method=self.command), timeout=1800)
                except urllib.error.HTTPError as e2:
                    _proxy_record(None, t0, None, None, model, self.path, stream, e2.code)
                    self._relay_error(e2)
                    return
                except Exception as e2:
                    _proxy_record(None, t0, None, None, model, self.path, stream, 502)
                    self.send_error(502, str(e2))
                    return
            else:
                _proxy_record(None, t0, None, None, model, self.path, stream, e.code)
                self._relay_error(e)
                return
        except Exception:
            _proxy_record(None, t0, None, None, model, self.path, stream, 502)
            self.send_error(502, "relay failed")
            return

        ct = resp.headers.get("Content-Type", "")
        if "event-stream" in ct:
            self.send_response(resp.status)
            self.send_header("Content-Type", ct)
            self.end_headers()
            first = last = None
            usage = None
            finish = None
            try:
                for raw in resp:
                    self.wfile.write(raw)
                    self.wfile.flush()
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    d = line[5:].strip()
                    if d == "[DONE]":
                        break  # some servers keep-alive after DONE — stop reading
                    if not d:
                        continue
                    try:
                        chunk = json.loads(d)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                        last = time.time()
                    for ch in chunk.get("choices") or []:
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
                        delta = ch.get("delta") or {}
                        if delta.get("content") or delta.get("reasoning_content") or ch.get("text"):
                            now = time.time()
                            if first is None:
                                first = now
                            last = now
            except Exception:
                pass  # client disconnected mid-stream; keep what we measured
            _proxy_record(usage, t0, first, last, model, self.path, True, resp.status, finish)
        else:
            data = resp.read()
            usage = None
            finish = None
            try:
                j = json.loads(data)
                usage = j.get("usage")
                ch = (j.get("choices") or [{}])[0]
                finish = ch.get("finish_reason")
            except Exception:
                pass
            self.send_response(resp.status)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            _proxy_record(usage, t0, None, None, model, self.path, False, resp.status, finish)

    do_GET = _relay
    do_POST = _relay


def start_proxy():
    srv = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), ProxyHandler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()


# ----------------------------------------------------------------------------
# Framework lifecycle
# ----------------------------------------------------------------------------
def _adopt_served_model_id(fw, retries=10, delay=2.0):
    """Align cfg['model'] with the id the server actually serves. Frameworks
    like OMLX serve cache-style ids (org--name) while the config/picker use
    repo-style (org/name); match by normalized key so requests resolve.
    Right after a start, the server's discovery scan of the HF cache may be
    only partially complete (a 225GB cache takes a while to walk) — retry
    for a while before concluding the model isn't served. Falls back to
    adopting the single served id, else warns."""
    cfg = FRAMEWORKS[fw]
    want = discovery.normalize_key(cfg["model"])
    ids = []
    for _ in range(retries):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{cfg['port']}/v1/models", timeout=5) as r:
                ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
        except Exception:
            return
        if cfg["model"] in ids:
            return
        match = next((i for i in ids
                      if discovery.normalize_key(i) == want), None)
        if match:
            log(f"{cfg['name']} serves “{cfg['model']}” as “{match}” — adopting",
                fw=fw, level="ok")
            cfg["model"] = match
            return
        time.sleep(delay)
    if len(ids) == 1:
        log(f"{cfg['name']} serves model id “{ids[0]}” — adopting", fw=fw, level="ok")
        cfg["model"] = ids[0]
    elif ids:
        log(f"⚠ configured model “{cfg['model']}” not in {cfg['name']}'s list "
            f"{ids} after {retries} checks — requests may fail", fw=fw, level="err")


def _served_model_ids(cfg):
    """Model ids the framework currently reports (empty when down/unknown)."""
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{cfg['port']}/v1/models", timeout=5) as r:
            return [m.get("id") for m in json.loads(r.read()).get("data", [])
                    if m.get("id")]
    except Exception:
        return []


def ensure_single_model(fw):
    """Sequential-testing guarantee: exactly ONE resident model in an OMLX
    pool — the selected one. Any other resident model is unloaded (two
    residents combine their RAM and trip the ceiling → 507s), and the
    selected model is warmed if the unload swept it out too."""
    if fw != "omlx":
        return
    cfg = FRAMEWORKS[fw]
    try:
        stats = _get_json(f"http://127.0.0.1:{cfg['port']}/admin/api/stats")
        loaded = [m.get("id") for m in
                  ((stats or {}).get("active_models") or {}).get("models", [])
                  if m.get("id")]
    except Exception:
        return
    want = discovery.normalize_key(cfg["model"])
    stale = [mid for mid in loaded
             if discovery.normalize_key(mid) != want]
    for mid in stale:
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{cfg['port']}/v1/models/"
                f"{urllib.parse.quote(mid)}/unload", method="POST")
            urllib.request.urlopen(req, timeout=120)
            log(f"unloaded non-selected resident {mid} — one model at a time",
                fw=cfg["name"])
        except Exception as e:
            log(f"unload {mid} failed: {e}", fw=cfg["name"], level="err")
    if stale and cfg["model"] in loaded:
        # the sweep took the selected model out with the rest — warm it back
        try:
            call_chat(fw, "Say OK.", {"temperature": 0.1, "max_tokens": 16})
            log("selected model re-warmed after the unload sweep", fw=cfg["name"])
        except Exception as e:
            log(f"re-warm failed: {e}", fw=cfg["name"], level="err")


def omlx_reasoning_apply(fw, level):
    """omlx has no CLI reasoning flag — the level lives in its per-model
    settings file (reasoning_file), applied at request time. Rewrite the
    served model's chat_template_kwargs[reasoning_key] to `level` (and force
    it via forced_ct_kwargs) before the run. Returns a restore() closure that
    puts the original value back, or None if the file was left untouched."""
    cfg = FRAMEWORKS[fw]
    path = os.path.expanduser(
        cfg.get("reasoning_file", "~/.omlx/model_settings.json"))
    key = cfg.get("reasoning_key", "reasoning_effort")
    model = cfg["model"]
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log(f"reasoning: cannot read {path} ({e}) — file left untouched",
            fw=fw, level="err")
        return None
    # The settings file is keyed by cache-style ids (org--name) while the
    # config may use repo-style (org/name) — match by normalized key.
    want = discovery.normalize_key(model)
    entry_key = next((k for k, v in (data.get("models") or {}).items()
                      if isinstance(v, dict)
                      and discovery.normalize_key(k) == want), None)
    entry = (data.get("models") or {}).get(entry_key) if entry_key else None
    if entry is None:
        log(f"reasoning: no settings entry for “{model}” in {path} — file "
            f"left untouched", fw=fw, level="err")
        return None
    orig_ct = entry.get("chat_template_kwargs")
    orig_forced = entry.get("forced_ct_kwargs")
    ct = dict(orig_ct) if isinstance(orig_ct, dict) else {}
    ct[key] = level
    entry["chat_template_kwargs"] = ct
    forced = list(orig_forced) if isinstance(orig_forced, list) else []
    if key not in forced:
        forced.append(key)
    entry["forced_ct_kwargs"] = forced
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        log(f"reasoning: cannot write {path} ({e})", fw=fw, level="err")
        return None
    log(f"reasoning: {os.path.basename(path)} → “{model}” {key}={level}",
        fw=fw)

    def restore():
        try:
            with open(path) as f:
                data2 = json.load(f)
            e2 = (data2.get("models") or {}).get(entry_key)
            if isinstance(e2, dict):
                if orig_ct is None:
                    e2.pop("chat_template_kwargs", None)
                else:
                    e2["chat_template_kwargs"] = orig_ct
                if orig_forced is None:
                    e2.pop("forced_ct_kwargs", None)
                else:
                    e2["forced_ct_kwargs"] = orig_forced
            with open(path, "w") as f:
                json.dump(data2, f, indent=2)
            log(f"reasoning: {os.path.basename(path)} restored to pre-run "
                f"value", fw=fw)
        except (OSError, json.JSONDecodeError) as e:
            log(f"reasoning: could not restore {path} ({e}) — check it "
                f"manually", fw=fw, level="err")
    return restore


def _omlx_unload_all(cfg):
    """Unload every model resident in OMLX's pool. A reused server (or the
    GUI's auto-started default) otherwise keeps its model resident, and a
    large model's load then exceeds the memory ceiling with the old weights
    still counted (observed: Flash-Next 103.94GB + resident 27B 29.34GB >
    114GB ceiling → HTTP 507 on every request)."""
    try:
        stats = _get_json(f"http://127.0.0.1:{cfg['port']}/admin/api/stats")
        loaded = ((stats or {}).get("active_models") or {}).get("models", [])
        for m in loaded:
            mid = m.get("id")
            if not mid:
                continue
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{cfg['port']}/v1/models/"
                    f"{urllib.parse.quote(mid)}/unload", method="POST")
                urllib.request.urlopen(req, timeout=120)
                log(f"unloaded resident model {mid} — freeing its memory",
                    fw=cfg["name"])
            except urllib.error.HTTPError as e:
                # 400 "Model not loaded" = discovered but not resident — fine
                body = e.read().decode("utf-8", "replace")[:100]
                if "not loaded" not in body:
                    log(f"unload {mid}: {e.code} {body}",
                        fw=cfg["name"], level="err")
            except Exception as e:
                log(f"unload {mid} failed: {e}", fw=cfg["name"], level="err")
    except Exception as e:
        log(f"omlx unload-all failed: {e}", fw=cfg["name"], level="err")


def start_framework(fw, reasoning_level=None):
    cfg = FRAMEWORKS[fw]
    if framework_healthy(fw):
        # omlx reads model_settings.json at SERVER START - if the file
        # changed after the server process started, the running server is
        # applying stale settings (observed: the oQ5e 507'd every request
        # on a pre-125GB-ceiling server). Restart instead of reusing.
        if fw == "omlx":
            ms_path = os.path.expanduser(cfg.get("reasoning_file", ""))
            srv_pids = subprocess.run(["lsof", "-ti", f":{cfg['port']}"],
                                      capture_output=True, text=True).stdout.split()
            if ms_path and srv_pids:
                try:
                    ms_mtime = os.path.getmtime(ms_path)
                    out = subprocess.run(["ps", "-o", "lstart=", "-p",
                                          srv_pids[0].strip()],
                                         capture_output=True, text=True).stdout.strip()
                    srv_start = time.mktime(time.strptime(
                        out, "%a %b %d %H:%M:%S %Y"))
                    if ms_mtime > srv_start:
                        log(f"{cfg['name']} server predates model_settings.json "
                            f"changes - restarting for the new settings",
                            fw=cfg["name"], level="err")
                        stop_framework(fw)
                except Exception:
                    pass
        if fw in PROCS and FW_REASONING.get(fw) != reasoning_level:
            # We started it for a different task type — the reasoning flags
            # are baked in at start, so restart with the right level.
            log(f"{cfg['name']} is up with reasoning={FW_REASONING.get(fw)} — "
                f"restarting for reasoning={reasoning_level}", fw=fw)
            stop_framework(fw)
        if fw in PROCS and fw != "omlx":
            # single-model servers run ONE model via CLI flags: if the served
            # model is not the newly selected one, a reuse would benchmark the
            # WRONG model — restart with the selection baked in
            served = _served_model_ids(cfg)
            want = discovery.normalize_key(cfg["model"])
            if served and not any(discovery.normalize_key(s) == want
                                  for s in served):
                log(f"{cfg['name']} serves {served} but {cfg['model']} is "
                    f"selected — restarting for the new model", fw=fw)
                stop_framework(fw)
        else:
            log(f"{cfg['name']} already running on port {cfg['port']} — reusing"
                + (f" (reasoning={FW_REASONING.get(fw)})"
                   if FW_REASONING.get(fw) else ""), fw=fw)
            _adopt_served_model_id(fw)
            adopt_served_meta(fw)
            if fw == "omlx":
                _omlx_unload_all(cfg)
                ensure_single_model(fw)
            set_fw_status(fw, "up")
            return True
    set_fw_status(fw, "starting")
    cmd = resolve_start_cmd(cfg, reasoning_level)
    if "{reasoning}" in cfg.get("start_cmd", []) and reasoning_level \
            and reasoning_level not in (cfg.get("reasoning_flags") or {}):
        log(f"⚠ no reasoning_flags for level “{reasoning_level}” — starting "
            f"without reasoning flags", fw=fw, level="err")
    log(f"starting {cfg['name']} (reasoning={reasoning_level or 'default'}): "
        f"{' '.join(cmd)}", fw=fw)
    # Server stdout/stderr goes to logs/<fw>-<ts>.log — a failed start is
    # diagnosable instead of silently discarded.
    try:
        logf = open(os.path.join(LOG_DIR, f"{fw}-{time.strftime('%Y%m%d-%H%M%S')}.log"), "w")
    except OSError:
        logf = subprocess.DEVNULL
    try:
        proc = subprocess.Popen(cmd, stdout=logf,
                                stderr=subprocess.STDOUT, start_new_session=True)
    except FileNotFoundError:
        log(f"{cfg['name']} CLI not found — check start_cmd in config.json", fw=fw, level="err")
        set_fw_status(fw, "down")
        return False
    with LOCK:
        PROCS[fw] = proc
        FW_LOGS[fw] = logf
    deadline = time.time() + 300  # allow up to 5 min for model load
    while time.time() < deadline and proc.poll() is None:
        if not RUN_FLAG.is_set():
            stop_framework(fw)
            return False
        if framework_healthy(fw):
            # Adopt the model id the server actually reports (e.g. OMLX serves
            # cache-style org--name, MTPLX a normalized id) and, for mlx-serve,
            # the served context length so ctx-fill % reflects the real window.
            _adopt_served_model_id(fw)
            adopt_served_meta(fw)
            if fw == "omlx":
                _omlx_unload_all(cfg)
            FW_REASONING[fw] = reasoning_level
            set_fw_status(fw, "up")
            log(f"{cfg['name']} healthy on port {cfg['port']} ({cfg['model']})", fw=fw, level="ok")
            return True
        time.sleep(2)
    log(f"{cfg['name']} failed to become healthy", fw=fw, level="err")
    stop_framework(fw)
    return False


def stop_framework(fw):
    cfg = FRAMEWORKS[fw]
    PROXY_STATE.update({"target": None, "fw": None, "model": None})
    set_fw_status(fw, "down")
    with LOCK:
        proc = PROCS.pop(fw, None)
        logf = FW_LOGS.pop(fw, None)
        FW_REASONING.pop(fw, None)
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        # double check: the port must actually go quiet before the next
        # framework loads into the freed memory
        if port_open(cfg["port"]):
            log(f"⚠ {cfg['name']} port {cfg['port']} still answering after "
                f"stop — memory may not be released", fw=fw, level="err")
        else:
            log(f"{cfg['name']} shut down — port {cfg['port']} free", fw=fw)
    elif port_open(cfg["port"]):
        # Server we didn't spawn (was already running or left over from a
        # previous backend) — stop it anyway: the next framework's model
        # needs the memory, and a resident model silently skews the
        # comparison. SIGTERM the listener, then SIGKILL if it lingers.
        pids = subprocess.run(["lsof", "-ti", f":{cfg['port']}"],
                              capture_output=True, text=True).stdout.split()
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        try:
            subprocess.run(["pkill", "-9", "-f", f"port {cfg['port']}"],
                           capture_output=True, timeout=10)
        except Exception:
            pass
        time.sleep(2)
        if port_open(cfg["port"]):
            log(f"⚠ {cfg['name']} port {cfg['port']} STILL answering after "
                f"pre-existing-server stop", fw=fw, level="err")
        else:
            log(f"{cfg['name']} pre-existing server on port {cfg['port']} "
                f"stopped — memory released", fw=fw)
    if logf:
        try:
            logf.close()
        except OSError:
            pass


# ----------------------------------------------------------------------------
# Harness execution
# ----------------------------------------------------------------------------
def _chat_request(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=7200)


def call_chat(fw, prompt, settings, messages=None, on_progress=None):
    """Streaming OpenAI-compatible chat completion. Time-to-first-token gives
    prompt-processing (prefill) speed; the remaining wall time gives token
    generation (decode) speed. Returns (text, metrics_dict). `messages`
    overrides the default single-user-message prompt (used by raw+
    continuations). `on_progress(elapsed, approx_tok, tail, think_chars)` is
    called ~every 10s while streaming so the UI tail can show generation
    progress — counters only, the way an agent TUI shows activity, not a
    firehose of the model's bytes."""
    cfg = FRAMEWORKS[fw]
    url = f"http://127.0.0.1:{cfg['port']}/v1/chat/completions"
    body = {
        "model": cfg["model"],
        "messages": messages or [{"role": "user", "content": prompt}],
        "temperature": settings.get("temperature", 0.7),
        "top_p": settings.get("top_p", 0.95),
        "max_tokens": settings.get("max_tokens", 65536),
        "stream": True,
    }
    try:
        resp = _chat_request(url, dict(body, stream_options={"include_usage": True}))
    except urllib.error.HTTPError as e:
        if e.code == 400:  # server doesn't know stream_options — retry plain
            resp = _chat_request(url, body)
        else:
            raise
    if not RUN_FLAG.is_set():
        resp.close()
        raise RunStopped()

    # Register the underlying socket so Stop can shut it and wake this read.
    sock = None
    try:
        sock = resp.fp.raw._sock
        with CUR_LOCK:
            CURRENT["sock"] = sock
    except (AttributeError, OSError):
        pass

    t0 = time.time()
    ttft = None
    parts = []
    usage = None
    finish = None
    think_chars = 0
    last_pb = 0.0
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    parts.append(delta["content"])
                elif delta.get("reasoning_content"):
                    # Thinking models stream reasoning first — prefill is done
                    # when the first token of ANY kind arrives.
                    if ttft is None:
                        ttft = time.time() - t0
                    think_chars += len(delta["reasoning_content"])
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
            if on_progress is not None and time.time() - last_pb >= 10.0:
                last_pb = time.time()
                streamed = "".join(parts)
                on_progress(time.time() - t0, len(streamed) // 4,
                            streamed[-70:], think_chars)
    except (OSError, urllib.error.URLError) as e:
        with CUR_LOCK:
            CURRENT["sock"] = None
        if not RUN_FLAG.is_set():
            raise RunStopped() from e
        raise
    finally:
        with CUR_LOCK:
            CURRENT["sock"] = None
        resp.close()
    # Stop manifests as a clean EOF on some platforms (shutdown → EOF), so a
    # ended stream with a cleared flag is a stop, not a short response.
    if not RUN_FLAG.is_set():
        raise RunStopped()

    total = time.time() - t0
    text = "".join(parts)
    ptok = (usage or {}).get("prompt_tokens") or len(prompt.split())
    ctok = (usage or {}).get("completion_tokens") or len(text.split())
    gen_time = max(total - (ttft or 0), 1e-6)
    metrics = {
        "pp": round(ptok / ttft, 1) if ttft else None,       # prefill tok/s
        "tgs": round(ctok / gen_time, 1) if ctok else None,  # decode tok/s
        "tps": round(ctok / total, 1) if ctok and total else None,  # overall
        "ttft": round(ttft, 3) if ttft else None,
        "prompt_tokens": ptok,
        "wall": round(total, 3),  # total call wall time (raw+ decode math)
    }
    return text, ctok, total, finish == "length", metrics


def plausible_html(s):
    """Sanity check that a candidate string is really a self-contained HTML
    artifact (not prose, not a random snippet, not a truncated sentence)."""
    s = (s or "").strip()
    return (len(s) > 120 and "<" in s and
            re.search(r"<(html|body|head|div|canvas|script|main|section)\b", s, re.I))


def _trim_to_html_close(s):
    """Cut anything after the last </html> so outputs are html and only html."""
    i = s.rfind("</html>")
    return s[: i + 7] if i != -1 else s


def extract_html(text):
    """Pull the first plausible HTML artifact out of a model response.
    Strategies, in order: ```html-labeled fences, bare fences containing a
    full document, raw <!DOCTYPE …> blobs, and bare <html>…</html> blocks.
    Every candidate must pass plausible_html. Fence matches are GREEDY (to
    the last closing fence) and closed documents are preferred: a doc whose
    own code contains fence markers (markdown compilers, code-sample spec
    tests) would otherwise be amputated at the first internal ```. Returns
    None if nothing plausible is found."""
    candidates = []
    for m in re.finditer(r"```(?:html|htm|x-html|html4strict)\s*\n(.*)```", text, re.S | re.I):
        candidates.append(m.group(1))
    for m in re.finditer(r"```\w*\s*\n(.*)```", text, re.S):
        candidates.append(m.group(1))
    m = re.search(r"(<!DOCTYPE html.*)", text, re.S | re.I)
    if m:
        candidates.append(m.group(1))
    m = re.search(r"(<html[\s>].*</html>)", text, re.S | re.I)
    if m:
        candidates.append(m.group(1))

    parsed = []
    for cand in candidates:
        cand = cand.strip()
        # Labeled/bare fences may carry prose before the document itself.
        m = re.search(r"(<!DOCTYPE html.*)", cand, re.S | re.I)
        if m:
            cand = m.group(1)
        else:
            m = re.search(r"(<html[\s>].*)", cand, re.S | re.I)
            if m:
                cand = m.group(1)
        cand = _trim_to_html_close(cand)
        if plausible_html(cand):
            parsed.append(cand)
    if not parsed:
        return None
    # A candidate that actually closes is the real artifact; an open one is
    # a truncated document — only fall back to it when nothing is closed.
    for cand in parsed:
        if "</html>" in cand.lower():
            return cand
    return parsed[0]


CLI_TIMEOUT = 7200  # long-horizon tasks need up to 120 min per cell

PI_CONFIG_DIR = os.path.join(ROOT, "harness-configs", "pi")
OC_CONFIG_ROOT = os.path.join(ROOT, "harness-configs", "opencode")


def inline_local_scripts(html, base_dir):
    """Replace <script src="…"> tags with the referenced file's contents when
    that file exists under base_dir (agents routinely emit index.html +
    app.js; the collected single artifact must carry the JS or QA and the
    Output button see a shell with no behaviour). Remote, protocol-relative
    and absolute-path srcs are left untouched; paths may not escape base_dir."""
    def _sub(m):
        src = (m.group(2) or "").split("?")[0].split("#")[0]
        if not src or re.match(r"^[a-z][a-z0-9+.-]*:|^//|^/", src, re.I):
            return m.group(0)
        p = os.path.realpath(os.path.join(base_dir, src))
        if not p.startswith(os.path.realpath(base_dir) + os.sep) \
                or not os.path.isfile(p):
            return m.group(0)
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                code = fh.read()
        except OSError:
            return m.group(0)
        log(f"inlined local script {src} ({len(code)}B) into artifact")
        return f"<script{m.group(1)}{m.group(3)}>\n{code}\n</script>"
    return re.sub(r"<script([^>]*?)\ssrc=[\"']([^\"']+)[\"']([^>]*)>(.*?)</script>",
                  _sub, html, flags=re.S | re.I)


def newest_html(workdir, since):
    """Newest .html file an agent harness created during its run (agents
    like opencode/pi write artifacts to disk instead of answering in chat)."""
    best = None
    for dirpath, dirnames, filenames in os.walk(workdir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if not fn.endswith((".html", ".htm")):
                continue
            p = os.path.join(dirpath, fn)
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if mt >= since and (best is None or mt > best[0]):
                best = (mt, p)
    # Prefer the newest plausible artifact; ignore stray/stub html files.
    if best:
        try:
            with open(best[1], encoding="utf-8", errors="replace") as f:
                if plausible_html(f.read(20000)):
                    return best[1]
        except OSError:
            return None
    return None


NODE_BIN = shutil.which("node")


def js_syntax_ok(js_source):
    """Run `node --check` on the page's inline JS. Returns (ok, error)."""
    if not NODE_BIN or not js_source.strip():
        return None, None  # node unavailable / no JS to check
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(js_source)
        path = f.name
    try:
        p = subprocess.run([NODE_BIN, "--check", path], capture_output=True, text=True, timeout=30)
        if p.returncode == 0:
            return True, None
        err = (p.stderr or "").strip().splitlines()
        return False, err[0][:200] if err else "syntax error"
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    finally:
        os.unlink(path)


_JS_TOKEN = re.compile(
    r"/\*.*?\*/"                    # block comment
    r"|//[^\n]*"                    # line comment
    r"|(?:'(?:\\.|[^'\\\n])*')"     # single-quoted string
    r'|(?:"(?:\\.|[^"\\\n])*")'     # double-quoted string
    r"|`(?:\\.|[^`\\])*`",          # template literal
    re.S)


def _js_code_skeleton(js):
    """JS with comments and string/template literals blanked, so brace/paren
    counting reflects CODE structure rather than text inside strings — a
    markdown compiler's spec tests (or any app with braces in its strings)
    would otherwise score as unbalanced. Leftmost-alternative matching makes
    ordering self-correcting: a quote inside a comment is comment text, a //
    inside a string is string text. Best-effort; syntax truth still comes
    from node --check."""
    return _JS_TOKEN.sub(" ", js)


def qa_artifact(path, requirements=None, probe_ctx=None):
    """Quality gate for a generated artifact. Returns three INDEPENDENT
    grades so a regression in one can never mask another:

      functionality — the release gate. A hard, results-focused verdict:
          did the artifact RUN, and did the required behaviors actually work?
          Structure: 60% runtime probe (headless Chrome interacting with the
          real artifact — the ONLY source that measures function rather than
          appearance), 40% declared-requirements (what the task asked for).
          No quality/defect signals are blended into this number.
      quality — code hygiene, reported separately (decomposition, no
          eval/document.write, modern declarations). Never folded into
          functionality.
      usable — release gate: functionality >= 90.

    Any failed runtime-probe check caps functionality below the usable line
    (a page that renders but has a broken core behavior is not usable)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            src = f.read()
    except OSError:
        return None

    req_missing, req_rate = [], None
    if requirements:
        req_missing = [label for label, pat in requirements
                       if not re.search(pat, src, re.I)]
        req_rate = round(100 * (len(requirements) - len(req_missing))
                         / len(requirements), 1)

    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", src, re.S | re.I)
    js = "\n;\n".join(scripts)
    has_inline_handlers = bool(re.search(r"\son[a-z]+\s*=\s*[\"']", src, re.I))

    skeleton = _js_code_skeleton(js)
    syn_ok, syn_err = js_syntax_ok(js)

    notes = []

    # ---- code hygiene (independent grade, reported but never blended) ----
    funcs = len(re.findall(r"\bfunction\b|=>", skeleton))
    qchecks = [
        ("decomposed", funcs >= 3, 40),
        ("no eval/doc.write", not re.search(r"\beval\s*\(|document\.write", js), 30),
        ("modern decls", bool(re.search(r"\b(const|let)\b", js)), 30),
    ]
    # ---- functionality ----
    # Step 1: runtime probe on a disposable copy (authoritative for function).
    probe = None
    if probe_ctx and not probe_ctx.get("static_report"):
        probe = run_qa_probe(path, probe_ctx.get("task"),
                             base_url=probe_ctx.get("base_url"))

    # Step 2: declared requirements (what the task asked for).
    req_ok = req_missing == [] if requirements else None

    # Step 3: hard verdicts that cannot be compensated by other checks.
    # A JS syntax error means the page categorically cannot work.
    qual = None
    if syn_ok is False:
        functionality = 20
        notes.append(f"JS syntax error: {syn_err}")
    elif probe and probe.get("ran") and probe.get("fails"):
        # the page loaded but core behaviors failed — that IS a broken artifact
        functionality = max(10, min(60, 100 - 15 * len(probe["fails"])))
        notes.append(f"runtime probe {probe['passed']}/{probe['total']} failed: "
                     + ", ".join(probe["fails"]))
        if probe.get("jsErr"):
            notes.append(f"probe saw JS error: {probe['jsErr']}")
    else:
        # page runs and required behaviors work; requirements decide the rest
        functionality = 100 if (requirements and req_ok) or not requirements else None
        if requirements:
            # non-requirement gaps don't break function; missing requirements do
            functionality = max(60, 100 - 10 * len(req_missing))
            if req_missing:
                notes.append("missing: " + ", ".join(req_missing))
            else:
                notes.append("all required features present")
        notes.append(f"runtime probe {probe['passed']}/{probe['total']} passed"
                     if probe and probe.get("ran") else None)

    # ---- quality: independent, reported separately ----
    funcs = len(re.findall(r"\bfunction\b|=>", skeleton))
    qchecks = [
        ("decomposed", funcs >= 3, 40),
        ("no eval/doc.write", not re.search(r"\beval\s*\(|document\.write", js), 30),
        ("modern decls", bool(re.search(r"\b(const|let)\b", js)), 30),
    ]
    qual = round(100 * sum(w for _, ok, w in qchecks if ok) / sum(w for _, _, w in qchecks))

    # ---- assemble ----
    if functionality is None:
        functionality = 0
    notes = [n for n in notes if n]

    out = {"qa_func": functionality, "qa_qual": qual, "qa_notes": "; ".join(notes[:6]),
           "usable": functionality >= 90,
           "req_missing": req_missing, "req_rate": req_rate}
    return out


def _arg_summary(args, cap=70):
    """Compact one-line summary of a tool call's arguments: the path when
    there is one (write/edit/read), else the command, else truncated JSON."""
    if not isinstance(args, dict) or not args:
        return ""
    for key in ("path", "filePath", "file"):
        if args.get(key):
            return os.path.basename(str(args[key]))
    if args.get("command"):
        return str(args["command"]).replace("\n", " ")[:cap]
    try:
        return json.dumps(args)[:cap]
    except (TypeError, ValueError):
        return ""


def _result_text(result, cap=110):
    """First text chunk of a pi tool result, for ✓/⚠ tail lines."""
    try:
        for c in result.get("content", []):
            if c.get("type") == "text" and c.get("text"):
                return c["text"].replace("\n", " ")[:cap]
    except (TypeError, AttributeError):
        pass
    return ""


_GOOSE_BANNER = re.compile(
    r"^\s*(_+\( O\)>|\\____\)|L L\s|─────)")          # goose ASCII banner
_GOOSE_RUST_NOISE = re.compile(
    r"^\s*(Text\(|TextContent \{|\}|(\},?))\s*$"       # Rust debug-dump frame
    r"|^\s*(meta|annotations|audience|priority|last_modified):"
    r"|^\s*(Some\(|None,|\),?)\s*$"
    r"|^\s*Annotations \{\s*$")


def _render_goose_line(s):
    """goose --debug narration (plain text) → shared tail vocabulary.
    '▸ write' + following 'path hi.txt' lines become one 🔧 line; the Rust
    tool-result dump collapses to ✓ text; the stderr stats become ⟳ lines."""
    m = re.search(r"●\s*new session · (.*)$", s)    # rides after the goose art
    if m:
        return f"● goose · {m.group(1)}"
    if _GOOSE_BANNER.match(s) or s in ("goose is ready", "Stats:"):
        return None
    if _GOOSE_RUST_NOISE.match(s):
        return None
    m = re.match(r"^\s*▸\s*(\S+)", s)
    if m:
        _GOOSE_STREAM["tool"] = m.group(1)
        return f"🔧 {m.group(1)}…"
    if s.startswith(("path ", "filePath ")):
        tool = _GOOSE_STREAM.get("tool") or "tool"
        return f"🔧 {tool} {os.path.basename(s.split(None, 1)[1])}"
    if s.startswith(("content:", "command:")):
        return None                                   # payload line: skip
    m = re.search(r'text:\s*"((?:[^"\\]|\\.)*)"', s)
    if m:
        try:
            txt = json.loads('"' + m.group(1) + '"')
        except json.JSONDecodeError:
            txt = m.group(1)
        _GOOSE_STREAM["tool"] = None
        return f"  ✓ {txt[:110]}"
    if s.startswith("Time to first token:"):
        return f"⟳ ttft {s.split(':', 1)[1].strip()}"
    if s.startswith("Tokens/sec:"):
        return f"⟳ decode {s.split(':', 1)[1].strip()} tok/s"
    if s.startswith("Output tokens:"):
        return f"⟳ {s.split(':', 1)[1].strip()} out tok"
    return s[:300]


_GOOSE_STREAM = {"tool": None}   # pending "▸ tool" name across debug lines


def harness_stream_format(line, harness=""):
    """Render one line of a harness stream as a compact TUI-style tail line —
    pi/opencode JSON events and goose --debug narration become the same
    vocabulary (🔧 tool calls, ◌ thinking, ● replies, ⟳ steps, ✓/⚠ results):
    the same event stream their TUIs render. Returns None for high-frequency
    deltas and banner noise (they would drown the tail); returns the raw line
    (truncated) when nothing specific matches, so plain-text narration (hart)
    passes through unchanged."""
    s = line.strip()
    if not s:
        return None
    if harness == "goose" and not s.startswith("{"):
        return _render_goose_line(s)
    if not s.startswith("{"):
        return s[:300]
    try:
        ev = json.loads(s)
    except json.JSONDecodeError:
        return s[:300]
    t = ev.get("type", "")

    if t == "tool_execution_start":                      # pi
        return f"🔧 {ev.get('toolName')} {_arg_summary(ev.get('args'))}"
    if t == "tool_execution_end":                        # pi
        name = ev.get("toolName", "tool")
        if ev.get("isError"):
            return f"⚠ {name} failed: {_result_text(ev.get('result') or {})}"
        return f"  ✓ {_result_text(ev.get('result') or {})}"
    if t == "message_update":                            # pi (assistantMessageEvent)
        # Semantic events only — what pi's TUI shows as readable activity.
        # The raw thinking/text deltas are the model's byte stream, not
        # activity; the end-of-phase summaries carry the same information
        # in one clean line.
        ame = ev.get("assistantMessageEvent") or {}
        st = ame.get("type", "")
        if st == "thinking_start":
            return "◌ thinking…"
        if st == "thinking_end":
            n = len(ame.get("content") or "")
            return f"◌ thinking done · {n:,} chars" if n > 200 else None
        if st == "text_start":
            return "● composing reply…"
        if st == "text_end":
            body = (ame.get("content") or "").strip()
            return f"● {body[:140]}" if body else None
        if st == "toolcall_start":
            return f"🔧 {ame.get('toolName', 'tool')}…"
        if st == "toolcall_end":
            tc = ame.get("toolCall") or {}
            return f"🔧 {tc.get('name', 'tool')} {_arg_summary(tc.get('arguments'))}"
        return None                                       # deltas: not activity
    if t in ("step_start", "session", "agent_start", "turn_start",
             "turn_end", "agent_settled"):
        return None
    if t == "agent_end":                                 # pi
        return None                                       # finalization handles it

    part = ev.get("part") or {}                          # opencode events
    if t == "tool_use":
        state = part.get("state") or {}
        status = state.get("status", "")
        if status not in ("completed", "error"):
            return None                                  # pending/running: skip
        name = part.get("tool", "tool")
        if status == "error":
            return f"⚠ {name} failed: {str(state.get('output'))[:110]}"
        return f"🔧 {name} {_arg_summary(state.get('input'))} ✓"
    if t == "text":
        body = (part.get("text") or "").strip()
        return f"● {body[:140]}" if body else None
    if t == "step_finish":
        tok = (part.get("tokens") or {})
        return (f"⟳ step {part.get('reason')} · "
                f"in {tok.get('input', '?')} / out {tok.get('output', '?')} tok")
    if t in ("step_start",):
        return None
    return None


def finalize_harness_stream(harness, out):
    """Extract (text, exact_output_tokens, extra_metrics) from a pi/opencode
    JSON event stream or goose's --stats block. Text = the assistant's chat
    reply (artifacts come from disk); tokens = the model's real output tokens
    summed per assistant message/step — far more honest than the word-count
    estimate. Falls back to (raw output, None, {}) when nothing parses."""
    if harness == "goose":
        met = {}
        m = re.search(r"Output tokens:\s*(\d+)", out)
        tokens = int(m.group(1)) if m else None
        m = re.search(r"Time to first token:\s*([\d.]+)", out)
        if m:
            met["ttft"] = float(m.group(1))
        m = re.search(r"Tokens/sec:\s*([\d.]+)", out)
        if m:
            met["tgs"] = float(m.group(1))
        return out, tokens, met
    if not out or not out.lstrip().startswith("{"):
        return out, None, {}
    text_parts, tokens, steps, last_input = [], 0, 0, None
    parsed_any = False
    for line in out.splitlines():
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            ev = json.loads(s)
        except json.JSONDecodeError:
            continue
        parsed_any = True
        t = ev.get("type", "")
        if harness == "pi":
            if t == "agent_end":
                for m in ev.get("messages", []):
                    if m.get("role") == "assistant":
                        for c in m.get("content", []):
                            if isinstance(c, dict) and c.get("type") == "text" \
                                    and c.get("text", "").strip():
                                text_parts.append(c["text"])
                        u = m.get("usage") or {}
                        tokens += u.get("output") or 0
                        last_input = u.get("input") or last_input
        else:                                            # opencode
            part = ev.get("part") or {}
            if t == "text" and (part.get("text") or "").strip():
                text_parts.append(part["text"])
            if t == "step_finish":
                steps += 1
                tok = part.get("tokens") or {}
                tokens += tok.get("output") or 0
                last_input = tok.get("input") or last_input
    if not parsed_any:
        return out, None, {}
    # A parsed stream with NO reply text is a real outcome (e.g. the final
    # turn hit the token cap mid-thinking and produced no text/action):
    # return "" so the cell records an honest empty-response failure instead
    # of dumping megabytes of JSON events as a fake .txt artifact.
    text = "".join(text_parts).strip()
    metrics = {}
    if steps:
        metrics["calls"] = steps
    if last_input:
        metrics["prompt_tokens"] = last_input
    return text, (tokens or None), metrics


def prep_pi(fw):
    """Point pi at this framework via an ISOLATED config dir (PI_CODING_AGENT_DIR)
    so the user's global ~/.pi/agent/models.json is never touched — safe for
    concurrent pi use and for multi-user deployments. Returns (cmd, env)."""
    cfg = FRAMEWORKS[fw]
    os.makedirs(PI_CONFIG_DIR, exist_ok=True)
    with open(os.path.join(PI_CONFIG_DIR, "models.json"), "w") as f:
        json.dump({"providers": {"bench": {
            "baseUrl": agent_base_url(fw),
            "api": "openai-completions",
            "apiKey": "bench",
            "models": [{
                "id": cfg["model"],
                # pi's input budget = contextWindow - maxTokens. On the
                # Flash-Next set (131072 window) pi's input reached 79,850
                # tokens; a 65536 output reserve made the request 145K >
                # the window -> the engine cut the reply mid-action and the
                # step looped. 32768 output leaves a 98K input budget.
                "maxTokens": 32768,
                # pi compacts against the window it believes - declare the model's REAL
                # served window (adopted at start) or a rounding mismatch
                # (served 131000 vs declared 131072) aborts mid-generation
                "contextWindow": cfg.get("ctx_tokens") or 131072,
            }],
        }}}, f, indent=2)
    cmd = ["pi", "--print", "--provider", "bench",
           "--model", cfg["model"] + PI_THINKING, "--no-session",
           "--mode", "json"]
    env = dict(os.environ, PI_CODING_AGENT_DIR=PI_CONFIG_DIR)
    return cmd, env


def prep_opencode(fw):
    """Generate an isolated opencode config pointing at this framework.
    Returns the XDG_CONFIG_HOME dir to run opencode from (verified working:
    opencode sends the configured model id through unchanged, even with
    slashes in the id)."""
    cfg = FRAMEWORKS[fw]
    cfgdir = os.path.join(OC_CONFIG_ROOT, fw, "opencode")
    os.makedirs(cfgdir, exist_ok=True)
    with open(os.path.join(cfgdir, "opencode.json"), "w") as f:
        json.dump({
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "bench": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": f"Bench {cfg['name']}",
                    "options": {"baseURL": agent_base_url(fw),
                                "apiKey": "bench"},
                    "models": {cfg["model"]: {
                "name": cfg["model"],
                # without a declared limit opencode caps each step at 32k
                # output and abandons the run on a mid-generation length stop
                "limit": {"context": cfg.get("ctx_tokens") or 131072,
                          "output": cfg.get("max_tokens") or 65536},
            }},
                }
            },
        }, f, indent=2)
    return os.path.join(OC_CONFIG_ROOT, fw)


GOOSE_CONFIG_ROOT = os.path.join(ROOT, "harness-configs", "goose")


def prep_goose(fw):
    """Generate an isolated goose config (XDG_CONFIG_HOME isolation, verified:
    goose's openai provider honors OPENAI_BASE_URL from the process env and
    the model from providers.openai.model). Lean extension set — developer
    only — so runs stay comparable with the other agent harnesses.
    Returns (cmd, env)."""
    cfg = FRAMEWORKS[fw]
    cfgdir = os.path.join(GOOSE_CONFIG_ROOT, fw, "goose")
    os.makedirs(cfgdir, exist_ok=True)
    with open(os.path.join(cfgdir, "config.yaml"), "w") as f:
        f.write(f"""\
GOOSE_TELEMETRY_ENABLED: false
active_provider: openai
providers:
  openai:
    enabled: true
    model: {cfg['model']}
    configured: true
OPENAI_BASE_URL: {agent_base_url(fw)}
OPENAI_API_KEY: bench
extensions:
  developer:
    enabled: true
    type: builtin
    name: developer
  computercontroller:
    enabled: false
    type: builtin
    name: computercontroller
  summon:
    enabled: false
    type: platform
    name: summon
  chatrecall:
    enabled: false
    type: platform
    name: chatrecall
""")
    return os.path.join(GOOSE_CONFIG_ROOT, fw)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_leading_fence(chunk):
    m = re.match(r"^\s*```(?:html|htm|xml|js|javascript|json)?\s*\n?", chunk)
    return chunk[m.end():] if m else chunk


def _seam_overlap(buf, chunk, window=600, min_overlap=8):
    """Longest suffix of buf that equals a prefix of chunk (model repeats)."""
    tail, head = buf[-window:], chunk[:window]
    for size in range(min(len(tail), len(head)), min_overlap, -1):
        if tail.endswith(head[:size]):
            return size
    return 0


CONTINUATION_PREAMBLE = (
    "\n\n---\nBEGINNING OF PREVIOUS OUTPUT (cut off mid-generation):\n"
    "{tail}\n---END OF PREVIOUS OUTPUT (cut mid-content).\n\n"
    "CONTINUATION: continue the file EXACTLY at the cut point — begin with "
    "the very next character. Do NOT repeat any prior content, do NOT "
    "re-open a code fence, do NOT summarize or re-introduce. Output only the "
    "remaining file content through to the very end."
    "\n```html\n")


def rawplus_generate(fw, prompt, settings, on_progress=None):
    """raw+ : one-shot generation with automatic continuation. The model's
    output already streams; the per-call max_tokens ceiling is a GENERATION
    limit, so when finish_reason=length, raw+ re-issues the request with the
    accumulated tail and continues the SAME artifact — streaming-buffer
    generation across calls. Per-chunk cap unchanged (uniform settings);
    total budget = RAWPLUS_MAX_ROUNDS x cap. Returns (text, tokens, wall,
    truncated, metrics, rounds)."""
    t0 = time.time()
    buffer = ""
    total_ctok = 0
    total_ptok = 0
    decode_s = 0.0
    first_metrics = {}
    rounds = 0
    truncated = True
    for rnd in range(RAWPLUS_MAX_ROUNDS):
        rounds = rnd + 1
        if rnd == 0:
            msgs = None
        else:
            tail = buffer[-3000:]
            msgs = [{"role": "user",
                     "content": prompt + CONTINUATION_PREAMBLE.format(tail=tail)}]
        text, ntok, gen, trunc, met = call_chat(fw, prompt, settings,
                                                messages=msgs,
                                                on_progress=on_progress)
        chunk = text
        cut = 0
        if rnd > 0:
            chunk = _strip_leading_fence(chunk)
            cut = _seam_overlap(buffer, chunk)
            if cut:
                chunk = chunk[cut:]
        buffer += chunk
        total_ctok += ntok
        total_ptok += met.get("prompt_tokens") or 0
        decode_s += max((met.get("wall") or 0) - (met.get("ttft") or 0), 0.01)
        if rnd == 0:
            first_metrics = met
        log(f"raw+ round {rounds}: +{ntok} tok"
            + (f", seam −{cut} chars" if cut else "")
            + (", finish=length → continuing" if trunc else ", finish=stop ✓"),
            fw=FRAMEWORKS[fw]["name"], harness="raw+")
        if not trunc:
            truncated = False
            break
    wall = time.time() - t0
    metrics = {"pp": first_metrics.get("pp"), "ttft": first_metrics.get("ttft"),
               "tgs": round(total_ctok / decode_s, 1) if decode_s > 0 else None,
               "prompt_tokens": total_ptok, "continuations": rounds - 1}
    return buffer, total_ctok, wall, truncated, metrics, rounds


def run_cli_abortable(cmd, env, cwd, timeout, line_cb=None, fw=None):
    """Run a harness CLI so that Stop works mid-flight AND its output streams
    live: each stdout line (stderr merged) is forwarded to line_cb as it is
    produced — the harness narration lands in the activity feed in real time
    instead of being swallowed by a captured pipe. The process is polled in
    short slices; on stop (or timeout or stall) its whole process group is
    killed. fw enables the stall watchdog for engines that expose a
    per-token completion counter (omlx, mlxserve): ~5 min of output silence
    plus a frozen counter means the model stream hung (observed on hybrid
    models whose abort path doesn't close the stream) — the harness is
    stopped and any artifact it wrote is kept. Returns
    (out, err, returncode, stopped, timed_out)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env, cwd=cwd, start_new_session=True,
                         bufsize=1)
    with CUR_LOCK:
        CURRENT["proc"] = p

    import queue as _queue
    lines_q = _queue.Queue()

    def _reader():
        try:
            for line in p.stdout:
                lines_q.put(line)
        except (OSError, ValueError):
            pass
        finally:
            lines_q.put(None)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    deadline = time.time() + timeout
    stopped = timed_out = False
    out_lines = []
    last_line_ts = time.time()
    last_probe = 0.0
    probe_base = None
    while True:
        # stop/timeout checked on EVERY iteration — steady CLI output can no
        # longer starve them (they previously ran only on idle ticks)
        if not RUN_FLAG.is_set():
            stopped = True
        elif time.time() >= deadline:
            timed_out = True
        # stall watchdog: ~10 min of output silence plus a frozen per-token
        # completion counter = the model stream hung; stop the harness and
        # keep whatever it delivered. 10 min (not 5) so a harness running a
        # legitimately silent local command is not killed mid-work.
        if (not stopped and not timed_out and fw in ("omlx", "mlxserve")
                and time.time() - last_line_ts > 600
                and time.time() - last_probe > 60):
            now = time.time()
            snap = server_snapshot(fw) if fw in FRAMEWORKS else None
            cur = (snap or {}).get("completion") if snap else None
            if cur is not None:
                if probe_base is not None and cur == probe_base:
                    timed_out = True
                    log(f"{cmd[0]} stream stalled — no harness output for "
                        f"{int(now - last_line_ts)}s and no model generation — "
                        f"stopping it (any artifact written is kept)",
                        level="err")
                probe_base = cur
            last_probe = now
        if stopped or timed_out:
            break
        try:
            line = lines_q.get(timeout=2)
        except _queue.Empty:
            line = "__poll__"
        if line is None:
            p.wait()
            break
        if line != "__poll__":
            out_lines.append(line)
            last_line_ts = time.time()
            if line_cb:
                try:
                    clean = ANSI_RE.sub("", line).rstrip()
                    if clean:
                        line_cb(clean)
                except Exception:
                    pass
            continue
        if p.poll() is not None:
            # drain anything the reader has left, then finish
            reader.join(timeout=2)
            while True:
                try:
                    line = lines_q.get(timeout=0.5)
                except _queue.Empty:
                    line = None
                if line is None:
                    break
                out_lines.append(line)
                if line_cb:
                    clean = ANSI_RE.sub("", line).rstrip()
                    if clean:
                        line_cb(clean)
            p.wait()
            break
    # stop / timeout / stall: kill the whole process group
    if stopped or timed_out:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            p.kill()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    with CUR_LOCK:
        CURRENT["proc"] = None
    if not timed_out and not RUN_FLAG.is_set():
        stopped = True
    return "".join(out_lines), "", p.returncode, stopped, timed_out


def run_harness(fw, harness, task_name, prompt, settings, task=None):
    """Run one harness test against the framework. pi and opencode are routed
    via generated per-framework configs; their stderr is captured and shown
    in the activity log so failures are diagnosable."""
    cfg = FRAMEWORKS[fw]
    label = HARNESS_LABELS.get(harness, harness)
    truncated = False
    artifact = None
    metrics = {}
    exact_tokens = None   # real model output tokens from pi/opencode streams

    if harness in ("raw", "rawplus"):
        # Raw is a single one-shot request with no agent to budget context —
        # give it its own (higher) token cap so thinking models can finish.
        # raw+ adds automatic continuation when the generation cap is hit.
        s = dict(settings)
        if settings.get("max_tokens_raw"):
            s["max_tokens"] = settings["max_tokens_raw"]
        # tool-aligned models sometimes respond to fixture tasks by CALLING a
        # tool they do not have (observed: a Qwen checkpoint emitting an XML
        # Bash tool-call to "cat" the file that is already inlined). Raw has
        # no tool executor, so say so explicitly.
        if "<source>" in prompt:
            prompt += ("\n\nNote: you have NO tools, shell, or file access in "
                       "this mode. Everything you need — including the complete "
                       "source inside the <source> block above — is already in "
                       "this conversation. Do not attempt tool calls or file "
                       "operations; produce your answer directly.")
        log(f"single streaming request · max_tokens {s.get('max_tokens')}",
            fw=cfg["name"], harness=label)

        def _raw_progress(el, approx_tok, tail, think_chars):
            # semantic progress into the activity feed — phase + counters,
            # never the model's byte stream (the artifact is the Output
            # button's job)
            if think_chars and not tail:
                log(f"◌ thinking… {el:.0f}s · {think_chars // 1000}k chars",
                    fw=cfg["name"], harness=label)
            else:
                log(f"⏳ writing… {el:.0f}s · ~{approx_tok:,} tok",
                    fw=cfg["name"], harness=label)

        if harness == "rawplus":
            text, ntok, gen, truncated, metrics, _rnds = \
                rawplus_generate(fw, prompt, s, on_progress=_raw_progress)
        else:
            text, ntok, gen, truncated, metrics = call_chat(
                fw, prompt, s, on_progress=_raw_progress)
        log(f"stream complete · {ntok} tokens · truncated={truncated}",
            fw=cfg["name"], harness=label)
    else:
        if harness == "pi":
            cmd, env = prep_pi(fw)
            cmd = cmd + [prompt]
        elif harness == "opencode":
            # --dir: opencode resolves its project dir itself and ignores the
            # subprocess cwd, so file artifacts must be routed explicitly.
            workdir = os.path.join(WORK_DIR, f"{fw}-{harness}-{uuid.uuid4().hex[:8]}")
            os.makedirs(workdir, exist_ok=True)
            cmd = ["opencode", "run", "--dir", workdir,
                   "--format", "json",
                   "--model", f"bench/{cfg['model']}", prompt]
            env = dict(os.environ, XDG_CONFIG_HOME=prep_opencode(fw))
        elif harness == "goose":
            xdg = prep_goose(fw)
            # --debug: TUI-style tool narration (forwarded to the activity feed);
            # --stats: real token counts + TTFT/decode rate on stderr
            cmd = ["goose", "run", "-t", prompt, "--no-session",
                   "--debug", "--stats"]
            env = dict(os.environ, XDG_CONFIG_HOME=xdg, OPENAI_API_KEY="bench",
                       OPENAI_BASE_URL=agent_base_url(fw))
        elif harness == "hart":
            if not os.path.isfile(HART_PATH):
                return {"status": "error", "latency": 0, "tokens": 0, "tps": 0.0,
                        "error": f"hart harness not found at {HART_PATH}"}
            workdir = os.path.join(WORK_DIR, f"{fw}-{harness}-{uuid.uuid4().hex[:8]}")
            os.makedirs(workdir, exist_ok=True)
            # Cell bounds: standard tasks 2×1500s; long-horizon tasks
            # 3×2400s (~2h ceiling, matching CLI_TIMEOUT). hart's internal
            # defaults (50 epochs) are for standalone use only.
            if settings.get("long_task"):
                t_budget, epochs = "2400", "3"
            else:
                t_budget, epochs = "1500", "2"
            cmd = [sys.executable, HART_PATH,
                   "--base-url", agent_base_url(fw),
                   "--model", cfg["model"],
                   "--workdir", workdir,
                   # uniform model settings: 64k per-call cap for every task
                   "--per-call-tokens", "65536",
                   "--time-budget", t_budget, "--epochs", epochs,
                   prompt]
            env = dict(os.environ)
        else:
            return {"status": "error", "latency": 0, "tokens": 0, "tps": 0.0,
                    "error": f"unknown harness {harness}"}

        if not shutil.which(cmd[0]):
            log(f"{label} CLI not found — using raw API fallback", fw=cfg["name"], harness=label)
            text, ntok, gen, truncated, metrics = call_chat(fw, prompt, settings)
        else:
            log(f"dispatching via CLI: {cmd[0]}", fw=cfg["name"], harness=label)
            if harness not in ("opencode", "hart"):  # these set their own workdir
                workdir = os.path.join(WORK_DIR, f"{fw}-{harness}-{uuid.uuid4().hex[:8]}")
                os.makedirs(workdir, exist_ok=True)
            if task and task.get("id") in TASK_FIXTURES and workdir:
                # per-cell copy: each harness works on its own pristine fixture
                src_name, dst_name = TASK_FIXTURES[task["id"]]
                shutil.copy2(os.path.join(ROOT, "fixtures", src_name),
                             os.path.join(workdir, dst_name))
            proxy_reset()
            t0 = time.time()

            def _line_handler(line):
                # JSON event streams (pi/opencode) and goose's debug narration
                # render as compact semantic activity lines; other plain text
                # (hart) passes through. Delta-class events return None and
                # are skipped so the feed stays readable.
                rendered = harness_stream_format(line, harness)
                if rendered is None:
                    return
                if harness == "hart":
                    if (line.startswith(("[", "AGEDIN", "resum", "[epoch"))
                            or line.startswith(("⚠", "⏳", "🔧"))):
                        log(rendered[:160], fw=cfg["name"], harness=label)
                else:
                    log(rendered[:160], fw=cfg["name"], harness=label)

            out, errtext, rc, stopped, timed_out = run_cli_abortable(
                cmd, env, workdir, CLI_TIMEOUT, line_cb=_line_handler, fw=fw)
            gen = time.time() - t0
            if stopped:
                log(f"{label} stopped by user after {gen:.0f}s",
                    fw=cfg["name"], harness=label, level="err")
                raise RunStopped()
            # Structured streams carry the assistant's chat reply and the
            # model's REAL token counts — replace the raw stdout blob and the
            # word-count estimate with them (artifacts still come from disk).
            if harness in ("pi", "opencode", "goose"):
                out, exact_tokens, ev_metrics = finalize_harness_stream(harness, out)
                metrics.update(ev_metrics)
            text = out
            # Model-side timing measured by the proxy across all the
            # agent's calls (the CLIs don't expose timing internals).
            st = proxy_read()
            if st["requests"]:
                metrics = {"prompt_tokens": st["prompt_tokens"],
                           "ttft": round(st["ttft_sum"] / st["requests"], 3)}
                if st["ttft_sum"] > 0:
                    metrics["pp"] = round(st["prompt_tokens"] / st["ttft_sum"], 1)
                if st["decode_sum"] > 0 and st["completion_tokens"] > 0:
                    metrics["tgs"] = round(st["completion_tokens"] / st["decode_sum"], 1)
                log(f"model traffic: {st['requests']} call(s), "
                    f"{st['prompt_tokens']}→{st['completion_tokens']} tok"
                    + (f", {st['length_hits']} hit the harness token cap"
                       if st.get("length_hits") else ""),
                    fw=cfg["name"], harness=label)
            if st.get("length_hits"):
                truncated = True
                log(f"⚠ {st['length_hits']} model call(s) ended finish=length — the HARNESS's "
                    f"per-call token cap (not the framework) cut the answer; thinking models "
                    f"spend the budget on reasoning first",
                    fw=cfg["name"], harness=label, level="err")
            # Agent harnesses deliver results as files. Tasks with a declared
            # NON-HTML artifact (SKILL.md, mcp_server.py) are collected by
            # exact name — newest_html only knows .html, and grading the
            # agent's chat summary instead of its written file produced
            # false 17-50 scores on the builder tasks.
            artifact = None
            declared = (task or {}).get("artifact")
            if declared and not declared.endswith(".html") and workdir:
                cand = os.path.join(workdir, os.path.basename(declared))
                if os.path.isfile(cand):
                    artifact = cand
                    log(f"agent wrote artifact → {os.path.basename(artifact)}",
                        fw=cfg["name"], harness=label)
            if not artifact:
                artifact = newest_html(workdir, t0)
            if artifact:
                with open(artifact, encoding="utf-8", errors="replace") as f:
                    ntok = len(text.split()) + len(f.read().split())
                log(f"agent wrote artifact → {os.path.basename(artifact)}",
                    fw=cfg["name"], harness=label)
            else:
                ntok = len(text.split())
            if exact_tokens:
                # The stream reported the model's real output token count —
                # beats the word-count heuristic (artifacts included only as
                # a fallback when the stream didn't parse).
                ntok = exact_tokens
            if timed_out:
                extra = (f"; model traffic so far: {st['requests']} call(s), "
                         f"{st['completion_tokens']} tok") if st["requests"] else ""
                if artifact:
                    # The killed process may still have delivered its artifact.
                    with open(artifact, encoding="utf-8", errors="replace") as f:
                        ntok = len(f.read().split())
                    truncated = True
                    log(f"{label} timed out after {CLI_TIMEOUT}s but the artifact "
                        f"was delivered — keeping it{extra}",
                        fw=cfg["name"], harness=label, level="err")
                else:
                    log(f"{label} timed out after {CLI_TIMEOUT}s{extra}",
                        fw=cfg["name"], harness=label, level="err")
                    return {"status": "error", "latency": round(time.time() - t0, 2),
                            "tokens": 0, "tps": 0.0,
                            "error": f"timeout after {CLI_TIMEOUT}s{extra}"}
            elif rc != 0:
                err = (errtext or out or "no output").strip()[-300:]
                if artifact:
                    # The agent finished the task (file delivered) but a
                    # later call failed — keep the artifact, don't fail.
                    log(f"{label} exited {rc} after delivering the "
                        f"artifact — keeping it ({err[:120]})",
                        fw=cfg["name"], harness=label, level="err")
                else:
                    log(f"{label} exited {rc}: {err}",
                        fw=cfg["name"], harness=label, level="err")
                    return {"status": "error", "latency": round(gen, 2), "tokens": 0,
                            "tps": 0.0, "error": err}

    # An agent may deliver the artifact as a file and print nothing to stdout
    # — that is a success, not an empty response. Only fail when there is
    # neither text nor a collected artifact.
    if not text.strip() and not artifact:
        return {"status": "error", "latency": gen, "tokens": 0, "tps": 0.0,
                "error": "empty response"}

    # hart reports accurate totals on its last stdout line — prefer them
    # over word-count estimates (applies after artifact/ntok fallback logic).
    if harness == "hart":
        m = re.search(r"HART_RESULT (\{.*\})", text, re.S)
        if m:
            try:
                ar = json.loads(m.group(1))
                if ar.get("tokens"):
                    ntok = ar["tokens"]
                metrics.update({"calls": ar.get("calls"),
                                "hart_status": ar.get("status")})
            except json.JSONDecodeError:
                pass

    result = {"status": "done", "latency": round(gen, 2), "tokens": ntok,
              "tps": round(ntok / gen, 1) if gen > 0 else 0.0,
              "truncated": truncated, **metrics}
    # iterations = how many model requests the harness needed. Sources per
    # harness: calls (pi/hart step counts), 1+continuations (raw+ rounds),
    # server_requests (opencode/goose via the framework's own counter),
    # 1 for raw. None when nothing measured (e.g. mlxlm cells).
    iters = (metrics.get("calls")
             or (1 + metrics["continuations"] if metrics.get("continuations")
                 else None)
             or metrics.get("server_requests")
             or (1 if harness in ("raw", "rawplus") else None))
    if iters:
        result["iterations"] = iters

    # Every successful run gets an Output button: the agent's written HTML
    # file if one was collected, else extracted HTML, else the response text.
    oid = uuid.uuid4().hex[:12]
    declared_md = (task or {}).get("artifact", "").endswith(".md")
    if artifact:
        with open(artifact, encoding="utf-8", errors="replace") as f:
            content = f.read()
        ext = os.path.splitext(artifact)[1] or ".html"
        if ext == ".html":
            content = inline_local_scripts(content, os.path.dirname(artifact))
        # keep the artifact's real extension: a review.md served as .html
        # renders as broken markup in the browser (observed on codereview)
        fname = f"{oid}{ext}"
    else:
        html = extract_html(text)
        if html:
            content, fname = html, f"{oid}.html"
        elif declared_md:
            content, fname = text, f"{oid}.md"
        else:
            content, fname = text, f"{oid}.txt"
    ctype = "text/html" if fname.endswith(".html") else "text/plain"
    with open(os.path.join(OUTPUT_DIR, fname), "w") as f:
        f.write(content)
    result["output_url"] = f"/output/{fname}"

    reqs = REQUIREMENTS.get(task.get("id") if task else None)
    if fname.endswith(".html") or reqs:
        # runtime probe context: the task's probe config + the live endpoint
        # (the framework server is still up at QA time, so the agentconsole
        # probe can exercise a real connect + chat round-trip)
        probe_ctx = None
        if reqs:
            probe_ctx = {"task": task.get("id"), "base_url": agent_base_url(fw)}
        qa = qa_artifact(os.path.join(OUTPUT_DIR, fname),
                         requirements=reqs, probe_ctx=probe_ctx)
        if qa:
            result.update(qa)
            log(f"QA: functionality {qa['qa_func']}%, quality {qa['qa_qual']}"
                + ("" if qa["usable"] else " — ⚠ BELOW 90% USABILITY THRESHOLD"),
                fw=FRAMEWORKS[fw]["name"], harness=HARNESS_LABELS.get(harness, harness),
                level="ok" if qa["usable"] else "err")

    result["_text"] = text[:5000]  # kept internally, not sent wholesale to UI
    return result


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def _validate_run(req):
    """Validate a /api/run body. Raises ValueError with a user-readable
    message on the first problem — keeps bad input from 500-ing the server."""
    if not isinstance(req, dict):
        raise ValueError("body must be a JSON object")
    harnesses = req.get("harnesses")
    if not isinstance(harnesses, list) or not harnesses:
        raise ValueError("harnesses: non-empty list required")
    bad = [h for h in harnesses if h not in HARNESS_LABELS]
    if bad:
        raise ValueError(f"unknown harness(es): {', '.join(map(str, bad))}")
    if not isinstance(req.get("task_id"), str) or not req.get("task_id"):
        raise ValueError("task_id: non-empty string required")
    if not isinstance(req.get("prompt"), str) or not req["prompt"].strip():
        raise ValueError("prompt: non-empty string required")
    settings = req.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError("settings: object required")
    for k in ("temperature", "top_p", "max_tokens", "max_tokens_raw", "repeats"):
        if k in settings and settings[k] is not None \
                and not isinstance(settings[k], (int, float)):
            raise ValueError(f"settings.{k}: number required")
    if "frameworks" in req and req["frameworks"] is not None \
            and not isinstance(req["frameworks"], list):
        raise ValueError("frameworks: list required")


def run_benchmark(req):
    _validate_run(req)
    harnesses = req["harnesses"]
    # optional framework subset (e.g. compare just two); unknown ids ignored,
    # empty/missing → all
    frameworks = [fw for fw in (req.get("frameworks") or list(FRAMEWORKS))
                  if fw in FRAMEWORKS] or list(FRAMEWORKS)
    task_id = req["task_id"]
    prompt = req["prompt"]
    settings = req.get("settings", {})
    try:
        repeats = max(1, min(5, int(settings.get("repeats", 1))))
    except (TypeError, ValueError):
        repeats = 1
    task_name = next((t["name"] for t in TASKS if t["id"] == task_id),
                     "Custom prompt" if task_id == "__custom__" else task_id)
    artifact = next((t.get("artifact") for t in TASKS if t["id"] == task_id), None)
    long_task = any(t.get("long") for t in TASKS if t["id"] == task_id)
    if long_task:
        settings = dict(settings, long_task=True)
    # Reasoning level is baked into the framework server at start (CLI flags
    # or omlx's settings file), so it is chosen per task TYPE, not per cell:
    # standard tasks run at the 'short' level, LONG tasks at the 'long' level.
    reasoning_level = reasoning_level_for(task_id)
    log(f"reasoning level for this run: {reasoning_level} "
        f"({'LONG' if long_task else 'standard'} task)", level="ok")

    with LOCK:
        STATE["running"] = True
        STATE["results"] = []
        STATE["run_started"] = time.time()
    RUN_FLAG.set()
    shutil.rmtree(WORK_DIR, ignore_errors=True)  # scratch dirs from prior runs

    # data-analysis tasks get their input fixture snapshotted from the
    # PREVIOUS run's logs before anything else touches the work dir
    if task_id == "logreport":
        _build_logreport_fixture()
        prompt = prompt.replace("@LOGREPORT_INLINE@", _logreport_inline())
    elif task_id == "bugfix":
        src_path = os.path.join(ROOT, "fixtures", "expenses-broken.html")
        try:
            with open(src_path, encoding="utf-8") as fh:
                prompt = prompt.replace("@BUGFIX_SOURCE@", fh.read())
        except OSError:
            prompt = prompt.replace("@BUGFIX_SOURCE@", "(fixture file missing)")
    elif task_id == "bugfix" or task_id == "codereview":
        # inline the fixture source; if the user edited the prompt and removed
        # the placeholder, append it anyway so the model always sees the code
        fname = "expenses-broken.html" if task_id == "bugfix" else "review-sample.py"
        ph = "@BUGFIX_SOURCE@" if task_id == "bugfix" else "@CODE_REVIEW_SOURCE@"
        try:
            with open(os.path.join(ROOT, "fixtures", fname), encoding="utf-8") as fh:
                body = fh.read()
        except OSError:
            body = "(fixture file missing)"
        if ph in prompt:
            prompt = prompt.replace(ph, body)
        elif "<source>" not in prompt:
            prompt += "\n\n<source>\n" + body + "\n</source>"

    # omlx-style frameworks keep the reasoning level in a per-model settings
    # file (applied at request time) — rewrite it for the run, restore after.
    reasoning_restores = []
    for fw in frameworks:
        if FRAMEWORKS[fw].get("reasoning_file"):
            r = omlx_reasoning_apply(fw, effective_reasoning(fw, reasoning_level))
            if r:
                reasoning_restores.append(r)

    try:
        for fw in frameworks:
            if not RUN_FLAG.is_set():
                log("benchmark stopped by user", level="err")
                break
            cfg = FRAMEWORKS[fw]
            with LOCK:
                STATE["current_step"] = f"starting {cfg['name']}…"
            if not FRAMEWORKS[fw].get("model_available", True):
                log(f"{cfg['name']} is not available for the active model set "
                    f"({CONFIG.get('model_set')}) — skipping", level="err")
                continue
            if not start_framework(fw, effective_reasoning(fw, reasoning_level)):
                continue
            PROXY_STATE.update({"target": f"http://127.0.0.1:{cfg['port']}",
                                "fw": cfg["name"], "model": cfg["model"]})
            log_device_info(fw)

            # Warmup so first-request overhead (graph compile, cache fill)
            # doesn't skew the first measured row.
            log("warmup request (not measured)…", fw=cfg["name"])
            try:
                call_chat(fw, "Say OK.", {"temperature": 0.1, "max_tokens": 16})
                # huge models carry a large N-gram/PLE component that only
                # offloads to SSD under memory pressure - a big-prompt
                # warmup creates that pressure BEFORE the measured cells,
                # so the resident drops to the model-card level first
                # (observed: pi's 80k-token prefill failed with 135GB
                # resident on the oQ5e until the PLE offload engaged)
                if (cfg.get("model_gb") or 0) >= 90:
                    log("memory pre-shrink warmup (large model: forces the "
                        "PLE SSD offload)…", fw=cfg["name"])
                    call_chat(fw, "word " * 14000,
                              {"temperature": 0.1, "max_tokens": 1})
            except RunStopped:
                raise
            except Exception as e:
                # a failed warmup usually means the reused server just died
                # (observed: the oMLX GUI quit 2s before the cell started -
                # the health check passed during its final moments, then
                # every request hit a dead port). Restart the framework ONCE
                # and retry the warmup instead of failing the first cell.
                log(f"warmup failed ({e}) - restarting "
                    f"{cfg['name']} once and retrying", fw=cfg["name"],
                    level="err")
                stop_framework(fw)
                if start_framework(fw, effective_reasoning(fw, reasoning_level)):
                    try:
                        call_chat(fw, "Say OK.",
                                  {"temperature": 0.1, "max_tokens": 16})
                        log("warmup ok after restart", fw=cfg["name"],
                            level="ok")
                    except Exception as e2:
                        log(f"warmup failed after restart too ({e2}) - "
                            f"continuing", fw=cfg["name"], level="err")

            for rep in range(repeats):
                for h in harnesses:
                    if not RUN_FLAG.is_set():
                        break
                    label = HARNESS_LABELS.get(h, h)
                    row = {"framework": fw, "model": cfg["model"], "harness": label,
                           "task": task_name + (f" #{rep + 1}" if repeats > 1 else ""),
                           "status": "running", "latency": None, "tokens": None,
                           "tps": None, "output_url": None}
                    with LOCK:
                        STATE["results"].append(row)
                        STATE["current_step"] = f"{cfg['name']} / {label} / {task_name}"
                    log(f"running task “{task_name}”", fw=cfg["name"], harness=label)
                    ensure_single_model(fw)
                    snap_before = server_snapshot(fw)
                    try:
                        r = run_harness(fw, h, task_name,
                                        prompt_for_harness(prompt, artifact, h), settings,
                                        task=next((t for t in TASKS if t["id"] == task_id), None))
                        row.update({k: v for k, v in r.items() if k != "_text"})
                        row["_text"] = r.get("_text", "")
                        snap_after = server_snapshot(fw)
                        srv = server_cell_delta(fw, snap_before, snap_after)
                        if fw == "omlx" and snap_after and \
                                not srv.get("server_tgs") and not srv.get("server_pp"):
                            # usage counters flush every ~5s — the diff window
                            # may have caught a flush boundary; settle + retry
                            time.sleep(6)
                            try:
                                srv = server_cell_delta(fw, snap_before,
                                                        server_snapshot(fw))
                            except Exception as e:
                                # a metrics bug must never overwrite a
                                # completed cell (the vllm 'cache' KeyError
                                # erased every result in its first run)
                                log(f"cell metrics unavailable: {e}",
                                    fw=cfg["name"], harness=label, level="err")
                                srv = {}
                        if srv:
                            row.update(srv)
                        # context fill % and effective-bandwidth estimate
                        if row.get("prompt_tokens") and cfg.get("ctx_tokens"):
                            row["ctx_fill_pct"] = round(
                                100.0 * row["prompt_tokens"] / cfg["ctx_tokens"], 1)
                        tgs_source = row.get("server_tgs") or row.get("tgs")
                        if tgs_source and cfg.get("model_gb"):
                            row["est_gbps"] = round(
                                tgs_source * cfg["model_gb"] / 1024, 1)
                        if srv or row.get("ctx_fill_pct") is not None:
                            log(f"cell metrics: server pp={srv.get('server_pp')} "
                                f"tgs={srv.get('server_tgs')}"
                                + (f" | ctx fill {row.get('ctx_fill_pct')}%"
                                   if row.get("ctx_fill_pct") is not None else "")
                                + (f" | ~{row.get('est_gbps')} GB/s est"
                                   if row.get("est_gbps") else ""),
                                fw=cfg["name"], harness=label)
                        log(f"done in {row['latency']}s @ {row['tps']} tok/s"
                            + (" — ⚠ truncated at max_tokens, raise the cap and rerun"
                               if row.get("truncated") else ""),
                            fw=cfg["name"], harness=label,
                            level="ok" if row["status"] == "done" else "err")
                    except RunStopped:
                        row.update({"status": "error", "error": "stopped by user"})
                        log(f"{label} stopped by user", fw=cfg["name"],
                            harness=label, level="err")
                        raise
                    except Exception as e:
                        row.update({"status": "error", "error": str(e)[:300]})
                        log(f"failed: {e}", fw=cfg["name"], harness=label, level="err")

            # framework's tests complete → shut it down before next framework
            with LOCK:
                STATE["current_step"] = f"stopping {cfg['name']}…"
            stop_framework(fw)
    finally:
        for r in reasoning_restores:
            r()
        for fw in list(PROCS):
            stop_framework(fw)
        with LOCK:
            STATE["running"] = False
            STATE["current_step"] = ""
            STATE["run_started"] = None
            rows = [{k: v for k, v in r.items() if k != "_text"} for r in STATE["results"]]
        # Persist run history so the community can compare across runs/days.
        if any(r["status"] == "done" for r in rows):
            try:
                os.makedirs(RUNS_DIR, exist_ok=True)
                fname = os.path.join(RUNS_DIR, time.strftime("%Y%m%d-%H%M%S") + ".json")
                with open(fname, "w") as f:
                    json.dump({"ts": time.time(), "task_id": task_id,
                               "task_name": task_name,
                               "reasoning_level": reasoning_level,
                               "model_set": CONFIG.get("model_set"),
                               "reasoning_overrides": {fw: effective_reasoning(fw, reasoning_level)
                                                       for fw in frameworks
                                                       if FRAMEWORKS[fw].get("reasoning_override")},
                               "harnesses": harnesses,
                               # frameworks_run = the subset actually run this
                               # time; frameworks = full config snapshot (the
                               # UI's history view reads this one).
                               "frameworks_run": frameworks,
                               "settings": settings,
                               "frameworks": {k: {"port": v["port"], "model": v["model"]}
                                              for k, v in FRAMEWORKS.items()},
                               "results": rows}, f, indent=2)
                log(f"run history saved → {fname}", level="ok")
            except OSError as e:
                log(f"could not save run history: {e}", level="err")
        log("benchmark complete", level="ok")


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
def _scan_models_payload(store_filter=None, extra=None):
    """Scan payload shared by the /api/models/scan route and model/select."""
    machine = discovery.machine_profile()
    free, _total = free_ram()
    free_gb = round(free / 1073741824, 1)
    cands = discovery.all_candidates(free_gb, FRAMEWORKS, machine)
    if extra and os.path.isdir(extra):
        for entry in sorted(os.listdir(extra)):
            d = os.path.join(extra, entry)
            if not os.path.isfile(os.path.join(d, "config.json")):
                continue
            cands.append({
                "id": entry, "store": "custom", "custom_path": d,
                "size_gb": _dir_size_gb_local(d),
                "context_length": _ctx_local(os.path.join(d, "config.json")),
                "frameworks": ["mlxlm", "mlxserve"],
                "in_cache": False, "served": False,
            })
    models_out = []
    for c in cands:
        store = c.get("source") or ("mtplx" if c.get("in_mtplx_store") else "hf")
        if c.get("custom_path"):
            store = "custom"
        if store_filter and store != store_filter:
            continue
        mf = (c.get("compat") or {}).get("machine_fit")
        models_out.append({
            "id": c.get("id"), "store": store,
            "size_gb": c.get("size_gb"),
            "context_length": c.get("context_length"),
            "family": c.get("family"), "moe": c.get("moe"),
            "moe_detail": c.get("moe_detail"),
            "frameworks": c.get("frameworks") or [],
            "verdict": (c.get("compat") or {}).get("verdict"),
            "machine_fit": mf, "need_gb": c.get("need_gb"),
            "served": bool(c.get("served")),
            "custom_path": c.get("custom_path"),
        })
    # compatibility authority: the shipped model sets encode which framework
    # each model FAMILY actually runs on (flash-next -> omlx+mtplx only, etc)
    fam_fw = {}
    for name, s in (CONFIG.get("model_sets") or {}).items():
        if name == "selected":
            continue   # a snapshot of user picks, not a compatibility statement
        fam = s.get("group") or s.get("label") or name
        for fw in (s.get("models") or {}):
            fam_fw.setdefault(fam, set()).add(fw)
    return {"machine": machine, "free_gb": free_gb, "models": models_out,
            "family_frameworks": {fam: sorted(fws)
                                  for fam, fws in fam_fw.items()}}


def _dir_size_gb_local(d):
    total = 0
    for dirpath, _dn, filenames in os.walk(d):
        for fn in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    return round(total / 1073741824, 1) if total else None


def _ctx_local(path):
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    for node in (cfg, cfg.get("text_config") or {}):
        if isinstance(node, dict):
            v = node.get("max_position_embeddings") or node.get("seq_len")
            if isinstance(v, int) and v > 0:
                return v
    return None




class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _safe(self, fn):
        """Run a route handler; bad input → 400, anything else → 500 JSON
        (never an HTML traceback page)."""
        try:
            fn()
        except ValueError as e:
            self._json({"ok": False, "error": str(e)[:200]}, 400)
        except BrokenPipeError:
            pass
        except Exception as e:
            log(f"HTTP {self.command} {self.path} failed: {e}", level="err")
            try:
                self._json({"ok": False, "error": f"internal error: {str(e)[:150]}"}, 500)
            except Exception:
                pass

    def do_OPTIONS(self):
        if self.path.startswith("/api/relay/"):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._safe(self._route_options) if hasattr(self, "_route_options") else             self.send_response(204) or self.end_headers()

    def do_GET(self):
        self._safe(self._route_get)

    def _relay(self):
        # CORS relay: browser pages (dashboard, generated consoles) cannot
        # call framework servers that reject browser origins (MTPLX 403s
        # them). The bench forwards the call and adds permissive CORS — the
        # forwarded request carries no Origin header, so origin-rejecting
        # frameworks accept it.
        from urllib.parse import urlparse
        parts = urlparse(self.path).path.split("/")
        try:
            port = int(parts[3])
        except (IndexError, ValueError):
            raise ValueError("relay URL must be /api/relay/<port>/<path>")
        known = {f.get("port") for f in FRAMEWORKS.values()}
        if port not in known:
            raise ValueError(f"relay: port {port} is not a framework port")
        sub = "/".join(parts[4:])
        target = f"http://127.0.0.1:{port}/{sub}"
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(target, data=body, method=self.command)
        for h in ("Content-Type", "Authorization"):
            if self.headers.get(h):
                req.add_header(h, self.headers[h])
        try:
            with urllib.request.urlopen(req, timeout=1800) as up:
                payload, code, ctype = up.read(), up.status, up.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            payload, code, ctype = e.read(), e.code, e.headers.get("Content-Type", "application/json")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _route_get(self):
        if self.path.startswith("/api/relay/"):
            self._relay()
            return
        if self.path.startswith("/api/fixture"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            task = (q.get("task") or [""])[0]
            if task not in TASK_FIXTURES:
                raise ValueError(f"no fixture for task {task!r}")
            fname = TASK_FIXTURES[task][0]
            with open(os.path.join(ROOT, "fixtures", fname), encoding="utf-8") as fh:
                self._json({"task": task, "file": fname, "source": fh.read()})
            return
        if self.path.startswith("/api/hold"):
            # same logic as the POST route: a real-time pause the QA probe's
            # page fetches so async page work completes before grading
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            try:
                ms = min(max(int(q.get("ms", ["3000"])[0]), 0), 15000)
            except ValueError:
                ms = 3000
            time.sleep(ms / 1000)
            self._json({"ok": True, "held_ms": ms})
            return
        if self.path == "/" or self.path == "/index.html":
            with open(os.path.join(ROOT, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-cache, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/health":
            with LOCK:
                self._json({"ok": True, "running": STATE["running"],
                            "ts": time.time(),
                            "frameworks": dict(STATE["framework_status"])})
        elif self.path == "/api/tasks":
            # fixture tasks: serve the prompt with the source ALREADY inlined,
            # so the dashboard can never show a raw @...SOURCE@ placeholder
            out = []
            for t in TASKS:
                t = dict(t)
                if t.get("id") in TASK_FIXTURES:
                    try:
                        with open(os.path.join(ROOT, "fixtures",
                                               TASK_FIXTURES[t["id"]][0]),
                                  encoding="utf-8") as fh:
                            body = fh.read()
                        ph = "@BUGFIX_SOURCE@" if t["id"] == "bugfix" \
                            else "@CODE_REVIEW_SOURCE@"
                        t["prompt"] = t["prompt"].replace(ph, body)
                    except OSError:
                        pass
                out.append(t)
            self._json(out)
        elif self.path == "/api/harnesses":
            # ordered harness id -> display label (UI renders from this)
            self._json(HARNESS_LABELS)
        elif self.path == "/api/reasoning":
            # task-type → reasoning level (drives the framework server start)
            self._json(REASONING_LEVELS)
        elif self.path == "/api/state":
            refresh_fw_status_idle()
            free, total = free_ram()
            with LOCK:
                snap = {**STATE, "results": [{k: v for k, v in r.items() if k != "_text"}
                                             for r in STATE["results"]]}
            snap["ram"] = {"free": free, "total": total}
            self._json(snap)
        elif self.path.startswith("/api/activity?since="):
            try:
                since = float(self.path.split("since=")[1])
            except (IndexError, ValueError):
                raise ValueError("since= must be a number")
            with LOCK:
                lines = [l for l in ACTIVITY if l["ts"] > since]
            self._json(lines)
        elif self.path == "/api/runs_history":
            import glob as _g
            runs = []
            for f in sorted(_g.glob(os.path.join(RUNS_DIR, "*.json")),
                            reverse=True):
                try:
                    with open(f) as fh:
                        d = json.load(fh)
                    rows = [{k: v for k, v in r.items() if k != "_text"}
                            for r in d.get("results", [])]
                    runs.append({"file": os.path.basename(f),
                                 "ts": d.get("ts"), "task": d.get("task_name"),
                                 "reasoning_level": d.get("reasoning_level"),
                                 "frameworks": {k: v.get("model")
                                                for k, v in (d.get("frameworks")
                                                             or {}).items()},
                                 "harnesses": d.get("harnesses"),
                                 "rows": rows})
                except (json.JSONDecodeError, OSError):
                    pass
            self._json(runs)
        elif self.path == "/api/campaign":
            c = STATE.get("campaign") or _campaign_load() or {}
            sets = list((CONFIG.get("model_sets") or {}).keys())
            plans = {name: {"label": (CONFIG.get("model_sets") or {})
                            .get(name, {}).get("label", name),
                            "group": (CONFIG.get("model_sets") or {})
                            .get(name, {}).get("group", name),
                            "runs": len(campaign_plan([name])),
                            "frameworks": list((CONFIG.get("model_sets") or {})
                                               .get(name, {}).get("models", {}).keys()),
                            "estimate": CAMPAIGN_ESTIMATES.get(name, "?")}
                     for name in sets}
            both = {"runs": sum(p["runs"] for p in plans.values()),
                    "estimate": CAMPAIGN_ESTIMATES.get("both", "5–7 days")}
            self._json({"campaign": c, "sets": plans, "both": both,
                        "active_set": CONFIG.get("model_set")})
        elif self.path == "/api/frameworks":
            with LOCK:
                self._json({fw: {"name": c["name"], "port": c["port"],
                                 "model": c["model"],
                                 "model_available": c.get("model_available", True),
                                 "model_set": CONFIG.get("model_set"),
                                 "repo": c.get("repo"),
                                 "ctx_tokens": c.get("ctx_tokens"),
                                 "model_gb": c.get("model_gb"),
                                 "start_cmd": " ".join(resolve_start_cmd(
                                     c, REASONING_LEVELS.get("short", "low"))),
                                 "notes": c.get("notes", ""),
                                 "status": STATE["framework_status"].get(fw)}
                            for fw, c in FRAMEWORKS.items()})
        elif self.path == "/api/machine":
            self._json(discovery.machine_profile())

        elif self.path.startswith("/api/models/scan"):
            # MODEL-FIRST discovery: every installed model across all stores
            # (HF cache, MTPLX store, optional extra folders), each tagged with
            # the frameworks that can serve it and the fit on this machine.
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            extra = (q.get("extra") or [""])[0]
            payload = _scan_models_payload(extra=extra or None)
            self._json(payload)
        elif self.path == "/api/proxy":
            with PROXY_LOCK:
                data = {"port": PROXY_PORT, **PROXY_STATE,
                        "totals": dict(PROXY_TOTALS), "log": list(PROXY_LOG)}
            self._json(data)
        elif self.path.startswith("/output/"):
            name = os.path.basename(self.path.split("?")[0])
            fp = os.path.join(OUTPUT_DIR, name)
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    body = f.read()
                ext_ctype = {".md": "text/markdown", ".py": "text/x-python"}
                ctype = (ext_ctype.get(os.path.splitext(name)[1], "text/html"
                         if name.endswith(".html") else "text/plain")) + "; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        self._safe(self._route_post)

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            raise ValueError("bad Content-Length")
        if length > 1_000_000:  # prompts are small; reject anything fat
            raise ValueError("body too large")
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            raise ValueError("body must be valid JSON")

    def _route_post(self):
        req = self._read_body()
        if self.path == "/api/run":
            _validate_run(req)  # 400 on bad input before spawning a thread
            with LOCK:
                already = STATE["running"]
            if already:
                self._json({"ok": False, "error": "already running"}, 409)
                return
            threading.Thread(target=run_benchmark, args=(req,), daemon=True).start()
            self._json({"ok": True})
        elif self.path == "/api/model/select":
            # MODEL-FIRST selection: one model -> the frameworks that can serve
            # it are enabled, everything else is marked unavailable.
            model = str((req or {}).get("model") or "")
            store = str((req or {}).get("store") or "hf")
            if not model:
                raise ValueError("model required")
            payload = _scan_models_payload()
            entry = next((mm for mm in payload["models"] if mm["id"] == model), None)
            if entry is None:
                raise ValueError("model %r not found in the scanned stores" % model)
            if entry.get("machine_fit") == "wont-fit":
                raise ValueError(
                    "%s needs ~%s GB but this machine has ~%s GB usable -- pick a "
                    "smaller quant of the same family" % (model, entry.get("need_gb"),
                    payload["machine"].get("usable_gb")))
            fws = entry.get("frameworks") or []
            if not fws:
                raise ValueError("no benchmark framework can serve %r" % model)
            for fw, c in FRAMEWORKS.items():
                c["model_available"] = fw in fws
                if fw in fws:
                    c["model"] = (model.replace("/", "--")
                                  if fw == "omlx" else model)
                    c["repo"] = model
                    if entry.get("context_length"):
                        c["ctx_tokens"] = min(entry["context_length"], 131072)
            set_name = "selected"
            # store the per-framework ids (OMLX cache-style) so the later
            # apply_model_set() re-application keeps the servable form
            CONFIG["model_sets"][set_name] = {
                "label": "Selected - " + model.split("/")[-1],
                "group": "Selected",
                "ctx_tokens": FRAMEWORKS[fws[0]].get("ctx_tokens"),
                "max_tokens": FRAMEWORKS[fws[0]].get("max_tokens", 65536),
                "models": {fw: FRAMEWORKS[fw]["model"] for fw in fws}}
            CONFIG["selected_model"] = {"id": model, "store": store}
            apply_model_set(set_name)
            self._json({"ok": True, "model": model, "frameworks": fws,
                        "set": set_name, "machine": payload["machine"]})
        elif self.path == "/api/stop":
            request_stop()
            log("stop requested — killing in-flight work", level="err")
            self._json({"ok": True})
        elif self.path.startswith("/api/hold"):
            # a deliberate real-time pause for the QA probe: virtual time
            # pauses while this response is pending, giving asynchronous
            # page work (IndexedDB callbacks) time to finish
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            try:
                ms = min(max(int(q.get("ms", ["3000"])[0]), 0), 15000)
            except ValueError:
                ms = 3000
            time.sleep(ms / 1000)
            self._json({"ok": True, "held_ms": ms})
        if self.path.startswith("/api/relay/"):
            self._relay()
            return
            # the resolved fixture source for prompt display: the dashboard
            # inlines it into the prompt box so users see what the model sees
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            task = (q.get("task") or [""])[0]
            if task not in TASK_FIXTURES:
                raise ValueError(f"no fixture for task {task!r}")
            fname = TASK_FIXTURES[task][0]
            with open(os.path.join(ROOT, "fixtures", fname), encoding="utf-8") as fh:
                self._json({"task": task, "file": fname, "source": fh.read()})
        elif self.path == "/api/qa_probe":
            # on-demand runtime-probe of a saved artifact (MCP/agent tooling)
            artifact = str((req or {}).get("artifact") or "")
            task = str((req or {}).get("task") or "")
            base_url = (req or {}).get("base_url")
            root = os.path.realpath(ROOT)
            path = os.path.realpath(os.path.join(ROOT, artifact.lstrip("/")))
            if not path.startswith(root + os.sep) or not os.path.isfile(path):
                raise ValueError(f"artifact not found: {artifact}")
            if task not in QA_PROBES:
                raise ValueError(f"no probe for task {task!r}")
            self._json(run_qa_probe(path, task, base_url=base_url))
        elif self.path == "/api/rescore":
            fname = str((req or {}).get("file") or "")
            root = os.path.realpath(ROOT)
            path = os.path.realpath(os.path.join(ROOT, fname.lstrip("/")))
            if not path.startswith(root + os.sep) or not os.path.isfile(path):
                raise ValueError(f"run file not found: {fname}")
            import rescore as _rescore
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _rescore.rescore_file(path)
            self._json({"ok": True, "log": buf.getvalue()[-2000:]})
        elif self.path == "/api/models/select":
            fw = req.get("fw") if isinstance(req, dict) else None
            model = req.get("model") if isinstance(req, dict) else None
            if fw not in FRAMEWORKS:
                raise ValueError(f"unknown framework: {fw}")
            if not isinstance(model, str) or not model.strip():
                raise ValueError("model: non-empty string required")
            cfg = FRAMEWORKS[fw]
            model = model.strip()
            cfg["model"] = model
            # MTPLX: the CLI --model flag loads by repo id from ~/.mtplx/models,
            # but the server serves a normalized id (auto-adopted on start). Track
            # the repo separately so start_cmd's {repo} placeholder resolves right.
            if cfg.get("model_source") == "mtplx":
                cfg["repo"] = model
            for key, cast in (("ctx_tokens", int), ("model_gb", float)):
                if isinstance(req.get(key), (int, float)) and req[key] > 0:
                    cfg[key] = cast(req[key])
            if not save_config():
                raise ValueError("model updated in memory but config.json save failed")
            log(f"model for {cfg['name']} set to {cfg['model']}", fw=fw, level="ok")
            self._json({"ok": True, "model": cfg["model"]})
        elif self.path == "/api/campaign":
            action = (req or {}).get("action")
            with LOCK:
                busy = STATE.get("running")
            camp = STATE.get("campaign") or {}
            if action == "start":
                mode = req.get("mode")
                if mode not in ("set", "all"):
                    self._json({"ok": False, "error": "mode must be set|all"}, 400)
                    return
                if busy or camp.get("active"):
                    self._json({"ok": False, "error": "a run or campaign is already active"}, 409)
                    return
                sets = (req.get("sets")
                        or (list((CONFIG.get("model_sets") or {}).keys())
                            if mode == "all"
                            else [req.get("set") or CONFIG.get("model_set", "qwen38-27b")]))
                unknown = [x for x in sets if x not in (CONFIG.get("model_sets") or {})]
                if unknown:
                    self._json({"ok": False, "error": f"unknown model set(s): {unknown}"}, 400)
                    return
                if not sets:
                    self._json({"ok": False, "error": "no model sets selected"}, 400)
                    return
                # fit gate: a ticked set whose model cannot fit this machine
                # would fail every cell — refuse with the numbers, not mid-run
                fitmap, machine = _fit_lookup()
                for s in sets:
                    for fw, model in ((CONFIG.get("model_sets") or {})
                                      .get(s, {}).get("models", {}).items()):
                        cand = fitmap.get(discovery.normalize_key(model))
                        mf = (cand.get("compat") or {}).get("machine_fit") if cand else None
                        if mf == "wont-fit":
                            self._json({"ok": False, "error": (
                                f"set {s!r}: {FRAMEWORKS[fw]['name']} model {model} needs "
                                f"~{cand.get('need_gb')} GB but this machine has "
                                f"~{machine.get('usable_gb')} GB usable — untick it or pick a "
                                f"smaller quant")}, 400)
                            return
                harnesses = [h for h in (req.get("harnesses") or [])
                             if h in HARNESS_LABELS] or None
                threading.Thread(target=campaign_runner,
                                 args=(mode, sets, 0, harnesses),
                                 daemon=True).start()
                self._json({"ok": True, "started": mode, "sets": sets})
            elif action == "resume":
                saved = _campaign_load()
                if not saved or not saved.get("sets"):
                    self._json({"ok": False, "error": "no campaign to resume"}, 400)
                    return
                if busy or camp.get("active"):
                    self._json({"ok": False, "error": "a run or campaign is already active"}, 409)
                    return
                idx = saved.get("index", 0)
                threading.Thread(target=campaign_runner,
                                 args=(saved.get("mode", "all"), saved.get("sets"),
                                       idx, saved.get("harnesses")),
                                 daemon=True).start()
                self._json({"ok": True, "resumed": True, "from": idx})
            elif action == "cancel":
                if camp.get("active"):
                    request_stop()
                cleared = {"active": False, "status": "cancelled",
                           "cleared": time.time()}
                STATE["campaign"] = cleared
                _campaign_save(cleared)
                self._json({"ok": True, "cancelled": True})
            else:
                self._json({"ok": False, "error": "action must be start|resume|cancel"}, 400)
        elif self.path == "/api/model_set":
            mmap = (req or {}).get("models")
            if isinstance(mmap, dict) and mmap:
                models_map = {}
                for fw, model in mmap.items():
                    if fw not in FRAMEWORKS:
                        raise ValueError(f"unknown framework {fw!r}")
                    if model:
                        models_map[fw] = str(model)
                # frameworks the selection leaves empty become unavailable —
                # their checkboxes clear (e.g. MLX-VLM/MLX-Serve under a
                # flash-next family pick)
                for fw in FRAMEWORKS:
                    FRAMEWORKS[fw]["model_available"] = fw in models_map
                try:
                    _validate_fit(models_map)
                except ValueError as e:
                    self._json({"ok": False, "error": str(e)}, 400)
                    return
                for fw, model in models_map.items():
                    cfg = FRAMEWORKS[fw]
                    if fw == "omlx":
                        model = model.replace("/", "--")   # OMLX cache-style id
                    cfg["model"] = model
                    cfg["repo"] = model
                    cfg["model_available"] = True
                    cache = discovery.cache_dirname(model)
                    declared = discovery._cached_context(cache) if cache else None
                    if declared and declared < (cfg.get("ctx_tokens") or 0):
                        cfg["ctx_tokens"] = declared   # model's native window wins
                    # NOTE: no live-server id adoption here — adopting would
                    # rename the selection to the OLD served model. A running
                    # server that doesn't serve the selection is restarted by
                    # start_framework's mismatch check at the next Run.
                CONFIG["model_set"] = "selected"
                first = next(iter(models_map))
                CONFIG["model_sets"]["selected"] = {
                    "label": "Selected models (dashboard dropdowns)",
                    "group": "Selected",
                    "ctx_tokens": FRAMEWORKS[first].get("ctx_tokens"),
                    "max_tokens": FRAMEWORKS[first].get("max_tokens", 65536),
                    "models": dict(models_map)}
                save_config()
                self._json({"ok": True, "set": CONFIG["model_set"],
                            "frameworks": {fw: {"name": c.get("name", fw),
                                                "model": c.get("model"),
                                                "model_available": c.get("model_available", True)}
                                           for fw, c in FRAMEWORKS.items()}})
                return
            sel = (req or {}).get("sets") or []
            if sel and isinstance(sel, list):
                try:
                    apply_set_selection([str(x) for x in sel])
                except ValueError as e:
                    self._json({"ok": False, "error": str(e)}, 400)
                    return
            else:
                name = (req or {}).get("set")
                if name not in (CONFIG.get("model_sets") or {}):
                    self._json({"ok": False, "error": f"unknown model set {name!r}"}, 400)
                    return
                apply_model_set(name)
            self._json({"ok": True, "set": CONFIG["model_set"],
                        "frameworks": {fw: {"name": c.get("name", fw),
                                            "model": c.get("model"),
                                            "model_available": c.get("model_available", True)}
                                       for fw, c in FRAMEWORKS.items()}})
        elif self.path == "/api/proxy/clear":
            with PROXY_LOCK:
                PROXY_LOG.clear()
                PROXY_TOTALS.clear()
                PROXY_TOTALS.update(PROXY_ZERO)
            self._json({"ok": True})
        else:
            self.send_error(404)

    def do_DELETE(self):
        self._safe(self._route_delete)

    def _route_delete(self):
        # /api/runs/<YYYYmmdd-HHMMSS>.json — permanently remove one saved run:
        # its JSON file plus any /output artifacts referenced only by it.
        m = re.fullmatch(r"/api/runs/(\d{8}-\d{6}\.json)", self.path)
        if not m:
            self._json({"ok": False, "error": "bad request"}, 400)
            return
        name = m.group(1)
        fp = os.path.join(RUNS_DIR, name)
        if not os.path.isfile(fp):
            self._json({"ok": False, "error": "no such run"}, 404)
            return
        refs = set()
        try:
            with open(fp) as fh:
                d = json.load(fh)
            for r in d.get("results", []):
                u = r.get("output_url")
                if u and "/output/" in u:
                    refs.add(os.path.basename(u.split("?")[0]))
        except (json.JSONDecodeError, OSError):
            pass
        os.remove(fp)
        # keep any output file another surviving run still points at
        import glob as _g
        for f in _g.glob(os.path.join(RUNS_DIR, "*.json")):
            if os.path.basename(f) == name:
                continue
            try:
                with open(f) as fh:
                    other = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            for r in other.get("results", []):
                u = r.get("output_url")
                if u and "/output/" in u:
                    refs.discard(os.path.basename(u.split("?")[0]))
        removed = 0
        for n in sorted(refs):
            op = os.path.join(OUTPUT_DIR, n)
            if os.path.isfile(op):
                try:
                    os.remove(op)
                    removed += 1
                except OSError:
                    pass
        log(f"deleted run {name}"
            + (f" + {removed} orphaned output file(s)" if removed else ""),
            level="ok")
        self._json({"ok": True, "deleted_outputs": removed})


def _shutdown(signum, _frame):
    log(f"received signal {signum} — shutting down frameworks", level="err")
    for fw in list(PROCS):
        stop_framework(fw)
    sys.exit(0)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(CONFIG.get("port", 7090))
    host = CONFIG.get("host", "127.0.0.1")
    if ROUTE_VIA_PROXY:
        start_proxy()
    # reflect real status of any servers already listening
    for fw in FRAMEWORKS:
        set_fw_status(fw, "up" if framework_healthy(fw) else "down")
    signal.signal(signal.SIGTERM, _shutdown)
    # auto-resume an interrupted campaign (week-long runs span restarts)
    saved = _campaign_load()
    if saved and saved.get("active") and saved.get("status") == "running" \
            and saved.get("sets"):
        log(f"resuming interrupted campaign at run "
            f"{(saved.get('index') or 0) + 1}/{saved.get('plan_total')} "
            f"({', '.join(saved['sets'])})", level="ok")
        threading.Thread(target=campaign_runner,
                         args=(saved.get("mode", "all"), saved.get("sets"),
                               saved.get("index", 0)), daemon=True).start()
    log("benchmark server ready"
        + (f" (measurement proxy on :{PROXY_PORT})" if ROUTE_VIA_PROXY else ""))
    if host in ("0.0.0.0", "::"):
        log("⚠ listening on ALL interfaces — anyone on your network can start "
            "runs and delete data. Use host 127.0.0.1 unless you know why.",
            level="err")
    print(f"\n  ➜ Open http://{'localhost' if host in ('127.0.0.1', '::1') else host}:{port}\n")
    try:
        ThreadingHTTPServer((host, port), Handler).serve_forever()
    except OSError as e:
        print(f"\n  ✗ Could not bind {host}:{port} — {e}\n"
              f"    Is another benchtest server (or a framework) already using it?\n"
              f"    Change 'port' in config.json or pass a different port: python3 server.py <port>",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        for fw in list(PROCS):
            stop_framework(fw)

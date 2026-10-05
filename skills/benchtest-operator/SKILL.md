---
name: benchtest-operator
description: Operate, monitor and debug the benchtest local-LLM benchmark (Apple Silicon; OMLX/MTPLX/MLX-VLM/MLX-Serve × raw/raw+/pi/opencode/Goose/hart). Use when the user asks to run, monitor, compare, rescore or debug benchmark runs, or when benchmark artifacts or logs need triage.
---

# benchtest operator

Local LLM benchmark on one Mac. A web backend (port 7090) drives framework
servers (OMXL :7001, MTPLX :7002, MLX-VLM :7003, MLX-Serve :7004), runs agent
harnesses against them, and grades the produced artifacts.

## Mental model

- **Model set** = one model per framework (`config.json → model_sets`). Two
  families: `qwen38-27b` (all 4 frameworks) and `flash-next` (MTPLX gets the
  Speed build, OMLX gets the 4bit-mtp).
- **Run** = one task × the selected frameworks × harnesses. Each cell = one
  framework/harness pair. **Campaign** = sets × all tasks, sequential.
- **QA gate** = regex feature checks (40%) + a headless-Chrome runtime probe
  (70%) that loads the artifact and exercises it. QA ≥ 90 = usable.
- Tasks: tetris, snake, pong, todo, fib, markdown, agentconsole, logreport
  (analyze the previous run's logs), webdb (single-file app over IndexedDB),
  bugfix (fix 4 planted bugs in a broken app, graded by its self-test).
  bugfix and markdown are the best model/harness discriminators.

## Key files

| Path | What |
|---|---|
| `config.json` | frameworks, model_sets, reasoning, ports |
| `runs/<ts>.json` | one completed run: per-cell status/latency/qa/error |
| `outputs/<oid>.html` | graded artifacts (referenced by `output_url`) |
| `logs/bench.log` | orchestrator log — the forensics source |
| `logs/<fw>-<ts>.log` | per-framework server logs |
| `campaign.json` | campaign state (resume/complete/paused) |
| `rescore.py` | re-grades saved runs with the current gate (no re-runs) |

The backend HTTP API (port 7090): `GET /api/state`, `GET /api/runs_history`,
`GET /api/frameworks`, `POST /api/run`, `POST /api/stop`,
`POST /api/campaign {action: start|resume|cancel, mode, sets, harnesses}`.

## Triage runbook (run this top-to-bottom on any failure)

1. **State first**: `GET /api/state` — is a run/campaign active, which cell?
2. **Classify the error** from the cell's `error` string:
   - `empty response` → the model spent the harness's output cap on hidden
     thinking before writing content. Check thinking budget (OMLX per-model
     settings) and the harness cap (pi 32K, opencode 32K).
   - `timeout after 7200s` → the harness ran out of time; scan the bench.log
     window: was it progressing (tool calls streaming) or stalled?
   - MTPLX `mtplx_stream_stall_break` in the server log → MTPLX killed a
     silent thinking phase at its 300s watchdog (`--stream-stall-deadline-s`).
   - `exited 1` / `no output` → the harness CLI died silently; check the
     framework server log for whether a request even arrived.
   - OMLX `memory-guard` / `forced to SSD` → model does not fit; check
     resident GB vs the wired ceiling (~121.6 GB on 128 GB machines).
3. **Read the QA notes**, never just the number: `runtime probe N/M failed: …`
   names the exact broken behavior; `missing: …` lists requirement gaps.
4. **Only then change code/settings**, and prefer, in order: probe bug
   (fix the gate), harness setting, framework flag, model swap. Re-verify by
   re-probing the artifact before re-running any expensive cell.
5. **Gate improved?** `python3 rescore.py runs/<file>.json` — artifacts are
   the ground truth; scores are recomputable without re-running models.

## Pitfalls (hard-won rules)

- `config.json` is rewritten by the backend's in-memory state (`save_config`
  on every model-set apply). **Never edit it while a backend lives** — the
  edit will be silently reverted. Sequence: stop backend → edit → start.
  (`~/.omlx/model_settings.json` is safe to edit while OMLX runs; the backend
  restarts OMLX on the file's mtime change.)
- pi and opencode cap per-request output (32K); thinking models can exhaust
  that on hard tasks before emitting content — that reads as `empty response`
  and is a cap setting, not a model bug.
- The QA probe presses Enter/arrow keys and clicks canvas/start buttons, but
  never Space (pong binds Space to pause). CHIP-8 is graded on its self-test
  panel, not canvas animation (test ROMs are static).
- A fresh model download must land where each framework reads it: OMLX and
  MLX-VLM/MLX-Serve read the HF cache; **MTPLX only reads `~/.mtplx/models`**.
- After changing the QA gate, rescore instead of re-running.

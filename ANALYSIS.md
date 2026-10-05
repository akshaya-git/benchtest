# benchtest — Code review & analysis

This is the record of the full review of the original codebase: the bugs found and
fixed, the redundant code assessed, and the design decisions made to turn a
single-machine script into a sturdy tool for many users.

## 1. Bugs found & fixed

Each item: what was wrong, why it mattered, and the fix.

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| **B1** | Duplicate `"frameworks"` key in the run-history JSON — a list (the subset run) and a dict (full config) under the same key; the list was silently discarded by `json.dump`. | Run history lost which frameworks were actually run; the dict survived by accident of ordering. | Renamed the list to `frameworks_run`; kept the dict as `frameworks` (the UI's history view reads it). |
| **B2** | `rawplus_generate` computed decode TGS from `met.get("wall")`, but `call_chat`'s metrics dict had no `wall` key → `None` → `max(None - ttft, …)` crashed or produced garbage. | raw+ TGS was wrong (or the cell errored). | Added `"wall": round(total, 3)` to `call_chat`'s metrics. raw+ now computes an honest decode rate. |
| **B3** | `_prom_scrape` was dead code — it skipped every labeled line, but Prometheus metrics are *all* labeled, so it parsed nothing. | MLX-Serve's rich Prometheus surface (TTFT/prefill/decode histograms) was never read. | Rewrote to strip labels, sum across series (correct for counters and `_sum`/`_count`), and drop `_bucket` lines. |
| **B4** | The "empty response" check failed a cell whenever stdout was empty — even when the agent had delivered the artifact as a file. | File-delivering agents (common for "write me a file" tasks) were marked as errors. | The check now fails only when there is **neither** text **nor** a collected artifact. |
| **B5** | `free_ram()` used `vm_stat`/`sysctl` — macOS only. | Broke on Linux; returned `(0,0)` everywhere else. | Cross-platform: `vm_stat` on macOS, `/proc/meminfo` on Linux. |
| **B6** | HTTP handlers had no exception handling — any unexpected error produced an HTML traceback page and could leave the connection in a bad state. | Ugly failures; no structured error for the UI. | Every route is wrapped (`_safe`): `ValueError` → 400 JSON, anything else → 500 JSON. Never an HTML traceback. |
| **B7** | `prep_pi` wrote the user's **global** `~/.pi/agent/models.json` (with a one-time `.bak-bench`). | Mutated the user's real pi config on every run; unsafe for concurrent pi use and for multi-user machines. | pi now runs with an **isolated** `PI_CODING_AGENT_DIR` under `harness-configs/pi`. The user's global config is never touched. |
| **B8** | Machine-specific hacks: a `_ms_snap` block that patched one hardcoded model's path, a hardcoded `HART_PATH`, and a hardcoded `/opt/homebrew/opt/python@3.14/bin/python3`. | Broke on any other machine; impossible to share. | All moved to `config.json` (`hart_path`, `start_cmd` with `{model}`/`{model_path}` placeholders). The snapshot patch is gone — `resolve_start_cmd` resolves paths generically. |
| **B9** | `subprocess.run(["which", cmd[0]])` to check for a CLI. | `which` isn't portable (not on minimal Linux). | `shutil.which(cmd[0])`. |
| **B10** | CSV export joined fields with `,` and no quoting. | Any model/task name with a comma or quote produced a malformed CSV. | Added a `csvq()` helper that RFC-4180-quotes fields containing `,` `"` or newline. |
| **B11** | `json.load(open(f))` in several places — file handles never closed. | Handle leaks over long-running sessions. | All replaced with `with open(f) as fh:` context managers. |
| **B12** | `refresh_fw_status_idle` wrote `STATE["framework_status"]` **without** the lock. | Data race with the orchestrator and other handlers. | Now goes through `set_fw_status`, which takes the lock. |
| **B13** | The UI hardcoded the framework list in three places (top strip, selection grid, chart colors). | Adding a framework required editing `index.html` in three spots. | The UI renders everything from `/api/frameworks` (and harnesses from `/api/harnesses`). Add a framework to `config.json` and it appears everywhere. |
| **B14** | `start_framework` sent the framework's stdout/stderr to `DEVNULL`. | A failed start was undiagnosable. | Framework output is captured to `logs/<fw>-<timestamp>.log`; handles tracked in `FW_LOGS` and closed on stop. |
| **B15** | `free_ram()` spawned **two** subprocesses on every call, and the UI polls `/api/state` every 1.2s. | Constant subprocess churn for a value that changes slowly. | Free RAM is cached for 3s; total RAM is computed once. |
| **B16** | No input validation on `POST /api/run` — a malformed body could 500 or spawn a thread that crashed. | Unstructured failures; a bad body could wedge state. | `_validate_run` checks types and ranges; bad input → 400 before any thread is spawned. Bodies are size-capped. |
| **B17** | `main()` bound to `0.0.0.0` unconditionally with no warning, and a port conflict was an opaque traceback. | Accidental network exposure; confusing startup failure. | Binds to `config.json → host` (default `127.0.0.1`), warns loudly on `0.0.0.0`, and turns a bind failure into a clear message. |
| **B18** | Model discovery only scanned the HF cache — MTPLX models live in a separate store (`~/.mtplx/models`), so the picker never showed them and MTPLX model selection was impossible. | Users couldn't pick an MTPLX model from the UI; the dropdown was wrong for MTPLX. | Added `discover_mtplx()` (reads `~/.mtplx/models`, using `.mtplx-source.json` for the repo id and `config.json` for context) and a unified `all_candidates()` that tags every model with the frameworks it can serve (by `model_source`). The UI now greys out models a framework can't serve. MTPLX's `start_cmd` uses a new `{repo}` placeholder so the CLI loads the right model from its store. |
| **B19** | The `hart` harness was not bundled — it lived at `~/Documents/hart/hart.py`, so a fresh clone from GitHub had no hart and the hart harness failed out of the box. | Fresh installs couldn't run the hart harness without a manual, undocumented copy. | Bundled `hart/` (hart.py + README + docs) into the repo; `hart_path` now defaults to `./hart/hart.py` (relative paths resolve against the repo root). `install.sh` verifies the bundled harness is present. The other agent harnesses (pi/opencode/goose) remain community installs. |
| **B20** | The raw / raw+ harness `finally` block referenced `ntok` unconditionally. When `call_chat` raised (e.g. a 404), the `finally` raised `UnboundLocalError: ntok`, **masking the real error** — the log showed `failed: cannot access local variable 'ntok'` instead of the actual cause. | Real failures were hidden behind a confusing Python error, making diagnosis hard. | The `finally` now only logs the "stream complete" line on success (`_raw_ok` flag); on failure the original exception propagates cleanly to the caller's handler, which logs the real cause. |
| **B21** | Model-id adoption only ran on the *start* path and only auto-adopted when a server exposed exactly **one** model. OMLX serves several models under **cache-style** ids (`org--name`) while the config/picker use **repo-style** (`org/name`), so with multiple models it just warned and every request (warmup + all harnesses) failed with `model not found`. Reused (already-running) servers never got the id aligned at all. | OMLX runs failed across the board (warmup 404, pi/opencode/goose "model not found") whenever the configured id's format didn't match the served id. | New `_adopt_served_model_id()` matches the configured model to a served id by `normalize_key` (so `org/name` ↔ `org--name`), falling back to the single-id case, else warns. It runs on **both** the reuse and start paths, so the warmup and every harness use the id the server actually accepts. |
| **B22** | MLX-Serve's `start_cmd` passed the **repo id** (`org/name`) to `--model`, but mlx-serve 26.x cannot resolve a repo id — it fails with a bare `error: FileNotFound` and never becomes healthy, so **all** MLX-Serve cells are silently missing from a run. (The same model loads fine when given the local HF-cache snapshot path.) | Every MLX-Serve cell in a multi-framework run was lost to an opaque startup failure. | MLX-Serve's `start_cmd` now uses the `{model_path}` placeholder (the local HF-cache snapshot path), which mlx-serve loads correctly. Verified end-to-end: MLX-Serve now starts healthy and completes cells. |

## 2. Redundant / duplicated code — assessed

The original was a single 1,900-line `server.py`. The review looked for duplication and
decided what to keep, merge, or move.

- **One file is fine — but it needed seams.** The backend is cohesive (one process, one
  set of shared state), so splitting it into many modules would add import churn without
  buying isolation. Instead, the one genuinely separable concern — **model discovery** —
  was extracted to `discovery.py`. It has no dependency on server state, is unit-testable
  on its own, and is the piece a user is most likely to extend.
- **Config vs. code.** The original mixed machine-specific values (models, ports,
  commands, the hart path) into the code. All of it now lives in `config.json` with
  built-in defaults in code. This is the single biggest "sturdiness for many users"
  win: the code is now machine-agnostic.
- **The measurement proxy** is optional and self-contained; it's kept but gated behind
  `route_via_proxy` so the default path is the simplest possible (agents → framework
  directly).
- **Frontend.** The single-file `index.html` (no build step) is a feature, not a bug —
  it's trivially deployable and has zero supply-chain surface. The review kept it
  single-file but removed the hardcoded framework/harness lists so it's data-driven.

**What's written best (kept as-is):**
- The **streaming `call_chat`** with TTFT/decode timing — clean and correct.
- The **raw+ continuation** logic (seam detection + de-dup) — subtle and well done.
- The **process-group start/stop** (SIGTERM → SIGKILL, `start_new_session`) — robust.
- The **QA artifact scoring** — a real differentiator; kept intact.
- The **run-history save/delete** with orphaned-output cleanup — careful and correct.

## 3. Sturdiness for hundreds of users / companies

The changes that matter most at scale:

1. **No global state mutation.** Agent harnesses use isolated config dirs (B7). Two
   people (or a benchmark and a human) can use pi/opencode/goose on the same machine
   without clobbering each other.
2. **Config-driven, machine-agnostic.** Nothing machine-specific in code (B8). A new
   user runs `./install.sh`, edits `config.json`, done.
3. **Structured errors.** Every endpoint returns JSON errors (B6, B16). The UI can
   surface them; scripts can rely on them.
4. **Graceful lifecycle.** `SIGTERM` shuts down spawned frameworks (B17); Stop works
   mid-flight; port conflicts are explained.
5. **Local-only by default** with a loud warning on exposure (B17). No telemetry, no
   egress beyond `127.0.0.1` and the local HF cache.
6. **Diagnosability.** Framework output is logged (B14); everything is timestamped in
   `logs/bench.log`.
7. **Cross-platform** RAM detection (B5) and CLI lookup (B9) — works on macOS and Linux.

## 4. End-user quality & benchmarking speed

- **Model picker with live compatibility verdicts** — users stop guessing whether a
  model fits; the UI tells them (ready/tight/too-large/unknown) from real free RAM.
- **Run-elapsed timer** in the status line — long runs feel less like a hang.
- **Settings persistence** (localStorage) — the UI remembers task, prompt, harnesses,
  frameworks, and sampling settings across reloads.
- **Faster idle polling** — `free_ram` caching (B15) removes per-poll subprocess churn.
- **Live framework status** — the top strip reflects real liveness (TTL-cached), so a
  killed server stops showing green.
- **CSV that opens cleanly** in Excel/Sheets (B10).

## 5. What was deliberately *not* changed

- **The six harnesses and their semantics** — raw / raw+ / pi / opencode / goose / hart
  are the product's core value; their behavior is preserved exactly.
- **The QA threshold (90% "usable")** — a product decision, not a bug.
- **One-framework-at-a-time execution** — intentional, so frameworks never compete for
  RAM/GPU and numbers are comparable.
- **Stdlib-only** — no dependencies is a hard constraint that keeps setup trivial and the
  supply-chain surface at zero.

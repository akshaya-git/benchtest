# Parked fixes & changes

Deferred on purpose. The qwen38-27b campaign is FROZEN (results/qwen38-27b/);
items here apply from the flash-next campaign onward unless noted.

## 1. Diagnose opencode exit-1 (empty/no-output)

Two data points from the qwen38-27b campaign: logreport failed on ALL FOUR
frameworks' opencode cells (empty response); markdown failed on mtplx only
(`exited 1: no output`). Never any other harness. Diagnose with a real
model up: reproduce one failing cell with a raised per-call cap, capture
opencode's own stderr, and pin the CLI versions we validate against (pi and
opencode both change event shapes between releases).

## 2. Task-aware time budgets / MTPLX stall watchdog

The cell cap is now configurable (Max cell run time; `cell_timeout_s`).
Still open: MTPLX's ~300s stream-stall watchdog killed silent thinking
phases on markdown (14 stall-breaks; hart churned to the 7200s timeout).
Options: expose `--stream-stall-deadline-s` per task class in the mtplx
start_cmd, or document the interaction in the results. The qwen dataset
keeps those two cells as honest errors.

## 5. Agent-loop KV working set is not modeled by the fit gate

Real incident (2026-10-10, kernel panic `watchdog timeout: no checkins
from watchdogd in 91 seconds`): MLX-VLM ran a 51.8 GB bf16 model — well
within the fit gate — but markdown × pi drove two interleaved ~58k-token
sessions whose KV caches, stacked on the weights, swap-stormed a 128 GB
machine into a hard panic. need_gb = weights × 1.1 + 2 models weights
only. Fix direction: budget KV for the configured context (ctx × per-token
KV bytes) into need_gb, or cap agent-session context (MLX-VLM ctx_tokens
65536 halves the exposure). Interim: documented here; users can lower
ctx_tokens per framework.

## 3. Config/state split (post-release)

`config.json` still persists the full merged tree (notes, start commands,
runtime state) rather than sparse overrides + a separate state file.
Upgrade-hazard: a fixed default does nothing for existing users whose
config already carries the old value. Split into sparse `config.json` +
`state.json` with a `_schema` version and migration.

## 4. Repeat aggregation (if variance is ever wanted)

The dashboard's Repeats setting was removed: it re-ran cells N times but
never aggregated. If variance measurement is wanted later, re-introduce
repeats WITH aggregation (median ± spread per cell); the backend still
accepts `repeats` (clamped 1–5, default 1).

## Resolved (for history)
- mlxlm→mlxvlm id rename — done (config auto-migrated on load).
- Goose/agent token accounting — agent cells without exact stream tokens
  now take the server-side completion delta (tokens_source: server).
- RAM fit drops cached-file credit — available_ram = free + speculative +
  purgeable, conservative per operator decision.
- Dashboard free RAM now = Physical − Memory Used (Activity Monitor
  semantics; cached files count toward free, not used).
- Dashboard free RAM now = Physical − Memory Used (Activity Monitor
  semantics; cached files count toward free, not used).
- Max cell run time configurable on the dashboard; iterations coverage
  documented in CONFIG.md.

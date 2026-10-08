# Parked fixes & changes

Deferred on purpose until both publication campaigns (qwen38-27b, then
flash-next) finish — none of these block the benchmark numbers, and touching
them mid-campaign would invalidate labels or risk the runs. In rough priority
order for the post-publication window.

## 1. Rename the framework id `mlxlm` → `mlxvlm`

The internal framework id is a historical misnomer: the wrapped engine is
**mlx-vlm** (that is what provides MTP support for the Qwen models). A
separate, different project — **MLX-LM** — exists in the ecosystem and does
NOT support MTP for these models; the `mlxlm` id invites users to confuse
the two. The dashboard display name is already "MLX-VLM"; this task renames
the *id* everywhere it appears (~18 sites): `config.json` +
`config.example.json` frameworks key (with a load-time migration for old
configs), `server.py` (engine-kind checks, allowlists, sweep lists),
`mcp/benchtest_mcp.py` (`framework_log` allowlist), `index.html`, README
ports table. Do NOT rename while a campaign is running — the run's cells
are labelled with the id.

## 2. Goose token accounting

Goose cells record ~300 client-side tokens while the server decodes ~13k
(the artifact is written to disk; only the short chat reply is counted), so
client `tps` reads ~1–2 tok/s and misrepresents the harness. Fix: count
Goose (and generally disk-artifact agents) via the server-side token delta
in the cell window, or reliably parse goose's stats block. Until then,
published agent charts must use `server_tgs`, never client `tps`.

## 3. opencode exits 1 with no output (second occurrence: markdown × mtplx)

opencode is the only harness with unexplained hard exits. Data points from
the qwen38-27b campaign: (a) logreport — empty response on ALL FOUR
frameworks' opencode cells; (b) markdown — `opencode exited 1: no output`
on mtplx (other three frameworks finished the same task). Points at an
opencode cap/config/CLI-stability interaction, not a model or framework
issue. Diagnose after the campaign: reproduce one failing cell with a
raised per-call cap, capture opencode's own stderr, and check whether the
CLI version we test against has a known exit-1 mode (the skill says to pin
the CLI versions we validate).

## 4. Task-aware time budgets

The 2-hour per-cell cap is uniform; `mlxlm/pi` on tetris used all 7200 s
(slowest framework × chatty agent). Consider per-harness or per-task time
budgets, or at least surfacing "likely to time out" in the fit/validate
panel. The timeouts themselves are legitimate performance data and stay in
the published numbers.

## 5. MTPLX stream-stall watchdog vs LONG thinking tasks

MTPLX kills any model stream that is silent for ~300s
(`mtplx_stream_stall_break`, its `--stream-stall-deadline-s`). On
`markdown` (LONG, heavy-thinking) the qwen-27B Quality build hit this
repeatedly: 14 stall-breaks in the current MTPLX log. Consequences
observed: `mtplx/hart` churned on stream-recovery until the 7200s cell
timeout (error row), and `mtplx/opencode` exited 1 earlier the same task
(see #3). The tool behaved correctly (honest error rows, artifacts kept) —
this is framework behavior under thinking-heavy tasks, and the published
numbers should say so. Fix after the campaigns: consider a per-task-class
stall deadline (e.g. 900s for LONG tasks) or documenting the interaction;
optionally re-run the two failed cells as a clearly-labeled supplementary
run with a raised deadline. Do NOT change MTPLX settings mid-campaign —
comparability with the 9 completed tasks wins.

## 6. Document `iterations` coverage

`iterations` only exists for opencode and hart (pi and goose do not emit
step counts). Not a bug — a metrics-dictionary note for the published
graphs so readers don't read missing iterations as zero.

## 9. Repeat aggregation (if variance is ever wanted)

The dashboard's Repeats setting was removed: it re-ran cells N times but
never aggregated the results — pure run-time multiplication with no median
or sigma computed. If variance measurement is wanted later, re-introduce
repeats WITH aggregation (median ± spread per cell) and a visible
indicator; the backend still accepts `repeats` (clamped 1–5, default 1).

## 10. RAM free calculation: no cached-file credit

User direction: cached files are fungible and dynamic — their size swings
constantly — so RAM calculations should not lean on them. Current state of
the two numbers: the dashboard **display** (`free_ram`) already excludes
cache (pages free + speculative only; verified against Activity Monitor);
the **fit verdicts** (`available_ram`, used by Validate/model-set fit
gating) DO credit inactive + purgeable pages — the cache — as claimable.
This parked change removes that credit: fit verdicts then judge against
hard-free RAM only, i.e. conservative. Trade-off to accept consciously:
models that would in practice load fine (macOS evicts clean cache on
demand — a 100 GB model on an 81 GB-free machine with 29 GB cached) will
read as tight/wont-fit. If implemented, keep the display/fit split (they
answer different questions) and re-verify the fit tags against a real
model load afterwards.

## 7. Config/state split (post-release)

`config.json` still persists the full merged tree (notes, start commands,
runtime state) rather than sparse overrides + a separate state file.
Upgrade-hazard for future releases: a fixed default does nothing for
existing users whose config already carries the old value. Split into
sparse `config.json` + `state.json` with a `_schema` version and migration.

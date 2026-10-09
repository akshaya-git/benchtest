# Release plan (endgame checklist)

The publication numbers come from exactly two campaigns: **qwen38-27b**
(running now) and **flash-next** (next). Nothing else in `runs/` gets
published. Work in this order — each phase gates the next.

## Phase 0 — during the campaigns (now)

- [x] Monitor periodically; append every new finding to `PARKED.md`
- [ ] Prepare HF packaging (do NOT push yet — numbers must freeze first):
      results Dataset repo (the run JSONs), a static graph Space with a
      prominent GitHub link, and a community-article draft
- [x] When qwen38-27b completes: snapshot its 11 run JSONs →
      `results/qwen38-27b/`; repeat for flash-next after campaign 2

## Phase 1 — final tool updates (after both campaigns)

- [x] `mlxlm`→`mlxvlm` id rename (config auto-migrates on load)
- [x] Goose/agent token accounting via server-side completion deltas
- [x] RAM fit drops cached-file credit; cell run time configurable
- [ ] opencode exit-1 diagnosis (needs a model up — do during flash-next prep)
- [ ] Decide (and apply) any "reduce complexity for community users"
      simplifications discovered on the fresh-user pass below

## Phase 2 — validate

- [ ] `make check` green (syntax + all unit tests)
- [ ] Browser click-through of the dashboard (picker, validate, run, stop,
      comparison, history)
- [ ] One short smoke run end to end against a locally loaded model

## Phase 3 — package & push

- [ ] Final GitHub push (tool + frozen `results/`)
- [ ] README positioning: cite Anubis OSS and apple-silicon-llm-bench as
      the raw-perf references; claim the framework×harness agentic niche

## Phase 4 — fresh-user test (before announcing)

- [ ] Rename the local folder / simulate a new machine; run `install.sh`,
      first-run config seeding, dashboard boot, model pick, one cell —
      exactly as a community user would
- [ ] Fix everything that tripped; re-validate (Phase 2)

## Phase 5 — publish

- [ ] HF dataset (results) + static Space (graphs → GitHub) + community
      article announcing the benchmark

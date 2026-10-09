# qwen38-27b campaign — frozen publication results

- Window: 2026-10-06 23:50 → 2026-10-08 17:09
- Model set: selected — one 27B model per framework:
  - omlx: Jundot--Qwen3.8-27B-oQ8e-mtp
  - mtplx: mtplx-qwen38-27b-optimized-quality
  - mlxlm: Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality
  - mlxserve: 300a4ac6c6058585e80571ff6c910819711843ec
- Tasks (11): logreport, tetris, snake, pong, todo, fib, webdb, bugfix, codereview, markdown, agentconsole
- Cells: 264 (11 tasks × 4 frameworks × 6 harnesses) — 258 done, 6 errors, 0 skipped
- Settings: temperature 0.5, top_p 0.95, max_tokens 65536, cell cap 7200 s, repeats 1

## Error inventory (honest rows, part of the dataset)
- [logreport] omlx/opencode: empty response
- [logreport] mtplx/opencode: empty response
- [logreport] mlxlm/opencode: empty response
- [logreport] mlxserve/opencode: empty response
- [markdown] mtplx/opencode: no output
- [markdown] mtplx/hart: timeout after 7200s

## Id semantics (recorded as-run)
- `mlxlm` = the MLX-VLM framework (port 7003). The framework id was renamed
  to `mlxvlm` in the tool right after this campaign; this dataset keeps the
  id as recorded.
- `mlxserve` rows record the model as a snapshot hash (`300a4ac6…`) — that is
  the served id MLX-Serve itself reports; the weights are
  Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality, the same build MLX-VLM ran.

## Metric notes for graphing
- Agent rows (pi/opencode/goose/hart): chart server-side decode (`server_tgs`) —
  client `tps` includes tool-execution time; goose client tokens undercount
  (artifact written to disk) in THIS campaign (fixed in code afterwards).
- `iterations` exists only for opencode/hart cells.
- MTPLX cells on markdown ran under its 300 s stream-stall watchdog; the two
  mtplx markdown errors trace to it (see repo PARKED.md history).
- Raw rows (Raw (api), raw+) are the clean apples-to-apples comparison.

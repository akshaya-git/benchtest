# benchtest MCP server

Exposes the benchmark to any MCP host (ZCode, Claude Desktop, ...) as tools:
live state, run history, run comparison, bench/framework log tails, artifact
QA re-probing, rescoring, campaign start/stop.

Pure HTTP client of the backend (host/port from config.json) — no benchtest imports, so
it never touches the benchmark's config state.

## Run

The backend must be running (`python3 server.py`). Then register the server
with ZCode — paths are computed from wherever you cloned the repo:

```
python3 scripts/register_mcp.py          # registers + creates mcp/venv if needed
python3 scripts/register_mcp.py --remove # unregister
```

For other MCP hosts, the shape is the same — point the stdio command at
`<repo>/mcp/venv/bin/python` with arg `<repo>/mcp/benchtest_mcp.py`.
The bench URL is read from `config.json` (host + port) at startup;
override with the `BENCHTEST_URL` environment variable.

Tools marked MUTATES in their description change benchmark state
(start_campaign, stop_run, rescore_run); everything else is read-only.


```
cd mcp && venv/bin/python ../tests/mcp_smoke.py        # list tools + live calls
```

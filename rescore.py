#!/usr/bin/env python3
"""Re-score saved run files with the current QA gate (including the runtime
probe) — without re-running a single model cell.

Use this whenever the QA gate has improved after cells were already recorded:
the artifacts on disk are the real deliverables, so their scores can simply be
recomputed. Only rows whose artifact file still exists in outputs/ are touched.
agentconsole rows are probed only if their framework's port is live (the probe
needs a real endpoint to connect to); otherwise they fall back to regex-only.

The updated file keeps the original as <name>.json.bak. Don't run this while a
benchmark run is writing the same run file (i.e. wait for the run to finish).

Usage:
  python3 rescore.py                                  # all runs/*.json
  python3 rescore.py runs/20260929-095401.json ...    # specific files
"""
import glob
import json
import os
import shutil
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import FRAMEWORKS, REQUIREMENTS, TASKS, qa_artifact  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def port_live(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def rescore_file(path):
    with open(path) as f:
        d = json.load(f)
    tid = d.get("task_id")
    reqs = REQUIREMENTS.get(tid or "")
    changed = 0
    for r in d.get("results", []):
        if r.get("status") != "done" or not r.get("output_url"):
            continue
        p = os.path.join(ROOT, "outputs", os.path.basename(r["output_url"]))
        if not os.path.exists(p):
            continue
        probe_ctx = None
        if reqs:
            task_def = next((t for t in TASKS if t["id"] == tid), None)
            probe_ctx = {"task": tid,
                         "static_report": bool(task_def and task_def.get("static_report"))}
            if tid == "agentconsole":
                port = FRAMEWORKS.get(r.get("framework"), {}).get("port")
                if port and port_live(port):
                    probe_ctx["base_url"] = f"http://127.0.0.1:{port}/v1"
                else:
                    probe_ctx = None  # no live endpoint — regex-only for this row
        qa = qa_artifact(p, requirements=reqs, probe_ctx=probe_ctx)
        if not qa:
            continue
        old = r.get("qa_func")
        if old != qa["qa_func"]:
            for k in ("qa_func", "qa_qual", "qa_notes", "usable",
                      "req_missing", "req_rate"):
                if k in qa:
                    r[k] = qa[k]
            changed += 1
            print(f"  {r.get('framework')}/{r.get('harness')}: {old} -> {qa['qa_func']}")
    if changed:
        shutil.copy2(path, path + ".bak")
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"{os.path.basename(path)}: {changed} cell(s) updated (backup: .bak)")
    else:
        print(f"{os.path.basename(path)}: already up to date")


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "runs", "*.json")))
    if not files:
        print("no run files found")
        return
    for f in files:
        if os.path.basename(f).endswith(".bak"):
            continue
        print(os.path.basename(f))
        try:
            rescore_file(f)
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"  SKIPPED: {e}")


if __name__ == "__main__":
    main()

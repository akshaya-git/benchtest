"""Reliability-only QA for benchmark cells — the deterministic half of the
old quality gate, in its own file so the rest of the tool touches it with a
single call (server.py: `row.update(qa.evaluate(row, path))`).

Design rules (learned from the flaky-gate era, 13 regressions):
- NO probes, NO browsers, NO heuristics about how a game "should" look.
  Only structural facts about the cell and the artifact file on disk.
- Every component is transparent: qa_notes names each check and its result
  so a score can be audited without re-running anything.
- Error cells score 0 — an error row is a failed delivery, by definition.

Score components (100 max):
  20  delivered     — artifact file exists and is non-empty
  20  substantial   — size in a plausible range for a working app/report
  20  complete      — HTML closes (</html> present near the end); text
                      artifacts pass when long enough to be whole
  30  self-test     — "N/M passed" line: M * passed/M; 10 (partial) when the
                      artifact carries no self-test line, so tasks without
                      one top out at 80 rather than being penalized blindly
  10  untruncated   — the cell did not hit the token cap
"""
import os
import re

_SELF_TEST = re.compile(r"(\d+)\s*/\s*(\d+)\s*passed", re.I)


def _artifact_checks(path, notes):
    """(delivered, substantial, complete) as 0/1 for the artifact file."""
    if not path or not os.path.isfile(path):
        notes.append("artifact: missing on disk")
        return 0, 0, 0
    try:
        size = os.path.getsize(path)
        with open(path, encoding="utf-8", errors="replace") as f:
            body = f.read(200000)
    except OSError as e:
        notes.append(f"artifact: unreadable ({e})")
        return 0, 0, 0
    if size == 0:
        notes.append("artifact: empty file")
        return 0, 0, 0
    delivered = 1
    is_html = bool(re.search(r"<html[\s>]|<!doctype html", body[:2000], re.I))
    if is_html:
        substantial = 1 if size >= 5000 else 0
        closed = body.rstrip()[-20:].lower().find("</html>") != -1 or \
            "</html>" in body[-2000:].lower()
        complete = 1 if closed else 0
        if not substantial:
            notes.append(f"substantial: html only {size} bytes")
        if not complete:
            notes.append("complete: no closing </html> near end — likely cut off")
    else:
        substantial = 1 if size >= 1000 else 0
        complete = 1 if substantial else 0
        if not substantial:
            notes.append(f"substantial: text only {size} bytes")
    notes.append(f"artifact: {size} bytes, {'html' if is_html else 'text'}")
    return delivered, substantial, complete


def evaluate(row, artifact_path=None):
    """Return {qa_score, qa_notes} for one finished cell. Deterministic:
    same row + same artifact file always yield the same score."""
    notes = []
    if row.get("status") != "done":
        return {"qa_score": 0,
                "qa_notes": "cell error: " + (row.get("error") or "unknown")[:90]}

    delivered, substantial, complete = _artifact_checks(artifact_path, notes)
    score = 20 * delivered + 20 * substantial + 20 * complete

    self_test = 0
    st_note = "self-test: no N/M passed line (+10 partial)"
    if artifact_path and os.path.isfile(artifact_path):
        try:
            with open(artifact_path, encoding="utf-8", errors="replace") as f:
                tail = f.read(200000)
            m = _SELF_TEST.search(tail)
        except OSError:
            m = None
        if m:
            passed, total = int(m.group(1)), int(m.group(2))
            if total > 0:
                self_test = round(30 * passed / total)
                st_note = f"self-test: {passed}/{total} passed (+{self_test})"
            else:
                st_note = "self-test: 0/0 passed line is malformed (+0)"
    score += self_test if self_test else 10
    notes.append(st_note)

    if row.get("truncated"):
        notes.append("truncated: reply hit the token cap (+0)")
    else:
        score += 10
        notes.append("untruncated (+10)")

    return {"qa_score": min(score, 100), "qa_notes": "; ".join(notes)}

#!/usr/bin/env python3
"""Acceptance tests for the organizer. Run with the venv python that has numpy.

    .venv/bin/python tests/test_organize.py

Gates (from the plan):
  calibration  the chosen gate is the largest in the sweep satisfying both floors
  plan         5/5 duplicates, >=8/10 destinations, stubs and PDFs reported
  staleness    warns on a large row delta, refuses on a model/dimension change
  apply/undo   byte-identical round trip, links rewritten, collision refused
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from make_fixture import build, tree_hash  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "organize.py")
ROOT = "/tmp/vault-fixture"

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def run(info: dict, *args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "VAULT_PATH": info["root"],
        "DB_DIR": info["db"],
        "VAULT_ORGANIZE_CACHE": info["cache_dir"],
    }
    return subprocess.run([sys.executable, SCRIPT, *args],
                          capture_output=True, text=True, env=env, timeout=600)


def load_cfg(info: dict) -> dict:
    with open(os.path.join(info["root"], ".vaultindex.json"), encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    info = build(ROOT)
    expected = info["expected"]

    # ── calibration ────────────────────────────────────────────────────────
    out = run(info, "calibrate")
    check("calibrate exits 0", out.returncode == 0, (out.stdout + out.stderr)[-400:])
    cfg = load_cfg(info)
    cal = cfg.get("calibration") or {}
    check("calibration stored", bool(cal.get("vote_gate")), str(cal)[:200])
    sweep = cal.get("sweep", [])
    min_votable = cal.get("min_votable", 10)
    ok_rows = [r for r in sweep if r["agreement"] >= 0.55 and r["votable"] >= min_votable]
    check("chosen gate is the largest satisfying both floors",
          bool(ok_rows) and cal["vote_gate"] == max(r["t"] for r in ok_rows),
          f"chosen {cal.get('vote_gate')} vs {[r['t'] for r in ok_rows]}")
    check("sweep recorded for audit", len(sweep) >= 10, str(len(sweep)))
    check("coverage reported, not gated", 0 < cal.get("coverage", 0) <= 1, str(cal.get("coverage")))

    # ── plan ───────────────────────────────────────────────────────────────
    out = run(info, "plan", "00-inbox", "--json", "--limit", "50")
    check("plan exits 0", out.returncode == 0, (out.stdout + out.stderr)[-400:])
    doc = json.loads(out.stdout)

    dups = {d["path"] for d in doc["duplicates"]}
    wanted_dups = {f"00-inbox/research-{i}-dup.md" for i in range(5)}
    check("duplicate recall 5/5", wanted_dups <= dups, str(sorted(wanted_dups - dups)))
    check("duplicate floor came from labelled pairs",
          cal.get("dup_source") == "labelled", str(cal.get("dup_source")))

    moved_paths = {m["path"] for m in doc["moves"]}
    check("companion .txt never proposed as a move",
          "00-inbox/paper-0.txt" not in moved_paths, str(sorted(moved_paths)))
    check("coverage denominator includes companions",
          doc["coverage"]["indexed_in_scope"] <= doc["coverage"]["text_on_disk"],
          str(doc["coverage"]))

    moved = {m["path"]: m["destination"] for m in doc["moves"]}
    want = {f"00-inbox/mover-{i}.md": ("06-research" if i < 5 else "13-concepts")
            for i in range(10)}
    hit = sum(1 for p, dest in want.items() if moved.get(p) == dest)
    check("destination accuracy >= 8/10", hit >= 8, f"{hit}/10 :: {moved}")

    cov = doc["coverage"]
    check("stubs counted as too short", cov["too_short"] == expected["stubs"], str(cov))
    check("PDFs reported as non-classifiable", cov["non_text"].get(".pdf") == expected["pdfs"], str(cov))
    comp = {c["path"]: c["destination"] for c in doc.get("companions", [])}
    check("PDF classified through its text companion",
          comp.get("00-inbox/paper-0.pdf") == "07-library", str(comp))
    check("no staleness warning on a fresh calibration", not doc["warnings"], str(doc["warnings"]))

    # A dead watcher: notes are newer than the last index write.
    state_file = os.path.join(info["root"], ".vaultindex", "state.json")
    good_state = json.load(open(state_file, encoding="utf-8"))
    stale_state = dict(good_state, updated="2020-01-01T00:00:00")
    with open(state_file, "w", encoding="utf-8") as fh:
        json.dump(stale_state, fh)
    doc_stale = json.loads(run(info, "plan", "00-inbox", "--json").stdout)
    check("warns when notes are newer than the index",
          any("watcher may be down" in w for w in doc_stale["warnings"]),
          str(doc_stale["warnings"]))
    with open(state_file, "w", encoding="utf-8") as fh:
        json.dump(good_state, fh)
    check("nothing unindexed except stubs/PDFs", cov["unindexed"] == 0, str(cov))

    # ── staleness behaviour ────────────────────────────────────────────────
    stale = load_cfg(info)
    stale["row_delta_trigger"] = 10
    stale["calibration"]["rows_at_calibration"] = cal["rows_at_calibration"] - 41
    with open(os.path.join(info["root"], ".vaultindex.json"), "w", encoding="utf-8") as fh:
        json.dump(stale, fh, indent=2)
    doc2 = json.loads(run(info, "plan", "00-inbox", "--json").stdout)
    check("warns when the row delta exceeds the trigger",
          any("calibrate" in w for w in doc2["warnings"]), str(doc2["warnings"]))

    stale["calibration"]["rows_at_calibration"] = cal["rows_at_calibration"]
    stale.pop("row_delta_trigger", None)
    stale["model"]["dims"] = 99
    with open(os.path.join(info["root"], ".vaultindex.json"), "w", encoding="utf-8") as fh:
        json.dump(stale, fh, indent=2)
    out = run(info, "plan", "00-inbox", "--json")
    check("refuses on a dimension change",
          out.returncode != 0 and "dimensions" in (out.stdout + out.stderr),
          (out.stdout + out.stderr)[-200:])

    stale["model"]["dims"] = cfg["model"]["dims"]
    with open(os.path.join(info["root"], ".vaultindex.json"), "w", encoding="utf-8") as fh:
        json.dump(stale, fh, indent=2)

    # A recorded index path that does not exist here must say so, not crash.
    # DB_DIR is deliberately unset so the config field is what gets consulted.
    moved = load_cfg(info)
    moved["db"] = "/tmp/index-that-moved"
    with open(os.path.join(info["root"], ".vaultindex.json"), "w", encoding="utf-8") as fh:
        json.dump(moved, fh, indent=2)
    env = {k: v for k, v in os.environ.items() if k not in ("DB_DIR",)}
    env["VAULT_PATH"] = info["root"]
    env["VAULT_ORGANIZE_CACHE"] = "/tmp/empty-cache-for-test"
    out = subprocess.run([sys.executable, SCRIPT, "plan", "00-inbox", "--json"],
                         capture_output=True, text=True, env=env, timeout=600)
    check("missing recorded index path is named",
          out.returncode != 0 and "/tmp/index-that-moved" in (out.stdout + out.stderr),
          f"rc={out.returncode} {(out.stdout + out.stderr)[-200:]}")

    # And the guard must actually be reached with DB_DIR unset.
    moved.pop("db")
    with open(os.path.join(info["root"], ".vaultindex.json"), "w", encoding="utf-8") as fh:
        json.dump(moved, fh, indent=2)

    # Same dimensions, different model: distances are not comparable.
    same_dims = load_cfg(info)
    same_dims["model"]["name"] = "some-other-embedder"
    with open(os.path.join(info["root"], ".vaultindex.json"), "w", encoding="utf-8") as fh:
        json.dump(same_dims, fh, indent=2)
    out = run(info, "plan", "00-inbox", "--json")
    check("refuses when the model name changed at equal dimensions",
          out.returncode != 0 and "not comparable" in (out.stdout + out.stderr),
          (out.stdout + out.stderr)[-200:])

    same_dims["model"]["name"] = cfg["model"]["name"]
    with open(os.path.join(info["root"], ".vaultindex.json"), "w", encoding="utf-8") as fh:
        json.dump(same_dims, fh, indent=2)

    # ── CLI surface ────────────────────────────────────────────────────────
    out = run(info, "00-inbox", "--limit", "1")
    check("bare folder means plan (default verb)", out.returncode == 0,
          (out.stdout + out.stderr)[-200:])

    out = run(info, "undo", "nope")
    check("undo of a missing token exits 2 with a message",
          out.returncode == 2 and "no manifest" in out.stderr,
          f"rc={out.returncode} {out.stderr[-160:]}")

    out = run(info, "plan", "--like", "00-inbox/mover-0.md", "--limit", "3", "--json")
    check("plan --like returns neighbours, not itself",
          out.returncode == 0
          and all(n["path"] != "00-inbox/mover-0.md"
                  for n in json.loads(out.stdout)["neighbours"]),
          (out.stdout + out.stderr)[-200:])

    # ── apply / undo ───────────────────────────────────────────────────────
    before = tree_hash(info["root"])
    plan_path = "/tmp/organize-plan.json"
    with open(plan_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)

    # Apply with reindex enabled: the fixture's DB_DIR is threaded through the
    # env, which is exactly the path that used to fall back to a phantom index.
    out = run(info, "apply", plan_path)
    check("apply exits 0", out.returncode == 0, (out.stdout + out.stderr)[-400:])

    hub = open(os.path.join(ROOT, "13-concepts/hub.md"), encoding="utf-8").read()
    check("path-qualified link rewritten", "[[06-research/mover-0]]" in hub, hub)
    check("aliased link keeps its alias", "[[06-research/mover-1|alias text]]" in hub, hub)
    check("embed rewritten", "![[06-research/mover-2]]" in hub, hub)
    check("basename link untouched", "[[mover-3]]" in hub, hub)

    moved_ok = os.path.exists(os.path.join(ROOT, "06-research/mover-0.md"))
    check("file actually moved", moved_ok)

    manifest_dir = os.path.join(ROOT, ".vaultindex", "applied")
    tokens = sorted(os.listdir(manifest_dir))
    check("one manifest written", len(tokens) == 1, str(tokens))
    manifest = json.load(open(os.path.join(manifest_dir, tokens[0]), encoding="utf-8"))
    check("manifest is marked complete", manifest.get("state") == "complete",
          str(manifest.get("state")))
    check("manifest records the planned count", manifest.get("planned") == len(doc["moves"]),
          f"{manifest.get('planned')} vs {len(doc['moves'])}")
    check("reindex recorded its db dir",
          "db_dir" in (manifest.get("reindex") or {}), str(manifest.get("reindex")))
    refused = {r["path"] for r in manifest["refused"]}
    check("basename collision refused", expected["collision"] in refused, str(refused))

    out = run(info, "undo", tokens[0].replace(".json", ""), "--no-index")
    check("undo exits 0", out.returncode == 0, (out.stdout + out.stderr)[-300:])
    after = tree_hash(ROOT)
    check("undo restores a byte-identical tree", before == after,
          str(sorted(set(before) ^ set(after))[:5]))

    # A basename shared across folders: destination does not exist, so only the
    # ambiguity rule can catch it. [[collide]] would re-point silently otherwise.
    ambiguous = dict(doc)
    ambiguous["moves"] = [{
        "path": "00-inbox/collide.md",
        "destination": "07-library",
        "target_dir": "07-library",
        "confidence": 0.9,
        "votes": 5,
        "neighbours": [],
    }]
    amb_path = "/tmp/organize-ambiguous.json"
    with open(amb_path, "w", encoding="utf-8") as fh:
        json.dump(ambiguous, fh)
    out = run(info, "apply", amb_path, "--no-index")
    check("ambiguous basename is refused",
          out.returncode == 0 and "ambiguous basename" in out.stdout,
          (out.stdout + out.stderr)[-300:])
    check("refused note did not move",
          os.path.exists(os.path.join(ROOT, "00-inbox/collide.md")))

    # The manifest must exist even when nothing moved, or a partially refused
    # plan leaves no record of what it decided.
    tokens = sorted(os.listdir(manifest_dir))
    refused_manifest = json.load(open(os.path.join(manifest_dir, tokens[-1]),
                                      encoding="utf-8"))
    check("refused plan still writes a complete manifest",
          refused_manifest.get("state") == "complete"
          and refused_manifest.get("applied") == []
          and len(refused_manifest.get("refused", [])) == 1,
          f"state={refused_manifest.get('state')} "
          f"applied={refused_manifest.get('applied')} "
          f"refused={refused_manifest.get('refused')}")

    out = run(info, "apply", amb_path, "--no-index", "--allow-ambiguous-basename")
    check("opt-out flag allows it",
          out.returncode == 0 and "applied 1 moves" in out.stdout,
          (out.stdout + out.stderr)[-300:])

    print("\nFAILURES:", failures or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if os.path.isdir(ROOT) and os.environ.get("KEEP_FIXTURE") is None:
            shutil.rmtree(ROOT, ignore_errors=True)

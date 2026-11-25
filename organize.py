#!/usr/bin/env python3
"""vault organize — classify notes by their nearest neighbours, plan moves, apply them.

One source of truth for the organizing behaviour. `vault_db.py` owns vectors and
index state; this module owns classification, planning, applying and undoing.

Design rules, all measured rather than chosen:

  classification  a note's destination is the modal top-level folder among its
                  nearest neighbours *outside* its own folder. Same-folder
                  neighbours never vote, and the queue/archive folders never win.
  vote gate       `calibration.vote_gate` is the largest distance at which the
                  median note still has >=2 legal neighbours and the modal
                  destination keeps >=55% agreement. Recompute with `calibrate`.
  dup floor       from labelled duplicate pairs if present, else a gap in the
                  within-folder distance histogram. Never invented.
  coverage        files the index cannot see (PDF, audio, images) are reported,
                  never silently dropped.

Verbs:

    organize.py calibrate [--vault V] [--db D]
    organize.py plan <folder> [--json]
    organize.py apply <plan.json> [--no-index]
    organize.py undo <token>

Nothing is moved without an explicit `apply`, and every apply writes a manifest
that `undo` reverses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from typing import Any

import numpy as np

TABLE = "notes"
CONFIG_NAME = ".vaultindex.json"
STATE_DIR = ".vaultindex"
STATE_NAME = "state.json"
APPLIED_DIR = "applied"

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_FOLDER = "00-inbox"
DEFAULT_NEVER = ("00-inbox", "99-archive")

# Calibration: agreement floor and coverage floor for the vote gate. These two
# numbers are the definition of the gate, not a tuning preference.
AGREEMENT_FLOOR = 0.55
COVERAGE_FLOOR = 0.50
ROW_DELTA_TRIGGER = 500


# ── paths and config ────────────────────────────────────────────────────────


def vault_root(explicit: str | None = None) -> str:
    return os.path.abspath(
        os.path.expanduser(
            explicit or os.environ.get("VAULT_PATH") or os.getcwd()
        )
    )


def db_dir_for(vault: str, explicit: str | None = None) -> str:
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    env = os.environ.get("DB_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    # Recorded at calibrate time, so `cd othervault && vault organize` works
    # without exporting anything (that vault's index may live anywhere).
    recorded = load_config(vault).get("db")
    if recorded:
        return os.path.abspath(os.path.expanduser(str(recorded)))
    return os.path.join(vault, STATE_DIR, "lancedb")


def config_path(vault: str) -> str:
    return os.path.join(vault, CONFIG_NAME)


def load_config(vault: str) -> dict:
    try:
        with open(config_path(vault), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        cfg = {}
    cfg.setdefault("version", 1)
    cfg.setdefault("default_folder", DEFAULT_FOLDER)
    cfg.setdefault("never_dest", list(DEFAULT_NEVER))
    cfg.setdefault("model", {"name": DEFAULT_MODEL, "dims": 0})
    cfg.setdefault("calibration", None)
    return cfg


def save_config(vault: str, cfg: dict) -> None:
    tmp = f"{config_path(vault)}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, config_path(vault))


# ── index state and vector cache ────────────────────────────────────────────


def state_path(db_dir: str) -> str:
    return os.path.join(os.path.dirname(db_dir.rstrip("/")), STATE_NAME)


def read_state(db_dir: str) -> dict | None:
    for path in (state_path(db_dir), os.path.join(db_dir, STATE_NAME)):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            continue
    return None


def write_state(db_dir: str, rows: int, model: str, dims: int) -> None:
    """Called by the indexer after every successful write. Increments the version."""
    prev = read_state(db_dir) or {}
    state = {
        "rows": int(rows),
        "version": int(prev.get("version", 0)) + 1,
        "model": model,
        "dims": int(dims),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    target = state_path(db_dir)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = f"{target}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, target)


def cache_path(db_dir: str) -> str:
    base = os.environ.get(
        "VAULT_ORGANIZE_CACHE", os.path.expanduser("~/.cache/vault-organize")
    )
    os.makedirs(base, exist_ok=True)
    key = hashlib.sha1(os.path.abspath(db_dir).encode()).hexdigest()[:10]
    return os.path.join(base, f"notes-{key}.npz")


def _unit(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def load_index(db_dir: str, vault: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (paths, unit_vectors, source-description).

    Validation ladder:
      1. state.json version matches the cache  -> serve (no lancedb import)
      2. state.json missing                    -> fall back to mtime + row count
      3. otherwise                             -> rebuild from LanceDB
    """
    cache = cache_path(db_dir)
    state = read_state(db_dir)

    z = None
    if os.path.exists(cache):
        try:
            z = np.load(cache, allow_pickle=False)
            needs = ("paths", "vectors", "version", "rows", "written_at")
            if any(k not in z.files for k in needs):
                z = None
        except Exception:
            z = None

    if z is not None:
        age = max(0, int(time.time()) - int(z["written_at"]))
        if state is not None and int(z["version"]) == int(state.get("version", -1)):
            return z["paths"], z["vectors"], f"cache v{int(z['version'])}, {_age(age)} old"
        if state is None:
            return z["paths"], z["vectors"], f"cache (no state), {_age(age)} old"

    # No usable cache: the table must exist, and saying so beats an import trace.
    if not os.path.isdir(os.path.join(db_dir, f"{TABLE}.lance")):
        raise SystemExit(
            f"no index at {db_dir} (expected {TABLE}.lance/) and no usable cache.\n"
            "  The path is machine-local: pass --db, set DB_DIR, or re-run "
            "`vault organize calibrate` on this machine"
        )

    import lancedb  # heavy; only on a cold cache

    tbl = lancedb.connect(db_dir).open_table(TABLE).to_arrow()
    paths = np.array(tbl.column("path").to_pylist())
    flat = tbl.column("vector").combine_chunks().flatten()
    vecs = _unit(np.asarray(flat).reshape(tbl.num_rows, -1).astype(np.float32, copy=False))

    version = int((state or {}).get("version", 0))
    tmp = f"{cache}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(fh, paths=paths, vectors=vecs,
                 version=np.int64(version), rows=np.int64(tbl.num_rows),
                 written_at=np.int64(time.time()))
    os.replace(tmp, cache)
    return paths, vecs, f"rebuilt v{version}"


def _age(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    if seconds < 172800:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


# ── corpus helpers ──────────────────────────────────────────────────────────


def top_level(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else path


def dirname(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def in_scope(path: np.ndarray, folder: str) -> np.ndarray:
    prefix = folder.strip("/")
    if prefix in ("", "."):
        return np.ones(len(path), dtype=bool)
    return np.char.startswith(path, prefix + "/")


DOCUMENT_COMPANIONS = {".pdf", ".docx", ".epub", ".djvu"}


def scan_files(vault: str, folder: str) -> dict:
    """What is on disk under `folder`: markdown, PDF companions, other binaries."""
    rel = folder.strip("/")
    base = os.path.join(vault, rel) if rel not in ("", ".") else vault
    out: dict[str, Any] = {"md": [], "companion_txt": [], "other": Counter(),
                           "documents": [], "newest_mtime": 0.0}
    if not os.path.isdir(base):
        return out
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        pdf_stems = {
            os.path.splitext(f)[0].lower() for f in filenames if f.lower().endswith(".pdf")
        }
        for name in filenames:
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            relpath = os.path.relpath(os.path.join(dirpath, name), vault).replace(os.sep, "/")
            if ext == ".md":
                out["md"].append(relpath)
                try:
                    mtime = os.stat(os.path.join(dirpath, name)).st_mtime
                    if mtime > out["newest_mtime"]:
                        out["newest_mtime"] = mtime
                except OSError:
                    pass
            elif ext == ".txt" and os.path.splitext(name)[0].lower() in pdf_stems:
                out["companion_txt"].append(relpath)
            else:
                out["other"][ext or "(none)"] += 1
                if ext in DOCUMENT_COMPANIONS:
                    out["documents"].append(relpath)
    return out


def stripped_len(vault: str, path: str) -> int:
    """Body length as the indexer sees it (frontmatter removed)."""
    try:
        with open(os.path.join(vault, path), encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except OSError:
        return 0
    if content.startswith("---"):
        lines = content.split("\n")
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                content = "\n".join(lines[i + 1 :])
                break
    return len(content.strip())


# ── calibration ─────────────────────────────────────────────────────────────


def _gate_stats(vecs: np.ndarray, idx: np.ndarray, top: np.ndarray,
                never: set[str], gate: float) -> tuple[int, float, int]:
    """(median legal votes, modal-agreement median, votable count) at one threshold.

    "Legal" mirrors the vote rule exactly: outside the note's own top-level
    folder, and never a queue/archive folder. `votable` counts notes with at
    least two such neighbours — the only ones whose agreement means anything.
    """
    votes, shares = [], []
    for i in idx:
        d = 1.0 - vecs @ vecs[i]
        legal = np.nonzero(
            (d <= gate) & (d > 0) & (top != top[i]) & ~np.isin(top, list(never))
        )[0]
        votes.append(len(legal))
        if len(legal) >= 2:
            shares.append(Counter(top[j] for j in legal).most_common(1)[0][1] / len(legal))
    votable = sum(1 for v in votes if v >= 2)
    return int(np.median(votes)), (float(np.median(shares)) if shares else 0.0), votable


def calibrate(paths: np.ndarray, vecs: np.ndarray, never: set[str],
              sample: int = 600) -> dict:
    """Pick the vote gate and duplicate floor from the corpus itself.

    The gate is the largest distance at which notes that *can* vote still reach
    the agreement floor. Coverage is reported, not gated: in a well-sorted vault
    most notes are already home and have no legal cross-folder neighbour, so a
    corpus-wide coverage floor would refuse to calibrate a healthy vault.
    """
    top = np.array([top_level(p) for p in paths])
    n = len(paths)
    if n < 40:
        raise SystemExit(f"cannot calibrate on {n} notes: need at least 40")
    rng = np.random.default_rng(0)
    idx = rng.choice(n, min(sample, n), replace=False)
    min_votable = max(10, int(0.05 * len(idx)))

    rows = []
    for gate in np.arange(0.16, 0.62, 0.02):
        med_votes, agreement, votable = _gate_stats(vecs, idx, top, never, float(gate))
        rows.append({
            "t": round(float(gate), 2),
            "median_votes": med_votes,
            "agreement": round(agreement, 3),
            "votable": votable,
            "coverage": round(votable / len(idx), 3),
        })

    chosen = None
    for row in rows:  # ascending t: keep the last that still satisfies both floors
        if row["agreement"] >= AGREEMENT_FLOOR and row["votable"] >= min_votable:
            chosen = row
    if chosen is None:
        raise SystemExit(
            f"cannot calibrate: no threshold keeps agreement >= {AGREEMENT_FLOOR} "
            f"with at least {min_votable} votable notes.\n"
            "  This corpus has no folder structure a nearest-neighbour vote can use.\n"
            "  Sweep:\n" + "\n".join(f"    {r}" for r in rows)
        )

    floor, source = dup_floor(paths, vecs, top)
    return {
        "vote_gate": chosen["t"],
        "dup_floor": floor,
        "dup_source": source,
        "rows_at_calibration": n,
        "samples": int(len(idx)),
        "min_votable": min_votable,
        "calibrated_at": time.strftime("%Y-%m-%d"),
        "agreement": chosen["agreement"],
        "coverage": chosen["coverage"],
        "sweep": rows,
    }


def dup_floor(paths: np.ndarray, vecs: np.ndarray, top: np.ndarray) -> tuple[float, str]:
    """(duplicate floor, provenance) — labelled pairs first, then a gap."""
    index = {str(p): i for i, p in enumerate(paths)}
    by_name: dict[str, int] = {}
    for i, p in enumerate(paths):
        by_name.setdefault(str(p).rsplit("/", 1)[-1].lower(), i)

    dists = []
    for i, raw in enumerate(paths):
        p = str(raw)
        if not p.endswith("-dup.md"):
            continue
        twin = index.get(p[: -len("-dup.md")] + ".md")
        if twin is None:  # the twin often lives in another folder
            base = p.rsplit("/", 1)[-1][: -len("-dup.md")].lower() + ".md"
            twin = by_name.get(base)
        if twin is not None and twin != i:
            dists.append(1.0 - float(vecs[i] @ vecs[twin]))
    if len(dists) >= 5:
        dists.sort()
        # p90 of labelled pairs, small margin, never above the measured
        # duplicate/near-miss boundary (0.08): real copies sit at <=0.054 while
        # same-topic notes start around 0.10, and a false duplicate hides a note
        # from the move planner entirely.
        p90 = float(np.percentile(dists, 90))
        return round(min(0.08, max(0.03, p90 * 1.2)), 3), "labelled"

    # No labels: look for a gap in the same-folder distance distribution, and
    # never exceed the measured duplicate boundary even if no gap exists.
    rng = np.random.default_rng(1)
    sample = rng.choice(len(paths), min(300, len(paths)), replace=False)
    near = []
    for i in sample:
        d = 1.0 - vecs @ vecs[i]
        d[i] = 9.0
        same = (top == top[i])
        if same.any():
            near.append(float(d[same].min()))
    near.sort()
    for a, b in zip(near, near[1:]):
        if b - a > 0.15 and a < 0.3:
            return round(min(0.08, a + (b - a) / 2), 3), "gap"
    return 0.08, "default"


# ── planning ────────────────────────────────────────────────────────────────


def plan(paths: np.ndarray, vecs: np.ndarray, cfg: dict, vault: str, scope: str,
         db_dir: str | None = None) -> dict:
    cal = cfg.get("calibration")
    if not cal:
        raise SystemExit("no calibration: run `vault organize calibrate` first")
    gate = float(cal["vote_gate"])
    dup = float(cal["dup_floor"])
    never = set(cfg["never_dest"])

    warnings = []
    rows_now = int(len(paths))
    rows_cal = int(cal.get("rows_at_calibration", 0) or 0)
    trigger = int(cfg.get("row_delta_trigger", ROW_DELTA_TRIGGER))
    if abs(rows_now - rows_cal) > trigger:
        warnings.append(
            f"calibration covers {rows_cal} rows, index has {rows_now} "
            f"({rows_now - rows_cal:+d}): run `vault organize calibrate`"
        )
    dims_cal = int(cfg.get("model", {}).get("dims", 0) or 0)
    dims_now = int(vecs.shape[1])
    if dims_cal and dims_now != dims_cal:
        raise SystemExit(
            f"index dimensions {dims_now} != calibration dimensions {dims_cal}; "
            "the model changed, so the old calibration is void — re-run calibrate"
        )
    # Same dimensions is not the same model: two 768-d embedders produce
    # incomparable distances, so compare the name the indexer recorded.
    state = read_state(db_dir or db_dir_for(vault))
    cal_model = str(cfg.get("model", {}).get("name") or "")
    idx_model = str((state or {}).get("model") or "")
    if cal_model and idx_model and cal_model != idx_model:
        raise SystemExit(
            f"index was built with model {idx_model!r}, calibration with {cal_model!r};\n"
            "  distances are not comparable — re-run calibrate"
        )

    top = np.array([top_level(p) for p in paths])
    dirs = np.array([dirname(p) for p in paths])
    idx = np.nonzero(in_scope(paths, scope))[0]

    disk = scan_files(vault, scope)
    indexed = set(paths.tolist())

    # A dead watcher is invisible otherwise: notes keep changing, the index does
    # not, and every plan is confidently wrong. State is written on each index
    # write, so compare it with the newest note in scope.
    if state:
        try:
            updated = time.mktime(
                time.strptime(str(state.get("updated", "")), "%Y-%m-%dT%H:%M:%S")
            )
        except Exception:
            updated = 0.0
        lag = float(disk["newest_mtime"]) - float(updated)
        if updated and lag > 3600:
            warnings.append(
                f"newest note in scope is {lag / 3600:.1f}h newer than the last index "
                f"write ({state.get('updated')}); the watcher may be down — run `vault sync`"
            )

    # A .txt companion is indexed only so its PDF becomes classifiable. It must
    # never be proposed as a move: that would separate it from the document.
    paired = set(disk["companion_txt"])

    # Ambiguity is a vault-wide property: [[name]] resolves by basename anywhere,
    # so a move can be refused at apply time even when nothing nearby shares it.
    # Label it here, or the plan you triage is not the plan that runs.
    vault_names: Counter[str] = Counter()
    for dirpath, dirnames, filenames in os.walk(vault):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.endswith(".md"):
                vault_names[name.lower()] += 1

    duplicates, moves, unclear = [], [], []
    for i in idx:
        if str(paths[i]) in paired:
            continue
        d = 1.0 - vecs @ vecs[i]
        d[i] = 9.0
        best = int(np.argmin(d))
        if d[best] <= dup:
            duplicates.append({
                "path": str(paths[i]), "duplicate_of": str(paths[best]),
                "distance": round(max(0.0, float(d[best])), 4),
            })
            continue
        legal = np.nonzero((d <= gate) & (top != top[i])
                           & ~np.isin(top, list(never)))[0]
        if len(legal) < 2:
            unclear.append({"path": str(paths[i]),
                            "nearest": round(max(0.0, float(d[best])), 4)})
            continue
        dest, agree = Counter(top[j] for j in legal).most_common(1)[0]
        target = Counter(dirs[j] for j in legal if top[j] == dest).most_common(1)[0][0]
        moves.append({
            "path": str(paths[i]), "destination": str(dest), "target_dir": str(target),
            "confidence": round(agree / len(legal), 3), "votes": int(len(legal)),
            "neighbours": [str(paths[j]) for j in legal[:3]],
            "ambiguous_basename": vault_names.get(
                os.path.basename(str(paths[i])).lower(), 0) > 1,
        })

    # when one is indexed, and emit no destination at all when the sibling has no
    # legal neighbours. Never emit a queue/archive folder as a destination.
    position = {str(p): k for k, p in enumerate(paths)}
    companions = []
    for doc in disk["documents"]:
        stem = doc.rsplit(".", 1)[0]
        for cand in (stem + ".txt", stem + ".md"):
            k = position.get(cand)
            if k is None:
                continue
            d = 1.0 - vecs @ vecs[k]
            d[k] = 9.0
            legal = np.nonzero(
                (d <= gate) & (top != top[k]) & ~np.isin(top, list(never))
            )[0]
            dest = (str(Counter(top[j] for j in legal).most_common(1)[0][0])
                    if len(legal) else None)
            companions.append({"path": doc, "via": cand, "destination": dest,
                               "votes": int(len(legal)),
                               "confidence": (round(Counter(top[j] for j in legal)
                                                    .most_common(1)[0][1] / len(legal), 3)
                                              if len(legal) else 0.0)})
            break

    text_on_disk = disk["md"] + disk["companion_txt"]
    missing = [f for f in text_on_disk if f not in indexed]
    short = [f for f in missing if f.endswith(".md") and stripped_len(vault, f) < 50]

    return {
        "scope": scope,
        "gate": gate,
        "dup_floor": dup,
        "dup_source": cal.get("dup_source"),
        "calibrated_at": cal.get("calibrated_at"),
        "warnings": warnings,
        "duplicates": sorted(duplicates, key=lambda x: x["distance"]),
        "moves": sorted(moves, key=lambda x: -x["confidence"]),
        "unclassified": unclear,
        "companions": companions,
        "coverage": {
            "indexed_in_scope": int(len(idx)),
            "text_on_disk": len(text_on_disk),
            "markdown_on_disk": len(disk["md"]),
            "companion_txt_on_disk": len(disk["companion_txt"]),
            "too_short": len(short),
            "unindexed": len(missing) - len(short),
            "non_text": dict(disk["other"]),
        },
    }


# ── apply / undo ────────────────────────────────────────────────────────────


def _link_patterns(old_rel: str, new_rel: str) -> list[tuple[re.Pattern, str]]:
    """Regexes for path-qualified links only. A bare [[basename]] is left alone."""
    out = []
    for old, new in (
        (old_rel, new_rel),
        (old_rel[:-3] if old_rel.endswith(".md") else old_rel,
         new_rel[:-3] if new_rel.endswith(".md") else new_rel),
    ):
        # The reference must end at a valid link boundary, otherwise
        # [[dir/a]] would corrupt [[dir/ab]].
        pat = re.compile(r"(\[\[|!\[\[)" + re.escape(old) + r"(?=[|\]#])")
        out.append((pat, r"\g<1>" + new))
    return out


def _rewrite_links(vault: str, mapping: list[tuple[str, str]]) -> int:
    """Rewrite path-qualified wikilinks and embeds for many moves in one pass.

    One tree walk per call regardless of how many files moved: applying 140 moves
    must not read every note 140 times.
    """
    patterns: list[tuple[re.Pattern, str]] = []
    for old_rel, new_rel in mapping:
        patterns.extend(_link_patterns(old_rel, new_rel))
    if not patterns:
        return 0
    changed = 0
    for dirpath, dirnames, filenames in os.walk(vault):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if not name.endswith(".md"):
                continue
            full = os.path.join(dirpath, name)
            try:
                with open(full, encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                continue
            new_text = text
            for pat, repl in patterns:
                new_text = pat.sub(repl, new_text)
            if new_text != text:
                with open(full, "w", encoding="utf-8") as fh:
                    fh.write(new_text)
                changed += 1
    return changed


def _git(vault: str, *args: str) -> bool:
    try:
        return subprocess.run(["git", "-C", vault, *args],
                              capture_output=True, text=True).returncode == 0
    except OSError:
        return False


def _basename_counts(vault: str) -> Counter[str]:
    """How many markdown files share each lowercase basename.

    A wikilink written as [[name]] resolves by basename, so moving a file whose
    name is not unique can silently re-point every such link to a different
    note. That is damage with no error and no manifest entry, so apply refuses.
    """
    counts: Counter[str] = Counter()
    for dirpath, dirnames, filenames in os.walk(vault):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.endswith(".md"):
                counts[name.lower()] += 1
    return counts


def apply_plan(vault: str, plan_doc: dict, reindex: bool = True,
               db_dir: str | None = None,
               allow_ambiguous_basename: bool = False) -> dict:
    """Move, relink, reindex. The manifest is written BEFORE the first move and
    rewritten after each one, so an interrupt or an exception can never leave
    moved files without an undo record — the one irreversible failure here."""
    moves = plan_doc.get("moves", [])
    applied, refused, pairs = [], [], []

    stamp = time.strftime("%Y%m%dT%H%M%S")
    token = f"apply-{stamp}"
    outdir = os.path.join(vault, STATE_DIR, APPLIED_DIR)
    os.makedirs(outdir, exist_ok=True)
    manifest_path = os.path.join(outdir, f"{token}.json")

    manifest: dict = {
        "token": token,
        "created": stamp,
        "vault": vault,
        "state": "in_progress",
        "planned": len(moves),
        "applied": applied,
        "refused": refused,
        "links_files_updated": 0,
        "scope": plan_doc.get("scope"),
    }

    def save() -> None:
        tmp = f"{manifest_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, manifest_path)

    save()  # an empty, valid manifest exists before anything moves
    names = _basename_counts(vault)
    try:
        for move in moves:
            src = move["path"]
            target_dir = move.get("target_dir") or move["destination"]
            dst = os.path.join(target_dir, os.path.basename(src)).replace(os.sep, "/")
            if not os.path.exists(os.path.join(vault, src)):
                refused.append({"path": src, "reason": "source missing"})
                save()
                continue
            if os.path.exists(os.path.join(vault, dst)):
                refused.append({"path": src, "reason": f"destination exists: {dst}"})
                save()
                continue
            shared = names.get(os.path.basename(src).lower(), 0)
            if shared > 1 and not allow_ambiguous_basename:
                refused.append({
                    "path": src,
                    "reason": (f"ambiguous basename: {shared} notes are named "
                               f"{os.path.basename(src)!r}, so [[name]] links would "
                               "silently re-point (use --allow-ambiguous-basename)"),
                })
                save()
                continue
            os.makedirs(os.path.join(vault, target_dir), exist_ok=True)
            if not _git(vault, "mv", src, dst):
                shutil.move(os.path.join(vault, src), os.path.join(vault, dst))
            applied.append({"from": src, "to": dst})
            pairs.append((src, dst))
            save()  # record each move as it happens, not after the loop

        if pairs:
            manifest["links_files_updated"] = _rewrite_links(vault, pairs)
            save()
        manifest["state"] = "complete"
        save()
    except BaseException:
        # KeyboardInterrupt included: the partial manifest is already on disk.
        manifest["state"] = "interrupted"
        save()
        raise

    if reindex and applied:
        manifest["reindex"] = _reindex(vault, db_dir)
        save()
    return manifest


def undo_apply(vault: str, token: str, reindex: bool = True,
               db_dir: str | None = None) -> dict:
    name = token if token.endswith(".json") else f"{token}.json"
    path = os.path.join(vault, STATE_DIR, APPLIED_DIR, name)
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    reverted, pairs = [], []
    for item in reversed(manifest.get("applied", [])):
        src, dst = item["from"], item["to"]
        if not os.path.exists(os.path.join(vault, dst)):
            continue
        os.makedirs(os.path.join(vault, os.path.dirname(src) or "."), exist_ok=True)
        if not _git(vault, "mv", dst, src):
            shutil.move(os.path.join(vault, dst), os.path.join(vault, src))
        reverted.append(item)
        pairs.append((dst, src))

    if pairs:
        _rewrite_links(vault, pairs)
        # Symmetric with apply: without this, undo leaves the index pointing at
        # paths that no longer exist and only the watcher repairs it.
        if reindex:
            _reindex(vault, db_dir)
    return {"token": manifest.get("token"), "reverted": reverted}


def _reindex(vault: str, db_dir: str | None = None) -> dict:
    """Ask the indexer to reconcile paths. Best effort, but never silent.

    DB_DIR must be explicit: without it the child falls back to the engine repo's
    sibling directory and cheerfully reindexes a phantom table while the real one
    stays stale.
    """
    db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vault_db.py")
    env = dict(os.environ, VAULT_PATH=vault)
    if db_dir:
        env["DB_DIR"] = os.path.abspath(db_dir)
    proc = subprocess.run([sys.executable, db, "sync", "--changed"],
                          capture_output=True, text=True, env=env)
    return {"ran": True, "returncode": proc.returncode,
            "db_dir": env.get("DB_DIR", "<inherited>"),
            "stderr": (proc.stderr or "")[-400:] if proc.returncode else ""}


# ── CLI ─────────────────────────────────────────────────────────────────────


def _banner(source: str, load_s: float) -> None:
    print(f"index: {source} · {load_s:.2f}s")


def _like(paths: np.ndarray, vecs: np.ndarray, target: str, top: int) -> dict | None:
    """Nearest notes to one note's own vector: single-file triage without a query."""
    exact = [k for k, p in enumerate(paths) if str(p) == target]
    matches = exact or [
        k for k, p in enumerate(paths) if str(p).endswith("/" + target)
    ]
    if not matches:
        return None
    k = matches[0]
    d = 1.0 - vecs @ vecs[k]
    order = np.argsort(d)
    out = []
    for j in order:
        if int(j) == k:
            continue
        out.append({"path": str(paths[j]), "distance": round(max(0.0, float(d[j])), 4)})
        if len(out) >= top:
            break
    return {"path": str(paths[k]), "neighbours": out}


def cmd_calibrate(args) -> int:
    vault = vault_root(args.vault)
    cfg = load_config(vault)
    paths, vecs, source = load_index(db_dir_for(vault, args.db), vault)
    t0 = time.time()
    cal = calibrate(paths, vecs, set(cfg["never_dest"]), sample=args.samples)
    cfg["calibration"] = cal
    cfg["db"] = os.path.abspath(db_dir_for(vault, args.db))
    cfg["model"] = {"name": cfg["model"].get("name", DEFAULT_MODEL),
                    "dims": int(vecs.shape[1])}
    save_config(vault, cfg)
    print(f"calibrated {len(paths)} notes in {time.time() - t0:.1f}s ({source})")
    print(f"  vote_gate {cal['vote_gate']}  dup_floor {cal['dup_floor']}  "
          f"samples {cal['samples']}")
    for row in cal["sweep"]:
        mark = " <-" if row["t"] == cal["vote_gate"] else ""
        print(f"  t={row['t']:.2f} votes={row['median_votes']:<3} "
              f"agree={row['agreement']:.2f} cover={row['coverage']:.2f}{mark}")
    return 0


def cmd_plan(args) -> int:
    vault = vault_root(args.vault)
    cfg = load_config(vault)
    t0 = time.time()
    db_dir = db_dir_for(vault, args.db)
    paths, vecs, source = load_index(db_dir, vault)

    if args.like:
        found = _like(paths, vecs, args.like, max(args.limit, 5))
        if found is None:
            print(f"not in the index: {args.like}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(found, indent=2, ensure_ascii=False))
        else:
            print(f"nearest notes to {found['path']} ({len(paths)} indexed, {time.time()-t0:.2f}s)")
            for n in found["neighbours"]:
                print(f"  {n['distance']:.3f}  {n['path']}")
        return 0

    doc = plan(paths, vecs, cfg, vault, args.folder or cfg["default_folder"], db_dir=db_dir)
    doc["index_source"] = source
    doc["load_seconds"] = round(time.time() - t0, 2)
    if args.json:
        # --limit also caps the arrays here: a 500 KB blob is unusable in an
        # agent's context. Totals stay exact, and `truncated` says what was cut
        # so `apply` can warn instead of silently moving a subset.
        limit = args.limit
        out = dict(doc)
        out["totals"] = {
            "duplicates": len(doc["duplicates"]),
            "moves": len(doc["moves"]),
            "moves_blocked": sum(1 for m in doc["moves"] if m.get("ambiguous_basename")),
            "unclassified": len(doc["unclassified"]),
            "companions": len(doc["companions"]),
        }
        out["truncated"] = {}
        for key in ("duplicates", "moves", "unclassified", "companions"):
            if limit and len(doc[key]) > limit:
                out[key] = doc[key][:limit]
                out["truncated"][key] = True
            else:
                out["truncated"][key] = False
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    cov = doc["coverage"]
    print(f"scope {doc['scope']} · gate {doc['gate']} · dup {doc['dup_floor']} "
          f"({doc.get('dup_source')}) · calibrated {doc['calibrated_at']}")
    for warning in doc.get("warnings", []):
        print(f"  WARNING: {warning}")
    print(f"  {len(doc['duplicates'])} duplicates · {len(doc['moves'])} moves "
          f"({sum(1 for m in doc['moves'] if m.get('ambiguous_basename'))} blocked: "
          f"shared filename) · {len(doc['unclassified'])} unclear")
    print(f"  coverage: {cov['indexed_in_scope']} indexed / {cov['text_on_disk']} text "
          f"({cov['markdown_on_disk']} md + {cov['companion_txt_on_disk']} companions), "
          f"{cov['too_short']} too short, {cov['unindexed']} unindexed")
    if cov["non_text"]:
        parts = ", ".join(f"{n} {ext}" for ext, n in sorted(cov["non_text"].items()))
        print(f"  not classifiable (needs text companion): {parts}")
    limit = 0 if args.limit == 0 else args.limit
    comps = doc.get("companions", [])
    shown_comps = comps if limit == 0 else comps[:limit]
    for c in shown_comps:
        dest = c["destination"] or "unclear"
        print(f"  via text  {c['path']} -> {dest}  (through {c['via']})")
    if len(shown_comps) < len(comps):
        print(f"  … {len(comps) - len(shown_comps)} more non-text files (--limit 0 for all)")
    dups = doc["duplicates"] if limit == 0 else doc["duplicates"][:limit]
    for d in dups:
        print(f"  dup {d['distance']:.3f} {d['path']}\n            = {d['duplicate_of']}")
    moves = doc["moves"] if limit == 0 else doc["moves"][:limit]
    for m in moves:
        flag = "  [blocked: shared filename]" if m.get("ambiguous_basename") else ""
        print(f"  {m['confidence']:.2f} {m['path']} -> {m['target_dir']}{flag}")
    return 0


def cmd_apply(args) -> int:
    vault = vault_root(args.vault)
    with open(args.plan, encoding="utf-8") as fh:
        doc = json.load(fh)
    db_dir = None if args.no_index else db_dir_for(vault, args.db)
    manifest = apply_plan(vault, doc, reindex=not args.no_index, db_dir=db_dir,
                          allow_ambiguous_basename=args.allow_ambiguous_basename)
    if doc.get("truncated", {}).get("moves"):
        print("WARNING: this plan was truncated, so only the listed moves ran. "
              "Re-run plan with --limit 0 for the full set.")
    print(f"applied {len(manifest['applied'])} moves (token {manifest['token']}, "
          f"{manifest['links_files_updated']} files relinked)")
    for item in manifest["applied"]:
        print(f"  {item['from']} -> {item['to']}")
    for item in manifest["refused"]:
        print(f"  REFUSED {item['path']}: {item['reason']}")
    print(f"undo: vault organize undo {manifest['token']}")
    return 0


def cmd_undo(args) -> int:
    vault = vault_root(args.vault)
    db_dir = None if args.no_index else db_dir_for(vault, args.db)
    try:
        result = undo_apply(vault, args.token, reindex=not args.no_index,
                            db_dir=db_dir)
    except FileNotFoundError:
        print(f"no manifest for {args.token} in {os.path.join(vault, STATE_DIR, APPLIED_DIR)}",
              file=sys.stderr)
        return 2
    print(f"reverted {len(result['reverted'])} moves for {result['token']}")
    return 0


def _add_common(p: argparse.ArgumentParser, suppress: bool) -> None:
    """Global options, accepted before or after the verb.

    On subparsers they use SUPPRESS so an unset value cannot overwrite what the
    top-level parser already read.
    """
    d = argparse.SUPPRESS if suppress else None
    p.add_argument("--vault", default=d, help="vault root (default $VAULT_PATH or cwd)")
    p.add_argument("--db", default=d, help="LanceDB dir (default <vault>/.vaultindex/lancedb)")
    p.add_argument("--json", action="store_true", default=d, help="machine-readable output")
    p.add_argument("--quiet", action="store_true", default=d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="organize",
        description="Classify notes by nearest neighbours; plan, apply, undo moves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  organize.py calibrate                  # derive the gates from this corpus
  organize.py plan 00-inbox --json       # duplicates, moves, coverage
  organize.py apply plan.json            # move + relink + reindex
  organize.py undo apply-20261124T101500 # reverse one apply
""",
    )
    _add_common(ap, suppress=False)
    sub = ap.add_subparsers(dest="cmd")

    c = sub.add_parser("calibrate", help="derive vote gate and duplicate floor")
    _add_common(c, suppress=True)
    c.add_argument("--samples", type=int, default=600)
    c.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("plan", help="duplicates, moves and coverage for a folder")
    _add_common(p, suppress=True)
    p.add_argument("folder", nargs="?", default=None)
    p.add_argument("--like", metavar="PATH", default=None,
                   help="Nearest notes to this note's own vector (single-file triage)")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_plan)

    a = sub.add_parser("apply", help="apply a plan file")
    _add_common(a, suppress=True)
    a.add_argument("plan")
    a.add_argument("--no-index", action="store_true")
    a.add_argument("--allow-ambiguous-basename", action="store_true",
                   help="Move even when [[name]] links could re-point to a twin")
    a.set_defaults(func=cmd_apply)

    u = sub.add_parser("undo", help="reverse one apply")
    _add_common(u, suppress=True)
    u.add_argument("token")
    u.add_argument("--no-index", action="store_true")
    u.set_defaults(func=cmd_undo)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 2
    return args.func(args)


VERBS = {"calibrate", "plan", "apply", "undo"}


def _default_verb(argv: list[str]) -> list[str]:
    """`organize.py 00-inbox` and `organize.py --json 00-inbox` both mean `plan`.

    A verb anywhere in argv wins, so flags before the verb still work; `--help`
    alone keeps showing the top-level help.
    """
    if not argv or argv[0] in ("-h", "--help"):
        return argv
    if any(arg in VERBS for arg in argv):
        return argv
    return ["plan", *argv]


if __name__ == "__main__":
    sys.exit(main(_default_verb(sys.argv[1:])))

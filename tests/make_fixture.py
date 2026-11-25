#!/usr/bin/env python3
"""Build a deterministic fixture vault for the organizer tests.

Vectors are synthesised, not embedded: the tests exercise classification,
planning, apply and undo, not the embedder. Every note gets a vector drawn near
its topic's base vector, so the "correct" destination is known by construction.

Layout produced under the given root:

    00-inbox/     10 movers (5 belong in 06-research, 5 in 13-concepts),
                  5 exact duplicates of research notes, 3 frontmatter-only
                  stubs, 2 PDFs with .txt companions, 1 basename collision
    06-research/  5 notes of topic "research" + collide.md
    13-concepts/  5 notes of topic "concepts" + the other collide.md + hub.md
    02-creative-ecosystem/ 5 notes of topic "creative"
    07-library/   5 notes of topic "library"
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

import numpy as np

DIMS = 32
WITHIN_TOPIC_NOISE = 0.10  # cos ~0.87 -> distance ~0.13 within a topic


def build(root: str) -> dict:
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)

    rng = np.random.default_rng(42)
    # Two related pairs of folders, so agreement degrades as the gate widens and
    # calibration has a knee. A corpus with four orthogonal topics never degrades
    # and would let the sweep pick the widest gate, which is the trap this avoids.
    g1 = norm(rng.normal(size=DIMS))
    g2 = norm(rng.normal(size=DIMS))
    bases = {
        "research": norm(g1 + 0.75 * rng.normal(size=DIMS)),
        "library": norm(g1 + 0.75 * rng.normal(size=DIMS)),
        "concepts": norm(g2 + 0.75 * rng.normal(size=DIMS)),
        "creative": norm(g2 + 0.75 * rng.normal(size=DIMS)),
    }

    def vec_for(topic: str) -> np.ndarray:
        return norm(bases[topic] + WITHIN_TOPIC_NOISE * rng.normal(size=DIMS))

    records: list[tuple[str, np.ndarray]] = []
    files: dict[str, str] = {}

    def add(path: str, topic: str, body: str, vector: np.ndarray | None = None) -> None:
        files[path] = body
        records.append((path, vector if vector is not None else vec_for(topic)))

    for i in range(6):
        add(f"06-research/research-{i}.md", "research",
            f"# Research {i}\n\nempirical study of systems and feedback loops {i}\n")
        add(f"13-concepts/concept-{i}.md", "concepts",
            f"# Concept {i}\n\nabstract definition of a cybernetic principle {i}\n")
        add(f"02-creative-ecosystem/creative-{i}.md", "creative",
            f"# Creative {i}\n\nsong sketch and lyrics fragment {i}\n")
        add(f"07-library/library-{i}.md", "library",
            f"# Library {i}\n\nreference note on a third-party source {i}\n")

    # Movers: the first five belong with research, the next five with concepts.
    for i in range(10):
        topic = "research" if i < 5 else "concepts"
        add(f"00-inbox/mover-{i}.md", topic,
            f"# Mover {i}\n\nunsorted note about {topic} belonging elsewhere {i}\n")

    # Exact duplicates of five research notes, named `<twin>-dup.md` so the
    # labelled branch of dup_floor can resolve them (basename fallback).
    research_vectors = {p: v for p, v in records if p.startswith("06-research/research-")}
    for path, vector in sorted(research_vectors.items())[:5]:
        stem = path.rsplit("/", 1)[-1][: -len(".md")]
        add(f"00-inbox/{stem}-dup.md", "research", files[path], vector.copy())

    # Stubs: frontmatter only, below the indexer's 50-character body threshold.
    for i in range(3):
        path = f"00-inbox/stub-{i}.md"
        files[path] = f"---\ntitle: stub {i}\ntags: []\n---\n\n"

    # A mover whose destination already holds a file of the same basename.
    add("13-concepts/collide.md", "concepts", "# Collide\n\nexisting concept note\n")
    add("00-inbox/collide.md", "concepts", "# Collide\n\nunsorted duplicate name\n")

    # Path-qualified links that must be rewritten, and a basename link that must not.
    files["13-concepts/hub.md"] = (
        "# Hub\n\n"
        "- [[00-inbox/mover-0]]\n"
        "- [[00-inbox/mover-1|alias text]]\n"
        "- ![[00-inbox/mover-2]]\n"
        "- [[mover-3]]\n"
    )
    records.append(("13-concepts/hub.md", vec_for("concepts")))

    # Non-classifiable files: PDFs whose .txt companions ARE indexed, so the
    # PDF becomes classifiable through its companion.
    for i in range(2):
        files[f"00-inbox/paper-{i}.pdf"] = "%PDF-1.3 not really\n"
        add(f"00-inbox/paper-{i}.txt", "library",
            f"extracted text about systems and references {i}\n")

    for path, body in files.items():
        full = os.path.join(root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(body)

    # Index state + vector cache, shaped exactly like the real ones.
    state_dir = os.path.join(root, ".vaultindex")
    os.makedirs(os.path.join(state_dir, "lancedb"), exist_ok=True)
    paths = np.array([p for p, _ in records])
    vecs = np.array([v for _, v in records], dtype=np.float32)
    state = {"rows": len(records), "version": 1, "model": "fixture",
             "dims": DIMS, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(os.path.join(state_dir, "state.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh)

    with open(os.path.join(root, ".vaultindex.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "version": 1,
            "default_folder": "00-inbox",
            "never_dest": ["00-inbox", "99-archive"],
            "model": {"name": "fixture", "dims": DIMS},
            "calibration": None,
        }, fh, indent=2)

    cache_dir = os.path.join(root, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(os.path.abspath(os.path.join(state_dir, "lancedb")).encode()).hexdigest()[:10]
    cache = os.path.join(cache_dir, f"notes-{key}.npz")
    with open(cache, "wb") as fh:
        np.savez(fh, paths=paths, vectors=vecs, version=np.int64(1),
                 rows=np.int64(len(records)), written_at=np.int64(1_700_000_000))

    return {
        "root": root,
        "db": os.path.join(state_dir, "lancedb"),
        "cache_dir": os.path.join(root, ".cache"),
        "expected": {
            "duplicates": 5,
            "movers": {"mover-0": "06-research", "mover-4": "06-research",
                       "mover-5": "13-concepts", "mover-9": "13-concepts"},
            "stubs": 3,
            "pdfs": 2,
            "collision": "00-inbox/collide.md",
        },
    }


def norm(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def tree_hash(root: str) -> dict[str, str]:
    """path -> sha1 of bytes, for the whole tree except .vaultindex/.cache."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".vaultindex", ".cache", ".git")]
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            with open(full, "rb") as fh:
                out[rel] = hashlib.sha1(fh.read()).hexdigest()
    return out


if __name__ == "__main__":
    import sys

    info = build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/vault-fixture")
    print(json.dumps({k: v for k, v in info.items() if k != "expected"}, indent=2))

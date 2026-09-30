#!/usr/bin/env python3
"""
Embed Obsidian vault notes with nomic-embed-text via Ollama.
Usage:
  python3 embed_vault.py --test 5     # embed 5 files, report timing
  python3 embed_vault.py --all        # embed entire vault
  python3 embed_vault.py --query "underwater acoustic instruments"  # semantic search
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

VAULT = os.environ.get("VAULT_PATH", "")
if not VAULT:
    sys.exit("ERROR: VAULT_PATH not set.\n"
             "  export VAULT_PATH=/path/to/your/vault")
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
EMBED_TIMEOUT = 120.0
STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")

# Truncate note content to ~8000 chars to stay within model context
MAX_CHARS = 8000


def embed_text(text: str) -> list[float]:
    """Call Ollama embed endpoint, return 768-dim vector."""
    payload = json.dumps({"model": MODEL, "input": text}).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
        data = json.loads(resp.read())
    return data["embeddings"][0]


def find_md_files(vault=VAULT) -> list[str]:
    """Find all .md files, excluding hidden dirs and common noise."""
    skip_prefixes = (".", "assets", "_Archive", "node_modules")
    md_files = []
    for root, dirs, files in os.walk(vault):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.endswith(".md"):
                rel = os.path.relpath(os.path.join(root, f), vault)
                if not rel.startswith(skip_prefixes):
                    md_files.append(rel)
    return sorted(md_files)


def read_note(path: str, vault=VAULT) -> str:
    """Read note, strip frontmatter, truncate."""
    full = os.path.join(vault, path)
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    # Strip YAML frontmatter (closing --- must be on its own line)
    if content.startswith("---"):
        import re
        m = re.match(r'^---\s*\n.*?\n---\s*\n', content, re.DOTALL)
        if m:
            content = content[m.end():]
    return content.strip()[:MAX_CHARS]


def run_test(n: int = 5):
    """Embed N files, report per-file and projected total timing."""
    files = find_md_files()
    sample = files[:n]
    print(f"Vault has {len(files)} .md files total.")
    print(f"Embedding {n} sample files with {MODEL}...\n")

    vectors = {}
    t0 = time.time()

    for i, rel in enumerate(sample):
        text = read_note(rel)
        char_count = len(text)
        t1 = time.time()
        vec = embed_text(text)
        elapsed = time.time() - t1
        vectors[rel] = vec
        print(
            f"  [{i+1}/{n}] {elapsed:.2f}s  {char_count:>5} chars  "
            f"{os.path.basename(rel)}"
        )

    total = time.time() - t0
    avg = total / n

    print(f"\n--- Timing ---")
    print(f"  Total:     {total:.2f}s for {n} files")
    print(f"  Per-file:  {avg:.2f}s avg")
    print(f"  Embed dim: {len(next(iter(vectors.values())))}")

    projected = avg * len(files)
    if projected < 60:
        proj_str = f"{projected:.0f}s"
    else:
        proj_str = f"{projected/60:.1f} min"

    print(f"  Projected full vault ({len(files)} files): ~{proj_str}")


def embed_all():
    """Embed entire vault, save to STORE_DIR as .npy + index.json."""
    os.makedirs(STORE_DIR, exist_ok=True)
    files = find_md_files()
    print(f"Embedding {len(files)} files with {MODEL}...")

    vectors = []
    paths = []
    t0 = time.time()

    for i, rel in enumerate(files):
        text = read_note(rel)
        if len(text) < 50:
            continue
        vec = embed_text(text)
        vectors.append(vec)
        paths.append(rel)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(files) - i - 1) / rate
            print(f"  [{i+1}/{len(files)}] {rate:.1f} files/s  ETA ~{eta:.0f}s")

    elapsed = time.time() - t0
    mat = np.array(vectors, dtype=np.float32)

    # Save
    np.save(os.path.join(STORE_DIR, "embeddings.npy"), mat)
    with open(os.path.join(STORE_DIR, "index.json"), "w") as f:
        json.dump(paths, f, indent=2)

    print(f"\nDone: {len(paths)} notes embedded in {elapsed:.1f}s")
    print(f"Matrix shape: {mat.shape}")
    print(f"Saved to {STORE_DIR}/")


def query(text: str, top_k: int = 10):
    """Semantic search: embed query, find nearest notes."""
    emb_path = os.path.join(STORE_DIR, "embeddings.npy")
    idx_path = os.path.join(STORE_DIR, "index.json")

    if not os.path.exists(emb_path):
        print("No embeddings found. Run --all first.")
        sys.exit(1)

    mat = np.load(emb_path)
    with open(idx_path) as f:
        paths = json.load(f)

    # Embed query
    qvec = np.array(embed_text(text), dtype=np.float32)

    # Cosine similarity
    mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    q_norm = qvec / np.linalg.norm(qvec)
    sims = mat_norm @ q_norm

    # Top-k
    top_idx = np.argsort(sims)[::-1][:top_k]

    print(f'Query: "{text}"\n')
    print(f"Top {top_k} results:\n")
    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1}. {sims[idx]:.4f}  {paths[idx]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed vault with nomic-embed-text")
    parser.add_argument("--test", type=int, metavar="N", help="Embed N files, report timing")
    parser.add_argument("--all", action="store_true", help="Embed entire vault")
    parser.add_argument("--query", type=str, metavar="TEXT", help="Semantic search")
    parser.add_argument("--top", type=int, default=10, help="Results for --query")
    args = parser.parse_args()

    if args.test:
        run_test(args.test)
    elif args.all:
        embed_all()
    elif args.query:
        query(args.query, args.top)
    else:
        parser.print_help()

#!/usr/bin/env python3
"""
Obsidian vault vector DB powered by LanceDB + nomic-embed-text (Ollama).

Usage:
  python3 vault_db.py index --test 5       # embed 5 files, report timing
  python3 vault_db.py index --all          # embed entire vault into LanceDB
  python3 vault_db.py index --missing     # embed only files not yet in DB
  python3 vault_db.py index --file PATH    # upsert single note
  python3 vault_db.py index --prune        # drop DB rows for deleted files
  python3 vault_db.py search "query text"  # semantic search
  python3 vault_db.py search "query" --folder 06-research --top 5
  python3 vault_db.py topics              # domain/folder/keyword overview
  python3 vault_db.py stats                # DB stats
"""

import argparse
import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import timedelta

import lancedb
import pyarrow as pa
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False


VAULT = os.environ.get("VAULT_PATH", "")
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
DB_DIR = os.environ.get("DB_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lancedb"))
TABLE = "notes"
MAX_CHARS = 8000

SKIP_PREFIXES = (".", "assets", "_Archive", "node_modules")
EMBED_TIMEOUT = 120.0

# ── Domain classification by folder prefix ──────────────────────────────────

def _load_domains():
    """Load domain→prefixes mapping from domains.json, falling back to defaults."""
    domains_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "domains.json")
    defaults = {
        "research": ["00-inbox", "06-research"],
        "projects": ["08-projects"],
        "archive": ["99-archive"],
        "library": ["07-library"],
        "journaling": ["10-journaling"],
    }
    try:
        with open(domains_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            return data
        print(f"Warning: {domains_path} has wrong format, using defaults.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if not isinstance(e, FileNotFoundError):
            print(f"Warning: {domains_path} is invalid JSON, using defaults.")
    return defaults


DOMAINS = _load_domains()


def domain_of(path: str) -> str:
    for dom, prefixes in DOMAINS.items():
        for p in prefixes:
            if path.startswith(p):
                return dom
    return "other"


# ── Embedding ──────────────────────────────────────────────────────────────


def embed_text(text: str) -> list[float]:
    """Call Ollama embed endpoint, return 768-dim vector."""
    payload = json.dumps({"model": MODEL, "input": text}).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
        data = json.loads(resp.read())
    return data["embeddings"][0]


# ── Ollama diagnostics ─────────────────────────────────────────────────────
#
# Embedding fails for three different reasons, and each needs a different fix.
# This used to report all of them as "is Ollama running?", which sends you
# looking in the wrong place when the server is up but its embed worker has
# wedged.

OLLAMA_BASE = OLLAMA_URL.rsplit("/api/", 1)[0]  # http://localhost:11434


def _ollama_model_names(timeout: float = 5.0) -> list[str]:
    """Names of the models Ollama has installed. Raises if it is unreachable."""
    req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return [m.get("name", "") for m in json.loads(resp.read()).get("models", [])]


def _ollama_loaded_models(timeout: float = 5.0) -> list[str]:
    """Models currently resident in memory. Raises if Ollama is unreachable."""
    req = urllib.request.Request(f"{OLLAMA_BASE}/api/ps")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return [m.get("name", "") for m in json.loads(resp.read()).get("models", [])]


def _model_installed(names: list[str], want: str | None = None) -> bool:
    """True if `want` (default MODEL) is installed, as bare name or any `:tag`."""
    want = want or MODEL
    return any(n == want or n.startswith(want + ":") for n in names)


def _warm_curl() -> str:
    """The exact command that reloads the embedding model."""
    body = json.dumps({"model": MODEL, "input": "warm up"})
    return f"curl -s --max-time 180 {OLLAMA_URL} -d '{body}'"


def _fault_down() -> tuple[str, str]:
    return (
        f"Ollama isn't running (nothing answering on {OLLAMA_BASE}).",
        "ollama serve",
    )


def _fault_no_model() -> tuple[str, str]:
    return (
        f"Ollama is running, but {MODEL} isn't installed.",
        f"ollama pull {MODEL}",
    )


def _fault_wedged(detail: str | None = None) -> tuple[str, str]:
    cause = (
        "Ollama is running and the model is installed, but its embed worker "
        "isn't responding — the server is wedged."
    )
    if detail:
        # Carry the real error: a timeout and a connection reset need
        # different reactions, and guessing between them misleads.
        cause += f" Underlying error: {detail}"
    return (
        cause,
        "launchctl kickstart -k gui/$(id -u)/local.ollama.serve\n"
        f"then warm the model:  {_warm_curl()}",
    )


def diagnose_ollama() -> tuple[str, str | None]:
    """Explain why an embed call just failed, as (cause, fix).

    Only meaningful AFTER a failure. It reads the server's model list, which
    cannot separate "healthy" from "wedged", so a reachable server with the
    model installed is reported as wedged. For a check that can tell those two
    apart, use `preflight_embed`.
    """
    try:
        names = _ollama_model_names()
    except Exception:
        return _fault_down()
    if not _model_installed(names):
        return _fault_no_model()
    return _fault_wedged()


def probe_embed(timeout: float = EMBED_TIMEOUT) -> tuple[int, float]:
    """Embed one short string. Returns (dimensions, seconds). Raises on failure."""
    body = json.dumps({"model": MODEL, "input": "health check"}).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        dims = len(json.loads(resp.read())["embeddings"][0])
    return dims, time.time() - t0


def preflight_embed() -> tuple[str, str | None]:
    """Check whether embedding works right now, as (cause, fix).

    `fix` is None only after a real embed succeeds, so unlike `diagnose_ollama`
    this separates a healthy server from a wedged one.
    """
    try:
        names = _ollama_model_names()
    except Exception:
        return _fault_down()
    if not _model_installed(names):
        return _fault_no_model()
    try:
        probe_embed()
    except Exception as e:
        return _fault_wedged(f"{type(e).__name__}: {e}")
    return ("embedding works", None)


def print_embed_failure(cause: str, fix: str | None, detail: str | None = None) -> None:
    """Print an embedding failure in one consistent shape."""
    print(f"ERROR: couldn't embed with {MODEL}.")
    print(f"  Cause: {cause}")
    if fix:
        for i, line in enumerate(fix.splitlines()):
            print(f"  {'Fix: ' if i == 0 else '      '}{line}")
    if detail:
        print(f"  Detail: {detail}")


def embed_checked(text: str) -> list[float]:
    """Embed for CLI entry points, reporting the real fault on failure.

    Batch paths (index_files, flush_batch) keep per-file handling instead;
    this is only for single-shot commands where a raw traceback is unhelpful.
    """
    try:
        return embed_text(text)
    except Exception as e:
        cause, fix = diagnose_ollama()
        print_embed_failure(cause, fix, str(e))
        sys.exit(1)


# ── File discovery & reading ───────────────────────────────────────────────


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


def _strip_frontmatter(content: str) -> str:
    """Remove a leading YAML frontmatter block.

    The opening delimiter is ``---`` at the very start; the closing delimiter
    is the next line whose stripped value is exactly ``---``. Requiring the
    closing marker on its own line avoids truncating on a ``---`` that appears
    *inside* a frontmatter value (e.g. ``description: split---here``).
    """
    if content.startswith("---"):
        lines = content.split("\n")
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return "\n".join(lines[i + 1 :])
    return content


def read_note(path: str, vault=VAULT) -> str:
    """Read note, strip YAML frontmatter, truncate."""
    full = os.path.join(vault, path)
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return _strip_frontmatter(content).strip()[:MAX_CHARS]


def extract_title(path: str, vault=VAULT) -> str:
    """Get note title from filename or first H1 (frontmatter stripped)."""
    basename = os.path.basename(path).replace(".md", "")
    full = os.path.join(vault, path)
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return basename
    content = _strip_frontmatter(content)
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return basename


def _classify_missing(unindexed: set) -> tuple[set, int]:
    """Split on-disk-but-unindexed paths into (genuinely missing, too-short count).

    `index_files()` skips notes shorter than 50 chars by design, so those paths
    never get indexed. Counting them as "missing" produces a permanent false
    alarm that no amount of `sync` can clear.
    """
    genuinely_missing: set = set()
    too_short = 0
    for rel in unindexed:
        try:
            if len(read_note(rel)) < 50:
                too_short += 1
            else:
                genuinely_missing.add(rel)
        except Exception:
            genuinely_missing.add(rel)  # unreadable: surface it, don't hide it
    return genuinely_missing, too_short


# ── LanceDB operations ─────────────────────────────────────────────────────


def get_db():
    db = lancedb.connect(DB_DIR)
    return db


def has_table(db, name: str = TABLE) -> bool:
    """True if a LanceDB table exists.

    Uses try/except because the return type of db.list_tables() is not a
    plain list across LanceDB versions (0.36 returns a ListTablesResponse),
    so ``name in db.list_tables()`` is unreliable.
    """
    try:
        db.open_table(name)
        return True
    except Exception:
        return False


def create_table(db):
    """Create or open the notes table with proper schema."""
    schema = pa.schema([
        pa.field("path", pa.string()),
        pa.field("title", pa.string()),
        pa.field("folder", pa.string()),
        pa.field("domain", pa.string()),
        pa.field("char_count", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), 768)),
    ])
    if has_table(db):
        return db.open_table(TABLE)
    else:
        # Create empty table with schema
        empty = pa.table({
            "path": pa.array([], pa.string()),
            "title": pa.array([], pa.string()),
            "folder": pa.array([], pa.string()),
            "domain": pa.array([], pa.string()),
            "char_count": pa.array([], pa.int32()),
            "vector": pa.array([], pa.list_(pa.float32(), 768)),
        })
        db.create_table(TABLE, empty)
        return db.open_table(TABLE)


def sql_quote(value: str) -> str:
    """Escape a string as a single-quoted SQL literal for LanceDB predicates."""
    return "'" + value.replace("'", "''") + "'"


def upsert_notes(table, records: list[dict]):
    """Upsert (insert or update) a batch of notes, keyed on `path`.

    Uses LanceDB's atomic merge_insert when available: a delete-then-add pair
    loses rows if the add fails after the delete. Falls back to that pair only
    on LanceDB builds without merge_insert.
    """
    if not records:
        return
    try:
        (
            table.merge_insert("path")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(records)
        )
        return
    except Exception:
        # Missing on older LanceDB, or it rejected this batch's shape. Fall
        # through to delete-then-add, whose failure mode matches the old
        # behaviour instead of aborting a bulk index run.
        pass

    paths = [r["path"] for r in records]
    try:
        if len(paths) == 1:
            table.delete(f"path = {sql_quote(paths[0])}")
        else:
            table.delete(f"path IN ({', '.join(sql_quote(p) for p in paths)})")
    except Exception:
        pass  # paths may not exist yet — a no-op match is fine for an upsert
    table.add(_records_to_table(records))


def _records_to_table(records: list[dict]) -> pa.Table:
    """Build the LanceDB table payload from note records."""
    return pa.table({
        "path": pa.array([r["path"] for r in records], pa.string()),
        "title": pa.array([r["title"] for r in records], pa.string()),
        "folder": pa.array([r["folder"] for r in records], pa.string()),
        "domain": pa.array([r["domain"] for r in records], pa.string()),
        "char_count": pa.array([r["char_count"] for r in records], pa.int32()),
        "vector": pa.array([r["vector"] for r in records], pa.list_(pa.float32(), 768)),
    })


# ── macOS notification ─────────────────────────────────────────────────────


def _notify(title: str, message: str) -> None:
    """Fire a macOS notification via osascript (built-in, no TCC issues)."""
    try:
        escaped_msg = message.replace("\\", "\\\\").replace('"', '\\"')
        escaped_title = title.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{escaped_msg}" with title "{escaped_title}" sound name "default"'],
            check=False, timeout=5,
        )
    except Exception:
        pass  # notification is best-effort; never crash on it


# ── Commands ──────────────────────────────────────────────────────────────


def _short(rel: str, n: int = 42) -> str:
    return rel if len(rel) <= n else "…" + rel[-(n - 1):]


def _dir_size(path: str) -> int:
    """Total bytes of every file under `path` (0 if it does not exist)."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _positive_int(value: str) -> int:
    """argparse type for counts: rejects zero and negatives.

    Without this, `--top -1` slices the result list to `[:-1]`, silently
    dropping the last hit instead of failing.
    """
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer")
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or greater (got {n})")
    return n


def _eta(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf") or seconds < 0:
        return "--"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def index_files(table, files: list[str], label: str = "Indexing") -> dict:
    """Embed and upsert a list of vault files with a rich progress display.

    Per-file failures are logged and skipped, never fatal.
    Falls back to plain prints when rich is unavailable.
    """
    t0 = time.time()
    done = skipped = errors = 0
    records: list[dict] = []
    total = len(files)

    def make_record(rel: str, text: str) -> dict:
        return {
            "path": rel,
            "title": extract_title(rel),
            "folder": os.path.dirname(rel) or ".",
            "domain": domain_of(rel),
            "char_count": len(text),
            "vector": embed_text(text),
        }

    if HAVE_RICH:
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("[cyan]{task.fields[rate]}"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[cur]}"),
        )
        with progress:
            task = progress.add_task(label, total=total, rate="", cur="")
            for rel in files:
                try:
                    text = read_note(rel)
                    if len(text) < 50:
                        skipped += 1
                    else:
                        records.append(make_record(rel, text))
                        done += 1
                except Exception as e:
                    errors += 1
                    progress.console.print(f"[red]  ✗ {rel}: {e}")
                elapsed = time.time() - t0
                processed = done + skipped + errors
                rate = f"{processed / elapsed:.1f}/s" if elapsed > 0 else ""
                progress.update(task, advance=1, rate=rate, cur=_short(rel))

            up_task = progress.add_task("Upsert", total=len(records), rate="", cur="")
            chunk = 500
            for start in range(0, len(records), chunk):
                batch = records[start : start + chunk]
                upsert_notes(table, batch)
                progress.update(up_task, advance=len(batch), rate="", cur="")
    else:
        for rel in files:
            try:
                text = read_note(rel)
                if len(text) < 50:
                    skipped += 1
                    continue
                records.append(make_record(rel, text))
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    eta = _eta(elapsed / done * (total - done))
                    print(f"  [{done + skipped + errors}/{total}]  {done/elapsed:.1f} files/s  ETA ~{eta}")
            except Exception as e:
                errors += 1
                print(f"  ✗ {rel}: {e}")
        chunk = 500
        for start in range(0, len(records), chunk):
            upsert_notes(table, records[start : start + chunk])

    return {
        "done": done,
        "skipped": skipped,
        "errors": errors,
        "elapsed": time.time() - t0,
    }


def _index_summary(result: dict):
    line = (
        f"[green]✓ {result['done']} embedded[/] · "
        f"[yellow]{result['skipped']} skipped (<50 chars)[/] · "
        f"[red]{result['errors']} errors[/] · {result['elapsed']:.1f}s"
    )
    if HAVE_RICH:
        Console().print(Panel(line, title="Summary", border_style="green"))
    else:
        print(
            f"\nDone: {result['done']} embedded, {result['skipped']} skipped, "
            f"{result['errors']} errors in {result['elapsed']:.1f}s"
        )


def cmd_index(args):
    db = get_db()
    table = create_table(db)

    if args.test is not None:
        if args.test <= 0:
            print("index --test requires a positive integer.")
            sys.exit(1)
        files = find_md_files()
        sample = files[: args.test]
        print(f"Vault has {len(files)} .md files total.")
        print(f"Embedding {args.test} sample files with {MODEL}...\n")

        t0 = time.time()
        for i, rel in enumerate(sample):
            text = read_note(rel)
            t1 = time.time()
            vec = embed_checked(text)
            elapsed = time.time() - t1
            print(
                f"  [{i+1}/{args.test}] {elapsed:.2f}s  {len(text):>5} chars  "
                f"{os.path.basename(rel)}"
            )

        total = time.time() - t0
        avg = total / args.test
        projected = avg * len(files)
        proj_str = (
            f"{projected:.0f}s"
            if projected < 60
            else f"{projected/60:.1f} min"
        )
        print(f"\n--- Timing ---")
        print(f"  Total:     {total:.2f}s for {args.test} files")
        print(f"  Per-file:  {avg:.2f}s avg")
        print(f"  Projected full vault ({len(files)} files): ~{proj_str}")

    elif args.file:
        rel = args.file
        full = os.path.join(VAULT, rel)
        if not os.path.exists(full):
            print(f"File not found: {rel}")
            sys.exit(1)
        print(f"Upserting: {rel}")
        text = read_note(rel)
        vec = embed_checked(text)
        folder = os.path.dirname(rel) or "."
        record = {
            "path": rel,
            "title": extract_title(rel),
            "folder": folder,
            "domain": domain_of(rel),
            "char_count": len(text),
            "vector": vec,
        }
        upsert_notes(table, [record])
        print(f"Done. Domain: {record['domain']}  Chars: {record['char_count']}")

    elif args.all:
        files = find_md_files()
        print(f"Vault: {len(files)} .md files · model {MODEL} · LanceDB at {DB_DIR}")
        result = index_files(table, files, label="Embedding")
        _index_summary(result)
        cmd_stats(None)
    elif args.missing:
        existing = set(table.to_arrow().column("path").to_pylist())
        files = [f for f in find_md_files() if f not in existing]
        if not files:
            print("Coverage complete — every vault file is already indexed.")
        else:
            print(f"{len(files)} files on disk but not in DB; embedding only those...")
            result = index_files(table, files, label="Missing")
            _index_summary(result)
    elif args.reclassify:
        _reclassify(table)
    elif args.prune:
        disk = set(find_md_files())
        db_paths = set(table.to_arrow().column("path").to_pylist())
        ghosts = sorted(db_paths - disk)
        missing = sorted(_classify_missing(disk - db_paths)[0])
        for rel in ghosts:
            table.delete(f"path = {sql_quote(rel)}")
            print(f"  ✗ pruned  {rel}")
        print(f"\nPruned {len(ghosts)} DB rows for files no longer on disk.")
        if missing:
            print(f"On disk but not indexed: {len(missing)} files (run: vault_db.py sync)")
    else:
        print("Nothing to do: `index` needs an action flag.\n")
        print("  --all         re-embed every note")
        print("  --missing     embed only files not yet in the DB")
        print("  --file PATH   re-embed one note whose content changed")
        print("  --prune       drop rows for files no longer on disk")
        print("  --reclassify  recompute domains after folder moves (no Ollama)")
        print("  --test N      time N embeds without writing anything")
        print("\nTo reconcile after adding/renaming/deleting notes, use: vault_db.py sync")
        sys.exit(2)


def _reclassify(table) -> int:
    """Recompute `domain` from path for rows whose folder moved. No Ollama.
    Returns number of rows reclassified. One delete + one add (2 versions)."""
    rows = table.to_arrow().to_pylist()
    changed = []
    for rec in rows:
        new_dom = domain_of(rec["path"])
        if new_dom != rec["domain"]:
            rec["domain"] = new_dom
            changed.append(rec)
    if not changed:
        print("No stale domains — all rows match their path-derived domain.")
        return 0
    affected = [rec["path"] for rec in changed]
    table.delete(f"path IN ({', '.join(sql_quote(p) for p in affected)})")
    table.add(_records_to_table(changed))
    from collections import Counter

    by_dom = Counter(r["domain"] for r in changed)
    print(f"Reclassified {len(changed)} rows (domain only, vectors untouched).")
    for dom, n in sorted(by_dom.items()):
        print(f"  → {dom:>12}: {n}")
    return len(changed)


def cmd_sync(args):
    """Reconcile the DB with the filesystem: reclassify stale domains,
    prune ghost rows, and embed files missing from the index."""
    db = get_db()
    table = create_table(db)

    disk = set(find_md_files())
    indexed = set(table.to_arrow().column("path").to_pylist())

    ghosts = sorted(indexed - disk)
    # Notes under 50 chars are skipped by design — they are not "missing".
    missing = sorted(_classify_missing(disk - indexed)[0])

    # 1. Stale domains (path-derived state)
    _reclassify(table)

    # 2. Ghost rows — one delete with an IN predicate.
    if ghosts:
        table.delete(f"path IN ({', '.join(sql_quote(p) for p in ghosts)})")
        print(f"\n  ✗ pruned {len(ghosts)} ghost rows (files no longer on disk).")
    else:
        print("\n  No ghost rows.")

    # 3. Files on disk but not indexed — embed unless --no-embed.
    embedded = short = errors = 0
    if missing:
        if args.no_embed:
            print(f"  {len(missing)} files missing from index (skipped, --no-embed).")
        else:
            result = index_files(table, missing, label="Syncing")
            _index_summary(result)
            embedded = result["done"]
            short = result["skipped"]
            errors = result["errors"]
    else:
        print("  All files on disk are indexed.")

    # 4. Content-edited notes — only with --changed.
    updated = 0
    if getattr(args, "changed", False):
        stored = {r["path"]: r["char_count"] for r in table.to_arrow().to_pylist()}
        edited = []
        for rel, old_len in stored.items():
            if rel not in disk:
                continue  # deletion, already handled in step 2
            try:
                if len(read_note(rel)) != old_len:
                    edited.append(rel)
            except Exception:
                continue
        if edited:
            print(f"  {len(edited)} notes changed on disk — re-embedding.")
            res = index_files(table, edited, label="Updating")
            _index_summary(res)
            updated = res["done"]
        else:
            print("  No content changes detected.")

    if missing and args.no_embed:
        tail = f"{len(missing)} left unembedded (--no-embed)"
    else:
        tail = f"{embedded} embedded, {short} too short, {errors} errors"
    if updated:
        tail += f", {updated} content-updated"
    print(f"\nSync complete: {len(ghosts)} ghosts pruned, {tail}.")
    if getattr(args, "notify", False):
        msg = f"{len(ghosts)} ghosts pruned, {embedded} embedded"
        if updated:
            msg += f", {updated} content-updated"
        _notify("Vault DB synced", msg)


def _snippet(path: str, query: str, width: int = 2, limit: int = 300) -> str:
    """Return the note lines most relevant to query tokens (read from disk)."""
    try:
        content = read_note(path)
    except Exception:
        return ""
    tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))
    lines = content.splitlines()
    best, best_hits = -1, 0
    for i, line in enumerate(lines):
        hits = sum(1 for t in tokens if t in line.lower())
        if hits > best_hits:
            best, best_hits = i, hits
    if best < 0 or best_hits == 0:
        first = next((l for l in lines if l.strip()), "")
        return first[:limit]
    lo, hi = max(0, best - width), min(len(lines), best + width + 1)
    return "\n".join(lines[lo:hi]).strip()[:limit]


def cmd_search(args):
    db = get_db()
    if not has_table(db):
        print("No index found. Run: python3 vault_db.py index --all")
        sys.exit(1)
    table = db.open_table(TABLE)

    qvec = embed_checked(args.query)

    # Build search
    search = table.search(qvec, vector_column_name="vector").limit(
        max(args.top * 3, 30)
    )

    # Apply folder filter if provided
    if args.folder:
        search = search.where(f"folder LIKE {sql_quote(args.folder + '%')}", prefilter=True)

    results = search.to_list()
    top = results[: args.top]
    show_snippet = not args.no_snippet

    if args.json:
        out = []
        for rank, row in enumerate(top):
            item = {
                "rank": rank + 1,
                "path": row.get("path", ""),
                "distance": round(float(row.get("_distance", 0.0)), 6),
                "domain": row.get("domain", ""),
                "title": row.get("title", ""),
            }
            if show_snippet:
                item["snippet"] = _snippet(item["path"], args.query)
            out.append(item)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(f'Query: "{args.query}"\n')
    if args.folder:
        print(f"Filter: folder prefix '{args.folder}'\n")
    print(f"Top {args.top} results:\n")
    for rank, row in enumerate(top):
        dist = row.get("_distance", 0.0)
        # Flat L2 scan (no ANN index): lower distance is better.
        title = row.get("title", "?")
        path = row.get("path", "?")
        domain = row.get("domain", "?")
        print(f"  {rank+1:>2}. dist={dist:.4f}  [{domain:>10}]  {title}")
        print(f"      {path}")
        if show_snippet:
            snip = _snippet(path, args.query)
            if snip:
                for line in snip.splitlines():
                    print(f"      │ {line}")
    print()


def cmd_stats(args):
    db = get_db()
    if not has_table(db):
        print("No index found. Run: python3 vault_db.py index --all")
        return
    table = db.open_table(TABLE)
    count = table.count_rows()
    print(f"Vault vector DB stats:")
    print(f"  Location: {DB_DIR}")
    # Only this table's directory: DB_DIR also holds sibling tables
    # (e.g. mannlab_website.lance), which are not part of the vault index.
    table_size = _dir_size(os.path.join(DB_DIR, TABLE + ".lance"))
    print(f"  Size on disk: {table_size / 1e6:.1f} MB")
    print(f"  Model: {MODEL} (768d)")
    print(f"  Notes indexed: {count}")

    # Domain breakdown (via pyarrow — no polars dependency)
    try:
        domains = table.to_arrow().column("domain").to_pylist()
        from collections import Counter

        print(f"\n  By domain:")
        for dom, n in Counter(domains).most_common():
            print(f"    {dom:>12}: {n}")
    except Exception as e:
        print(f"\n  (domain breakdown unavailable: {e})")

    if getattr(args, "health", False):
        print("\n── Health ───────────────────────────────────────────────")
        try:
            rows = table.to_arrow().to_pylist()
            n = len(rows)

            short = sorted(
                (r for r in rows if r["char_count"] < 50),
                key=lambda r: r["char_count"],
            )
            print(f"  Short notes (char_count < 50): {len(short)}")
            for r in short[:10]:
                print(f"    - {r['path']} ({r['char_count']})")

            other = [r for r in rows if r["domain"] == "other"]
            print(f"  Domain 'other' (unclassified): {len(other)}")

            stale = [r["path"] for r in rows if domain_of(r["path"]) != r["domain"]]
            print(f"  Stale domains (path says otherwise): {len(stale)}")
            if stale:
                print("    Fix: python3 vault_db.py index --reclassify")

            on_disk = set(find_md_files())
            indexed = {r["path"] for r in rows}
            missing, too_short = _classify_missing(on_disk - indexed)
            ghosts = indexed - on_disk
            print(
                f"  Coverage: {len(indexed)}/{len(on_disk)} on disk "
                f"({100.0 * len(indexed) / max(len(on_disk), 1):.1f}%) "
                f"· {len(missing)} missing · {len(ghosts)} ghosts "
                f"· {too_short} too short to index"
            )
        except Exception as e:
            print(f"  (health unavailable: {e})")


def cmd_doctor(args):
    """One-pass health check: environment, Ollama, model, embed latency, index.

    Prints one line per check and a final PASS/FAIL. Exits 1 when anything
    needs attention, so it works as a guard in scripts.
    """
    failures: list[str] = []

    def ok(label: str, detail: str = ""):
        print(f"  ok    {label:<8} {detail}".rstrip())

    def bad(label: str, detail: str, fix: str = ""):
        failures.append(label)
        print(f"  FAIL  {label:<8} {detail}".rstrip())
        for i, line in enumerate(fix.splitlines()):
            print(f"        {'→ ' if i == 0 else '  '}{line}")

    print("vault-tools doctor\n")

    # 1. Python environment — this file needs lancedb + pyarrow
    try:
        import lancedb as _lancedb
        import pyarrow as _pyarrow

        ok(
            "python",
            f"{sys.version.split()[0]} · lancedb {_lancedb.__version__} · "
            f"pyarrow {_pyarrow.__version__}",
        )
    except Exception as e:
        bad(
            "python",
            f"missing dependency: {e}",
            "run with the bundled venv: .venv/bin/python vault_db.py doctor",
        )

    # 2. Vault
    if os.path.isdir(VAULT):
        files = find_md_files()
        ok("vault", f"{len(files):,} markdown files in {VAULT}")
    else:
        files = []
        bad("vault", f"not found: {VAULT}")

    # 3. Ollama reachable, and is the model installed?
    names = None
    try:
        names = _ollama_model_names()
        ok("ollama", f"reachable at {OLLAMA_BASE}")
    except Exception:
        bad(
            "ollama",
            "not reachable",
            "ollama serve\n"
            "or restart the service: "
            "launchctl kickstart -k gui/$(id -u)/local.ollama.serve",
        )
    if names is not None:
        if _model_installed(names):
            ok("model", f"{MODEL} installed")
        else:
            bad("model", f"{MODEL} is not installed", f"ollama pull {MODEL}")

    # 4. Live embed — the check that catches a wedged server
    if names is not None and _model_installed(names):
        try:
            dims, secs = probe_embed()
            ok("embed", f"{dims}-dim vector in {secs:.2f}s")
        except Exception as e:
            loaded = []
            try:
                loaded = _ollama_loaded_models()
            except Exception:
                pass
            if loaded:
                bad(
                    "embed",
                    f"no response in {EMBED_TIMEOUT:.0f}s with {loaded[0]} "
                    f"loaded ({type(e).__name__}: {e})",
                    "launchctl kickstart -k gui/$(id -u)/local.ollama.serve\n"
                    f"then warm it:  {_warm_curl()}",
                )
            else:
                bad(
                    "embed",
                    f"no response in {EMBED_TIMEOUT:.0f}s ({e})",
                    _warm_curl(),
                )

    # 5. Index state
    try:
        db = get_db()
        if not has_table(db):
            bad("index", "no index yet", "vault_db.py index --all")
        else:
            table = db.open_table(TABLE)
            total = table.count_rows()          # same source as `stats`
            rows = table.to_arrow().to_pylist()
            indexed = {r["path"] for r in rows}
            missing, too_short = _classify_missing(set(files) - indexed)
            ghosts = indexed - set(files)
            dupes = total - len(indexed)        # upserts key on path; >0 means corruption
            if dupes:
                bad(
                    "index",
                    f"{total:,} rows for only {len(indexed):,} distinct paths "
                    f"({dupes:,} duplicate rows)",
                    "vault_db.py index --all",
                )
            elif missing or ghosts:
                bad(
                    "index",
                    f"{total:,} rows · {len(missing):,} missing · "
                    f"{len(ghosts):,} ghosts",
                    "vault_db.py sync",
                )
            else:
                ok(
                    "index",
                    f"{total:,} rows, in sync with disk"
                    + (f" · {too_short:,} too short to index" if too_short else ""),
                )
            stale = sum(1 for r in rows if domain_of(r["path"]) != r["domain"])
            if stale:
                bad(
                    "domains",
                    f"{stale:,} rows have a stale domain",
                    "vault_db.py index --reclassify",
                )
            else:
                ok("domains", "all current")
    except Exception as e:
        bad("index", f"unreadable: {e}")

    print()
    if failures:
        print(
            f"RESULT: FAIL — {len(failures)} check(s) need attention: "
            f"{', '.join(failures)}"
        )
        sys.exit(1)
    print("RESULT: PASS — environment, Ollama, and index all check out.")
    sys.exit(0)


STOPWORDS = frozenset("""
    the and for with that this from into your you are was were not but has have
    will can all about after before more other some such only than then them
    they its his her our out over under when where which while who why how what
    been being both each few most own same very via etc notes note draft final
    copy new old index readme welcome untitled
    de la el los las un una y o en por para con sin sobre como cuando donde
    quien cual desde hasta entre segun tras del al lo se su sus le les mas qué
    cómo cuándo dónde
""".split())


def _bar(count: int, max_count: int, width: int = 18) -> str:
    if max_count <= 0:
        return "░" * width
    filled = round(width * count / max_count)
    return "█" * filled + "░" * (width - filled)

def _clean_name(name: str) -> str:
    """Strip wide/non-BMP chars (emoji) that break terminal column math."""
    return "".join(ch for ch in name if ord(ch) <= 0xFFFF)

def cmd_topics(args):
    """Topic overview: domains, folders, and title keywords with bar charts."""
    from collections import Counter

    db = get_db()
    if not has_table(db):
        print("No index found. Run: python3 vault_db.py index --all")
        return
    table = db.open_table(TABLE)
    arrow = table.to_arrow()
    n_rows = arrow.num_rows
    domains_all = arrow.column("domain").to_pylist()
    paths_all = arrow.column("path").to_pylist()
    titles_all = arrow.column("title").to_pylist()

    disk_all = find_md_files()
    if args.domain:
        idx = [i for i, d in enumerate(domains_all) if d == args.domain]
        if not idx:
            print(f"No notes in domain '{args.domain}'.")
            return
        paths = [paths_all[i] for i in idx]
        titles = [titles_all[i] for i in idx]
        sections = []
        header = f"domain: {args.domain} · {len(idx):,} / {n_rows:,} notes"
        # Coverage relative to that domain's files on disk, not the whole vault.
        disk = [p for p in disk_all if domain_of(p) == args.domain]
    else:
        paths, titles = paths_all, titles_all
        domain_counts = Counter(domains_all)
        sections = [("Domains", domain_counts.most_common(args.top), n_rows)]
        header = f"{n_rows:,} notes"
        disk = disk_all

    indexed = len(set(paths))
    coverage = 100.0 * indexed / len(disk) if disk else 0.0
    header += f" · coverage {coverage:.0f}% ({indexed:,}/{len(disk):,} files)"

    folder_counts = Counter("/".join(p.split("/")[:2]) for p in paths)
    kw_counts = Counter()
    for t in titles:
        for w in re.findall(r"[a-záéíóúñü]{4,}", (t or "").lower()):
            if w not in STOPWORDS:
                kw_counts[w] += 1
    sections.append(("Folders (level 2)", folder_counts.most_common(args.top), len(paths)))
    sections.append(("Keywords (titles)", kw_counts.most_common(args.top), len(titles)))

    palette = ("cyan", "green", "yellow", "magenta", "blue", "red")
    if HAVE_RICH:
        console = Console()
        console.print(Panel(header, title="Topics", border_style="cyan"))
        for si, (title, items, denom) in enumerate(sections):
            if not items:
                continue
            color = palette[si % len(palette)]
            mx = items[0][1]
            tbl = Table(title=title, show_header=False, box=None, padding=(0, 1))
            tbl.add_column("name", style="bold", no_wrap=True, max_width=38, overflow="ellipsis")
            tbl.add_column("bar", no_wrap=True)
            tbl.add_column("n", justify="right")
            tbl.add_column("pct", justify="right", style="dim")
            for name, cnt in items:
                tbl.add_row(
                    _clean_name(name),
                    f"[{color}]{_bar(cnt, mx)}[/{color}]",
                    f"{cnt:,}",
                    f"{100.0 * cnt / denom:.1f}%",
                )
            console.print(tbl)
            console.print()
    else:
        print(f"== {header} ==")
        for title, items, denom in sections:
            if not items:
                continue
            print(f"\n{title}")
            mx = items[0][1]
            for name, cnt in items:
                print(
                    f"  {name:<44} {_bar(cnt, mx)} {cnt:>7,} "
                    f"{100.0 * cnt / denom:>6.1f}%"
                )
        print()



# ── Watch (fswatch + debounce) ─────────────────────────────────────────────

# Debounce: wait this long after last event for a file before processing it
DEBOUNCE_SEC = float(os.environ.get("WATCH_DEBOUNCE_SEC", "3.0"))
# Batch flush: process accumulated changes at least this often
BATCH_FLUSH_SEC = float(os.environ.get("WATCH_BATCH_SEC", "10.0"))
# Run table.optimize() (compact + cleanup old versions) every N batch flushes.
OPTIMIZE_EVERY = int(os.environ.get("WATCH_OPTIMIZE_EVERY", "20"))


def cmd_watch(args):
    """Watch vault for .md changes, debounce, re-embed in batches."""
    # Verify fswatch
    if not shutil.which("fswatch"):
        print("ERROR: fswatch not found. Install with: brew install fswatch")
        sys.exit(1)

    # Verify Ollama can actually embed right now — a wedged server is the
    # common failure, and a tags-only check cannot see it.
    cause, fix = preflight_embed()
    if fix:
        print_embed_failure(cause, fix)
        sys.exit(1)

    db = get_db()
    table = create_table(db)

    # fswatch flags:
    #   --event Created Updated Removed Renamed   (coalesce labels)
    #   --exclude '\.obsidian'                     (Obsidian internals)
    #   --exclude '\.smart-env'                    (SC plugin internals)
    #   --exclude '\.hermes'                       (Hermes internals)
    #   --exclude '/\.'                            (hidden files/dirs)
    #   --recursive
    #   --latency 0.5                              (poll interval)
    fswatch_cmd = [
        "fswatch",
        "--event", "Created",
        "--event", "Updated",
        "--event", "Removed",
        "--event", "Renamed",
        "--exclude", r"\.obsidian",
        "--exclude", r"\.smart-env",
        "--exclude", r"\.hermes",
        "--exclude", r"/\.[^/]",  # hidden files but allow root
        "--recursive",
        "--latency", "0.5",
        VAULT,
    ]

    print(f"Watching vault for .md changes...")
    print(f"  Vault: {VAULT}")
    print(f"  Debounce: {DEBOUNCE_SEC}s  |  Batch flush: {args.batch}s")
    print(f"  Press Ctrl+C to stop.\n")

    # pending: {rel_path: ("upsert"|"remove", last_event_time)}
    pending: dict[str, tuple[str, float]] = {}
    last_flush = time.time()
    flush_count = 0

    proc = subprocess.Popen(fswatch_cmd, stdout=subprocess.PIPE)
    fd = proc.stdout.fileno()
    buf = ""
    notify = getattr(args, "notify", False)
    batch_sec = args.batch

    try:
        while True:
            # Wake on fswatch data OR the next flush deadline, so pending
            # changes flush even when no further events arrive.
            if pending:
                now = time.time()
                deadlines = [
                    last_flush + batch_sec,
                    min(t for _, t in pending.values()) + DEBOUNCE_SEC,
                ]
                timeout = max(0.0, min(deadlines) - now)
            else:
                timeout = None  # block until the next event

            ready, _, _ = select.select([fd], [], [], timeout)
            if ready:
                chunk = os.read(fd, 65536)
                if not chunk:
                    print("fswatch exited unexpectedly.")
                    break
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith(VAULT):
                        continue
                    rel = os.path.relpath(line, VAULT)
                    if not rel.endswith(".md"):
                        continue
                    if rel.startswith(SKIP_PREFIXES):
                        continue
                    full = os.path.join(VAULT, rel)
                    if os.path.exists(full):
                        pending[rel] = ("upsert", time.time())
                    else:
                        pending[rel] = ("remove", time.time())

            # Flush evaluation — runs on every wake, event or timer.
            now = time.time()
            if not pending:
                continue
            all_settled = all(now - t >= DEBOUNCE_SEC for _, t in pending.values())
            if all_settled or now - last_flush >= batch_sec:
                try:
                    flush_batch(table, pending, notify=notify)
                except Exception as e:
                    # Hold changes and re-settle: bounded retry in DEBOUNCE_SEC,
                    # no hot loop, nothing dropped.
                    print(f"  ! flush failed, retrying: {e}")
                    for k in pending:
                        pending[k] = (pending[k][0], time.time())
                else:
                    pending.clear()
                last_flush = now
                # Periodically compact the table so the accumulated
                # delete+add versions don't balloon the on-disk size.
                flush_count += 1
                if flush_count % OPTIMIZE_EVERY == 0:
                    try:
                        table.optimize(cleanup_older_than=timedelta(hours=1))
                        print("  ♻ optimized table (compacted versions)")
                    except Exception as e:
                        print(f"  ! optimize failed: {e}")

    except KeyboardInterrupt:
        print("\nStopping watcher...")
    finally:
        if pending:
            print(f"Flushing {len(pending)} pending changes...")
            try:
                flush_batch(table, pending, notify=notify)
            except Exception as e:
                print(f"  ! final flush failed: {e}")
        proc.terminate()
        proc.wait()
        proc.stdout.close()
        print("Stopped.")


def flush_batch(table, pending: dict[str, tuple[str, float]], notify: bool = False):
    """Process a batch of pending changes (upserts + removes) in bulk.

    Embs are still done per-file (with per-file error isolation), but the
    writes are batched: one delete (IN) + one add for upserts, one delete (IN)
    for removes — instead of 2 dataset versions per file. Falls back to
    per-row writes if the batch write fails, so one bad row never loses the rest.
    """
    upsert_records = []
    removes = []

    for rel, (action, _) in pending.items():
        if action == "upsert":
            full = os.path.join(VAULT, rel)
            if not os.path.exists(full):
                # File was deleted after being queued as upsert
                removes.append(rel)
                continue
            try:
                text = read_note(rel)
            except Exception as e:
                print(f"  ! read failed  {rel}: {e}")
                continue
            if len(text) < 50:
                continue
            try:
                vec = embed_text(text)
            except Exception as e:
                print(f"  ! embed failed  {rel}: {e}")
                continue
            upsert_records.append({
                "path": rel,
                "title": extract_title(rel),
                "folder": os.path.dirname(rel) or ".",
                "domain": domain_of(rel),
                "char_count": len(text),
                "vector": vec,
            })
        elif action == "remove":
            removes.append(rel)

    # Removes — one delete with an IN predicate.
    if removes:
        try:
            table.delete(f"path IN ({', '.join(sql_quote(r) for r in removes)})")
            for rel in removes:
                print(f"  ✗ removed  {rel}")
        except Exception as e:
            print(f"  ! remove batch failed, isolating: {e}")
            for rel in removes:
                try:
                    table.delete(f"path = {sql_quote(rel)}")
                    print(f"  ✗ removed  {rel}")
                except Exception as e2:
                    print(f"  ! remove failed  {rel}: {e2}")

    # Upserts — one batched write (single delete + add).
    if upsert_records:
        t0 = time.time()
        try:
            upsert_notes(table, upsert_records)
        except Exception as e:
            # Batch failed — isolate per-record so one bad row doesn't lose the rest.
            print(f"  ! update batch failed, isolating: {e}")
            for rec in upsert_records:
                try:
                    upsert_notes(table, [rec])
                    print(f"  ✓ updated  {rec['path']}")
                except Exception as e2:
                    print(f"  ! update failed  {rec['path']}: {e2}")
            return
        for rel in (r["path"] for r in upsert_records):
            print(f"  ✓ updated  {rel}")
        elapsed = time.time() - t0
        print(f"  (batch {len(upsert_records)} files, {elapsed:.2f}s)")
        if notify:
            names = [os.path.basename(r["path"]) for r in upsert_records]
            if len(names) <= 3:
                detail = ", ".join(names)
            else:
                detail = f"{', '.join(names[:3])} +{len(names)-3} more"
            removed_part = f", {len(removes)} removed" if removes else ""
            _notify("Vault DB updated", f"{detail}{removed_part} · {elapsed:.1f}s")

    # Remove-only batch notification
    if notify and removes and not upsert_records:
        names = [os.path.basename(r) for r in removes]
        if len(names) <= 3:
            detail = ", ".join(names)
        else:
            detail = f"{', '.join(names[:3])} +{len(names)-3} more"
        _notify("Vault DB updated", f"✗ {detail} removed")


def cmd_remove(args):
    """Manually remove a note's embedding from the DB."""
    db = get_db()
    if not has_table(db):
        print("No index found.")
        return
    table = db.open_table(TABLE)
    # Confirm the row exists before claiming a removal — a delete on a path
    # that was never indexed is a silent no-op, which hides typos.
    present = any(r["path"] == args.path for r in table.to_arrow().to_pylist())
    if not present:
        print(f"Not in the index: {args.path}")
        print("  Nothing removed. Paths are vault-relative and case-sensitive.")
        sys.exit(1)
    try:
        table.delete(f"path = {sql_quote(args.path)}")
        print(f"Removed: {args.path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_plist(args):
    """Print or install a launchd plist with current paths filled in."""
    import xml.sax.saxutils as saxutils
    py = shutil.which("python3") or sys.executable
    script = os.path.abspath(__file__)
    wd = os.path.dirname(script)
    vault = VAULT or "$VAULT_PATH"

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.vault.index-watcher</string>

    <key>ProgramArguments</key>
    <array>
        <string>{saxutils.escape(py)}</string>
        <string>-u</string>
        <string>{saxutils.escape(script)}</string>
        <string>watch</string>
        <string>--notify</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{saxutils.escape(wd)}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/tmp/vault-index-watcher.log</string>

    <key>StandardErrorPath</key>
    <string>/tmp/vault-index-watcher.err</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>VAULT_PATH</key>
        <string>{saxutils.escape(vault)}</string>
    </dict>
</dict>
</plist>"""

    if args.install:
        dest = os.path.expanduser("~/Library/LaunchAgents/com.vault.index-watcher.plist")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            f.write(plist)
        print(f"✓ Wrote {dest}")
        subprocess.run(["launchctl", "load", dest], check=False)
        subprocess.run(["launchctl", "start", "com.vault.index-watcher"], check=False)
        print("✓ Watcher started — index will auto-update on file changes.")
        print(f"  Logs: /tmp/vault-index-watcher.log")
    else:
        print(plist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obsidian vault vector DB (LanceDB + nomic-embed-text)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  vault_db.py --db ./lancedb-work sync              # separate DB per vault
  vault_db.py doctor                                # is my setup healthy?
  vault_db.py search "deleuze fold" --top 5
  vault_db.py search "sax harmonizer" --folder 06-research
  vault_db.py sync                                  # after adding/renaming/deleting notes
  vault_db.py sync --changed                        # also re-embed notes you edited
  vault_db.py index --file 06-research/foo.md       # re-embed one note
  vault_db.py stats --health

note:
  `index` alone does nothing; it needs an action flag (--all/--missing/--file/
  --prune/--reclassify/--test). For the everyday reconcile, use `sync`.
""",
    )
    parser.add_argument(
        "--db", type=str, default=None, metavar="DIR",
        help="LanceDB directory (default: ./lancedb, or $DB_DIR)",
    )
    sub = parser.add_subparsers(dest="command")

    p_index = sub.add_parser(
        "index", help="Index notes into LanceDB (needs an action flag; see --help)"
    )
    p_index.add_argument("--test", type=int, metavar="N", help="Timing test on N files")
    p_index.add_argument("--all", action="store_true", help="Index entire vault")
    p_index.add_argument("--file", type=str, metavar="PATH", help="Upsert single note")
    p_index.add_argument("--prune", action="store_true", help="Remove DB rows for deleted files")
    p_index.add_argument("--missing", action="store_true", help="Index only files not yet in DB")
    p_index.add_argument("--reclassify", action="store_true", help="Recompute domain for rows whose path changed folders (no Ollama)")

    p_search = sub.add_parser("search", help="Semantic search")
    p_search.add_argument("query", type=str, help="Search query")
    p_search.add_argument("--top", type=_positive_int, default=10, help="Number of results")
    p_search.add_argument(
        "--folder", type=str, default=None, help="Filter by folder prefix"
    )
    p_search.add_argument("--json", action="store_true", help="Emit structured JSON")
    p_search.add_argument(
        "--no-snippet", action="store_true", help="Skip snippet extraction (faster)"
    )

    p_stats = sub.add_parser("stats", help="Show DB stats")
    p_stats.add_argument("--health", action="store_true", help="Show index health report")

    p_doctor = sub.add_parser(
        "doctor", help="Check the environment, Ollama, the model, and index health"
    )

    p_sync = sub.add_parser("sync", help="Reconcile DB with filesystem")
    p_sync.add_argument(
        "--no-embed", action="store_true",
        help="Only reclassify domains and prune ghosts; skip embedding missing files",
    )
    p_sync.add_argument(
        "--changed", action="store_true",
        help="Also re-embed notes whose content changed (compares stored length vs disk)",
    )
    p_sync.add_argument(
        "--notify", action="store_true",
        help="Fire macOS notification on completion (built-in osascript, no deps)",
    )

    p_topics = sub.add_parser("topics", help="Topic overview: domains, folders, keywords")
    p_topics.add_argument("--top", type=_positive_int, default=10, help="Rows per section")
    p_topics.add_argument("--domain", type=str, default=None, help="Restrict folders/keywords to one domain")

    p_watch = sub.add_parser("watch", help="Watch vault for changes (auto re-embed)")
    p_watch.add_argument(
        "--batch", type=float, default=BATCH_FLUSH_SEC,
        help=f"Max seconds between batch flushes (default {BATCH_FLUSH_SEC})"
    )
    p_watch.add_argument(
        "--notify", action="store_true",
        help="Fire macOS notification on each batch update",
    )

    p_remove = sub.add_parser("remove", help="Remove a note's embedding")
    p_remove.add_argument("path", type=str, help="Vault-relative path to remove")

    p_plist = sub.add_parser("install-plist", help="Print or install a ready-to-use launchd plist")
    p_plist.add_argument("--install", action="store_true", help="Write to ~/Library/LaunchAgents and load (no sudo needed)")

    args = parser.parse_args()

    if args.db:
        DB_DIR = args.db

    if not VAULT and args.command not in ("install-plist", "doctor", None):
        sys.exit("ERROR: VAULT_PATH not set. Export it or pass the vault path:\n"
                 "  export VAULT_PATH=/path/to/your/vault\n"
                 "  python3 vault_db.py sync")

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "sync":
        cmd_sync(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "topics":
        cmd_topics(args)
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "install-plist":
        cmd_plist(args)
    else:
        parser.print_help()
        sys.exit(2)

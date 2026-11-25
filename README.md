# VaultIndex

Semantic search and live indexing for Obsidian vaults. Embeds markdown notes via Ollama (`nomic-embed-text`), stores vectors in LanceDB, and watches for changes with `fswatch` so the index stays current. macOS notifications on every batch update.

```
vault_db.py search "deleuze fold" --top 5
vault_db.py watch --notify
```

## How it works

```
┌──────────────┐    fswatch     ┌──────────────┐    embed     ┌──────────┐
│  Obsidian    │ ──────────────→ │  vault_db.py │ ──────────→ │  LanceDB │
│  vault .md   │  Created/Upd/  │  debounce     │  nomic-     │  vectors │
│  files       │  Removed/Ren   │  3s · batch   │  embed-text │  + meta  │
└──────────────┘                │  10s          │             └──────────┘
                                │  notify       │
                                └──────────────┘
                                      │
                                      ▼
                                 macOS notification
                               "README.md, ssca.md · 4.2s"
```

- **`sync`** — one-shot reconcile: prunes deleted files, reclassifies moved folders, embeds new/changed notes.
- **`watch`** — persistent daemon: monitors the vault with `fswatch`, debounces writes (Obsidian saves in bursts), re-embeds in batches every 10s max.
- **`search`** — semantic search: embed your query → cosine similarity over the LanceDB index.
- **`stats`** — index health: total notes, domain breakdown, disk size.

## Setup

### Prerequisites

**Ollama must be running** with the embedding model. Without it, nothing works — embedding is the core pipeline.

```bash
# Install and start Ollama
brew install ollama
ollama serve &                    # must stay running
ollama pull nomic-embed-text      # 768-dim vectors, ~274MB

# fswatch (for watch mode)
brew install fswatch

# Python deps
pip install -r requirements.txt
```

### Configuration

`VAULT_PATH` is **required** — there is no default. Set it before running any command:

```bash
export VAULT_PATH="/path/to/your/obsidian/vault"
```

Other tunables at the top of `vault_db.py`:

```python
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"      # 768-dim vectors
DB_DIR = "./lancedb"            # or set DB_DIR env var
```

### Multiple vaults

Each vault needs its own LanceDB database — they don't mix. Use `--db` to switch:

```bash
# First vault — default location
python3 vault_db.py sync

# Second vault — separate database
python3 vault_db.py --db ./lancedb-work sync
python3 vault_db.py --db ./lancedb-work search "quarterly report"
```

Or set `DB_DIR` in your environment for persistent switching. The `--db` flag overrides it per-command.

### Changing the embedding model

Any Ollama embedding model works — just pull it and update `MODEL`:

```bash
ollama pull mxbai-embed-large      # 1024-dim, strong multilingual
ollama pull bge-m3                 # 1024-dim, state-of-the-art
ollama pull nomic-embed-text       # 768-dim, fast, default
```

Then edit the `MODEL` constant at the top of `vault_db.py`. After switching, re-index from scratch:

```bash
rm -rf lancedb/
python3 vault_db.py sync
```

Vector dimensions are model-specific. LanceDB schema is built from the first embedding call — no manual config needed.

### Domain classification

`domains.json` maps vault folder prefixes to domain labels used in `stats` and `topics`. Edit it to match your vault structure. If the file is missing or invalid JSON, built-in defaults are used. Everything works without it — unclassified folders show as "other."

### First index

```bash
python3 vault_db.py sync
```

This embeds every `.md` note in the vault (~5,000 notes takes ~10–15 minutes). Subsequent runs only embed what's new or changed.

## The `vault` command

`bin/vault` is the single entry point. It finds this repo from its own location
(symlinks resolved), resolves the vault from the working directory, and routes
subcommands to `vault_db.py` or `organize.py`. Bare words become a search.

```bash
cp bin/vault ~/bin/vault && chmod +x ~/bin/vault   # copy, never symlink
cd ~/any/vault && vault organize plan 00-inbox --json
```

Copy rather than symlink: the wrapper finds this repo from its own path either
way, but a symlink turns any write to `~/bin/vault` into a write to the tracked
file, and a shim that execs itself looks identical to a working install until it
hangs. Record machine paths in `~/.config/vault/config` instead.

Everything instance-specific is overridable, so the repo stays generic:

| Variable | Meaning |
|---|---|
| `VAULT_TOOLS` | where the venv lives |
| `VAULTINDEX_HOME` | this checkout (default: the script's parent) |
| `VAULT_PATH` | the vault to act on (default: resolved from cwd) |
| `VAULT_DEFAULT` | fallback when the cwd has no vault |
| `DB_DIR` | index location (default: the vault's `.vaultindex.json`) |
| `DOMAINS_PATH` | taxonomy file (default: the vault's own, else built-in) |

Exit codes: 0 success, 1 unhealthy or a missing engine/venv, 2 bad usage.

## Commands

### `sync` — reconcile index with filesystem

```bash
python3 vault_db.py sync              # prune ghosts, embed new files
python3 vault_db.py sync --changed    # also re-embed edited notes
python3 vault_db.py sync --notify     # macOS notification on completion
python3 vault_db.py sync --no-embed   # only prune + reclassify, skip embedding
```

### `watch` — live monitor (daemon)

```bash
python3 vault_db.py watch             # run in foreground
python3 vault_db.py watch --notify    # + macOS notification per batch
python3 vault_db.py watch --batch 15  # flush every 15s instead of 10s
```

The watcher uses macOS kernel-level file events (FSEvents via `fswatch`) — no polling. When you save a note in Obsidian, here's what happens:

1. **fswatch** detects the change (Created/Updated/Removed/Renamed)
2. **Debounce 3s** — Obsidian writes in bursts (save → auto-save → plugin), so the watcher waits for the file to settle before re-embedding
3. **Batch flush ≤10s** — if you edit 5 files rapidly, they're grouped into one Ollama embedding batch
4. **Re-embed** — note content (minus YAML frontmatter) is sent to Ollama, the 768-dim vector is stored in LanceDB
5. **Notify** — if `--notify`, macOS notification shows which files were indexed

Hidden dirs (`.obsidian`, `.smart-env`, `.hermes`) are excluded. If fswatch dies, the watcher exits — launchd restarts it automatically.

**Tuning intervals:** See `.env.example` — set `WATCH_DEBOUNCE_SEC` (1–5s) and `WATCH_BATCH_SEC` (5–30s) for your workflow.

For persistent background watching:

```bash
python3 vault_db.py install-plist --install
```

That's it — generates the plist with all paths filled in, writes it to `~/Library/LaunchAgents/`, and starts the watcher. Logs go to `/tmp/vault-index-watcher.log`.

### `search` — semantic search

```bash
python3 vault_db.py search "supply chain sustainability" --top 5
python3 vault_db.py search "sousveillance" --folder 06-research
```

### `stats` — index overview

```bash
python3 vault_db.py stats             # total notes, domain breakdown
python3 vault_db.py stats --health    # + Ollama status, embed latency
```

### `doctor` — full health check

```bash
python3 vault_db.py doctor
```

Runs these checks in order, with actionable fixes for each failure:

| Check | What it verifies | If it fails |
|---|---|---|
| Python environment | fswatch installed, LanceDB importable | `brew install fswatch` / `pip install lancedb` |
| Ollama reachable | Server responding at `OLLAMA_URL` | `ollama serve` |
| Model installed | `nomic-embed-text` (or your `MODEL`) pulled | `ollama pull nomic-embed-text` |
| Model loaded | Model resident in GPU/RAM | Prints warm-up `curl` command |
| Embed latency | Real embed call, measured response time | Diagnoses wedged worker vs cold load |
| Index integrity | Ghost rows (deleted files still indexed), missing files (on disk not embedded), disk size | Run `sync` to reconcile |

### `topics` — domain overview

```bash
python3 vault_db.py topics            # domains, top folders, top keywords
python3 vault_db.py topics --domain research
```

### `index` — targeted operations

```bash
python3 vault_db.py index --file 06-research/foo.md   # re-embed one note
python3 vault_db.py index --missing                    # embed only unindexed
python3 vault_db.py index --prune                      # remove ghost rows
python3 vault_db.py index --reclassify                 # fix domains after moves
```

### `remove` — delete one embedding

```bash
python3 vault_db.py remove path/to/deleted-note.md
```

### `install-plist` — background watcher (macOS)

```bash
python3 vault_db.py install-plist --install   # one-step: generate + write + load
python3 vault_db.py install-plist             # print plist to stdout for inspection
```

All commands accept `--db DIR` to use a different LanceDB directory.

## Organizing: `organize.py`

Classify notes by their nearest neighbours, then plan, apply and undo moves.
It reads embeddings, never note bodies, and it refuses to invent thresholds.

```bash
python organize.py calibrate                 # derive gates from this corpus (~10 s)
python organize.py plan 00-inbox --json      # duplicates, moves, coverage
python organize.py apply plan.json           # move + relink + reindex
python organize.py undo apply-20261124T101500
```

`plan` reports three things per scope: **duplicates** (nearest neighbour inside
`--dup`), **moves** (modal destination among neighbours outside the note's own
folder), and **unclassified**. It also reports coverage: indexed notes against
markdown plus indexed `.txt` companions, notes too short to index, genuinely
unindexed notes, and non-text files — PDFs, docx, epub — which are classifiable
only through a text sibling of the same basename. `.txt` companions themselves are
never proposed as moves, or a PDF would be separated from its text.

`plan --like <path>` answers the single-file question: nearest notes to that note's
own vector, no query string, no scope.

`--json` on `plan` is bounded: `--limit` (default 20) caps each array, `totals`
stays exact and `truncated` says what was cut. On a 5,000-note vault the default is
about 19 KB instead of 523 KB.

### Calibration

Two numbers come from the corpus, not from taste:

- **vote gate** — the largest distance at which notes that *can* vote (two or more
  legal neighbours) still keep modal agreement at or above 0.55. Coverage is
  reported, not gated: a well-sorted vault has most notes already home, and a
  coverage floor would refuse to calibrate a healthy corpus.
- **duplicate floor** — from labelled `*-dup.md` pairs when they exist (p90 × 1.2,
  capped at the measured 0.08 boundary), otherwise a gap in the within-folder
  histogram.

Both are stored in `.vaultindex.json` with the sweep table, sample size and date, so
the choice is auditable rather than remembered. `plan` warns when the index has
drifted more than `row_delta_trigger` rows since calibration and refuses outright
when the model's dimensions changed.

Calibration costs about 10 s at 600 samples over ~5 k notes, so it is never run by
the watcher: `index`/`sync` bump a version in `.vaultindex/state.json`, the matrix
cache validates against that version, and calibration runs only on `calibrate`, a
model change, or a large row delta.

### Apply and undo

`apply` moves files with `git mv` when the file is tracked, rewrites path-qualified
`[[dir/name]]` links and `![[...]]` embeds in one tree pass, refuses moves that would
collide with an existing basename, and writes a manifest under
`.vaultindex/applied/`. `undo` reverses one manifest, including the link rewrites.
`plan` never mutates anything.

### Tests

```bash
python tests/test_organize.py        # fixture vault, 22 acceptance checks
```

The fixture synthesises vectors so the correct destination is known by
construction, and covers duplicates, movers, stubs, PDF companions, a basename
collision, link rewriting, and a byte-identical apply→undo round trip.

## embed_vault.py — lightweight alternative

A standalone embedder that doesn't need LanceDB or `rich`. Good for one-shot embedding into a numpy array when you just want vectors.

```bash
python3 embed_vault.py "your query"   # search from CLI
```

```python
from embed_vault import VaultEmbedder
ve = VaultEmbedder("/path/to/vault")
ve.build_index()
results = ve.search("deleuze fold", top_k=5)
```

## Notifications

When `--notify` is passed, `vault_db.py` fires native macOS notifications via `osascript` after each batch. Shows which files were indexed:

> **Vault DB updated**
> `README.md, 2021-03-ssca.md, literature-review.md +5 more · 8.1s`

No third-party tools, no TCC permission dance. Just built-in `osascript`.

## Architecture

**Watching (live mode)**

```
fswatch → debounce 3s → batch flush every 10s → embed → upsert LanceDB → notify
```

Events: `Created`, `Updated`, `Removed`, `Renamed`. Hidden dirs (`.obsidian`, `.smart-env`, `.hermes`) are excluded. The 3s debounce handles Obsidian's burst-write pattern. The 10s batch cap prevents unbounded accumulation.

**Syncing (one-shot)**

```
disk files ─┐
            ├─→ diff → prune ghosts → reclassify domains → embed missing → done
index rows ─┘
```

Ghost rows (indexed but file deleted) are pruned in one SQL delete. Domain classification is recomputed from path for any row whose folder changed. Only genuinely new files trigger Ollama embedding.

**Search**

```
query → Ollama embed → L2 distance over LanceDB → top-k results
```

Results include path, title, folder, domain, and a context snippet from disk.

## Files

| File | Purpose |
|---|---|
| `vault_db.py` | Main CLI — sync, watch, search, stats, doctor, topics |
| `embed_vault.py` | Lightweight embedder — numpy-based, no LanceDB |
| `domains.json` | Folder → domain mapping for stats (edit for your vault) |
| `.env.example` | All configurable env vars with suggested values |
| `requirements.txt` | Python dependencies |
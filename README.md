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
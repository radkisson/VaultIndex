# Reproducing this setup

What is reproducible: the engine, the per-vault configuration, the calibration
procedure, and the acceptance tests. What is **not** reproducible yet: a vault's
own structure (folders, hubs, templates). You supply that, or copy an existing
vault's shape.

Read this as a checklist. Each section ends with something observable, so you can
tell whether it worked instead of hoping.

---

## 1. Machine prerequisites (macOS)

| Need | Why | Check |
|---|---|---|
| Python 3.10+ | engine | `python3 -V` |
| Ollama + `nomic-embed-text` | embeddings | `ollama list` shows the model |
| `fswatch` | the watcher | `which fswatch` |
| `pdftotext` (poppler) | PDFs become classifiable through text companions | `which pdftotext` |

```bash
brew install ollama fswatch poppler
ollama serve &            # or the app
ollama pull nomic-embed-text
```

**Observable:** `curl -s localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":"x"}'`
returns a vector, not an error.

## 2. Install the engine

```bash
git clone https://github.com/radkisson/VaultIndex.git ~/Projects/VaultIndex
cd ~/Projects/VaultIndex
python3 -m venv ../vault-tools-venv || true
../vault-tools-venv/bin/pip install -r requirements.txt
```

Put the entry point on PATH **as a copy, never a symlink** — a write through a
symlink silently overwrites the engine's wrapper with itself:

```bash
cp bin/vault ~/bin/vault && chmod +x ~/bin/vault
```

Then record where things live on *this* machine, outside the repo:

```bash
mkdir -p ~/.config/vault
printf 'VAULT_TOOLS=%s\nVAULT_DEFAULT=%s\nVAULTINDEX_HOME=%s\n' \
    "$HOME/Projects/vault-tools-venv" "$HOME/notes" "$HOME/Projects/VaultIndex" \
    > ~/.config/vault/config
```

`VAULTINDEX_HOME` is not optional for a `cp` install: a copy in `~/bin` cannot
find the repo from its own path, so it must be told where the engine is.

`bin/vault` sources `~/.config/vault/config` (or `$XDG_CONFIG_HOME/vault/config`)
before anything else, so no machine paths live in the public repo. Supported keys:
`VAULT_TOOLS`, `VAULT_DEFAULT`, `VAULTINDEX_HOME`, `DB_DIR`, `DOMAINS_PATH`.

**Observable:** `vault --help` prints the command list; `scripts/check.sh` runs
the fixture suite and ends with `done`. With no config and no vault in the
working directory, `vault` explains what to set rather than guessing.

Install the guards, or the checks only run when you remember to run them:

```bash
scripts/install-hooks.sh
```

`pre-commit` runs `check.sh` whenever the engine, wrapper, scripts or tests are
staged. `pre-push` refuses to publish a commit whose diff contains a
machine-local path or your username, and refuses any ref named like a backup.
Both are local to a clone; `--no-verify` bypasses them deliberately.

## 3. Point it at a vault

A vault is any directory of markdown. Conventions live in
`<vault>/.vaultindex.json`:

```json
{
  "version": 1,
  "default_folder": "00-inbox",
  "never_dest": ["00-inbox", "99-archive"],
  "db": "/absolute/path/to/index/lancedb",
  "model": { "name": "nomic-embed-text", "dims": 768 },
  "calibration": null
}
```

`db` may be omitted; then the index lives at `<vault>/.vaultindex/lancedb`.
`never_dest` must name the folders that can never be a destination (a queue, an
archive) — with none, every note is eligible to be moved into the queue.

```bash
cd /path/to/vault
vault index --all            # first build; ~5,000 notes is 10-15 min
vault organize calibrate     # ~8 s at 5k notes
vault organize plan 00-inbox # read the plan, change nothing
```

**Observable:** `vault organize plan` prints a gate, a dup floor with its source
(`labelled` or `gap`), counts, and a coverage line whose numerator never exceeds
its denominator.

## 4. Keep it current

```bash
vault install-plist          # prints a launchd job
vault install-plist --install
```

The plist must carry real `VAULT_PATH`, `DB_DIR` and `DOMAINS_PATH` values —
launchd performs no substitution, and a literal `$VAULT_PATH` is a silent no-op.
Each vault needs its own label (`com.vault.index-watcher.<vault>`), or installing
the second one replaces the first.

**Observable:** create a scratch note, wait ~15 s, and `state.json` shows a higher
`version`. If it does not, the watcher is not running the engine.

## 5. Invariants that must hold after setup

- `scripts/check.sh` passes (fixture suite).
- `vault doctor` ends in `RESULT: PASS`.
- `plan` refuses when there is no calibration, and refuses when the model name
  or dimensions differ from the calibration.
- `plan` warns when the newest note in scope is more than an hour newer than the
  last index write — that is the dead-watcher signal.
- `apply` on a dry fixture moves files and `undo` restores a byte-identical tree.

## 6. Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `no index at … and no usable cache` | recorded `db` is from another machine | pass `--db`, or re-run `calibrate` here |
| `cannot calibrate: no threshold keeps agreement >= 0.55` | the corpus has no usable folder structure (everything in one folder) | fix the folders; the tool refuses rather than guessing |
| `index dimensions … != calibration dimensions` | model swapped | `vault index --all` then `calibrate` |
| plan counts look wrong, notes know they changed | watcher down | `vault sync`; check the plist and `state.json` age |
| `vault: cannot find the wrapper` | volume not mounted / repo moved | mount it, or set `VAULTINDEX_HOME` |
| every invocation hangs with no output | the wrapper execs itself (a shim written through a symlink) | `rm` the symlink, `git checkout` the wrapper, copy it instead |

## 7. What is deliberately not covered

- **Vault scaffold.** `vault new` does not exist. Folder taxonomy, hub notes and
  templates are yours to supply; the organizer only reasons about what exists.
- **`apply` on a real vault.** Proven on the synthetic fixture (byte-identical
  undo) but never run against 5,000 live notes with real wikilinks.
- **CI.** `scripts/check.sh` is the acceptance gate; nothing runs it for you.
- **Per-domain calibration.** One gate serves the whole corpus.

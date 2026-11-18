# Decisions

This file records WHY the system is built the way it is: non-obvious choices, the owner's rules,
workarounds for external limitations. How processes work step by step — `docs/09-PROCESSES.md`,
what exists in the code — `02-ARCHITECTURE.md`, facts about Forge Neo and upstream — `docs/external-apis/`.
**Only decisions in force live here.** Revoked ones are deleted; history is in git. No numbers or dates.
New entry: `### Short title` followed by `Why: …` (1–5 lines, plus the pitfall if there is one).
Entries marked *reconstructed from code* were inherited from the original author; the reason is our best reading.

---

## Process and documentation

### Documentation only in `docs/`, written for the current state
Why: a log that grows as a chronicle becomes unreadable within a month — you cannot find the rule in force.
Documents are rewritten for the current state; history is in git. Decision numbers and dates are not used in code or
documents. The `.claude/` folder is for Claude Code settings, not documentation.

### Everything a developer reads is in English
Why: the original code and README were in Chinese; the extension is public and maintained by people who read English.
Docs, comments, docstrings and commit messages — English; the owner is addressed in Russian.

### Tests mirror the code folders
Why: from a file's path you immediately see where its tests are, and vice versa.

### `pytest.ini` lives in `tests/`, tests run as `python -m pytest tests`
Why: the root holds ComfyUI's package `__init__.py` (ComfyUI-only); with the ini in the root pytest imports it and fails.
A root `conftest.py` hook cannot stop that (hooks for the root come from its parent), and `--rootdir` in `addopts` is ignored.
Default import mode: importlib mode would register `tests/seedvr2_vendor` as the `seedvr2_vendor` package and hide the real one.

### Branches: simplified Git Flow
Why: one developer, no versioned releases yet. `main` + short `feature/*` / `fix/*` branches merged back with green CI.
Switch to full Git Flow when releases are tagged and users pin versions. Agreed with the owner.
Pitfall: the extension is public and Forge's "Check for updates" pulls `main` directly, so `main` *is* production.
A branch is merged into `main` only after a live run on Forge Neo and with the owner's consent; pushes — only with consent.

### No personal data of the owner in the repository
Why: the repository is public. Hardware, names, contacts and similar details of the owner are not written into docs, code or
commits; hardware targets are described as a class ("8 GB GPU without bf16"), not as someone's machine.

## Vendored pipeline

### SeedVR2 code is vendored in `seedvr2_vendor/`, not installed as a package
Why (*reconstructed from code*): the upstream ComfyUI port is not a pip package and imports ComfyUI only optionally;
copying it makes the extension self-contained. Pitfall: upstream fixes do not arrive by themselves — local patches are
listed in `docs/external-apis/seedvr2-upstream.md` so a sync does not lose them.

### The vendored package is named `seedvr2_vendor`, not upstream's `src`
Why: Forge keeps every extension folder on `sys.path`, and other SeedVR2 ports also ship a top-level `src`; whichever
loads first is reused by both, so ours ran without its patches. Imports inside the package are relative, so the rename
touches only the code that imports it from outside. A sync copies upstream `src/` into `seedvr2_vendor/`. The runtime check
for a foreign `src` was removed with the rename.

### Fix in `scripts/` first, patch `seedvr2_vendor/` only when unavoidable
Why: every change to vendored code makes the next upstream sync harder. A small marked patch is preferred over
monkey-patching `seedvr2_vendor/` from the script: a patch conflict on sync is visible, a broken monkey-patch fails silently.
Our own helpers that need no Forge imports live in `seedvr2_neo/` (a `.py` in `scripts/` would be loaded by Forge as a script).

### Only Forge Neo is supported; ComfyUI files stay but are not maintained
Why: ComfyUI already has its own SeedVR2 node pack (the upstream). The root `__init__.py` and `seedvr2_vendor/interfaces/` are ComfyUI entry
points, never loaded by Forge; they are kept as close to upstream as possible so `diff -r` stays clean (the root `__init__.py`
only follows the package rename). Not fixed, not tested here.

### The whole repository is Apache-2.0; `NOTICE` credits the sources and lists our changes
Why: the owner's choice — the same license as the vendored code, so the repository has one license. Apache-2.0 §4(b)
change notices: a `Local patch (Forge Neo)` comment at each change plus the list in `external-apis/seedvr2-upstream.md`.
No per-file license headers in our own files: that is ASF guidance, not a §4 condition, and it needs a named copyright holder.
Existing ByteDance / HuggingFace headers in vendored files are kept as they are (§4(c)).

### `requirements.txt` is our own informational list, not upstream's
Why: Forge Neo does not read it (`install.py` installs what is missing), so it only informs a reader; upstream's copy listed
packages the code never imports (`peft`, `matplotlib`).

## Forge integration

### Models live in Forge's `models/SeedVR2`; the extension's own `models/` is only a fallback
Why: the owner's choice — users keep all models in one place, and it survives reinstalling the extension. The folder comes
from `shared.models_path` (follows `--model-ref` / `--data-dir`), never from the working directory. The first candidate
holding both a DiT and a VAE wins, so a stray or partial set in one place does not hide a full set in the other.

### 3B or 7B is read from the checkpoint header, not from the file name
Why: users rename files; a 7B built with the 3B config fails with a shape error. Block count (32 / 36) in the header is
cheap to read and exact. The file name is only a fallback when the header cannot be read.

### A selectable Script, not an always-on script or an Extras upscaler
Why (*reconstructed from code*): a Script owns `run()` and can replace the whole generation, which lets img2img skip
diffusion and txt2img chain "generate → upscale" in one click.

### txt2img calls `process_images` from inside `run()`
Why (*reconstructed from code*): the base image must be produced by Forge with all other extensions applied; the script is
removed from `p.scripts.scripts` for that call and restored in `finally`.

### Interruption checks only `shared.state.interrupted`
Why (*reconstructed from code*): the original author removed the `stopping_generation` check deliberately. Likely reason:
"Stop generating" in Forge means "finish the current image", which should not abort the upscale.

## VRAM and caching

### Unload the SD checkpoint before SeedVR2 by default
Why (*reconstructed from code*): SD and a 7B SeedVR2 do not fit together into 24 GB. Forge reloads the checkpoint on the
next generation.

### DiT and VAE stay cached on the GPU between runs
Why (*reconstructed from code*): repeated upscales skip model loading. Pitfall: the VRAM is not returned to Forge
(`07-RISKS.md` §3) — to be revisited in B06.

### Fixed pipeline settings: batch 1, LAB color correction, `sdpa`, no `torch.compile`, no BlockSwap
Why (*reconstructed from code*): the simplest configuration that works on 24 GB without optional packages.

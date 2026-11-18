# B04 — Repo hygiene

**Status:** 🔄 in progress
**Weight:** 2 (from `docs/03-ROADMAP.md`)
**Decisions this block rests on:** "SeedVR2 code is vendored in `src/`, not installed as a package"; "Fix in `scripts/`
first, patch `src/` only when unavoidable"; "Only Forge Neo is supported; ComfyUI files stay but are not maintained";
"Our `src` must be the one Python loaded; otherwise the run stops with a one-line error".

## Why this block
Removes noise before the real work of B06/B07: compiled files in git, an unused import, a misleading `requirements.txt`,
and — the user-visible part — the `src` package name that makes the extension unusable next to another SeedVR2 extension.
Also closes the legal gap: `src/` is Apache-2.0 code shipped without its license and notice.

## What we do (steps)
1. Branch `feature/b04-repo-hygiene` off `main`.
2. `git rm -r --cached` every tracked `__pycache__/*.pyc`; confirm `.gitignore` (UTF-8) keeps them out.
3. Remove the unused `download_weight` import from `scripts/seedvr2_script.py`.
4. `requirements.txt`: decide with the owner (see open questions) — trim to real dependencies or leave the upstream copy.
5. Rename the vendored package `src` → new name (owner's choice) with `git mv`; update the imports in
   `scripts/seedvr2_script.py`, `tests/src/` (→ mirrored folder), `seedvr2_neo/package_check.py`. Check that nothing inside
   the package imports itself absolutely and that file paths (configs, embeddings) do not depend on the folder name.
6. Package check after the rename: remove or keep (owner's choice); rewrite the decision and the README warning.
7. LICENSE (Apache-2.0 text) + NOTICE (upstream, ByteDance SeedVR, original Forge extension, list of our changes) per
   the `fact-checker` findings; change notice for modified upstream files.
8. Docs: `02-ARCHITECTURE`, `05-DECISIONS`, `external-apis/seedvr2-upstream.md` (rename as a local change), `07-RISKS`
   (close R6 `.pyc`, `src` name, LICENSE, unused import, `requirements.txt`), `01-VISION` REQ-5, README, CLAUDE.md.
9. Full test run; live check in Forge Neo by the owner (3B GGUF upscale in img2img).

## Acceptance criteria
- [ ] `git ls-files | grep -c '\.pyc$'` prints `0`; after a Forge run `git status` shows no `__pycache__`.
- [ ] `grep -rn "download_weight" scripts/` finds nothing.
- [ ] No top-level package named `src` in the repo; `grep -rn "from src\|import src" scripts seedvr2_neo tests` finds nothing.
- [ ] `LICENSE` and `NOTICE` are in the repo root; every modified upstream file carries a change notice.
- [ ] `requirements.txt` matches the owner's choice.
- [ ] All tests pass (same count as before ± moved/removed package-check tests).
- [ ] Owner's live check: img2img upscale with 3B GGUF works in Forge Neo after the rename.
- [ ] Docs list no `src/` path that no longer exists.

## Which checks are needed on `/block-done`
Always: `code-auditor`, `fresh-eyes-reviewer`, `plan-drift-checker`. Additionally (tick what applies):
- [x] `integration-checker` — package rename touches every import path; the extension must still load in Forge
- [ ] `security-auditor` — user data, authorization, external requests, files, money
- [ ] `stress-tester` — user input, money, external services

## Dependencies
B05 closed (done). Owner's answers to the open questions. Owner available for a live check in Forge.

## Out of scope for this block
- Fixing the `finally` keys and `argparse.Namespace` dead code in the script (B06 / B07).
- Deleting ComfyUI files (`__init__.py`, `src/interfaces/`) — kept by decision.
- Upstream sync to a newer version.
- ruff / CI (B09).

## Hidden assumptions
- Vendored code uses only relative imports internally (grep: no absolute `src.` imports inside the package) — to be
  confirmed for `from ...` that reach above the package and for `__file__`-based paths.
- Apache-2.0 obligations (§4: license copy, NOTICE, change notices) — to be confirmed by `fact-checker`, including whether
  upstream ships a `NOTICE` file.
- The original Forge extension (first commits in history) has no license file.

## Owner's answers
- License: Apache-2.0 for the whole repository; the original Forge extension author is credited in NOTICE.
- Package check: removed after the rename (with its tests, decision and README warning rewritten).
- `requirements.txt`: rewritten to the real dependencies; the difference from upstream is listed as a local change.

## Open questions
1. New package name (internal Python package, not the extension name).

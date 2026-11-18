# B04 — code-auditor report

Scope: `git diff -M main...HEAD` on `feature/b04-repo-hygiene` (commit `b1a5b39`), plus a look at the uncommitted
working tree at the time of the audit. Card: `docs/roadmap-blocks/B04-repo-hygiene.md`. Rules: `docs/process/code-style.md`.

Tests: `uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q` →
**254 passed, 1 skipped, 2 xfailed** (was 159/1/2: +100 parametrized cases of `test_no_reference_to_the_old_package_name`,
−5 package-check tests).

## Verified clean (no finding)

- **File set of the rename.** Every non-`.pyc` file of `main:src/` exists in `HEAD:seedvr2_vendor/` (same relative paths);
  all are R100 except `core/model_configuration.py` (R099, the VAE-path patch). `seedvr2_vendor/models/` has 40 tracked
  files, same as `main:src/models/` (the `/models/` ignore is root-anchored). `git ls-files | grep -c '\.pyc$'` → `0`.
- **Imports inside the package.** AST scan of every `seedvr2_vendor/**/*.py`: no relative import climbs above the package,
  no absolute self-import (`src.`, `seedvr2_vendor.`, `common.`, `core.` …), no `sys.path` edits.
- **Dynamic imports / config strings.** `configs_3b|7b/main.yaml` use `__object__.path` keys `dit_3b.nadit`,
  `dit_7b.nadit`, `video_vae_v3.modules.attn_video_vae`, resolved through `MODEL_CLASSES` in
  `seedvr2_vendor/utils/model_registry.py` (relative imports) — no package name involved. No `__inherit__` in any YAML.
- **`__file__`-relative paths.** `get_script_directory()` (`seedvr2_vendor/utils/constants.py:35`) walks three levels up to
  the repo root — name-independent; configs and `pos_emb.pt`/`neg_emb.pt` are joined to it. The only literal `'src/...'`
  path (VAE config) is patched at `seedvr2_vendor/core/model_configuration.py:1121-1124` and resolves to an existing file.
- **Pickled module paths in the embeddings.** `pos_emb/data.pkl` and `neg_emb/data.pkl` reference only `torch._utils`,
  `_rebuild_tensor_v2`, `torch.BFloat16Storage`, `collections.OrderedDict` — nothing from `src`; `weights_only=True` loads them.
- **Package-check removal in `scripts/seedvr2_script.py`.** `importlib` import, `find_shadowed_package_error`, the
  `package_check` import and the `or find_shadowed_package_error()` are all gone; no `VENDORED_PACKAGE` /
  `describe_shadowed` / `package_check` reference left in tracked code, tests or docs (only untracked local `__pycache__`).
  `download_weight` import removed; no remaining use of it.
- **`requirements.txt` vs imports.** Third-party imports on the Forge path (`scripts/`, `seedvr2_neo/`,
  `seedvr2_vendor/` minus `interfaces/`): torch, torchvision, PIL, numpy, safetensors, tqdm, psutil, einops, omegaconf,
  diffusers, rotary_embedding_torch, cv2, gguf (+ optional triton/flash/sage/xformers/bitsandbytes under try, + Forge's own
  gradio/modules). All are listed; `peft` and `matplotlib` are imported nowhere. The header claims match
  `external-apis/forge-neo.md` (file not read by Forge; `gguf` bundled) and the REQ-1 footnote in `01-VISION.md`.
- **LICENSE** is the stock Apache-2.0 text (MD5 `3b83ef96387f14655fc854ddc3c6bd57`, the canonical `LICENSE-2.0.txt`).
- **`.gitignore`** is UTF-8 without BOM.

## Findings

### IMPORTANT

**I-1. 🟡 R6 (S3×L2) — the only runtime code change of the block has no test that fails when it is broken.**
`tests/seedvr2_vendor/test_package_paths.py:27-29` vs `seedvr2_vendor/core/model_configuration.py:1122-1124`.
`test_vae_config_is_where_model_configuration_looks` re-states the expected path (`PACKAGE_DIR / "models" / ...`) and checks
that the file exists; it never touches the expression in `model_configuration.py`. The comment "model_configuration.py builds
this path from its own package folder" is an assertion the test does not verify.
Scenario: during the next upstream sync the patch is re-applied (the upstream doc says "Re-apply after every upstream sync")
as `os.path.dirname(os.path.abspath(__file__))` (one level) or with `'src'`/`'seedvr2_vendor'` doubled → every run with a
VAE not already cached raises on `load_config(...)` (missing YAML) → upscale broken for every user; the test stays green.
Also against code-style "tests mirror code folders": a change to `seedvr2_vendor/core/model_configuration.py` has no test under
`tests/seedvr2_vendor/core/`.
Fix: move the path into a tiny torch-free function next to `get_script_directory()` in `seedvr2_vendor/utils/constants.py`
(marked `Local patch (Forge Neo)`), call it from `model_configuration.py`, and have the test call that function and assert
`is_file()` (then the test fails if the computation is wrong).

### MINOR

**M-1. 🟢 R3 (S3×L1) — the old-name guard regex misses common forms and does not scan YAML.**
`tests/seedvr2_vendor/test_package_paths.py:12` and `:15-19`. The docstring promises "nothing may still reach it by the old
name", but the pattern only catches `from/import [.]src` at line start and a quote followed by `src/` or `src\`.
Proven false negatives (run against `OLD_NAME_REFERENCE`, all return no match):
`os.path.join(script_directory, 'src', 'models', 'video_vae_v3', 'a.yaml')`, `importlib.import_module('src.models.dit_3b.nadit')`,
`Path(root) / "src" / "models"`, `import os, src.utils`, and the YAML form `path: src.models.dit_3b.nadit`. `_python_files()`
never reads `configs_3b/`, `configs_7b/` or `seedvr2_vendor/**/*.yaml`, which are exactly where `config.import_item`
resolves dotted module paths. Scenario: an upstream sync brings a split-join path or a dotted `src.` string → runtime
`FileNotFoundError` / `ImportError`, test green.
Fix: add `['"]src['"]` (separate path component) and `['"]src\.` (dotted string) alternatives, `,\s*src\b` for multi-import,
and include `*.yaml` under `configs_*` and `seedvr2_vendor/` in the scanned files; add the missed forms to
`test_pattern_catches_the_old_forms`.

**M-2. 🟢 R2 (S2×L1) — update via `git pull` aborts for users whose Python rewrote the tracked `.pyc`.**
Deleted files: all `src/**/__pycache__/*.cpython-311.pyc`, `scripts/__pycache__/seedvr2_script.cpython-311.pyc`.
Python 3.11 rewrites a stale `.pyc` (source mtime after checkout ≠ the one in the committed `.pyc` header), so in a 3.11 install
those tracked files are locally modified. Reproduced in a scratch repo: modified tracked `f.pyc`, upstream commit deletes it,
`git pull` → `error: Your local changes to the following files would be overwritten by merge: f.pyc … Aborting`.
Forge's UI update (A1111-derived `fetch` + `reset --hard`; not recorded in `external-apis/forge-neo.md`) is not affected.
After any update path, untracked `.pyc` of modules added after the original commit (e.g. `src/utils/__pycache__/dit_size.*.pyc`)
keep an orphan `src/` folder of `__pycache__` dirs in users' installs — harmless (no `__init__.py`, a namespace portion
never wins over a regular `src` package), only clutter.
Fix: one line in the README / release note: "if `git pull` refuses, run `git checkout -- .` then pull; an empty `src/`
folder left in the extension can be deleted". Optionally verify Neo's update mechanism and record it in `forge-neo.md`.

**M-3. 🟢 R1 (S1×L1) — "kept untouched" statements about the root `__init__.py` are now false.**
`docs/01-VISION.md:24`, `docs/02-ARCHITECTURE.md:118` ("keep untouched"), `docs/04-STATE.md:35` ("ComfyUI files kept
untouched"). The root `__init__.py` is modified in this block (imports `.seedvr2_vendor`); `05-DECISIONS.md` was already
rewritten to "kept as close to upstream as possible … only follows the package rename".
Fix: use the same wording as `05-DECISIONS.md` in these three places.

**M-4. 🟢 R1 (S1×L1) — NOTICE: `requirements.txt` listed as a change to upstream files it does not list, without the marker it promises.**
`NOTICE:16-17`. "Changes to the files above" includes `requirements.txt`, but the "files above" (lines 8-9) do not list it;
and line 16 says changed files carry a `Local patch (Forge Neo)` comment — `requirements.txt` carries only an
"Informational" header. A reader cannot tell whether `requirements.txt` is upstream-derived.
Fix: add `requirements.txt` to the upstream file list on lines 8-9 (or drop line 17 and declare it ours, as
`seedvr2-upstream.md:12` already does — "Ours: … `requirements.txt`" contradicts line 26 of the same file, which lists it as a
local patch; pick one).

**M-5. 🟢 R1 (S1×L1) — stale `src/` paths and test count (card criterion "Docs list no `src/` path that no longer exists").**
At `HEAD`: `.gitignore:23`, `seedvr2_neo/__init__.py:3`, `seedvr2_neo/model_discovery.py:5` — **already fixed in the
uncommitted working tree** at audit time (not yet committed). Still stale: `docs/04-STATE.md:18` (`src/` and the `.pt`
embeddings match upstream…) and `docs/04-STATE.md:22` (159 passed → now 254). Expected to be done on `/block-done`; listed
so it is not missed.
Fix: commit the working-tree fixes; update STATE lines 18 and 22.

**M-6. 🟢 R2 (S2×L1) — LICENSE scope covers code whose author granted no license (owner/legal call, not code).**
`LICENSE`, `NOTICE:3` ("Licensed under the Apache License, Version 2.0"), `README.md` "License". The first commits
(`865879f`, `d6035dc` by the original Forge extension author) contain no license file; `scripts/seedvr2_script.py` descends
from that code. The repository now states Apache-2.0 for all of it. The card records the owner's choice and credits the author
in NOTICE, but crediting is not a grant. Scenario: the original author objects → the claim cannot be backed for those parts.
Fix: owner's decision — ask the original author for permission (or confirm the original repo's terms), and record the answer in
`05-DECISIONS.md`; route through `fact-checker` if needed.

## Status against the uncommitted working tree (re-checked at the end of the audit)

The main chat was editing in parallel; at the end of the audit the working tree (not committed) already contained:
- **M-2** — addressed: README "Updating" step (`git reset --hard origin/main` for hand-cloned installs) and a verified
  `fetch_and_reset_hard` fact in `external-apis/forge-neo.md`. The orphan `src/` `__pycache__` folder is still not mentioned
  (harmless).
- **M-3** — addressed in `01-VISION.md:24` and `02-ARCHITECTURE.md:118`; `04-STATE.md:35` still says "kept untouched".
- **M-4** — addressed: the `requirements.txt` line is removed from NOTICE, and `seedvr2-upstream.md` now says it is not synced
  from upstream (no contradiction with line 12 any more).
- **M-5** — `.gitignore`, `seedvr2_neo/__init__.py`, `seedvr2_neo/model_discovery.py` fixed; `04-STATE.md:18,22` remain.
- **M-6** — recorded as the owner's acceptance in `05-DECISIONS.md` (licence section). Closed as a finding; the legal
  question itself stays with the owner.
- **I-1** — **open.** **M-1** — **open**: only the docstring changed ("no import or path string may still use the old name");
  `os.path.join(..., 'src', 'models', ...)` is a path string and still passes the regex.
- New in the working tree, checked: `scripts/seedvr2_script.py:208` now prints the `ImportError` before returning — fine
  (the error is no longer silent in the console).

## Not findings (checked, rejected)
- `requirements.txt` without `peft`/`matplotlib` cannot break anything: Forge does not read the file (`forge-neo.md`).
- `scripts/seedvr2_script.py:16-20`: the `sys.path` insert is discarded by Forge at load time (`forge-neo.md`, "Script
  loading"); the comment says so ("only covers a host that does not") — accurate.
- `tests/pytest.ini` / `05-DECISIONS.md` claim that importlib mode would register `tests/seedvr2_vendor` as `seedvr2_vendor`:
  consistent with the rootdir being `tests/`; default prepend mode passes (`find_spec("seedvr2_vendor")` resolves to the real
  regular package — a namespace portion could not win anyway).

## Re-check after `b078dcb`

Tests: **271 passed, 1 skipped, 2 xfailed** (`uv run ... pytest tests -q`).

- **I-1 — closed.** `get_vae_config_path()` (`seedvr2_vendor/utils/constants.py:38-44`) is the single source; it is used at
  `seedvr2_vendor/core/model_configuration.py:1121` (import at `:79`, both marked `Local patch (Forge Neo)`). The test calls
  the function and asserts `is_file()` plus `parent == PACKAGE_DIR/models/video_vae_v3`. A wrong `dirname` count gives
  `seedvr2_vendor/utils/models/...` or `<repo>/models/...` (neither exists), so `is_file()` fails; the cwd test pins that the path
  is absolute and cwd-independent.
- **M-1 — closed, with a narrow leftover (new MINOR N-1 below).** All five forms I reported now match; YAML under
  `seedvr2_vendor/` and `configs_*/` is scanned; the negative cases (`src=src`, `mean_src`, `'srcset'`) do not match.
- **M-2 — closed** (README update step, `forge-neo.md` fact, two 🟢 entries in `07-RISKS.md` incl. the orphan `src/` folder).
- **M-3 — closed** in `01-VISION.md`, `02-ARCHITECTURE.md`; `04-STATE.md:35` left for `/block-done`, as agreed.
- **M-4 — closed** (`requirements.txt` line removed from NOTICE; `seedvr2-upstream.md` no longer contradicts itself).
- **M-5 — closed** (`.gitignore`, `seedvr2_neo` docstrings); `04-STATE.md:18,22` left for `/block-done`, as agreed.
- **M-6 — closed** as the owner's recorded decision in `05-DECISIONS.md`.
- New in the fix, checked and fine: the `ImportError` print in `scripts/seedvr2_script.py:208`; the new `07-RISKS.md` entries
  match the code.

### New findings

**N-1. MINOR 🟢 R2 (S2×L1) — unquoted YAML values are still not caught.**
`tests/seedvr2_vendor/test_package_paths.py` (`OLD_NAME_REFERENCE`). The string branch needs a quote before `src`, but YAML
scalars are usually unquoted — the configs themselves have `path: projects.video_diffusion_sr.train` unquoted
(`configs_3b/main.yaml:2`, `configs_7b/main.yaml:2`). Verified no match for `path: src.models.dit_3b.nadit`,
`path: src/models/x.yaml`, `__inherit__: src/models/a.yaml`. Scenario: an upstream sync writes an `__object__.path` or
`__inherit__` as unquoted `src....` → `import_item`/`load_config` fail at runtime, test green.
Fix: add a YAML branch such as `:\s*src[/\.]` (multiline) and those three lines to `test_pattern_catches_old_name_forms`.
Side note, not a finding: the `import` branch also matches `import torch.nn.functional as src` and `kwargs['src']` is matched by
the string branch — false positives only, they fail loudly and cost nothing today (no such line exists).

**N-2. MINOR 🟢 R1 (S1×L2) — the local-patch list names the wrong function.**
`docs/external-apis/seedvr2-upstream.md:27` says the VAE-path patch is in `_create_new_runner`; it is in `_setup_vae_model`
(`seedvr2_vendor/core/model_configuration.py:1069`, use at `:1121`). Scenario: re-applying patches after an upstream sync, the
reader looks for the VAE path in the wrong function. (Present since `b1a5b39`, I missed it in the first pass.)
Fix: replace `_create_new_runner` with `_setup_vae_model` on line 27.

### N-1 fix (main chat)
Fixed: `:\s*src[/\.]` branch added to `OLD_NAME_REFERENCE`; the three YAML lines above are now parametrized cases in
`test_pattern_catches_old_name_forms`. Full run: 274 passed, 1 skipped, 2 xfailed; no real file matches.

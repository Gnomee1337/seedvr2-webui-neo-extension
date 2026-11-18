# Integration check — B04 (rename `src` → `seedvr2_vendor`)

Branch `feature/b04-repo-hygiene` at `b1a5b39`, compared with `main` (`git diff -M main...HEAD`).
Static checks only (no torch in the test env); the owner's live 3B fp8 run on this branch is taken as given.

## Verdict

No BLOCKER, no IMPORTANT. The seams hold: imports, file paths, config object paths and test layout all resolve
after the rename. Two MINOR leftovers (stale `src/` mentions in a comment and two docstrings — the docstrings are
already being fixed in the working tree, uncommitted).

## What was checked and how

| Seam | Method | Result |
|---|---|---|
| Relative imports inside `seedvr2_vendor` | AST walk of all 94 `.py` files, 212 `from .`/`from ..` imports resolved against the file tree; imported names checked against target module top-level names | All resolve inside the package; none climb above `seedvr2_vendor/`. One false alarm: `core/infer.py:33 from ..models.dit_3b import na` targets a folder without `__init__.py` — an implicit namespace package, same as on `main` (the `__init__.py` set is identical before/after: diffed `git ls-tree main -- src` vs `git ls-files seedvr2_vendor`) |
| Absolute self-imports | grep / AST for `import src`, `from src`, `seedvr2_vendor.` inside the package | None |
| File set parity | `git ls-tree -r main -- src` (minus `.pyc`) vs `git ls-files seedvr2_vendor` | 95 = 95, every file renamed at 100% similarity except `core/model_configuration.py` (99%, the VAE path patch) |
| `get_script_directory()` | `seedvr2_vendor/utils/constants.py:35` — three `dirname` up from `seedvr2_vendor/utils/constants.py` | Still the repository root (depth unchanged) |
| DiT config path | `model_configuration.py:724` → `script_directory` + `CONFIG_DIR_BY_DIT_SIZE` (`./configs_3b`, `./configs_7b`) + `main.yaml` | Both files exist at the root |
| VAE yaml path | `model_configuration.py:1122-1124` — now `dirname(dirname(__file__))` = `seedvr2_vendor/` + `models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml` | Exists; also covered by `tests/seedvr2_vendor/test_package_paths.py::test_vae_config_is_where_model_configuration_looks` |
| Text embeddings | `generation_utils.py:537-538`, called from `scripts/seedvr2_script.py:267` and `generation_phases.py:594` with `script_directory` | `pos_emb.pt`, `neg_emb.pt` exist at the root |
| OmegaConf `__object__.path` | `configs_3b/main.yaml:8,48`, `configs_7b/main.yaml:8,45` vs `MODEL_CLASSES` keys (`utils/model_registry.py:17-21`); `common/config.py:103` looks the key up first | `dit_3b.nadit`, `dit_7b.nadit`, `video_vae_v3.modules.attn_video_vae` all present; keys are package-relative, not tied to the folder name. The top-level `projects.video_diffusion_sr.train` is never passed to `create_object` (only `model_loader.py:452` calls it, on the dit/vae sub-configs) — unchanged from `main` |
| Dynamic imports by string | grep `importlib`, `sys.modules`, quoted `"src.`/`'src/` | Only `common/config.py:108` fallback (not reached for the three registered keys) |
| Script imports | AST of `generation_utils.py` / `generation_phases.py` top level vs `scripts/seedvr2_script.py:167-169` | All 9 names + `utils/debug.py` exist. Removed `importlib`, `download_weight`, `package_check` names have no remaining users in `scripts/`, `seedvr2_neo/`, `tests/` |
| Root `__init__.py` (ComfyUI-only) | Targets of lines 7-8 | `ensure_triton_compat` (`optimization/compatibility.py:14`), `comfy_entrypoint`, `SeedVR2Extension` (`interfaces/__init__.py:14,27`) exist. Forge never imports it (`forge-neo.md`) |
| Import at run time in Forge | `forge-neo.md` §Script loading: extension root stays on `sys.path` | The lazy `import seedvr2_vendor...` inside `run()` resolves; no other extension ships a `seedvr2_vendor` package |
| `.gitignore` | `git check-ignore` on `seedvr2_vendor/models/dit_3b/nadit.py`; `git ls-files \| grep -c .pyc` | Not ignored (`/models/` is root-anchored); 0 tracked `.pyc` |
| Test layout | `tests/pytest.ini` (prepend mode, `pythonpath = ..`), no `__init__.py` under `tests/` | `tests/seedvr2_vendor/` cannot shadow the real package (namespace portion loses to the regular package on the path) |
| Compile | `compile()` over every `.py` in the repo | No syntax errors |
| Tests | `uv run ... python -m pytest tests -q` | 254 passed, 1 skipped, 2 xfailed (count grew from 159 because `test_package_paths.py` is parametrized per file; the removed `test_package_check.py` accounts for the rest) |

## Findings

### MINOR — stale `src/` path in `.gitignore` comment
R = S1 × L2 = 2. `.gitignore:23`: "Anchored to the root: src/models/ is source code and must stay tracked."
Scenario: a maintainer reads the comment, does not find `src/`, and "simplifies" the anchored `/models/` to `models/`,
which would silently ignore `seedvr2_vendor/models/` (all DiT/VAE source code) on the next add.
Fix: change to `seedvr2_vendor/models/`.

### MINOR — stale `src/` in `seedvr2_neo` docstrings (fix already in the working tree, uncommitted)
R = S1 × L1 = 1. `seedvr2_neo/__init__.py:3`, `seedvr2_neo/model_discovery.py:5` at `b1a5b39` still name the vendored
pipeline `src/`. The working tree has both changed to `seedvr2_vendor/` but not committed (written with LF; git warns it
will convert to CRLF — harmless). Not caught by `test_package_paths.py` because the regex looks for quoted `'src/'` or
`import src`, not backticked docstring text. Fix: commit the pending edit.

### Note (no action) — leftover `src/` folder on users' disks after update
If a user's Python ever wrote untracked bytecode into `src/**/__pycache__/` (any version other than the tracked
`cpython-311` files), the update removes the sources but leaves `src/` holding only `__pycache__` folders. Without an
`__init__.py` and without sources it is at most an empty namespace portion: our code no longer imports `src`, and a
regular `src` package of another extension still wins over it. Cosmetic only; could be mentioned in release notes.

### Note (no action) — namespace sub-packages
12 folders inside `seedvr2_vendor` have no `__init__.py` (`models/`, `models/dit_3b`, `data/`, …), exactly as upstream.
They work as implicit namespace packages inside a regular package; unchanged by the rename. Worth knowing for any
future tooling (e.g. ruff/mypy package discovery in B09).

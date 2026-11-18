# B05 — integration check (model discovery)

Scope: `git diff 4dc896e` on `feature/b05-model-discovery` — `scripts/seedvr2_script.py`, `seedvr2_neo/model_discovery.py`,
the `src/` patches (`src/utils/constants.py`, `src/utils/dit_size.py`, `src/core/model_configuration.py`), tests, `requirements-dev.txt`.
Focus: paths not covered by the owner's live run (img2img, extension-folder fallback, 7B, caching with a changed folder).
Forge facts cross-checked against a local Forge Neo checkout (`<Forge root>`, Gradio 4.40.0) and `docs/external-apis/forge-neo.md`.

## What was run
- `python -m pytest tests` in a clean venv with only `pytest` + `gguf` (as `requirements-dev.txt`): **46 passed**.
  No `__pycache__` / `.pytest_cache` written into the repository (`PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`).
- An end-to-end simulation outside the repo (cwd set to an unrelated temp folder):
  Forge `models/` without `SeedVR2`, extension `models/seedvr2` (lower case) with a 36-block header named `renamed_big`,
  a 32-block header named `model_7b_but_small`, a VAE, and a **decoy** `./models/SEEDVR2/renamed_big.safetensors` (32 blocks)
  relative to cwd. Result: `scan_models` picks the extension folder; `find_selection_error` is empty;
  `find_model_file(name, scan.folder)` returns the file inside `scan.folder` for both DiTs (decoy ignored);
  `choose_dit_size` → `renamed_big` 7B, `model_7b_but_small` 3B, both "from checkpoint header".
- Forge source read: `modules/scripts.py::load_scripts` / `list_scripts`, `modules/script_loading.py`, `modules/ui_components.py::ToolButton`,
  `modules/gradio_extensions.py` (tooltip patch), `modules/txt2img.py`, `modules/img2img.py`; Gradio 4.40 `components/dropdown.py`, `blocks.py`.

## Seams verified as fitting (no finding)
- **Folder → loader.** `scan.folder` (absolute) → `args.model_dir` (`scripts/seedvr2_script.py:175`) → `prepare_runner(model_dir=)` →
  `configure_runner(base_cache_dir=)` (`src/core/generation_utils.py:494`). All five `find_model_file` call sites in the Forge path
  pass `base_cache_dir` (`src/core/model_configuration.py:721, 1036, 1057, 1109, 1137`); the patch returns the explicit path first
  (`src/utils/constants.py:121-126`), so ComfyUI `folder_paths` and cwd `./models/SEEDVR2` are never consulted when the file exists.
- **Remaining cwd-relative code** (`get_base_cache_dir`, `get_validation_cache_path`, `download_weight`) is reached only from
  `src/interfaces/video_upscaler.py` (ComfyUI node) — not from Forge. `download_weight` is imported but never called by the script (known, B04).
- **No other `"7b" in name` logic** in `src/`: `src/optimization/compatibility.py:780` reads the model module from the loaded config,
  which now follows the header. Loader format detection is by suffix (`src/core/model_loader.py:106,140,590`), consistent with the
  case-sensitive `MODEL_FILE_EXTENSIONS`.
- **Config path.** `CONFIG_DIR_BY_DIT_SIZE` joined with `script_directory` (= extension root via `get_script_directory`) gives the same
  `configs_3b/main.yaml` / `configs_7b/main.yaml` as before. `choose_dit_size` is only reached in `_create_new_runner`, i.e. on a model change.
- **Imports at Forge load.** Forge's `load_scripts` permanently prepends every extension root (`ScriptFile.basedir` = extension path)
  to `sys.path` and additionally puts the current root first while executing each script, so `from seedvr2_neo...` at module level
  resolves to this extension. `seedvr2_neo` needs only the stdlib. `scripts/` contains only the one script, so Forge does not load
  helpers as scripts.
- **Runtime deps.** `dit_size` imports `gguf` lazily and only for `.gguf`; Neo ships its own `gguf` (forge-neo.md); an `ImportError`
  is caught and falls back to the name. `install.py` needs nothing new; `requirements-dev.txt` is test-only and never read by Forge.
- **Refresh handler.** `inputs=[dit_model, vae_model]` → `_refresh_models(current_dit, current_vae)` returns three `gr.update`s for
  `outputs=[dit_model, vae_model, missing_models_note]`. Gradio 4.40 stores the updated component per session (`blocks.py` state),
  Dropdown `type="value"` does no choice validation in `preprocess`, `value=None` with empty choices is accepted.
  `ToolButton(tooltip=...)` works because Forge's `gradio_extensions.py` pops `tooltip` from component kwargs.
  `self.elem_id(...)` differs per tab, so txt2img/img2img buttons do not clash.
- **img2img / API.** The early `Processed(p, [], p.seed, msg)` is the same shape Forge itself builds in `modules/img2img.py`
  (`Processed(p, [], p.seed, "")`); img2img batch mode extends from `proc.images` (empty is fine). API calls bypass Gradio, and
  `run()` re-validates the selection against a fresh scan, so a stale name gives the one-line error.
- **Stored data format.** Dropdown values are still bare file names (ui-config.json, infotext "SeedVR2 Model") — unchanged.

## Findings

### IMPORTANT
1. 🟡 **R6 (S3×L2) — The `src` name collision now breaks the run instead of being harmless.**
   `src/core/model_configuration.py:80`, `scripts/seedvr2_script.py:161-164, 202-203`.
   Scenario: the user also has another extension exposing a top-level `src` — most likely the original (unforked) SeedVR2 Forge
   extension or another port of the same upstream, which fork users often keep installed. At `run()` time all extension roots are on
   `sys.path` and the first one listed (Forge prepends later-loaded extensions in front) or the first `src` already in `sys.modules`
   wins. Before B05 both `src` copies were identical upstream code, so the wrong one still worked. Now our pipeline needs the
   patched `src`: `src.core.generation_utils` → `model_configuration` → `from ..utils.dit_size import ...` raises
   `ImportError: No module named 'src.utils.dit_size'`, caught at line 202 → result "Error: ..." with no console line; upscaling
   is impossible until the other extension is removed. If the other `src` were imported without failing, `find_model_file` would
   be the unpatched cwd-relative one.
   Fix: the real fix is B04's rename of the vendored package; until then (cheap, in this block) — in `run()`, after importing
   `src`, check `os.path.dirname(os.path.dirname(src.__file__))` equals `EXTENSION_ROOT` and return a one-line error naming the
   conflicting folder; add a README note "remove other SeedVR2 Forge extensions"; update R6 in `07-RISKS.md` §5 (higher impact
   now, scenario as above).

### MINOR
2. 🟡 **R4 (S1×L4) — The module-level `sys.path` insert is a no-op under Forge, and the comment / R6 text describe something that does not happen.**
   `scripts/seedvr2_script.py:17-20`. Forge already has the root on `sys.path` (so the `not in` check is false), and during script
   load it works on a temporary list that it replaces with the saved one afterwards (`sys.path = syspath` in `load_scripts`), so even
   a real insert there would be discarded. At `run()` time the root is on `sys.path` but **not necessarily first**. Harmless today
   (the old run-time insert was a no-op for the same reason), but `07-RISKS.md` §5 R6 says "the script puts the extension root first",
   which misleads the B04 analysis. Fix: reword the comment ("Forge puts every extension root on sys.path; this is a fallback for
   loading outside Forge") and the R6 text.

3. 🟢 **R2 (S2×L1) — Model caches are keyed by bare file name while the folder is now re-resolved on every run.**
   `scripts/seedvr2_script.py:175` with `src/core/model_configuration.py:597-601` (DiT), `:614-619` (VAE), `:650-657` (runner template).
   Scenario: a same-named file appears in a different folder (models later copied into Forge's `models/SeedVR2`, which then wins
   over the extension folder) or is replaced in place with different weights (e.g. a 7B renamed to the cached 3B's name). The cached
   DiT/VAE and the runner template (with the old 3B/7B config) are reused, the header is not re-read, `_dit_checkpoint` points to the
   new path but the old weights run. "Force Reload" does not help — it clears only `self.runner_cache` (R9, B06). Identical copies are harmless.
   Fix (B06, together with the real Force Reload): remember `(scan.folder, name, size, mtime)` of the last run and drop the global
   DiT/VAE/runner cache when it changes.

4. 🟢 **R2 (S2×L1) — Silent behaviour change: the "offer all files" fallback is gone.**
   `seedvr2_neo/model_discovery.py:57-58`. The old `ui()` offered every file in the VAE dropdown when no file name contained `vae`
   (and every file as DiT when all contained `vae`). Now a VAE renamed without `vae` is listed as a DiT and the VAE list is empty;
   the note says "No VAE model in <folder>" although the file is there. Not in the card or README.
   Fix: state the rule in README ("the VAE file name must contain `vae`") and in the note text, or keep the old fallback.

5. 🟢 **R2 (S2×L1) — A partial set in Forge's folder hides a complete set in the extension folder, and the note does not say so.**
   `seedvr2_neo/model_discovery.py:69-72`, `:121-125`. If Forge's `models/SeedVR2` holds only a VAE while the extension folder has
   DiT + VAE (users migrating from the old README), the Forge folder is chosen and the DiT list is empty. This follows the recorded
   decision; only the message is incomplete. Fix: when the chosen folder misses a kind, mention that the other candidate folder is not used.

6. 🟢 **R2 (S2×L1) — Empty `Processed` from the early selection error raises in Forge's txt2img "✨ upscale" (hires) button path.**
   `scripts/seedvr2_script.py:94`. `modules/txt2img.py` (`txt2img_upscale`) ends with `processed.infotexts[0]`; with `images=[]`
   and no `infotexts`, Forge builds `infotexts=[]` → `IndexError` in the Gradio handler. Needs the script selected, the ✨ button used
   and no/invalid model. Same pre-existing behaviour for the `ImportError` and runtime-error returns (lines 203, 330).
   Fix: pass `infotexts=[message]` in the empty `Processed` returns.

7. 🟢 **R2 (S1×L2) — On Windows/macOS the shown folder uses the requested spelling, not the one on disk.**
   `seedvr2_neo/model_discovery.py:86-88`. `os.path.isdir(parent/"SeedVR2")` is true for an on-disk `seedvr2` on
   case-insensitive file systems, so the log/note print `...\models\SeedVR2` while Explorer shows `seedvr2`. Loading is unaffected
   (verified in the simulation). Fix: optional — resolve the entry via `os.listdir` first, so the printed path matches the disk.

## Not verified
- 7B GGUF tensor names (no file available); detection falls back to the file name if they differ from `blocks.<N>.`.
- A live run on a case-sensitive file system (Linux); covered only by tests.
- A live img2img run and a live 3B → 7B switch in one session (code path is upstream's name-based cache logic, unchanged).

## Re-check (fixes in the working tree after `5f9b162`)

Scope: `git diff 5f9b162` (`scripts/seedvr2_script.py`, `seedvr2_neo/model_discovery.py`, comments in `src/`), new
`seedvr2_neo/package_check.py` and its tests. Forge facts re-read in `<Forge root>`: `modules/scripts.py` (`list_scripts`,
`load_scripts`), `modules/script_loading.py`, `modules/extensions.py`, `modules/paths_internal.py`, `modules/processing.py`
(`Processed`), `modules/txt2img.py`, `modules/img2img.py` (batch), `modules/shared_options.py`.

### What was run
- Full suite on Python 3.10 (`uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q -p no:cacheprovider`):
  **158 passed, 1 skipped, 2 xfailed**. `scripts/seedvr2_script.py` parses.
- Import-resolution simulation outside the repo (a fake `extensions/` with our root, another extension with a regular `src`
  package, and one with a namespace `src` folder), real `describe_shadowed_package`, each case in a fresh `python -I`
  process on Python 3.10 and 3.14, `sys.path` built the way `load_scripts` leaves it:
  | case | result |
  |---|---|
  | only ours | no error |
  | other regular `src` earlier on `sys.path` | error naming the other folder |
  | ours earlier | no error |
  | other namespace `src` (no `__init__.py`) earlier | no error — a regular package anywhere wins over namespace portions |
  | `sys.path` entry upper-case / script path lower-case (Windows) | no error (`normcase`) |
  | forward slashes, trailing separator, relative `--data-dir` | no error (`abspath`) |

### Earlier findings
1. **IMPORTANT `src` shadowing — resolved.** `run()` calls `find_shadowed_package_error()` (`scripts/seedvr2_script.py:33-39`,
   used at `:108`) after the selection check and before the base image, and returns a one-line `empty_result`.
   - *Forge loading:* our root is `ext.path`; the script's `__file__` is `os.path.join(ext.path, "scripts", name)` and the
     `sys.path` entry is the same `ext.path` string (`extensions.py:182-188`, `scripts.py:505-506`), so `EXTENSION_ROOT` and
     `src.__file__` derive from one spelling — junctions/symlinks and 8.3 names stay consistent on both sides.
   - *First run:* `src` is never imported at script load (only inside `run()`), so the first `import src` resolves by
     `sys.path` order; if another extension's `src` comes first or is already in `sys.modules`, the check fires — a true positive.
   - *`sys.modules` cache:* after our `src` is cached the check is a dict lookup; a foreign cached `src` is reported every run.
     Submodules resolve through `src.__path__`, so checking the top package is enough.
   - *False positives for a normal install:* none found (table above). `src/__init__.py` is empty, so the import itself cannot raise
     anything but `ImportError`, which is caught.
2. **MINOR `sys.path` comment — resolved.** `scripts/seedvr2_script.py:17-19` now matches Forge: the root is already on
   `sys.path` (so the insert is skipped; if spellings differed, the insert lands on the temporary list that `load_scripts`
   discards), and it explicitly says it does not decide which `src` wins. `07-RISKS.md` §5 and `forge-neo.md` "Script loading"
   say the same.
3. **MINOR cache keyed by file name — open** (B06). The new "first complete folder wins" rule adds one more way the folder can
   change between runs (completing a set in Forge's folder), same consequence.
4. **MINOR VAE-by-name rule — mostly resolved.** README now states it ("any file with `vae` in its name is listed as a VAE"); the old "offer all files" fallback is gone by design. The in-UI note still says only "No VAE model in <folder>" — optional wording tweak, stays for `07-RISKS.md`.
5. **MINOR partial set hides a complete set — resolved.** `find_model_folder` (`seedvr2_neo/model_discovery.py:62-79`) picks
   the first candidate holding both kinds; `scan_models` lists the same folder it returns, and `run()` passes that
   `scan.folder` to `args.model_dir` (`scripts/seedvr2_script.py:192`) — the pipeline still loads from exactly the folder the
   dropdowns were built from in that scan. `find_selection_error` now names a missing kind before "not selected".
   `05-DECISIONS.md`, `09-PROCESSES.md`, README describe the same rule.
6. **MINOR empty `Processed` in the ✨ path — resolved for the crash.** All empty returns go through
   `empty_result()` (`scripts/seedvr2_script.py:42-44`; uses at `:111, :153, :220`, runtime error near the end of `run()`);
   `Processed.__init__` keeps the passed `infotexts` (`processing.py:553`), so `txt2img_upscale_function`'s
   `processed.infotexts[0]` no longer raises. The normal txt2img/img2img handlers show `processed.info` and an empty gallery —
   unchanged. See new finding N1 for the side effect.
7. **MINOR shown folder spelling — resolved.** `find_subfolders_ignoring_case` builds paths from `os.listdir` names.

### New findings

N1. 🟢 **R2 (S1×L2) — An empty result with one infotext misaligns galleries in the ✨ path and in img2img batch.**
`scripts/seedvr2_script.py:44`. In Forge's ✨ handler (`modules/txt2img.py`, `txt2img_upscale_function`) with the default
"replace the selected image" (`hires_button_gallery_insert=False`), the selected image is dropped from the gallery, nothing is
inserted, but `[message]` is inserted into `infotexts` — the image vanishes from the gallery (still on disk) and every later
image is shown/saved with its neighbour's infotext. In img2img batch (`modules/img2img.py:165-167`) a runtime error on one
file adds an infotext without an image, shifting all later ones. Before the fix the ✨ case raised `IndexError` instead
(gallery untouched). Needs the script selected plus an error, so rare; today's success infotext is "SeedVR2 Upscaled" for all
images, which limits the damage.
Fix (B07, with the infotext work): in the ✨ path return the source image unchanged
(`[p.firstpass_image]` when `getattr(p, "txt2img_upscale", False)`), keeping `infotexts=[message]`; or record it in `07-RISKS.md`.

N2. 🟢 **R1 (S1×L1) — "txt2img UI reads infotexts[0]" is imprecise.** `scripts/seedvr2_script.py:43`,
`docs/external-apis/forge-neo.md:98-99`. Only the ✨ handler (`txt2img_upscale_function`) reads `processed.infotexts[0]`;
the main `txt2img_function` uses `processed.info`. Fix: name the ✨ handler in both places.

N3. 🟢 **R1 (S1×L1) — The shadowing message is less helpful than it could be.** `seedvr2_neo/package_check.py:19-25`.
The path is printed after `normcase`, i.e. lower-cased on Windows (`c:\users\...\other-ext\src`), and a foreign *namespace*
`src` already in `sys.modules` (`__file__` is `None`) is reported as "an unknown location" although `src.__path__` names it.
Detection itself is correct. Fix: print `os.path.dirname(package_file)` un-normalised; pass `list(src.__path__)` when
`__file__` is `None`.

N4. 🟢 **R2 (S1×L2) — `07-RISKS.md` §5 LICENSE risk says `src/` is an unmodified copy.** It now carries local patches
(`src/utils/constants.py`, `src/utils/dit_size.py`, `src/core/model_configuration.py`); Apache-2.0 §4(b) asks modified files to
carry a change notice. Fix: update the risk text so B04 covers the notices (the list already lives in `seedvr2-upstream.md`).

### Not verified
- A live Forge run with a second extension shipping `src` (covered by the simulation only).
- A live ✨ / img2img-batch run with an error (N1 is from reading Forge's handlers).

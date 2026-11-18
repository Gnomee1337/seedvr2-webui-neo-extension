# Fresh-eyes review — B05 model discovery

Scope: `git diff 4dc896e` (commit `5f9b162` plus the uncommitted B05 card edit). Code, tests and docs read directly;
Forge Neo behaviour checked against a local `neo` checkout (`modules/scripts.py`, `modules/script_loading.py`,
`modules/gradio_extensions.py`, `modules/ui_common.py`, `modules/shared.py`). Tests were not executed here (no `pytest`
in the reviewer's environment); the counts in the docs are the author's.

## How I understood the result

On every UI build, 🔄 click and Generate, `seedvr2_neo.scan_models` looks for a `SeedVR2` folder (any letter case)
first in Forge's `shared.models_path`, then in `<extension>/models`, and takes the first one that holds at least one
`.safetensors`/`.gguf` file; files with `vae` in the name go to the VAE dropdown, the rest to DiT. If nothing is usable,
a Markdown note names the folder; `run()` validates the selection before any base image is generated and returns an
empty `Processed` with a one-line reason. The pipeline gets the absolute folder; a patch in `find_model_file` makes that
folder win over the cwd-relative scan, and a new `src/utils/dit_size.py` chooses `configs_3b`/`configs_7b` from the
transformer block count in the checkpoint header (fallback: `"7b"` in the name). This matches the task; nothing extra
beyond docs and a dev requirements file.

## Findings

### IMPORTANT

**F1. A partial Forge folder hides a complete set in the extension folder** — `seedvr2_neo/model_discovery.py:69-72`
(and the note at `:127`).
What I saw: a candidate is accepted when it holds *any* model file, not when it holds a DiT *and* a VAE.
Scenario (the upgrade path the new README invites): an existing user followed the old README ("1. extension folder")
and keeps `ema_vae_fp16` + DiTs in `extensions/<ext>/models/SeedVR2`. They download one new DiT and, following the new
README, put it into `<WebUI>/models/SeedVR2`. After 🔄 (or on the next Generate, since `run()` rescans) the whole
extension set disappears, the VAE dropdown is empty, and the run fails with "VAE model 'ema_vae_fp16.safetensors' is not
in <forge folder>. Press 🔄 and choose again" — pressing 🔄 cannot fix it. The note says "No VAE model in `<forge
folder>`" and does not mention that a VAE exists in the fallback folder. The docstring ("an empty `SeedVR2` folder in one
place does not hide models in the other") reads as if partial folders were handled too.
Why it matters: the user did nothing wrong by the current README, and the message points them at the wrong fix.
Options: prefer the first candidate that has both kinds (else the first with any), or name the ignored fallback folder
in the note.
Rating: 🟡 R4 (S2×L2).

### MINOR

**F2. Priority silently flipped for users with models in both folders** — `README.md:22`, `seedvr2_neo/model_discovery.py:145-146`.
The old code (and old README) used the extension folder first; now Forge's folder wins. A user with sets in both places
gets a different list after updating; a previously selected file that only lives in the extension folder now fails with
"not in <folder>". The decision is recorded in `05-DECISIONS.md`, but the README does not tell upgraders that the order
changed. Rating: 🟢 R2 (S2×L1).

**F3. Misleading run error when the folder has no file of one kind** — `seedvr2_neo/model_discovery.py:138-141`.
Folder found, no DiT files: dropdown value is `None` → "No DiT model selected." (the real cause is "no DiT file in
`<folder>`"). With a stale value saved in Forge's `ui-config.json`/an API call: "...is not in <folder>. Press 🔄 and
choose again" — 🔄 cannot help when that kind is absent. The UI note says the right thing; the Generate message does not.
Rating: 🟢 R2 (S1×L2).

**F4. Renamed VAE without `vae` in its name can no longer be chosen as a VAE** — `seedvr2_neo/model_discovery.py:57-58, 113-114`.
Old code: if no file contained `vae`, every file was offered in the VAE dropdown. New code: such a file is listed only as
a DiT; the VAE dropdown is empty. If picked as a DiT, `choose_dit_size` finds no `blocks.N` and falls back to the name →
3B config → load error. Not mentioned as a behaviour change anywhere (README states the `vae` rule, but this user
previously worked). Rating: 🟢 R2 (S2×L1).

**F5. Only the first case variant of the folder is tried** — `seedvr2_neo/model_discovery.py:81-97` with `:69-72`.
On a case-sensitive FS with both `SeedVR2` (empty, e.g. created by the user from the README) and `seedvr2` (with
models) in the same parent, the exact spelling wins, has no files, and the loop moves on to the *next parent*, not the
next variant — models are "not found" although they sit right next to it. Rating: 🟢 R2 (S2×L1).

**F6. `sys.path` comment describes something that does not happen** — `scripts/seedvr2_script.py:17-20`.
"on sys.path so `seedvr2_neo` and the vendored `src` can be imported" — in Forge Neo `load_scripts()` already prepends
every extension's basedir to `sys.path` permanently (lines ~504-506 of `modules/scripts.py`), and after loading each
script it restores `sys.path = syspath`. So this insert is a no-op (path already present) or, if the strings differ, is
discarded right after import. `seedvr2_neo` and the later `import src...` in `run()` work because of Forge, not because
of this line; the `run()`-time insert that used to persist was removed. A reader will believe the extension controls its
import path (and its priority over other extensions' `src` packages) — it does not. Worth a fact in
`external-apis/forge-neo.md` and an honest comment. Rating: 🟢 R1 (S1×L1).

**F7. TOCTOU between the scan and the load** — `src/utils/constants.py:121-133`.
`run()` checks that the file is in `scan.folder`; if it disappears before `find_model_file` runs (file being moved/
re-downloaded), the patched function silently falls back to upstream's cwd-relative `./models/SEEDVR2` scan and may
load a same-named file from a folder the UI never showed. Rare; noted as a hidden assumption ("the file is still there at
load time"). Rating: 🟢 R2 (S2×L1).

**F8. `del reader` comment overclaims** — `src/utils/dit_size.py:91`.
`del` only drops the local name, which the function return would drop anyway; whether the memory map is released depends
on no other references (e.g. tensors) surviving. A reader will think this line is what guarantees the file is closed on
Windows. Rating: 🟢 R1 (S1×L1).

**F9. Doc inconsistencies**
- `docs/04-STATE.md:14` — "Code behaviour is unchanged." is now false (B05 changes discovery, loading and config choice).
- `docs/external-apis/seedvr2-upstream.md:20` and the B05 card step 5 name the function `create_new_runner`; it is
  `_create_new_runner` (`src/core/model_configuration.py:688`) — a grep during the next upstream sync will miss it.
- `src/core/model_configuration.py:704` — "header unreadable → "7b" in the name (case-insensitive) as upstream":
  upstream was case-sensitive, and the fallback also fires when the block count is unrecognised, not only when the header
  is unreadable.
- B05 card acceptance 1 says "the run log shows the full path of the loaded DiT/VAE"; the script itself prints the
  folder and the file names separately (`scripts/seedvr2_script.py:95`) — fine if the upstream debug log is what was
  meant, but the card does not say so.
Rating: 🟢 R1 (S1×L1).

## Checked and fine
- `ToolButton(..., tooltip=...)`: Neo's `modules/gradio_extensions.py` pops `tooltip`; `refresh_symbol` exists in
  `modules/ui_common.py`; `shared.models_path` is re-exported in `modules/shared.py`.
- `Processed(p, [], p.seed, ...)` before `process_images` mirrors the existing img2img-interrupt path.
- A stale dropdown value saved by Forge's ui-config is rejected by Forge itself when not in choices, and by
  `find_selection_error` at run time.
- Safetensors header length is bounded by both 64 MB and the file size; any read error falls back to the name.
- The runner template is reused only when the DiT/VAE names match, so a size change always goes through
  `_create_new_runner` and the new detection.

## Re-check

Scope: `git diff 5f9b162` (working tree) plus the untracked `seedvr2_neo/package_check.py` and new tests. The docs were
still being edited during the re-check (README, 02/03/05/07/08/09 appeared in the diff mid-review); this reflects the
tree as last read. Tests run: `158 passed, 1 skipped, 2 xfailed`. Forge facts checked against a local `neo` checkout
(`modules/scripts.py`, `modules/script_loading.py`, `modules/processing.py`, `modules/txt2img.py`).

### Status of earlier findings
| # | Status | Evidence |
|---|---|---|
| F1 partial folder hides full set | **Resolved** | `seedvr2_neo/model_discovery.py:71-79`: first folder with both kinds wins, else first with any; tests `test_complete_extension_folder_wins_over_partial_forge_folder`, `test_partial_forge_folder_is_used_when_no_folder_is_complete`. Residual — see N1. |
| F2 priority flip not told to upgraders | **Resolved** | `README.md:24-25` now says the order changed and which folder wins. |
| F3 misleading Generate error | **Resolved** | `model_discovery.py:140-141` checks "no file of this kind" before "nothing selected"/"not in folder"; test `test_selection_error_names_the_missing_kind_when_folder_lacks_it`. |
| F4 VAE without `vae` in name | **Open, recorded** | `docs/07-RISKS.md` §1 (R2). |
| F5 only first case variant tried | **Resolved** | `find_subfolders_ignoring_case` returns all variants (`model_discovery.py:87-99`); test `test_empty_exact_spelling_does_not_hide_other_spelling_with_models`. |
| F6 `sys.path` comment | **Resolved** | `scripts/seedvr2_script.py:17-19` now says Forge provides the path and the insert gives no priority. |
| F7 TOCTOU fallback in `find_model_file` | **Open, recorded** | `docs/07-RISKS.md` §1 (R2), with a strict-xfail repro for the related `NotADirectoryError`. |
| F8 `del reader` comment | **Resolved** | `src/utils/dit_size.py:91-93` now explains the real reason (traceback frame keeps the reader alive). |
| F9 doc inconsistencies | **Partly** | Fixed: `_create_new_runner` in `seedvr2-upstream.md:20`; `base_cache_dir` docstring; upstream case-sensitivity; card acceptance 1 answered by "Live run results". Still open: `src/core/model_configuration.py:704` and `docs/external-apis/seedvr2-upstream.md:19` say the name fallback fires on an "unreadable header" only — it also fires on an unrecognised block count (`src/utils/dit_size.py:49-50`); card step 5 (`B05-model-discovery.md:26`) still says `create_new_runner`; `docs/04-STATE.md:14` still "Code behaviour is unchanged", and STATE (30%, "next: B05") now contradicts `03-ROADMAP.md` (B05 ✅, 39%). |

### New findings

**N1. A partial WebUI folder is now silently ignored** — `seedvr2_neo/model_discovery.py:77-78`. 🟢 R2 (S1×L2).
The F1 fix flips the failure: an upgrader with a full set in `extensions/<ext>/models/SeedVR2` downloads one new DiT into
`<WebUI>/models/SeedVR2` (as the README now says), presses 🔄 — the extension folder (complete) wins, the new DiT never
appears, and the note is empty because the chosen folder lacks nothing. The README rule explains it, but nothing on
screen does. Related edge: DiT only in one folder, VAE only in the other — neither is complete, the WebUI folder is
used, and the note "No VAE model in `<WebUI folder>`" does not mention the VAE sitting in the fallback. Option: when a
skipped candidate has model files, name it in the note (or in the console line printed on every run).

**N2. `src`-shadow check compares path strings, not files** — `seedvr2_neo/package_check.py:18-20`. 🟢 R2 (S2×L1).
`normcase(abspath(...))` does not resolve symlinks, junctions or 8.3 short names. It works today because Forge derives
both the `sys.path` entry (`scripts.py:505-527`, `scriptfile.basedir`) and the script's `__file__`
(`script_loading.load_module(path)`) from the same extensions-dir string. If the two ever differ for the same folder,
every Generate is refused with "Another extension's 'src' package (<our own folder>) is loaded" and the advised fix
(disable the other extension) cannot help. `os.path.samefile` / `realpath` would remove the assumption. Also note this
check is outside the B05 card (it belongs to B04's `src` naming item); it is recorded in `05-DECISIONS.md`,
`07-RISKS.md` §5 and the README, so it is acknowledged scope growth rather than a hidden one.

**N3. `empty_result` docstring is vague, and the infotext list can run one longer than the images** —
`scripts/seedvr2_script.py:42-44`. 🟢 R1 (S1×L1).
The only `infotexts[0]` read on this path is in Forge's hires "Upscale" button handler (`modules/txt2img.py:129`); the
docstring reads as if the normal Generate needed it. In the same handler, with the `hires_button_gallery_insert` option
on, the gallery keeps the original image while `infotexts` gains the message with no image, so the infotexts of later
gallery images shift by one. Default is off, and before this change the same path raised `IndexError`, so it is a net
improvement; worth naming the handler in the docstring.

### Checked and fine
- `find_subfolders_ignoring_case`: `isdir` filter drops a same-named file; real spelling is returned; Windows
  (one variant only) behaves as before except that the path now shows the folder's real case — documented in
  `09-PROCESSES.md`.
- `find_shadowed_package_error`: `src/__init__.py` is empty, so the extra `import src` on every Generate has no side
  effects for our package; a foreign package is imported anyway by the later `import src...` in the old flow.
- Every `Processed(p, [], ...)` return in `run()` now goes through `empty_result`; no other return path changed.

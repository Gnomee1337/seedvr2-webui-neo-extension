# Code audit — B05 (model discovery)

Scope: `git diff 4dc896e` (commit `5f9b162` + uncommitted card edits). Rules: `docs/process/code-style.md`, `CLAUDE.md`.
Tests: `python -m pytest tests` — 46 passed (also `tests/seedvr2_neo` alone 18 passed, `test_dit_size.py` 8 passed,
GGUF test not skipped). Extra probe (not committed): corrupt GGUF headers with tensor count 2^40, KV count 2^40 and a
2^62 name length — each falls back to the file name in under 0.2 s, and the file can be deleted right after (no open
memory map on Windows).

**Totals:** BLOCKER 0 · IMPORTANT 0 · MINOR 9.

No logic error found in the main paths: folder resolution, refresh handler (values always within the new choices, so
Gradio 4 dropdown validation does not fire), selection error path (runs before the base image, returns `Processed`
like the existing `ImportError` path), header readers (every failure is caught and logged through `debug.log(force=True)`;
safetensors length bounded by `min(64 MB, file size)`), `find_model_file` patch (only short-circuits when the file
exists in the given folder).

---

## MINOR

### M1. 🟢 R3 (S3×L1) — no test ties `CONFIG_DIR_BY_DIT_SIZE` / `DIT_SIZE_BY_BLOCK_COUNT` to the real configs
`src/utils/dit_size.py:25-26`
The block counts 32/36 duplicate `num_layers` in `configs_3b/main.yaml` / `configs_7b/main.yaml`, and the size → folder
map is not tested at all. Scenario: someone swaps the two values in `CONFIG_DIR_BY_DIT_SIZE` (or an upstream sync changes
`num_layers`) → all 46 tests stay green, a 7B checkpoint is built with the 3B config → shape error at load — exactly the
bug this block fixes.
Fix: add a test that reads `num_layers` from each `configs_*/main.yaml` (plain YAML scan or OmegaConf) and asserts
`DIT_SIZE_BY_BLOCK_COUNT[num_layers]` maps back to the same folder through `CONFIG_DIR_BY_DIT_SIZE`.

### M2. 🟢 R2 (S2×L1) — an empty exact-case folder hides a populated case-variant folder (Linux)
`seedvr2_neo/model_discovery.py:81-97` together with `:62-73`
`find_subfolder_ignoring_case` returns the exact `SeedVR2` as soon as it is a directory, without checking it has models;
`find_model_folder` then skips the whole parent. Scenario (case-sensitive FS): `<models>/SeedVR2/` exists but is empty
(e.g. created after reading the note) and `<models>/seedvr2/` holds the models → no folder found, the note says
"No SeedVR2 models found. Put … into `<models>/SeedVR2`" while the models are next to it. This contradicts the stated rule
"an empty folder in one place does not hide models in the other".
Fix: let the case-insensitive lookup return all matching subfolders (exact first) and have `find_model_folder` take the
first one with model files.

### M3. 🟢 R1 (S1×L1) — test name promises what it does not check
`tests/seedvr2_neo/test_model_discovery.py:77`
`test_exact_spelling_wins_over_other_case` creates only `SeedVR2` and looks up `SeedVR2`; no other spelling exists.
Mutation: delete the exact-match shortcut (`exact = …; if os.path.isdir(exact): return exact`) → the loop still finds
`SeedVR2`, the test passes. Also the acceptance criterion names `seedvr2` **or** `SEEDVR2`; only `seedvr2` is tested.
Fix: build the case with the same `case_sensitive_isdir` monkeypatch plus a fake `os.listdir` returning both
`seedvr2` and `SeedVR2` and assert the exact one wins; or rename the test to what it checks.

### M4. 🟢 R2 (S2×L1) — update changes which folder is used for users with models in both places
`seedvr2_neo/model_discovery.py:62-73` (`_candidate_parents`, `:145-146`)
Before the block the extension folder won (`MODEL_DIR` = extension folder if it exists); now Forge's folder wins (owner's
decision, recorded). Scenario: a user has a full set in `extensions/<ext>/models/SeedVR2` and an older or partial set in
`<models>/SeedVR2` → after "Check for updates" the dropdowns list the other folder; if the saved `ui-config.json` value is
not there, the default DiT silently changes; if Forge's folder has only a VAE, the note says "No DiT model". Nothing breaks
for the common single-folder setup. Not a code defect — a release-note item.
Fix: mention the new priority in the README "Models" section / release notes when merging to `main`.

### M5. 🟢 R3 (S3×L1) — vendored `src` package now shadows other `src` packages from WebUI start
`scripts/seedvr2_script.py:18-20`
`sys.path.insert(0, EXTENSION_ROOT)` moved from `run()` to import time. Before, the extension root reached `sys.path` only
after the first SeedVR2 run; now it is first in `sys.path` from startup. Scenario: another extension that appends its own
root (`sys.path.append`) and imports a top-level `src` at load time, loaded after this one → it imports our `src` and fails
with `ImportError`. Rare, and the root cause (top-level package named `src`) is already planned in B04.
Fix: nothing in this block; add this scenario to the B04 rename item in `07-RISKS.md` so the earlier exposure is recorded.

### M6. 🟢 R1 (S1×L1) — duplicated knowledge: kinds, refresh symbol, extensions
`seedvr2_neo/model_discovery.py:134-137` and `:150`; `:122`, `:127`, `:141`; `:15`; `src/utils/dit_size.py:28`
- The `("DiT", dit_models) / ("VAE", vae_models)` pairs are spelled twice (`find_selection_error`, `_missing_kinds`).
- The note and errors hard-code `🔄`, while the button uses Forge's `refresh_symbol`; if Forge changes the symbol the text
  points to a button that no longer exists.
- `MODEL_FILE_EXTENSIONS` and `GGUF_EXTENSION` repeat `SUPPORTED_MODEL_EXTENSIONS` in `src/utils/constants.py`.
Fix: one private `_kinds(scan)` helper returning the pairs; refer to "the refresh button" in text or pass the symbol in;
optionally import the extensions from `src/utils/constants.py` (no Forge imports there).

### M7. 🟢 R2 (S1×L2) — docs diverge from code / from the card
- `docs/external-apis/seedvr2-upstream.md:20` and the card line 26 name `create_new_runner`; the function is
  `_create_new_runner` (`src/core/model_configuration.py:688`). A sync relying on the name will not find it by search.
- `docs/external-apis/seedvr2-upstream.md:39` "GGUF files not checked yet" — the card's live run records a real 3B Q6_K
  GGUF detected from the header.
- `src/core/model_configuration.py:706` docstring: `base_cache_dir` "(not used directly but passed for context)" — it is now
  used directly for the header read.
- `docs/04-STATE.md:11` "waiting for the owner's live run" — the card says the live run is done.
Fix: correct the four lines (STATE is normally rewritten on `/block-done`).

### M8. 🟢 R1 (S1×L1) — 64 MB header cap is not exercised by any test
`src/utils/dit_size.py:33`, `:76`
All "absurd length" cases are rejected by the `os.path.getsize` bound; removing `_MAX_SAFETENSORS_HEADER_BYTES` from the
`min()` keeps every test green. Without the cap a large corrupt file (e.g. a 6 GB file whose length field says 5 GB) would
make `file.read()` allocate gigabytes. Risk is low; the cap exists, it is just unguarded.
Fix: a stress test with a sparse file slightly above 64 MB (`f.seek(n); f.write(b"\0")`) and a length just above the cap,
asserting the "file name" fallback.

### M9. 🟢 R1 (S1×L1) — `del reader` comment overstates its effect
`src/utils/dit_size.py:91`
On normal return the local is freed anyway; `del` only matters when iterating `reader.tensors` raises (the traceback
would keep the frame). The comment reads as if the file stayed open without it. Behaviour is correct (verified: the probe
file is deletable right after a read).
Fix: reword to "drop the reader before the exception leaves the frame so the memory map is released".

---

## Checked, no finding
- Header readers never raise out of `choose_dit_size`: zero-byte, truncated, 2^63 length, length past EOF, non-UTF-8,
  JSON array, HTML file, missing file, three corrupt GGUF layouts — all fall back to the file name, logged with the reason.
- `_BLOCK_INDEX` cannot match VAE names (`down_blocks.N.` is preceded by `_`, not `.`); DiT modules have one `blocks`
  list each (`src/models/dit_*/nadit.py`).
- `find_model_file` patch: `fallback_dir` falsy or file absent → upstream path unchanged; test fails without the patch.
- Refresh: `_refresh_models` outputs match `[dit_model, vae_model, missing_models_note]`; values are always in the new
  choices or `None`, so Gradio 4 dropdown preprocessing does not reject them.
- `run()`: unchanged return list order of `ui()`, unchanged labels (ui-config keys), unchanged title; the check runs
  before the base image and before the SD checkpoint unload.
- No dead or commented-out code added; no debug output beyond the intended folder/model line; no TODOs.
- `tests/pytest.ini` is found for `tests`, `tests/seedvr2_neo` and single-file runs from the repository root.

---

## Re-check

Scope: `git diff 5f9b162` plus the untracked `seedvr2_neo/package_check.py`, `tests/seedvr2_neo/test_package_check.py`,
`tests/src/utils/test_constants_stress.py`. Tests: `python -m pytest tests` — 158 passed, 1 skipped (trailing-space folder,
Windows only), 2 xfailed (strict, both recorded in `07-RISKS.md` §1). Forge facts checked against a local Neo checkout
(`modules/scripts.py::load_scripts`, `modules/processing.py::Processed`, `modules/txt2img.py`).

**Totals now:** BLOCKER 0 · IMPORTANT 0 · MINOR open: M6, M7 (partly), N1, N2, N3.

| Finding | Status | Evidence |
|---|---|---|
| M1 size map vs `num_layers` | resolved | `tests/src/utils/test_dit_size.py:93-99` reads `num_layers` from the folder chosen by `CONFIG_DIR_BY_DIT_SIZE`; swapping the two folders gives 32 ≠ 36 and fails |
| M2 empty exact-case folder hides a populated variant | resolved | `find_subfolders_ignoring_case` returns all spellings, `find_model_folder` filters by model files; `test_empty_exact_spelling_does_not_hide_other_spelling_with_models` |
| M3 exact-spelling test | resolved | faked case-sensitive FS with `seedvr2` / `SEEDVR2` / `SeedVR2`; a plain alphabetical sort fails it |
| M4 folder priority changes on update | resolved | "complete set wins" removes the partial-Forge-folder case; the both-complete case is in README "Models" |
| M5 import-time `sys.path` insert | withdrawn | Neo prepends every extension root to `sys.path` for good and discards a script's import-time edits (temporary list), so the insert is a no-op inside Forge; the collision is now detected in `run()` and recorded (07-RISKS §5, forge-neo.md) |
| M6 duplicated knowledge | open, recorded | 07-RISKS §1 "Small UI/listing roughness" covers extensions and 🔄; the DiT/VAE split is now spelled a third time in `_has_both_kinds` (`seedvr2_neo/model_discovery.py:153-155`) — not mentioned there |
| M7 docs vs code | partly resolved | upstream doc lines and the `_create_new_runner` docstring fixed; still stale: `docs/04-STATE.md:11` ("waiting for the owner's live run"), card `:15` (old folder rule) and `:26` (`create_new_runner`) — to be handled on `/block-done` |
| M8 64 MB cap untested | resolved | `test_declared_length_over_the_cap_in_a_big_file_is_rejected_unread`: without the cap the 64 MB read pushes the peak over 1 MB and the source is not "implausible" |
| M9 `del reader` comment | resolved | comment now says what `del` actually prevents |

New code checked without findings: `describe_shadowed_package` (normcase + abspath on both sides; both paths derive from
the same `basedir` string Forge uses, so a junction/symlink install still matches; a namespace `src` elsewhere loses to our
regular package), `find_shadowed_package_error` (also covers a foreign `src` already in `sys.modules`), the new
"no models of this kind" branch in `find_selection_error` (test fails without it), `find_model_folder` rewrite
(complete-first, then first with any file; tests for both).

### N1. 🟢 R2 (S2×L1) — empty result in the hires "upscale" button path drops the image and shifts infotexts
`scripts/seedvr2_script.py:42-44`; `docs/external-apis/forge-neo.md:96-98`
Only `txt2img_upscale_function` (the gallery ✨ hires-fix button, `modules/txt2img.py:129`) reads `processed.infotexts[0]`;
the main `txt2img_function` reads `processed.info` (`:157`). The comment and the forge-neo.md fact say "txt2img's UI".
In that button path Forge rebuilds the gallery as `new_gallery.extend(processed.images)` /
`new_infotexts.extend(processed.infotexts)`. Scenario: gallery [a, b, c], ✨ on b with SeedVR2 selected and a selection or
package error → gallery becomes [a, c] (b is replaced by nothing), infotexts [ia, message, ic] → c now shows the error text
as its generation info, and saving c from the UI writes it into the PNG. Before the change the same case raised
`IndexError` and left the gallery untouched. Rare and cosmetic (files on disk unaffected).
Fix: name the hires-upscale handler in the comment and in forge-neo.md, and record the gallery side effect in 07-RISKS.

### N2. 🟢 R1 (S1×L1) — the package name `src` is spelled in two places
`scripts/seedvr2_script.py:36` (`import src`) and `seedvr2_neo/package_check.py:13` (`VENDORED_PACKAGE = "src"`)
Scenario: the planned rename in B04 changes the constant but not the import (or the reverse) → the check compares the
wrong package or fails with "Cannot import the SeedVR2 pipeline" on every run; no test covers the script.
Fix: `importlib.import_module(VENDORED_PACKAGE)` in `find_shadowed_package_error`.

### N3. 🟢 R1 (S1×L1) — README overstates the fallback
`README.md:23-24` "If that folder does not hold both a DiT and a VAE, `extensions/<this extension>/models/SeedVR2/` is used
instead." Code (`seedvr2_neo/model_discovery.py:77-78`) uses the extension folder only if it is complete; otherwise the
first folder with any model file. Scenario: WebUI folder has only a DiT, extension folder empty → README says the
extension folder is used; the code keeps the WebUI folder (the note does name it correctly).
Fix: "…is used instead if it holds both".

# Fresh-eyes review — B07 "Correct results"

Scope: `git diff main...HEAD` on `feature/b07-correct-results` (commits e132827, 568208e, 8b44152, efcf507).
Checked against the Forge Neo source in a local `neo` checkout (`modules/img2img.py`, `modules/images.py`,
`modules/processing.py`, `modules/api/models.py`) and the vendored `generation_phases.py`.

## How I understood the result
txt2img: Forge generates the batch as before; the script then takes the generated images (skips the grid at
`index_of_first_image`, ignores anything after `len(p.all_seeds)`). img2img: every distinct `init_images` object is
an input (the Batch tab's `[img] * batch_size` collapses to one). Inputs are converted to RGB/RGBA, grouped by
size+mode, and each group goes through the pipeline as one N-frame tensor; one SeedVR2 seed for the whole run.
The result has exactly one image and one infotext per input; images that were not upscaled (error/interrupt) come
back as the input with `SeedVR2 Error` in the infotext and are not saved. Infotexts are Forge's base infotext (txt2img)
or prompt + negative (img2img) with SeedVR2 keys appended. Saving follows `p.save_samples()`; `-original` is written
only when Forge did not save the base. This matches the task. Nothing extra was done beyond doc updates.

## IMPORTANT

### F1. README still promises a file on every run; API callers silently lose it
`README.md:52` — "The result is saved with the `-seedvr2` suffix." Now the file is written only when
`p.save_samples()` is true. Neo's API default is `save_images = False` (`modules/api/models.py:92,111`), so every API
caller that did not pass `save_images: true` used to get a `-seedvr2` file on disk and now gets none; UI users with
"Always save all generated images" off also stop getting files. The README also does not say that a batch now
upscales every image (run time × N) or that a failed run returns the inputs. Existing users will read this as a
regression. `forge-neo.md` still lists "Default value of `save_images` … was not read" under "Not verified" — it is
False. R = S2 × L3 = 6 🟡

### F2. A failed or interrupted run returns the inputs looking like results
`scripts/seedvr2_script.py:185`, `:331`; `seedvr2_neo/run_results.py:569-572`. Before, an error gave `images == []`
and the info panel said `Runtime Error: …`. Now an API caller gets N images byte-identical to its inputs, and the only
sign of failure is `SeedVR2 Error: …` at the end of one params line inside the `info` JSON string. Scenario: an API
batch job on an 8 GB GPU hits OOM → the client stores the un-upscaled inputs as "upscaled". In the img2img UI the
gallery shows the input unchanged with the error buried in the parameter line. The decision ("failures come back as the
input") is recorded, but callers are not told how to detect failure (README / `processed.comments` / a top-level `info`
marker). R = S2 × L2 = 4 🟡

### F3. "Always save" off + "Save Extra Copy of Original" on → only originals are written
`seedvr2_neo/run_results.py:546-550`, pinned by `tests/seedvr2_neo/test_run_results.py:17`
(`test_txt2img_with_saving_off_writes_only_the_requested_original`). With saving off, txt2img writes `-original`
files but no `-seedvr2`; img2img likewise writes `-original` with `opts.samples_save` off. The block card's acceptance
criterion says "'Always save all generated images' off → nothing written by the extension". Code, test and
`09-PROCESSES.md` agree with each other and disagree with the criterion; a reader cannot tell which is intended, and
the user ends up with the less valuable file of the two. Needs an explicit decision (and the criterion or the code fixed).
R = S2 × L2 = 4 🟡

## MINOR

### F4. The save plan is fixed before the upscale; docstring says it follows "interruption"
`scripts/seedvr2_script.py:312`, docstring `seedvr2_neo/run_results.py:543`. `p.save_samples()` is read before the
pipeline runs, so after an interrupt the upscaled images of finished groups are still saved (Forge itself would not,
unless `save_incomplete_images`). Behaviour is defensible, the docstring is not accurate. Related edge: `state.skipped`
is reset only at the top of each Forge batch (`processing.py:497` uses it), so Skip on the last txt2img batch leaves it
True → `p.save_samples()` is False → no `-seedvr2` for any image of the run, and with "Save Extra Copy of Original"
`-original` duplicates of the earlier batches Forge already saved. R = S2 × L1 = 2 🟢

### F5. txt2img API `override_settings` are restored before the script reads them
`scripts/seedvr2_script.py:312`; Neo `processing.py:839` restores opts after `process_images`. An API txt2img with
`override_settings: {"samples_save": true}` (user's option off): Forge saves the base images, the script does not save
results, and `-original` duplicates the base. Same for `samples_format` (base jpg, result png). `07-RISKS.md` lists this
only for img2img. R = S2 × L1 = 2 🟢

### F6. `07-RISKS.md` overstates what img2img ignores
"img2img ignores `p.override_settings` except `save_to_dirs`" — Neo's `images.save_image` reads
`p.override_settings["samples_filename_pattern"]` itself (`modules/images.py:716-717`), so the Batch tab's
"use original name" (`img2img.py:155-157`, `opts.img2img_batch_use_original_name`) does reach our files. The
`forge-neo.md` Batch-tab excerpt misses that override and the loop's `stopping_generation` break (`img2img.py:66`).
The `save_result_image` docstring ("that one override is passed on") is correct but does not say why the others need
no passing. R = S1 × L2 = 2 🟢

### F7. `07-RISKS.md` "img2img infotext 'Seed' is the SeedVR2 seed"
The img2img infotext has no `Seed:` key at all (`describe_img2img_inputs` → `BaseInfotext.from_prompt`, only
`SeedVR2 …` keys). The SeedVR2 seed reaches the UI through `processed.js()` `all_seeds = [actual_seed]` (♻️ button),
not through the infotext, so "Send to puts it into the seed field" is likely wrong as written. R = S1 × L2 = 2 🟢

### F8. Comment about stale `p.all_seeds` in img2img has no visible cause
`scripts/seedvr2_script.py:301` — "`p.all_seeds` there may be left from the previous Batch tab file". Per
`forge-neo.md`, `all_seeds` is set only in `process_images_inner`, which img2img with this script never calls, and the
script no longer sets it. A reader cannot see how it could be stale; either name the source or say "not set in img2img".
R = S1 × L2 = 2 🟢

### F9. Docstrings say "models load once / unload after the last" — true only per group
`scripts/seedvr2_script.py:354` and `:401-402`. In "Unload" mode the vendored phases delete the DiT after each group
(`generation_phases.py:796`, `cleanup_dit(cache_model=False)`), so mixed-size inputs reload it per group (correctly
noted in `07-RISKS` and `09-PROCESSES`). The two docstrings read as "once per run". R = S1 × L2 = 2 🟢

### F10. A save failure outside `OSError`/`ValueError` now escapes `run()`
`scripts/seedvr2_script.py:167-171`. `save_image` calls other extensions' `before_image_saved_callback`
(`images.py:749`); any other exception there now propagates from `build_result` out of `run()` and the whole result
(already computed) is lost with a Forge error. Before, saving was inside the broad `try` and gave an empty
"Runtime Error" result. VRAM is already released by then, so only the result is lost. R = S2 × L1 = 2 🟢

### F11. Out-of-range UI seed is silently replaced
`seedvr2_neo/run_inputs.py:475`. A typed seed outside `[0, 2**32-1]` (e.g. `-5`, `5000000000`) counts as "not chosen"
and the run follows the base seed with no message; the infotext shows the seed actually used. A console line would
remove the surprise. R = S1 × L1 = 1 🟢

### F12. Block card and `forge-neo.md` carry stale statements that may migrate on `/block-done`
`docs/roadmap-blocks/B07-correct-results.md:24-25` (step 3 "fill what `create_infotext` needs"), `:66-67`
("so the script fills them"), `:61` (out of scope "Batching several images through the pipeline in one pass" — the code
now does exactly that, as frames with pipeline batch 1). `forge-neo.md` "Prompt/seed lists" still advises calling
`p.setup_prompts()` and passing lists to `create_infotext`, contradicting the new fact that `create_infotext` fails
with the checkpoint unloaded. R = S1 × L2 = 2 🟢

## Hidden assumptions (checked, hold today)
- Batch tab passes `[img] * p.batch_size` (same object) — confirmed `img2img.py:88`; single img2img uses `[image]`
  (`:242`). Dedup by `id()` would upscale N identical copies if an extension ever replaces them with copies.
- Grid only at index 0 and extras outside `processed.images` in txt2img — matches `forge-neo.md`; masks land in
  `images` only in img2img's `process_images`, which this script skips.
- Pillow converts every mode tried (`LAB`, `HSV`, `F`, `I`, `1`, `YCbCr`, `I;16`) to RGB without raising, so one odd
  input does not fail the whole group via `to_pipeline_image`.
- `extra_images` (e.g. ControlNet maps) of the txt2img base are not passed to the result — same as before, not a regression.

## Missed changes
- `README.md` (F1).
- `forge-neo.md` "Not verified" `save_images` default can be closed (False); Batch-tab excerpt (F6).

## Re-check after e455f0c (code) and ce8b2de (docs)

Scope: `git diff efcf507..HEAD`; Forge facts re-checked in the local Neo checkout (`processing.py:317-318, 510, 580-597`,
`ui_tempdir.py:26-48`, `api/api.py:100-125`, `txt2img.py:157`, `img2img.py:299`, `images.py:783`).

### Original findings
| # | Status | Note |
|---|---|---|
| F1 README | closed | `README.md:44-50` describes every-image batches, no grid, saving rule, `-original` for img2img only, failures. |
| F2 failures look like results | closed for the UI, partly for the API | `p.comment` shows under the result in the UI (`txt2img.py:157`, `img2img.py:299`). The API returns `info = processed.js()`, which has no `comments` key; an API caller still learns of a failure only by finding `SeedVR2 Error` in `infotexts[i]` (now also embedded in each base64 PNG). Acceptable if the README says so for API users — it does not yet. R = S2 × L1 = 2 🟢 |
| F3 originals with saving off | closed | `plan_saving` (`run_results.py:44-57`): `save_originals` requires `save_results` and img2img. |
| F4 plan before upscale / Skip | closed | No longer `p.save_samples()`; docstring explains the Skip case; `07-RISKS` states finished passes are saved after an interrupt. |
| F5 txt2img overrides | closed | `read_option` for `samples_save`, `samples_format`, `save_to_dirs`; `07-RISKS` lists which overrides reach the saves. |
| F6 / F7 07-RISKS entries | closed | Both rewritten correctly; `forge-neo.md` gains the `save_samples`, overrides, API defaults facts. |
| F8 stale `all_seeds` comment | closed | `seedvr2_script.py:329`. |
| F9 "load once" docstrings | closed | `_upscale_images` / `_upscale_tensor` now say per pass. |
| F10 narrow save catch | closed | `except Exception` with a reason (`seedvr2_script.py:~186`). |
| F11 seed message | closed | `is_unusable_seed` + console line (`seedvr2_script.py:~327`). |
| F12 stale card / forge-neo | mostly closed | `forge-neo.md` "Prompt/seed lists" fixed; the card still lists "Batching several images through the pipeline in one pass" as out of scope and step 3 still says "fill what `create_infotext` needs" — the card is deleted on `/block-done`, so only a risk if copied. |

### New findings

**N1 IMPORTANT — the live acceptance ticks predate the fix commit.** `docs/roadmap-blocks/B07-correct-results.md`
(criteria ticked, "Progress"). Every live run listed was done before e455f0c, which rewrote the run path of the script
(`_collect_inputs`, `RunInputs`, `read_option`, `embed_infotext`, `p.comment`, pass splitting, settings parsing). `run()`
has no unit tests (Forge stubs are B09), so a mistake there breaks every run for every user on update. The card also
still says the Batch-tab `save_to_dirs` re-check is pending. Before merge: at least one txt2img batch and one img2img
run (one with an interrupt) live on HEAD, plus the Batch tab. R = S3 × L1 = 3 🟢 by the matrix, but the cost of being
wrong is every user's run; treat as a merge precondition.

**N2 MINOR — `describe_img2img_inputs` docstring now false.** `scripts/seedvr2_script.py:173-174` says "`p` is not
changed — the Batch tab reuses it"; `build_result` now calls `p.comment(...)` (`:215`). Consequence in the Batch tab:
`process_batch` keeps the first file's `Processed` and only extends its images/infotexts (`img2img.py:164-169`), so a
failure note from file 2+ never appears under the result, while `p.comments` keeps accumulating on the reused `p`.
R = S1 × L2 = 2 🟢

**N3 MINOR — "-original" and failed results are the converted image, not the input.** `save_originals` gets
`collected.images` after `to_pipeline_image` (`seedvr2_script.py:~282`, `:331`). A 16-bit grayscale, CMYK or palette
input is saved as `-original` in 8-bit RGB/RGBA, and comes back converted on failure. `09-PROCESSES` mentions the
failure case; the "Extra Copy of Original" label and README promise the original. R = S1 × L2 = 2 🟢

**N4 MINOR — a not-upscaled txt2img image loses its link to Forge's saved file.** `embed_infotext`
(`seedvr2_script.py:~219-226`) returns `image.copy()`; Pillow's `copy()` keeps `info` but not the `already_saved_as`
attribute Forge set on the base image (`images.py:783`), so the gallery writes a temp copy (`ui_tempdir.py:27`)
instead of serving the saved file. Only on failure; extra temp file, no user data harmed. R = S1 × L1 = 1 🟢

**N5 MINOR — `is_unusable_seed("")` is True.** `seedvr2_neo/run_inputs.py:~123`: an empty string from an API caller
prints "Seed '' is outside 0 … 2^32-1", while the docstring says empty means "not chosen". Cosmetic. R = S1 × L1 = 1 🟢

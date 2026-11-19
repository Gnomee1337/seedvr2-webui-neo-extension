# B07 — Correct results

**Status:** 🔄 in progress
**Weight:** 3 (from `docs/03-ROADMAP.md`)
**Decisions this block rests on:** "txt2img calls `process_images` from inside `run()`"; "Fixed pipeline settings: batch 1,
LAB color correction, `sdpa`, no `torch.compile`"; "SeedVR2 is unloaded after every run by default"; "Models are dropped
after an error or an interrupt, even with Keep in RAM"; "Fix in `scripts/` first, patch `seedvr2_vendor/` only when unavoidable".

## Why this block
Today a txt2img batch upscales only the first image — with "return grid" on (Forge's default) that is the grid itself;
img2img upscales only the first input; the result loses its generation parameters; saving ignores the user's settings;
grayscale/palette inputs crash. After the block every generated image is upscaled once, carries its own parameters and
is saved by the user's rules. These are the top user-visible risks in `07-RISKS.md` §4.

## What we do (steps)
1. Pure helpers in `seedvr2_neo/` (no Forge imports), with tests in `tests/seedvr2_neo/`:
   - which images of the base result are generated ones (skip the grid at `index_of_first_image`, skip extras after the batch);
   - image mode → pipeline input (`L`/`P`/`I;16`/`CMYK`/`LA` → `RGB`; `RGBA` and `P` with transparency → `RGBA`);
   - save decision (`opts.samples_save`, `p.do_not_save_samples`, API calls) and the "original" copy rule;
   - per-image infotext inputs (index in batch, SeedVR2 parameters for `extra_generation_params`).
2. `scripts/seedvr2_script.py`: upscale every generated image (txt2img) / every `init_images` entry (img2img) in one run;
   the models load once and are unloaded after the last image (B06 lifecycle unchanged); interrupt between images returns
   the images already done.
3. Per-image infotexts in `Processed` (`infotexts=[...]`, `index_of_first_image`), SeedVR2 parameters added; in img2img
   fill what `create_infotext` needs instead of failing in a bare `except`.
4. Saving by the user's settings; no duplicate base image; img2img Batch tab does not get double files.
5. Empty base result and per-image errors: no crash outside `try`; a failed image comes back as its input with the error in
   its infotext, so gallery infotexts do not shift.
6. Docs: `02-ARCHITECTURE`, `09-PROCESSES` (generation flow), `07-RISKS` §4 (remove closed items), `external-apis/forge-neo.md`.
7. Live check in the owner's Forge (test branch), then `/block-done`.

## Acceptance criteria
- [x] txt2img, batch size 2 × batch count 2, "return grid" on: 4 upscaled images in the gallery, no upscaled grid, each
      with its own seed/prompt in the info panel and in the saved PNG. *(API: 4 images 1024 px, 4 infotexts with 4
      seeds; PNG Info of a saved file: Seed 1003, Steps 6, SeedVR2 keys.)*
- [x] img2img with 2 input images (API or Batch tab): both are upscaled; the Batch tab writes each result once.
      *(API: 3 inputs → 3; Batch tab by the owner: 2 → 2, after the `save_to_dirs` fix straight into the output folder.)*
- [x] img2img with a grayscale (`L`) and a palette (`P`) PNG: upscaled, output is RGB; an RGBA PNG keeps its alpha. *(API.)*
- [x] "Always save all generated images" off → nothing written by the extension; on → one `-seedvr2` file per image;
      "Save Extra Copy of Original" never writes a second copy of a file Forge already saved. *(API, both settings;
      `save_images=False` too.)*
- [x] img2img result infotext has the prompt and `SeedVR2 …` keys; Forge's PNG Info parser (`/sdapi/v1/png-info`, the
      tab with "Send to txt2img") reads them back from the saved file (the SeedVR2 keys are not read back into the
      script's controls).
- [x] `process_images` returning no images → a clear message, no traceback, VRAM released. *(Stubbed `run()` only —
      cannot be produced through the API; the return happens before any SeedVR2 model is loaded.)*
- [x] Interrupt during the upscale of a batch → every image comes back (the not-upscaled ones as their inputs with
      `SeedVR2 Error: Interrupted`), the gallery count matches, models unloaded. *(API, img2img with 2 images; VRAM
      0.59 GB after. txt2img uses the same code path after the base images.)*
- [ ] All tests pass (`uv run … python -m pytest tests -q`), new helpers covered including hostile inputs
      (0 images, grid only, extras after batch, odd modes, 1×1 image, huge index).

## Which checks are needed on `/block-done`
Always: `code-auditor`, `fresh-eyes-reviewer`, `plan-drift-checker`. Additionally (tick what applies):
- [x] `integration-checker` — the result format returned to Forge changes; the block must run in Forge
- [x] `security-auditor` — files are written into the user's output folders
- [x] `stress-tester` — user input: image modes and sizes, batch shapes, settings combinations

## Dependencies
B06 (closed): model lifecycle and memory release. A live Forge Neo for the acceptance run (the owner's test branch).

## Out of scope for this block
- Forge progress bar during SeedVR2 and interrupt inside the DiT step — B08.
- Forge stubs to unit-test `run()` itself and CI — B09 (here only helpers are unit-tested; `run()` is checked live).
- An upscaled grid of the results (see open questions); per-image different SeedVR2 settings.
- Batching several images through the pipeline in one pass (pipeline batch stays 1).

## Hidden assumptions
From `fact-checker` (`external-apis/forge-neo.md`, "Results, infotexts and saving"; read through a page summariser):
- The UI shows `infotexts[selected index]` with no offset; a short `infotexts` list breaks Save/zip → one per image.
- `create_infotext` merges `p.extra_generation_params`; img2img never sets `all_prompts` / `all_seeds` / `all_subseeds`
  (only `process_images_inner` does), so the script fills them.
- Batch tab: calls `run()` per file with `init_images = [img] * batch_size`, does not save the result itself → the
  script's own save is the only one.
- `save_image` does not check `opts.samples_save`; `p.save_samples()` does; the API sets `do_not_save_samples = not save_images`.
- Grid is only at index 0 (`index_of_first_image = 1`); extras (ControlNet maps) are in `extra_images`, not `images`.
Own assumptions:
- The pipeline with batch 1 and no overlap treats N frames independently (read in `generation_phases.py`; upstream
  uses this for image batches). One seed per pass.
- Not verified: `save_images` API default, whether a script must call `fix_seed`.

## Progress
Steps 1–6 done: helpers + tests (1069 passed), script rewritten, stubbed smoke run of `run()` (batch + grid, odd modes,
error in the second group, empty base, Batch-tab repeats), docs updated.
Step 7, live run through Forge's API (3B GGUF, 8 GB GPU), all passed: img2img L/P/RGBA (3 images, RGB/RGB/RGBA, alpha
kept, 3 files, prompt + SeedVR2 keys in infotexts); img2img `save_images=False` → no files; img2img interrupt during
the upscale → 2 inputs back with `SeedVR2 Error: Interrupted`, no files; txt2img 2×2 with grid on, 512→1024 → 4
images, 4 seeds, 4 `-seedvr2` files, no `-original` duplicate; txt2img with "Always save" off → no files. VRAM after
each run 0.55–0.85 GB. The live run found that `create_infotext` fails with the checkpoint unloaded → infotexts are
now built from Forge's base infotext (`BaseInfotext`).
PNG Info parse of saved files (img2img from the UI, txt2img from the API): prompt, Forge's seed/steps, SeedVR2 keys.
Batch tab by the owner: 2 files in → 2 `-seedvr2` files out (1080 px), but in a date subfolder — fixed (`save_to_dirs`
override passed on), re-check pending.
Empty base result: stubs only (cannot be produced through the API).

## Agreed with the owner
- No grid of the upscaled images: only the images themselves are returned.
- "Save Extra Copy of Original" stays; it saves the base image only when Forge did not already save it.
- Acceptance criteria as above.

## Open questions
None.

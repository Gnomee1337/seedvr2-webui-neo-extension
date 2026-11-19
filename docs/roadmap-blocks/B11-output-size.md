# B11 — Output size: scale factor and long edge

**Status:** 🔄 in progress (code, tests, docs, live check done; UI look — owner)
**Weight:** 2 (from `docs/03-ROADMAP.md`)
**Decisions this block rests on:** "Same-sized images go through the pipeline in passes of up to 16 MP of output";
"API values of the numeric settings are clamped to the slider ranges"; "VAE tile 512 is the default, on a slider
labelled 'VAE Tile Size'" (the `ui-config.json` trap); "No 6K / 8K output sizes until the DiT works in tiles".

## Why this block
Today the output size is set only as a shortest edge in px, so "twice the size" or "2K wide" has to be worked out by
hand for every input size. After the block the user picks a factor or a long edge, and each input size gets its own
resolution. B08 rearranges the UI around this control, so it goes first.

## What we do (steps)
1. `seedvr2_neo/upscale_settings.py`: a size mode (Shortest edge / Scale factor / Long edge), bounds for the factor
   (×1.0–×4.0, step 0.1, default ×2.0) and the long edge (px, default 2048, max 3840 or lower after the live check);
   `UpscaleSettings` keeps the mode and its value; one function `size → shortest-edge resolution` for an input size,
   clamped to `RESOLUTION` (512–3840). Unknown mode / garbage from the API → the default mode / clamped value.
2. `seedvr2_neo/run_inputs.py`: `split_into_passes` and `estimate_output_pixels` take the per-size resolution
   instead of one int. `seedvr2_neo/upscale_run.py`: each pass runs with its group's resolution
   (`dataclasses.replace(settings, ...)`), no vendored change (the context is built per pass).
3. `scripts/seedvr2_script.py`: a Radio "Output Size Mode" (default "Shortest edge"), the factor slider with quick
   ×1.5 / ×2 / ×4 buttons, the long-edge slider; only the active mode's control is visible. The label "Upscale
   Resolution (Shortest Edge)" stays unchanged. New arguments appended at the end of `script_args`. A console line
   with the computed output size per group; a note when an input under 512 px is raised to 512.
4. Infotext: the actual resolution per image (not one per run) plus `SeedVR2 Size Mode` and the factor / long edge.
5. Tests in the mirrored folders: mapping per mode, clamps, mixed img2img sizes (one resolution per group, correct pass
   budget), positional order of `script_args` (old callers unchanged), hostile API values (unknown mode, NaN,
   string, negative, huge).
6. Live check via the Forge API, 8 GB GPU without bf16, 3B GGUF, tile 512: ×2 on img2img and long edge 3840 on a
   16:9 txt2img; numbers into README; the long-edge maximum set by the result.
7. Docs: README (new controls, measured times), `02-ARCHITECTURE`, `09-PROCESSES`, `05-DECISIONS`, remove
   "Scale presets" from `06-IDEAS`, `07-RISKS` if anything new.

## Acceptance criteria
- [ ] (owner, in the browser) In txt2img and img2img the UI shows "Output Size Mode" with three choices; a fresh install and an existing
      `ui-config.json` both start in "Shortest edge" with the saved px value (screenshot / UI check).
- [x] Factor ×2 on a 512×384 img2img input gives 1024×768 (±1 px from even rounding); long edge 2048 on a 16:9 input
      gives 2048×1152 — tests and one live run.
- [x] img2img with two input sizes in one request: each size gets its own resolution; the pass budget uses it — tests.
- [x] An API call with the old argument list (no new arguments) behaves as today — positional-order test.
- [x] Hostile API values for the new arguments are clamped or fall back to the default, never crash — tests.
- [x] No mode produces an output long edge above 3840 (or the measured cap) except "Shortest edge", which is unchanged.
- [x] Infotext of every result names the resolution actually used for that image and the size mode.
- [x] Live numbers (time, clean image) for ×2 and the long-edge maximum on the 8 GB target are in README.
- [ ] Full test run green (1751 passed locally); CI green — after push.

## Which checks are needed on `/block-done`
Always: `code-auditor`, `fresh-eyes-reviewer`, `plan-drift-checker`. Additionally (tick what applies):
- [x] `integration-checker` — `script_args` grows, `split_into_passes` signature changes
- [ ] `security-auditor` — user data, authorization, external requests, files, money
- [x] `stress-tester` — new API input (mode, factor, long edge)

## Dependencies
B10 closed (done). The owner's local Forge with 3B GGUF for the live check (available).

## Out of scope for this block
- 6K / 8K and anything that needs DiT tiling — B12.
- Reading the new infotext keys back into the UI ("send to txt2img") — B08.
- GPU profiles, moving controls into Advanced — B08.
- Lowering `RESOLUTION.minimum` below 512 (true ×1 for small inputs) — inputs under 512 are raised to 512 with a note.

## Hidden assumptions
- Forge keeps slider/radio values in `ui-config.json` under the label path (`external-apis/forge-neo.md`): new
  controls have new labels, so nobody's saved value is reset; the old resolution label is kept as is.
- A short `script_args` is not padded by Forge: `run()` gets only the values sent, and only a *required* parameter
  left out fails with HTTP 500 (`external-apis/forge-neo.md`). So the new `run()` parameters get Python defaults equal
  to today's behaviour, and an old caller's list keeps working. README lists the new full argument order.
- txt2img: the script calls `process_images` inside `run()` and gets the final images, so the factor applies to the
  image after Hires. fix (agreed with the owner).
- The vendored pipeline rounds output sides to even numbers; ×1.5 of odd sizes may differ by one pixel.
- No fact-check run: Gradio 4.40 `Radio`, `.click`, `.change`, `gr.update(visible=...)` and `ui-config.json` are
  already verified in `external-apis/forge-neo.md`.

## Live results (8 GB GPU without bf16, 3B Q6_K GGUF, tile 512)
- img2img 512×384 ×2 → 1024×768, 51 s with `--cuda-malloc`; ×1.5 of 512×384 + 640×640 → 768×576 + 960×960;
  16 old arguments (shortest edge 768) → 1024×768, infotext "Shortest edge".
- txt2img 1024×576 SDXL, Long edge 3840 → 3840×2160, clean: with `--cuda-malloc` 1185 s total (encode 39 s, DiT
  10.5 min, decode 6.6 min); without it 580 s total (encode 44 s, DiT 5.7 min, decode 1.5 min). VRAM full, no OOM.
  Long-edge maximum kept at 3840, with the numbers in README.

## Open questions
- None: the 3840 cap and the ×2.0 / 2048 defaults were agreed with the owner.

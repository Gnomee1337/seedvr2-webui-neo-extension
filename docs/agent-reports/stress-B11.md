# Stress tests — B11 (output size modes)

Branch `feature/b11-output-size`. Only tests were written; no production code changed. Real bugs are strict xfails.

Full run: **2409 passed, 1 skipped, 32 xfailed** (before: 1674 / 1 / 3). The 29 new xfails cover 3 bugs.

## Bugs found

### 1. 🟡 R4 (S2×L2) A very thin image gets a 0 px output side and the run fails (new in Scale factor / Long edge; already possible in Shortest edge)
`seedvr2_neo/upscale_settings.py:74-77` (`OutputTarget.output_size`) with `:118` (`max_resolution = min(resolution * 4, 3840)`).
Input 1×4096, any mode (Scale factor ×2, Long edge 2048): resolution 512 → torchvision gives 512×2097152 → scaled to fit
2048 → `round(0.5) = 0` → planned **0×2048**. Checked against the vendored `SideResize`: it raises
`RuntimeError: Input and output sizes should be greater than 0 ... output (H: 2048, W: 0)`, so the run ends with
`SeedVR2 Error: Runtime Error: RuntimeError` (images returned unscaled). It also builds a 512×2 097 152 intermediate frame
first. Threshold: aspect ratio past ~1365:1 in Scale factor / Long edge (2048 px cap), past ~2900:1 in Shortest edge (4320).
The preview shows `Output size: **0×2048** from 1×4096`; `output_pixels` is 0, so the pass budget no longer splits such
frames (the old `estimate_output_pixels` counted resolution²×ratio). Likelihood is low (API or a hand-made strip image).
Possible fix: a minimum of 2 px per side in `output_size` is not enough (the vendor resize still produces 0); reject or
pad inputs past a ratio limit before the run, or cap the ratio of the requested size.
Tests: `test_upscale_settings_stress.py::test_extreme_aspect_ratios_never_give_a_zero_side` (14 cases),
`test_size_preview_stress.py::test_the_preview_never_names_a_zero_side` (6).

### 2. 🟡 R4 (S2×L2) "Raised to the 512 px minimum" while the output shortest edge is far below 512 (duplicate of code-auditor I-1/M-1, fresh-eyes 5, integration I2)
`seedvr2_neo/upscale_settings.py:118-119`; messages at `size_preview.py:30`, `upscale_run.py:104`. Input 16000×1000,
Scale factor ×2 → `OutputTarget(512, 2048, raised=True)` → output **2048×128**; preview, console say "raised to the 512 px
minimum" and the infotext says `SeedVR2 Resolution: 512`. Same for 8000×1000 ×1 → 2048×256 (a downscale at "×1"),
Long edge 1000×16000 → 128×2048. Confirmed by tests, not a new finding.
Tests: `test_upscale_settings_stress.py::test_a_raised_shortest_edge_is_really_at_least_512` (4),
`test_size_preview_stress.py::test_a_named_raise_means_a_512_px_shortest_edge` (1).

### 3. 🟢 R1 (S1×L1) The preview raises `OverflowError` for a huge source size
`seedvr2_neo/upscale_settings.py:74` (`int(self.resolution * longest / shortest)` is int/int true division) reached from
`size_preview.describe_output_size` → `preview_img2img_output_size`. Input `(1e308, 1)`, `(1, 1e308)`, `(10**400, 1)` as
the browser width/height (only by calling the Gradio endpoint by hand; the JS sends natural image sizes) → `OverflowError`
inside the Gradio event (error toast, stale line). `_is_usable_size` accepts any positive int. The run itself cannot get
such a size (PIL images). Fix: treat sides above some limit (e.g. 10^6) as unknown in `_is_usable_size`.
Test: `test_size_preview_stress.py::test_huge_source_sizes_do_not_raise` (4).

### Notes (not bugs)
- Long edge mode passes the chosen long edge when the 512 px floor applies (1920×800 at 1024 → 1228×512): the
  documented minimum; the test checks it stays within 4×512 and is named as a raise.
- `finish_images` zips images, details and the new per-image `params`: a shorter `params` list would silently drop
  images. The script always builds one per image (tested); only a future caller could break it.
- `txt2img_final_size` treats any truthy `enable_hr` (e.g. the string "false") as on — Gradio sends a bool, so moot.
- `parse_size_mode` falls back to Shortest edge for look-alikes (`Long edge`, Cyrillic `е`, `LONG_EDGE`); intended.

## Scenarios covered (all pass unless listed above)
- Garbage (None, "", whitespace, 10 000-char emoji, "nan"/"1e999", NaN/±inf, negatives, 0, 10^400, 1e308, bools, lists,
  dicts, bytes, objects, complex, numpy NaN/int64, Decimal, Fraction) in `size_mode`, `scale_factor`, `long_edge`, alone
  and together: settings stay in range, through `Script.run` no crash and no `SeedVR2 Error`, one infotext per image,
  pipeline resolution 512–3840, `max_resolution` ≤ 3840 outside Shortest edge.
- Infotext injection: `"Long edge, Steps: 999"`, `"Scale factor\nSteps: 1"` never reach the infotext; params read back
  with Forge's parser; Scale / Long Edge keys only in their own mode; per-image Resolution in mixed batches, incl. a
  failed image (error key plus its own size params).
- Extreme sizes in every mode (1×1, 1×1300, 12×4096, 1000×16000, 16000×1000, 20000×20000, 0/negative sides): sides
  positive, even, long edge ≤ 3840 in Scale factor / Long edge, orientation kept, `output_pixels` = w×h.
- `output_size` equals the vendored `SideResize` (stubbed `memory_manager`) for 15 sizes × 3 modes × 4 settings.
- Properties over thousands of random inputs: scale factor follows the input within rounding; monotonic in factor and
  in long edge; long edge never exceeded unless raised.
- Pass budget: 150–5000 mixed-size inputs — each pass one size, its own target, ≤ 16 Mpx unless a single frame, every
  index once, in order; transposed sizes in separate passes.
- Preview = run: random raw values (4000 cases), extreme sources, all Hires. fix combinations, and through the script
  the preview line equals the console's planned size for img2img and txt2img in every mode.
- Preview handlers (txt2img 10 args, img2img 6 args) and odd `res_step` option: never raise (except bug 3).
- Old API callers: 13, 16 arguments keep the shortest edge; 17 (mode only) and 18 (no long edge) use slider defaults;
  JSON-style strings `" long EDGE "`, `"3"`, `"1536"` are read.
- State: six consecutive runs with different modes on one Script give their own resolutions; concurrent previews in 8
  threads match sequential ones; separate Script instances keep separate `after_component` sources; wrong / near-miss
  elem_ids are ignored; a rebuilt UI uses the newest controls; the mode switch always shows exactly one column.

## New test files
- `tests/seedvr2_neo/test_upscale_settings_stress.py` (new)
- `tests/seedvr2_neo/test_size_preview_stress.py` (new)
- `tests/seedvr2_neo/test_upscale_run_stress.py` (new)
- `tests/scripts/test_seedvr2_script_stress.py` (new)
- `tests/seedvr2_neo/test_run_results_stress.py` (section "per-image size parameters" appended, `from PIL import Image` added)

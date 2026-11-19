# B07 — security-auditor report

(Recorded by the main chat from the agent's returned report; the agent did not write files. Line numbers refer to
`scripts/seedvr2_script.py` at efcf507.)

1. **IMPORTANT R6 (S3×L2) — no size limit per pipeline pass.** One group = all same-sized images; `torch.stack` of
   float32 tensors (:130), whole output to float32 + clamp copy (:135), groups unbounded (:367-368). txt2img Batch
   count 100 × size 8 = 800 images; API any number of `init_images` → host RAM exhausted, Forge frozen or killed.
   Fix: chunk groups by a frame/megapixel budget; convert output frame by frame; optional cap with a message.
2. **MINOR R4 — output size unbounded.** `int(resolution)` without clamp (API ignores the 3840 slider), `max_resolution=0`;
   `resolution=100000` or a 1×4096 input → huge allocations (pre-existing, more inputs now). Fix: clamp [512, 3840] /
   reject planned output above N MP before loading models.
3. **MINOR R3 — full exception text in `SeedVR2 Error` infotext** (:380). Not saved by the extension, but reaches the API
   `info`, gallery and a manual gallery Save; CUDA OOM / file errors contain GPU sizes and local model paths. Fix:
   infotext gets `type(e).__name__` + short text; full text stays in the console.
4. **MINOR R2 — img2img file names can collide** (:150-151, all inputs share seed + prompt). With "Add number to
   filename" off, results overwrite each other / previous runs. Default settings unaffected.
5. **MINOR R2 — interrupt does not stop saving of finished groups** (:312 plan before the upscale). With
   `save_incomplete_images` off Forge would not save after an interrupt.
6. **MINOR R2 — request conversions moved outside the `try`** (:322-326, after `unload_sd`). API `"abc"` resolution →
   `ValueError`, HTTP 500, SD already unloaded; no SeedVR2 VRAM held.
7. **INFO — `save_to_dirs` override (efcf507) is safe** (only subfolder on/off inside Forge's folder). Gaps: API string
   `"false"` counts as true; non-dict `override_settings` raises outside `try`.
8. **INFO — narrow catch in `save_result_image`** (:170, `OSError`/`ValueError` only): another exception from
   `save_image` loses the whole result (models already released).
9. **Not verified:** Pillow `MAX_IMAGE_PIXELS` in Neo (decoding happens before `run()`); converted copies of all inputs
   at once add to item 1.

Checked and clean: no personal data, local paths, models, binaries or `__pycache__` in the diff; `pillow` only in
`requirements-dev.txt`; saving rules ("Save original" off by default, `parse_flag`, `save_images=false` → no files);
seed parsing rejects nan/inf/bool/out of range; infotext quoting keeps the params line parseable and a multi-line
prompt cannot fake parameters; `finish_run` in `finally` covers all groups and errors.

## Re-check (after e455f0c)
Closed: F1 (16 MP pass budget, per-frame conversion), F3 (exception type only), F6 (settings parsed before
generation, never raise), F7 (`read_option` tolerant, `save_to_dirs` via `parse_flag`), F8 (broad save catch).
F4 / F5 kept as documented behaviour — agreed (Forge's Batch tab collides the same way; finished passes are complete
images and saving still follows `samples_save` / `save_images`).
Remaining, all MINOR R2, fixed by the main chat right after:
- F2 rest: a frame above the budget had no upper limit (`max_resolution=0`) → `max_resolution = 4 × resolution`
  (`UpscaleSettings.max_resolution`).
- Pipeline import error put the full message (local paths) into the result → type only.
- `embed_infotext` copied the script's own upscaled images (2× peak RAM) → only inputs / base images are copied.
INFO: `samples_format` from the API's `override_settings` adds no new surface (Forge applies it to its own images).
Repo check: no personal data, absolute paths or model files in the diff.

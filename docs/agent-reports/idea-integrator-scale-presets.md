# idea-integrator — output size as a scale factor / long-edge target

Idea: next to "Upscale Resolution (Shortest Edge)" (px, default 1080) let the user pick the output size as a scale
factor (x1, x1.5, x2, x3, x4) or a target class (2K / 4K / 8K), UI idea only from the related Forge Neo SeedVR2
extension (no code taken).

## 1. Already there?

- **`docs/06-IDEAS.md`, "Scale presets (2x / 3x / 4x, long edge)"** — the same idea, weight 1, "Part of B08 if taken".
  This assessment supersedes that entry; the weight 1 there is too low (see section 3).
- **"Image tiling for very large outputs"** (06-IDEAS, weight 3) — the related extension's 6K / 8K presets only work
  because it tiles the DiT. Our DiT sees the whole image. 8K presets depend on that idea.
- **B08** (settings and feedback, weight 2): no overlap in content; touches the same UI and the infotext read-back of
  `SeedVR2 ...` keys (`04-STATE.md` lists "SeedVR2 keys read back" under B08), which will have to know the new keys.
- **B10** (VAE tile 512 default, renamed slider, weight 1): touches the same file and the same `ui-config.json`
  mechanism, but a different control. This idea needs **no rename** (see section 4).
- **Code:** `RESOLUTION = Bounds(512, 3840, 1080, 64)` and `max_resolution = resolution * 4`
  (`seedvr2_neo/upscale_settings.py`); one resolution per run, used by `split_into_passes` for the 16 MP budget
  (`seedvr2_neo/run_inputs.py`) and by every pass in `seedvr2_neo/upscale_run.py`. The generation context (and the
  vendored `video_transform` built from `resolution`) is created fresh per pass, so a per-pass resolution needs no
  vendored change.
- **Rejected by the owner:** the list is empty. `05-DECISIONS.md`: "API values of the numeric settings are clamped to
  the slider ranges" applies to the new values too; "Same-sized images go through the pipeline in passes" makes a
  per-group resolution natural (groups are already by size).

## 2. Does it fit the goal

Yes for x1-x4 and long-edge targets: it is UI convenience for the core use (`01-VISION.md`, "In scope").
**No for 8K (and "~6K")**: the vision promises 8 GB GPUs "without out-of-memory errors" (REQ-4). Measured on the
target: a 2048 px output took 212 s (VAE decode 156 s with `--cuda-malloc`); 1080 px already spills with
`--cuda-malloc` (R6). 8K UHD is 7680x4320 = 33 MP, 8x the measured 2048x2048, whole-image DiT, and its shortest edge
(4320) is above our own 3840 cap. Offering an 8K button on this extension **would mislead users**: on 8 GB it would run
for a very long time or spill/OOM, and the slider bounds would silently clamp it to less than 8K anyway.
**4K** (3840x2160, 8.3 MP, about 2x the measured 2048x2048) is already reachable today (slider max 3840 shortest edge
is even more than 4K for 16:9) but not measured on 8 GB; it can be offered only with a measurement and an honest note.

## 3. Weight and options

| Option | What | Weight |
|---|---|---|
| A. Factor only | "Size mode" radio (Shortest edge px / Scale factor) + factor dropdown x1, x1.5, x2, x3, x4; per-group resolution = round(factor x shortest edge of that group), clamped to `RESOLUTION`; passes estimated per group; infotext key; tests; README | 2 |
| **B. Factor + long edge (recommended)** | A + a third mode "Long edge (px)" with a slider (e.g. 1024-3840) and the 2K (2560) / 4K (3840) values as slider ticks or documented values, not marketing buttons; a live measure of x2 and 3840 long edge on 8 GB with 3B GGUF, tile 512, numbers into README | 2 (3 if the 4K live measure turns ugly and needs a UI note) |
| C. Full presets incl. 6K / 8K | B + presets above 3840 | 3 + DiT tiling idea (3) first = 6; **reject for now** |

Why B: the long edge is what "2K/4K" actually mean (3840x2160 has a 2160 shortest edge, so mapping "4K" to our
shortest-edge slider would produce 6827x3840 for 16:9 — a wrong and much heavier image). A px long edge is honest and
cheap to add once per-group resolution exists. Showing "2K/4K" as named buttons invites "8K" next; a px slider does not.

### How a factor maps to our `resolution`
- Per **group** (same size and mode, already built by `group_by_shape`): `resolution = round(factor * min(w, h))`;
  long-edge mode: `resolution = round(target_long * min(w, h) / max(w, h))`. Then clamp to `RESOLUTION` (512-3840)
  and keep `max_resolution` as the absolute long-edge guard.
- img2img with inputs of different sizes: each group gets its own resolution — the per-pass context already makes this
  work; `split_into_passes` must take a per-group resolution (or a function size -> resolution) instead of one int.
- txt2img: the factor applies to the image Forge produced, **including Hires. fix output** (x4 on a 2048 hires image
  asks for 8192 -> clamped to 3840 shortest edge). The UI/console must say what was actually used.
- **x1** = SeedVR2 as a restorer at the same size; valid and useful in both tabs. Inputs whose shortest edge is under 512
  (img2img, SD1.5 sizes like 384) get clamped up to 512, i.e. not really x1 — either note it or lower `RESOLUTION.minimum`
  (needs a check that the pipeline is fine below 512).
- The vendored pipeline rounds the output to even sizes; x1.5 of odd sizes may differ by a pixel — acceptable.

## 4. Dependencies and what it touches

- **After B10.** B10 removes the top risk (R9) with one setting and is REQ-4; it must not wait for a UX feature. B10's
  fact-check of `ui-config.json` (how saved values are keyed — by label path, so a renamed label drops the saved value)
  answers this idea's question too: **keep the label "Upscale Resolution (Shortest Edge)"**, add new controls; existing
  users keep their saved px value and the default mode stays px. So no need to merge with B10.
- **Before B08 preferably**, so B08's infotext read-back covers the final keys (`SeedVR2 Size Mode`, `SeedVR2 Scale`).
  If B08 goes first, this block must extend the read-back.
- **API (positional `script_args`, order tested):** append the new arguments **at the end**, after
  `swap_io_components`, with defaults = current behaviour (mode "Shortest edge"). Old callers sending a px value in the
  `resolution` slot keep working unchanged. Do not change the meaning of the `resolution` slot and do not put a string
  mode into it. Unknown mode strings from the API -> default mode; factor clamped to 1-4 (or the dropdown's set).
- Complicates: `split_into_passes` signature, `estimate_output_pixels` callers, `UpscaleSettings` (resolution becomes
  per group — e.g. `dataclasses.replace(settings, resolution=...)` per pass), the infotext (resolution per image already
  per image), the positional-order test, `run()` signature.

## 5. Risks (R = S x L, `07-RISKS.md`)

| Risk | Rating |
|---|---|
| Offering 8K / 6K on 8 GB GPUs: hours-long runs, spill, OOM, and a silent clamp below the promised size | 🟡 R6 (S2xL3) — avoided by option B |
| Surprise heavy outputs: x3/x4 on a large txt2img (Hires. fix) or img2img input -> 3840 shortest edge, very slow on 8 GB with `--cuda-malloc` (multiplies R6 decode spill, and in batches section 4 R6) | 🟡 R6 (S2xL3); mitigate: console line with the computed size, info text under the control, clamp |
| 4K long edge not measured on 8 GB: may spill or take many minutes | 🟡 R4 (S2xL2); mitigate: live measure in the block, README numbers |
| API compatibility if new args are not appended at the end | 🟠 R9 (S3xL3) if done wrong; 🟢 R2 (S2xL1) with append + defaults + order test |
| Wrong per-group resolution -> wrong pass budget (RAM) or wrong size for some inputs of a mixed img2img batch | 🟢 R2 (S2xL1) with tests per group |
| x1 below 512 px silently becomes an upscale | 🟢 R2 (S1xL2) |
| Saved UI values: new controls only, no rename | 🟢 R1 (S1xL1) |

No risk to users' files or SD checkpoint beyond today's; the main live-user risk is long runs / VRAM spill.

## 6. Percentage

Now 21 of 24 = 87%. As its own block of weight 2: 21 of 26 = **80%**. If it grows to 3: 21 of 27 = 77%.
Merging into B08 (B08 2 -> 4) gives the same 80%, but makes B08 a two-day block — too big for one session.

## 7. Recommendation

Into the roadmap as **its own UX block (e.g. B11 "Output size: scale factor and long edge", weight 2)**, option B,
order B10 -> B11 -> B08. Without 8K/6K; those wait for the DiT tiling idea (and then only with measurements on
12-16 GB). Remove the "Scale presets" entry from `06-IDEAS.md` when taken; add "8K/6K presets — only after DiT tiling"
to the tiling idea.

## 8. Questions for the owner

1. Factor only (A), or factor + long edge (B)? Named "2K/4K" buttons or a px long-edge slider with 2560/3840 noted?
2. Agree that 8K/6K is not offered until DiT tiling exists?
3. Default for new installs: keep "Shortest edge 1080" (no change for anyone) or switch to x2?
4. txt2img with Hires. fix: factor applies to the hires image (natural, but heavy) — OK?
5. x1 for inputs under 512 px: clamp to 512 with a note, or lower the minimum (needs a check)?

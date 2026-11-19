# Risks and known limitations

Only what can break **right now**. Fixed items are removed from here (history is in git).
Bugs found and not yet fixed also go here, in the section for their area, with reproduction steps.
How processes work — `docs/09-PROCESSES.md`; facts about Forge Neo and upstream — `docs/external-apis/`.
Within a section, items go from higher risk to lower; a summary of the most dangerous ones is at the end of the file.

## How risk is rated: severity × likelihood

Every risk and every bug found gets a rating **R = S × L** and an icon based on it. Notation: `🟠 R8 (S4×L2)`.

**Severity (S)** — what happens if it fires:
| S | Meaning |
|---|---|
| 4 — critical | the WebUI crashes or becomes unusable until restart; the user's files are overwritten or lost |
| 3 — high | the upscale does not work at all for a whole class of users / models; the next SD generation fails (OOM) |
| 2 — medium | wrong or partial result, wrong metadata, a confusing error the user can work around |
| 1 — low | cosmetics, extra log lines, developer inconvenience |

**Likelihood (L)** — how often it fires:
| L | Meaning |
|---|---|
| 4 — almost certain | on every run with the given setup |
| 3 — often | a common setup or setting (GGUF model, batch > 1, GPU < 24 GB) |
| 2 — sometimes | needs a less common setting or input (grayscale image, renamed model, Linux paths) |
| 1 — rarely | an unlikely combination |

**Icon by R:**
| R | Icon | What to do |
|---|---|---|
| 12–16 | 🔴 critical | fix first |
| 8–9 | 🟠 high | fix in the next block |
| 4–6 | 🟡 medium | put in the plan when there is time |
| 1–3 | 🟢 low | keep on the list |

## 1. Installation and model discovery
- 🟡 R4 (S4×L1) **A corrupt GGUF array length makes the header read grow without limit.** `gguf.GGUFReader` (Neo's
  bundled `gguf`, `gguf_reader.py`, array loop) keeps reading past the end of the file for the declared length: a 61-byte
  file declaring 10^6 items took ~650 MB and 23 s; a value near 2^63 hangs the WebUI. Both `seedvr2_vendor/utils/dit_size.py` and
  the loader (`seedvr2_vendor/core/model_loader.py`) use it. Truncated downloads do not trigger it (the header stays intact).
  Repro: `test_gguf_with_absurd_array_length_does_not_grow_memory` (strict xfail). Close: bounded own GGUF header parser.
  The header is also read for the 7B low-VRAM note: for the default DiT when Forge builds the UI, and on every DiT
  selection — so such a file as the default DiT hangs Forge at startup.
- 🟢 R2 (S2×L1) **A VAE without `vae` in its name is listed as a DiT**; the note then says no VAE was found. Rename the file.
- 🟢 R2 (S2×L1) **`find_model_file` falls back to the working-directory scan if the chosen file vanished** between the
  scan and the load (`seedvr2_vendor/utils/constants.py`), and raises `NotADirectoryError` if `./models/SEEDVR2` there is a file.
  Repro: `test_cwd_models_entry_that_is_a_file_does_not_raise` (strict xfail).
- 🟢 R2 (S1×L2) **The extension-folder fallback is checked by tests only**, not in a live Forge run.
- 🟢 R2 (S1×L2) **A skipped folder is not mentioned.** With a full set in the extension folder, a new DiT put into
  `<WebUI>/models/SeedVR2` alone never appears (that folder is incomplete, so the complete one wins); with the DiT in
  one folder and the VAE in the other, the note names only one. Close: name the skipped folder in the note / console.
- 🟢 R1 (S1×L1) **Small UI/listing roughness:** a backtick in the folder path breaks the note's Markdown; zero-byte model
  files are listed; the file extensions are defined in `model_discovery.py`, `dit_size.py` and `constants.py`, the
  DiT/VAE split is spelled three times in `model_discovery.py`, and the note hard-codes 🔄 instead of `refresh_symbol`.

## 2. Dependencies
- 🟢 R2 (S1×L2) **`gguf` comes from Forge Neo, not from `install.py`.** `import gguf` resolves to Neo's bundled copy
  (`external-apis/forge-neo.md`); if Neo drops or moves it, `.gguf` models stop loading. Close: a test that imports it. (B09)

## 3. VRAM and model lifecycle
Live-checked on an 8 GB GPU without bf16, 16 GB RAM, 1080px output, VAE tile 512: 3B GGUF Q6_K — DiT step 7 s;
3B fp8 — 5–7 s; 7B fp8 with BlockSwap 16 — 34–42 s (780 s without BlockSwap). All gave clean images; after a run with
"Unload" the GPU drops back to about 0.6 GB used. Between phases the reserved VRAM goes back to ~0.5 GB.

- 🟠 R9 (S3×L3) **VAE tile 1024 (the default) with `--cuda-malloc` can stall a run on 8 GB GPUs.** Live, 3B GGUF Q6_K,
  1024px shortest edge (img2img 512×384 → 1365×1024): VAE encode or decode ran 6–22+ minutes at full VRAM and 100% GPU
  without finishing, from a fresh Forge start too, on the old and the new code alike; the same encode once took 9 s
  after a txt2img run. Without `--cuda-malloc`, at 1080px: encode peak 6.9 GB, 1.3 GB paged (16 s vs 5.7 s with tile
  512); decode 2.9 GB paged. README recommends 512; the default is unchanged. An interrupt does not stop a running
  phase (B08). Close: tile 512 as the default, or a VRAM-based default; re-measure on the owner's setup.
- 🟡 R6 (S2×L3) **With `--cuda-malloc` (cudaMallocAsync) VAE decoding spills into shared memory.** At 1080px with tile
  512 the decode reserves 8.7 GB for a 5.2 GB peak, pages 0.7–2.9 GB and takes 25–137 s; without the flag 6.7 GB
  reserved, no spill, 12.4 s. The flag is opt-in in Forge but common on 8 GB setups; it makes the 7B DiT step faster
  with few BlockSwap blocks (8 blocks: 21 s vs 122 s). README gives the trade-off. Close: release the cache inside the
  VAE decode between tiles, or measure `expandable_segments`.
- 🟡 R4 (S2×L2) **7B with too few BlockSwap blocks falls off a cliff.** 7B fp8 on 8 GB: 8 blocks fit only with
  `--cuda-malloc` (0 GB free), 4 blocks spill (297 s). Larger outputs need more blocks. The note recommends 16.
- 🟡 R4 (S2×L2) **"Keep in RAM" holds the models in system RAM until the next run** (3–4 GB for 3B fp8, ~16 GB for
  7B fp16). Together with Forge's offloaded SD checkpoint a 16 GB system dropped to 0.03–0.7 GB free RAM. Default is
  "Unload SeedVR2"; the option's hint and the README say what it costs. An interrupt before the upscale starts (during
  the base image) leaves the previous run's kept models in RAM until the next run.
- 🟡 R4 (S2×L2) **GPUs without bf16 (RTX 20xx / GTX 16xx) run the pipeline in emulated bf16.** The bf16 probe in
  `seedvr2_vendor/optimization/compatibility.py` passes on current PyTorch, so DiT and VAE compute in bf16; results are clean.
  Not measured: how much faster fp16 compute would be on these GPUs. Idea: `06-IDEAS.md`.
- 🟡 R4 (S2×L2) **The 7B low-VRAM note judges the model family, not its size.** The 11 GiB threshold is sized for 7B fp8
  (~7.9 GB of weights): 7B fp16 (~16 GB) on 12–16 GB GPUs gets no note, a small 7B GGUF (Q4) on 8 GB gets one it may not
  need. The note is computed for the default DiT when the UI is built; a DiT restored from `ui-config.json` may show
  the other model's note until the selection changes (not checked in Forge). Close: threshold from the DiT file size.
- 🟢 R3 (S3×L1) **An unloaded model's VRAM is freed only when nothing references it.** `release_model_memory` does not
  free CUDA storage (`external-apis/seedvr2-upstream.md`, local patches); the memory returns when the runner drops the
  model and `gc.collect()` + empty-cache run (live: the DiT's 3.2 GB stays allocated right after Phase 2 and is gone
  before Phase 3). A stray reference would keep the VRAM held with no warning.
- 🟢 R3 (S1×L3) **"Unload SD Checkpoint" (on by default) costs a full checkpoint reload on the next generation** — about
  2 minutes with a large checkpoint. It is the price of freeing VRAM; the checkbox hint says so. Idea for a partial unload: `06-IDEAS.md`.
- 🟢 R2 (S1×L2) **"Keep in RAM" reloads both models when only one changes.** Changing only the VAE drops the cached DiT
  too (a 7B reloads from disk). After Forge's "Reload UI" the kept models stay in RAM untracked until the next run drops
  them. Switching Keep in RAM → Unload with the same files also reads them from disk once (the vendored cache drops its
  entries when caching is off).
- 🟢 R2 (S2×L1) **"Keep in RAM" + an error or interrupt in Phase 2 copies the DiT to RAM just to drop it.** The phase's
  `finally` caches it (`cache_model=True`), then the run is failed and the models are unloaded: a RAM spike of the DiT's
  size and a few seconds after Interrupt. Result correct (live). Avoiding it needs a patch of the vendored `finally`.
- 🟢 R2 (S2×L1) **Not run live:** Keep in RAM with BlockSwap on; Keep in RAM with the block count changed between runs
  (upstream's `_handle_blockswap_change`); interrupt in Unload mode.
- 🟢 R1 (S1×L1) **Edge cases of the B06 helpers** (stress tests in `tests/seedvr2_neo/*_stress.py`): a same-size file
  replaced within one mtime step (100 ns NTFS, 2 s FAT32) is trusted; a symlinked model path counts as another file
  (one extra reload); `"8.0"` as a string from the API means BlockSwap off; an error from the memory release in the
  script's `finally` replaces the run's result with a raw exception.
- 🟢 R2 (S2×L1) **Calls with the old argument list** (API payloads saved before the new controls) get the defaults
  for "After run" and BlockSwap: unload, BlockSwap off.

## 4. Generation flow and results
- 🟡 R6 (S2×L3) **With `--cuda-malloc` a batch multiplies the VAE-decode spill** (section 3): live, 3B GGUF, 4 × 512→1024 px,
  tile 512 on an 8 GB GPU — each frame decoded in 90–160 s with VRAM full (vs ~10 s without the flag); the run took
  ~12 min. Not caused by batching, but users with batches feel it most.
- 🟡 R4 (S2×L2) **A pass may still need a few GB of RAM.** Passes are capped at 16 MP of output (`PASS_BUDGET_PIXELS`
  in `seedvr2_neo/run_inputs.py`), but the pipeline keeps its own copies (latents, decoded frames, alpha) per pass;
  peak RAM per pass is not measured. Huge inputs (a decompression bomb through the API) are decoded by Forge before
  `run()`; whether Neo keeps Pillow's `MAX_IMAGE_PIXELS` limit is not verified.
- 🟢 R3 (S1×L3) **An interrupt or an error loses the whole pass in progress.** Its images come back un-upscaled (as
  inputs with `SeedVR2 Error` and a result comment); only earlier passes survive. Finished passes are saved even after
  an interrupt, like Forge's earlier batches.
- 🟢 R2 (S1×L2) **Each extra pass reloads the models in "Unload" mode** (different sizes, or more than 16 MP of
  output). "Keep in RAM" reuses them.
- 🟢 R2 (S1×L2) **A batched image is not reproduced exactly from its "SeedVR2 Seed".** The VAE encode draws its sample
  from one seeded generator per pass (`generation_phases.py`, `set_seed(seed + 1000000)` before the loop), so frame k's
  noise depends on k: upscaling one image alone gives a slightly different result. The images do not affect each
  other's content.
- 🟢 R2 (S1×L2) **img2img file names can collide** when Forge's "Add number to filename" is off: all inputs of a run
  share the SeedVR2 seed and the prompt, so a pattern like `[seed]-[prompt]` repeats and a later file overwrites an
  earlier one. Default settings add a number.
- 🟢 R2 (S1×L2) **Only some `override_settings` reach the extension's own saves:** `samples_save`, `samples_format`,
  `save_to_dirs` (read by the script) and `samples_filename_pattern` (read by `save_image`). Others
  (`directories_filename_pattern`, `save_images_add_number`, …) use the user's global setting.
- 🟢 R2 (S1×L2) **Empty results carry one infotext and no image** (setup errors, nothing to upscale): with the ✨
  (hires upscale) button the selected image drops out of the gallery (still on disk); in an img2img batch such a file
  shifts the infotexts of later files.
- 🟢 R2 (S1×L2) **The SeedVR2 keys are not read back into the script's controls** ("Send to" / PNG Info restore only
  Forge's fields; B08 candidate). An img2img infotext has the prompt and the SeedVR2 keys, no Forge `Seed`.
- 🟢 R2 (S1×L2) **`-original` copies and not-upscaled results are the converted input** (8-bit RGB / RGBA), not the
  file as given: a 16-bit, CMYK or palette input loses its original form there. The upscale itself needs the conversion.
- 🟢 R1 (S1×L1) **A failure note from the second file on is not shown in the img2img Batch tab:** Forge keeps the
  `Processed` (and its comments) of the first file only; each image's infotext still has `SeedVR2 Error`.
- 🟢 R1 (S1×L1) **A 16-bit grayscale PNG loses its transparency key** (it is scaled correctly, but comes back opaque).
  Repro: `test_16_bit_png_with_transparency_is_scaled_and_keeps_alpha` (strict xfail).
- 🟢 R3 (S1×L3) **Interrupt reacts only between phases** and the Forge progress bar is not updated during SeedVR2.
  Close: set `ctx['interrupt_fn']` to a Forge check and update `shared.state.job*` from `progress_callback`. (B08)

## 5. Code hygiene
- 🟢 R2 (S1×L2) **Upscaled pixels are truncated, not rounded, to 8 bits** (`tensor_to_images` in
  `seedvr2_neo/upscale_run.py`: `* 255` then `astype(uint8)`): channels come out up to one level darker (a flat 10
  comes back as 9). Not visible by eye; fix together with a pipeline-output change, not in a refactor.
- 🟢 R3 (S1×L3) **The script's tests run against stand-ins of Forge, Gradio and the pipeline** (`tests/scripts/forge_stubs.py`).
  A Forge Neo change of a name or signature the script uses, or an upstream change of `GlobalModelCache` /
  `prepare_runner` / `validate_blockswap_config`, leaves all tests green. Mitigation: the stand-ins follow
  `external-apis/forge-neo.md`; a live run after a Forge or upstream update.
- 🟢 R3 (S1×L3) **Leftover `src/` folder after an update.** Forge's update (`reset --hard`) removes the tracked files but
  keeps untracked `src/**/__pycache__`. Harmless: nothing imports `src`, and a folder without `__init__.py` loses to
  another extension's real `src` package. Users may delete it by hand.
- 🟢 R2 (S1×L2) **Manual `git pull` can stop on the removed `.pyc` files** that Python rewrote locally. Forge's own update
  is not affected (`external-apis/forge-neo.md`); the README gives `git reset --hard origin/main`.
- 🟢 R2 (S1×L2) **Same Script title as the original SeedVR2 Forge extension** ("SeedVR2 Native Upscaler (24G)"): with both
  installed the entries look alike, and `run()` matches scripts by title when it removes itself before txt2img. README
  asks to install only one; the title changes in B08.
- 🟢 R1 (S1×L1) **Our top-level package names `seedvr2_neo` / `seedvr2_vendor` can still collide** with a future extension
  that uses the same names; no check exists. Unlikely with these names.

## Top risks
1. 🟠 R9 VAE tile 1024 (default) with `--cuda-malloc` can stall a run for many minutes on 8 GB — section 3.
2. 🟡 R6 VAE decode spills with `--cuda-malloc` — section 3.
3. 🟡 R6 A batch multiplies the `--cuda-malloc` decode spill — section 4.
4. 🟡 R4 Corrupt GGUF header can hang the WebUI — section 1.
5. 🟡 R4 7B with too few BlockSwap blocks — section 3.
6. 🟡 R4 A pass may still need a few GB of RAM (not measured) — section 4.

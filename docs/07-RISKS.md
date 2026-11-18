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
- 🟢 R2 (S2×L1) **Cached models are matched by file name only.** The folder is re-chosen on every run; a same-named file
  in the other folder, or a file replaced in place, keeps the old weights and config. Force Reload does not clear it. (B06)
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
Live-checked on an 8 GB GPU without bf16, 16 GB RAM, 1080px output: 3B GGUF Q6_K — DiT step 7 s; 3B fp8 — 10 s;
7B fp8 — 780 s. All three gave clean images. After the run the GPU drops back to about 1 GB used.

- 🟠 R9 (S3×L3) **7B is practically unusable on 8 GB: no low-VRAM path.** 7B fp8 weights take 7.9 GB, the rest spills
  into shared memory and the DiT step is ~80× slower than 3B. BlockSwap, DiT/VAE offload exist in `seedvr2_vendor/` but are fixed
  to "off" in `run()`; the UI offers 7B with no hint. Close: BlockSwap/offload in the UI, a warning for 7B on small GPUs. (B06)
- 🟠 R9 (S3×L3) **"Force Reload" does not reload models.** It clears only `self.runner_cache` (the context);
  `GlobalModelCache` keeps the DiT/VAE, so a broken or swapped file with the same name is reused. (B06)
- 🟡 R6 (S2×L3) **Reserved VRAM grows between phases and spills into shared memory.** PyTorch's allocator keeps freed
  blocks reserved (4.7 GB after VAE encode with 0.5 GB in use, 8.8–9 GB at the end), the upstream logs "VRAM overflow:
  0.8–1 GB paged to system RAM". Seen: VAE decode of 1962×1080 took 470 s against 43 s for 1080×1080 in the same session;
  1080×1080 VAE decode took 38 s in the first run and 117 s in the second (4.8 GB reserved before decode, 2.9 GB paged).
  Close: release the cache between phases, check with a live run. (B06)
- 🟡 R6 (S2×L3) **SeedVR2 models stay in system RAM after the run.** With no offload device the cleanup moves DiT and VAE
  to CPU for caching (`seedvr2_vendor/optimization/memory_manager.py`, cleanup of DiT/VAE), so 3–8 GB stay in RAM. Together with
  Forge's offloaded SD checkpoint a 16 GB system dropped to 0.03 GB free RAM. Close: "unload SeedVR2 after run". (B06)
- 🟡 R6 (S2×L3) **"Unload SD Checkpoint" (on by default) costs a full checkpoint reload on every run** — about 2 minutes
  with a large checkpoint on the next generation. It is the price of freeing VRAM; the UI does not say so. (B06)
- 🟡 R4 (S2×L2) **GPUs without bf16 (RTX 20xx / GTX 16xx) run the pipeline in emulated bf16.** The bf16 probe in
  `seedvr2_vendor/optimization/compatibility.py` passes on current PyTorch, so DiT and VAE compute in bf16; results are clean.
  Not measured: how much faster fp16 compute would be on these GPUs. (B06)
- 🟡 R4 (S2×L2) **Cached context survives exceptions.** After an OOM mid-run, `self.runner_cache['ctx']` keeps partial
  state; the `finally` block deletes key names that do not exist (`images`, `encoded_latents`, `upscaled_latents`). (B06)
- 🟡 R4 (S2×L2) **Device hard-coded to `cuda:0`.** Ignores Forge's `--device-id` and non-CUDA backends; Forge's own device
  is `modules.devices.device`. (B06)

## 4. Generation flow and results
- 🟠 R8 (S2×L4) **Only the first image is upscaled.** txt2img with batch size / count > 1: `processed_base.images[0]` —
  and with "return grid" on (default) that is the grid, so the grid gets upscaled. img2img: only `init_images[0]`. (B07)
- 🟡 R6 (S2×L3) **Result infotext is "SeedVR2 Upscaled".** `Processed(..., info="SeedVR2 Upscaled")` without `infotexts`:
  the generation-info panel and "Send to" lose the parameters. In img2img `p.all_prompts` is unset, `create_infotext`
  fails inside a bare `except`, so the saved PNG has no parameters either. (B07)
- 🟡 R4 (S2×L2) **Grayscale / palette inputs fail.** `transforms.ToTensor()` on an `L`/`P` image gives 1 channel; the
  pipeline expects 3 (or 4). Close: `convert("RGB")` unless RGBA. (B07)
- 🟡 R4 (S2×L2) **Saving ignores the user's settings.** The result is always saved (ignores "Always save all generated
  images"); "Save Extra Copy of Original" saves the base image a second time. (B07)
- 🟡 R4 (S2×L2) **Empty base result crashes.** If `process_images` returns no images, `images[0]` raises outside the `try`. (B07)
- 🟢 R2 (S1×L2) **An error result shifts gallery infotexts.** Empty results carry one infotext and no image. With the ✨
  (hires upscale) button the selected image drops out of the gallery (still on disk) and later images show their
  neighbour's infotext; in an img2img batch a failed file shifts the infotexts of later files. Close: return the input
  image with the error in its infotext. (B07)
- 🟢 R3 (S1×L3) **Interrupt reacts only between phases** and the Forge progress bar is not updated during SeedVR2.
  Close: set `ctx['interrupt_fn']` to a Forge check and update `shared.state.job*` from `progress_callback`. (B08)

## 5. Code hygiene
- 🟢 R3 (S1×L3) **No CI; `scripts/seedvr2_script.py` has no tests** (needs Forge stubs). (B09)

## Top risks
1. 🟠 R9 7B practically unusable on 8 GB, no low-VRAM path — section 3.
2. 🟠 R9 "Force Reload" does not reload models — section 3.
3. 🟠 R8 Only the first image (or the grid) is upscaled — section 4.
4. 🟡 R6 Reserved VRAM grows and spills into shared memory — section 3.
5. 🟡 R6 SeedVR2 models stay in system RAM after the run — section 3.
6. 🟡 R6 SD checkpoint reload on every run — section 3.
7. 🟡 R6 Result infotext lost — section 4.
8. 🟡 R4 Corrupt GGUF header can hang the WebUI — section 1.

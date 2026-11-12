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
- 🔴 R12 (S3×L4) **README pointed to a folder the code never reads.** The original README said `./model/seedvr2`;
  the script reads `models/SeedVR2` (extension root, then WebUI root). Following the old README gives empty dropdowns.
  README fixed; the code still has three spellings (`SeedVR2` in the script, `SEEDVR2` in `src/utils/constants.py`,
  case-sensitive on Linux). At load time `find_model_file` (`src/utils/constants.py`) first looks in the
  working-directory-relative `./models/SEEDVR2` and only then in the script's folder, so a same-named file there wins. Close: one absolute folder, resolved in one place, with a visible
  message when it is empty. (B05)
- 🟡 R6 (S2×L3) **No refresh for the model list.** Models are scanned in `ui()`; a new file needs a UI restart. (B05)
- 🟡 R4 (S2×L2) **Model config chosen by file name.** `model_configuration.py`: `"7b" in dit_model` (case-sensitive).
  A 7B file renamed to `...7B...` or without `7b` loads the 3B config and fails with a shape error. (B05)

## 2. Dependencies
- 🟢 R2 (S1×L2) **`gguf` comes from Forge Neo, not from `install.py`.** `import gguf` resolves to Neo's bundled copy
  (`external-apis/forge-neo.md`); if Neo drops or moves it, `.gguf` models stop loading. Close: a test that imports it. (B09)
- 🟢 R2 (S1×L2) **`requirements.txt` lists unused packages** (`peft`, `matplotlib`) and is not read by Forge at all. (B04)

## 3. VRAM and model lifecycle
Live-checked on an 8 GB GPU without bf16, 16 GB RAM, 1080px output: 3B GGUF Q6_K — DiT step 7 s; 3B fp8 — 10 s;
7B fp8 — 780 s. All three gave clean images. After the run the GPU drops back to about 1 GB used.

- 🟠 R9 (S3×L3) **7B is practically unusable on 8 GB: no low-VRAM path.** 7B fp8 weights take 7.9 GB, the rest spills
  into shared memory and the DiT step is ~80× slower than 3B. BlockSwap, DiT/VAE offload exist in `src/` but are fixed
  to "off" in `run()`; the UI offers 7B with no hint. Close: BlockSwap/offload in the UI, a warning for 7B on small GPUs. (B06)
- 🟠 R9 (S3×L3) **"Force Reload" does not reload models.** It clears only `self.runner_cache` (the context);
  `GlobalModelCache` keeps the DiT/VAE, so a broken or swapped file with the same name is reused. (B06)
- 🟡 R6 (S2×L3) **Reserved VRAM grows between phases and spills into shared memory.** PyTorch's allocator keeps freed
  blocks reserved (4.7 GB after VAE encode with 0.5 GB in use, 8.8–9 GB at the end), the upstream logs "VRAM overflow:
  0.8–1 GB paged to system RAM". Seen: VAE decode of 1962×1080 took 470 s against 43 s for 1080×1080 in the same session.
  Close: release the cache between phases, check with a live run. (B06)
- 🟡 R6 (S2×L3) **SeedVR2 models stay in system RAM after the run.** With no offload device the cleanup moves DiT and VAE
  to CPU for caching (`src/optimization/memory_manager.py`, cleanup of DiT/VAE), so 3–8 GB stay in RAM. Together with
  Forge's offloaded SD checkpoint a 16 GB system dropped to 0.03 GB free RAM. Close: "unload SeedVR2 after run". (B06)
- 🟡 R6 (S2×L3) **"Unload SD Checkpoint" (on by default) costs a full checkpoint reload on every run** — about 2 minutes
  with a large checkpoint on the next generation. It is the price of freeing VRAM; the UI does not say so. (B06)
- 🟡 R4 (S2×L2) **GPUs without bf16 (RTX 20xx / GTX 16xx) run the pipeline in emulated bf16.** The bf16 probe in
  `src/optimization/compatibility.py` passes on current PyTorch, so DiT and VAE compute in bf16; results are clean.
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
- 🟢 R3 (S1×L3) **Interrupt reacts only between phases** and the Forge progress bar is not updated during SeedVR2.
  Close: set `ctx['interrupt_fn']` to a Forge check and update `shared.state.job*` from `progress_callback`. (B08)

## 5. Code hygiene
- 🟡 R6 (S2×L3) **`.gitignore` is UTF-16**, so git ignored none of its rules and `__pycache__/*.pyc` (Python 3.11) are
  committed. `.gitignore` is rewritten in UTF-8; the tracked `.pyc` files still need `git rm -r --cached`. (B04)
- 🟡 R6 (S3×L2) **Top-level package name `src`.** The script puts the extension root first on `sys.path` and imports `src.*`;
  any other extension or library that also exposes `src` and is imported first wins (`sys.modules` cache). (B04)
- 🟡 R4 (S2×L2) **No LICENSE / attribution.** `src/` is an unmodified copy of ComfyUI-SeedVR2_VideoUpscaler (Apache-2.0),
  which requires shipping the license text and a notice. (B04)
- 🟢 R3 (S1×L3) **No tests, no CI.** (B09)
- 🟢 R2 (S1×L2) **Unused `download_weight` import** in `scripts/seedvr2_script.py`. (ComfyUI files are kept on purpose — `05-DECISIONS.md`.) (B04)

## Top risks
1. 🔴 R12 Model folder mismatch (README fixed, code still inconsistent, relative path) — section 1.
2. 🟠 R9 7B practically unusable on 8 GB, no low-VRAM path — section 3.
3. 🟠 R9 "Force Reload" does not reload models — section 3.
4. 🟠 R8 Only the first image (or the grid) is upscaled — section 4.
5. 🟡 R6 Reserved VRAM grows and spills into shared memory — section 3.
6. 🟡 R6 SeedVR2 models stay in system RAM after the run — section 3.
7. 🟡 R6 SD checkpoint reload on every run — section 3.
8. 🟡 R6 Result infotext lost — section 4.
9. 🟡 R6 `.gitignore` in UTF-16, `.pyc` committed — section 5.
10. 🟡 R6 Package name `src` can collide — section 5.

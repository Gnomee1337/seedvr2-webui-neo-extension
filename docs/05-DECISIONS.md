# Decisions

This file records WHY the system is built the way it is: non-obvious choices, the owner's rules,
workarounds for external limitations. How processes work step by step — `docs/09-PROCESSES.md`,
what exists in the code — `02-ARCHITECTURE.md`, facts about Forge Neo and upstream — `docs/external-apis/`.
**Only decisions in force live here.** Revoked ones are deleted; history is in git. No numbers or dates.
New entry: `### Short title` followed by `Why: …` (1–5 lines, plus the pitfall if there is one).
Entries marked *reconstructed from code* were inherited from the original author; the reason is our best reading.

---

## Process and documentation

### Documentation only in `docs/`, written for the current state
Why: a log that grows as a chronicle becomes unreadable within a month — you cannot find the rule in force.
Documents are rewritten for the current state; history is in git. Decision numbers and dates are not used in code or
documents. The `.claude/` folder is for Claude Code settings, not documentation.

### Everything a developer reads is in English
Why: the original code and README were in Chinese; the extension is public and maintained by people who read English.
Docs, comments, docstrings and commit messages — English; the owner is addressed in Russian.

### Tests mirror the code folders
Why: from a file's path you immediately see where its tests are, and vice versa.

### `pytest.ini` lives in `tests/`, tests run as `python -m pytest tests`
Why: the root holds ComfyUI's package `__init__.py` (ComfyUI-only); with the ini in the root pytest imports it and fails.
A root `conftest.py` hook cannot stop that (hooks for the root come from its parent), and `--rootdir` in `addopts` is ignored.
Default import mode: importlib mode would register `tests/seedvr2_vendor` as the `seedvr2_vendor` package and hide the real one.

### The Forge script is tested against stand-ins of Forge, Gradio and the pipeline modules
Why: Forge cannot be installed in pytest. `tests/scripts/forge_stubs.py` copies only the names and signatures the script
uses, as recorded in `external-apis/forge-neo.md` (a stand-in that drifts from Forge lets tests pass against an API that
does not exist). The pipeline is replaced in `sys.modules` at the `seedvr2_vendor.*` module level, not inside the script,
so the tests survive moving code. Tests drive `Script.run()` and check what users get, not internal calls.

### CI runs only pytest; no linter, no required checks on GitHub
Why: one developer, tests already run before every commit; CI's value is a clean checkout (a file left out of a commit,
a package missing from `requirements-dev.txt`) and outside pull requests. The owner chose this minimal form.
Python 3.10 (the lowest supported) and 3.13 (current Forge Neo installs) with CPU torch from the PyTorch index; torch is in `requirements-dev.txt` because the
script imports it.

### torch is imported in `seedvr2_neo/` only by `upscale_run.py`
Why: the pipeline run moved out of the script needs tensors; the rest of `seedvr2_neo/` stays free of torch, so its
tests stay fast and readable. Forge-only things (device, memory release, interrupt flag) come in as `HostHooks`.

### Branches: simplified Git Flow
Why: one developer, no versioned releases yet. `main` + short `feature/*` / `fix/*` branches merged back with green CI.
Switch to full Git Flow when releases are tagged and users pin versions. Agreed with the owner.
Pitfall: the extension is public and Forge's "Check for updates" pulls `main` directly, so `main` *is* production.
A branch is merged into `main` only after a live run on Forge Neo and with the owner's consent; pushes — only with consent.

### No personal data of the owner in the repository
Why: the repository is public. Hardware, names, contacts and similar details of the owner are not written into docs, code or
commits; hardware targets are described as a class ("8 GB GPU without bf16"), not as someone's machine.

## Vendored pipeline

### SeedVR2 code is vendored in `seedvr2_vendor/`, not installed as a package
Why (*reconstructed from code*): the upstream ComfyUI port is not a pip package and imports ComfyUI only optionally;
copying it makes the extension self-contained. Pitfall: upstream fixes do not arrive by themselves — local patches are
listed in `docs/external-apis/seedvr2-upstream.md` so a sync does not lose them.

### The vendored package is named `seedvr2_vendor`, not upstream's `src`
Why: Forge keeps every extension folder on `sys.path`, and other SeedVR2 ports also ship a top-level `src`; whichever
loads first is reused by both, so ours ran without its patches. Imports inside the package are relative, so the rename
touches only the code that imports it from outside. A sync copies upstream `src/` into `seedvr2_vendor/`. The runtime check
for a foreign `src` was removed with the rename.

### Fix in `scripts/` first, patch `seedvr2_vendor/` only when unavoidable
Why: every change to vendored code makes the next upstream sync harder. A small marked patch is preferred over
monkey-patching `seedvr2_vendor/` from the script: a patch conflict on sync is visible, a broken monkey-patch fails silently.
Our own helpers that need no Forge imports live in `seedvr2_neo/` (a `.py` in `scripts/` would be loaded by Forge as a script).

### Only Forge Neo is supported; ComfyUI files stay but are not maintained
Why: ComfyUI already has its own SeedVR2 node pack (the upstream). The root `__init__.py` and `seedvr2_vendor/interfaces/` are ComfyUI entry
points, never loaded by Forge; they are kept as close to upstream as possible so `diff -r` stays clean (the root `__init__.py`
only follows the package rename). Not fixed, not tested here.

### The whole repository is Apache-2.0; `NOTICE` credits the sources and lists our changes
Why: the owner's choice — the same license as the vendored code, so the repository has one license. Apache-2.0 §4(b)
change notices: a `Local patch (Forge Neo)` comment at each change plus the list in `external-apis/seedvr2-upstream.md`.
No per-file license headers in our own files: that is ASF guidance, not a §4 condition, and it needs a named copyright holder.
Existing ByteDance / HuggingFace headers in vendored files are kept as they are (§4(c)). The original Forge extension
had no license file; its author is credited in `NOTICE` and `README`, and the owner accepted publishing under Apache-2.0 on that basis.

### `requirements.txt` is our own informational list, not upstream's
Why: Forge Neo does not read it (`install.py` installs what is missing), so it only informs a reader; upstream's copy listed
packages the code never imports (`peft`, `matplotlib`).

## Forge integration

### Models live in Forge's `models/SeedVR2`; the extension's own `models/` is only a fallback
Why: the owner's choice — users keep all models in one place, and it survives reinstalling the extension. The folder comes
from `shared.models_path` (follows `--model-ref` / `--data-dir`), never from the working directory. The first candidate
holding both a DiT and a VAE wins, so a stray or partial set in one place does not hide a full set in the other.

### 3B or 7B is read from the checkpoint header, not from the file name
Why: users rename files; a 7B built with the 3B config fails with a shape error. Block count (32 / 36) in the header is
cheap to read and exact. The file name is only a fallback when the header cannot be read.

### A selectable Script, not an always-on script or an Extras upscaler
Why (*reconstructed from code*): a Script owns `run()` and can replace the whole generation, which lets img2img skip
diffusion and txt2img chain "generate → upscale" in one click.

### txt2img calls `process_images` from inside `run()`
Why (*reconstructed from code*): the base image must be produced by Forge with all other extensions applied; the script is
removed from `p.scripts.scripts` for that call and restored in `finally`.

### Every generated or input image is upscaled; no grid of the results
Why: the owner's choice. A grid of upscaled images is a huge extra file that costs memory and time, and the gallery
already shows each image. In img2img each distinct input object is upscaled once: the Batch tab repeats one file
`batch_size` times, and the same input with the same seed gives the same result.

### Same-sized images go through the pipeline in passes of up to 16 MP of output, as frames of one tensor
Why: the models load once per pass, while the DiT is still dropped before the VAE decode (what lets 8 GB GPUs run).
Upscaling image by image would reload the models from disk for each image in "Unload" mode, or keep the DiT on the GPU
during decoding if cached. With batch 1 and no overlap the frames do not affect each other. The cap keeps RAM bounded
for huge batches (a typical batch of 8 at 1080p is one pass); other sizes get their own pass.

### A result has exactly one image and one infotext per input; failures come back as the input
Why: Forge pairs `processed.images` and `processed.infotexts` by index (gallery, saving, "Send to"); a missing image
shifts every later infotext. A not-upscaled image carries `SeedVR2 Error` in its infotext, is not saved, and the result
gets a comment so the failure is visible. The infotext holds only the exception type: the message can contain local
paths or GPU details, and the infotext leaves the machine with the image.

### The infotext is Forge's base infotext plus SeedVR2 keys, not rebuilt with `create_infotext`
Why: `create_infotext` reads the SD model, which "Unload SD Checkpoint" has already unloaded (live: it raised and the
prompt was lost). txt2img extends the infotexts Forge built for the base images; img2img writes the prompt in Forge's layout.

### Saving follows "Always save all generated images" and the API's `save_images`; with them off nothing is written
Why: the user's setting must hold for the upscaled images too. Read from the options (and the request's
`override_settings`), not `p.save_samples()`, whose skipped flag from the last txt2img batch would stop saving.
"Save Extra Copy of Original" only adds the img2img input: txt2img base images are saved by Forge by the same rule
(the owner left this narrower rule to Claude; earlier it also saved txt2img originals with "Always save" off).

### API values of the numeric settings are clamped to the slider ranges
Why: the UI cannot send more than the sliders allow, an API call could (an unbounded resolution means unbounded
memory). One set of bounds (`seedvr2_neo/upscale_settings.py`) builds the sliders and checks the values.

### Interruption checks only `shared.state.interrupted`
Why (*reconstructed from code*): the original author removed the `stopping_generation` check deliberately. Likely reason:
"Stop generating" in Forge means "finish the current image", which should not abort the upscale.

## VRAM and caching

### Unload the SD checkpoint before SeedVR2 by default
Why (*reconstructed from code*): SD and a 7B SeedVR2 do not fit together into 24 GB. Forge reloads the checkpoint on the
next generation.

### SeedVR2 is unloaded after every run by default; "Keep in RAM" is opt-in
Why: the owner's choice. Forge has no hook to ask an extension for its memory back, so whatever SeedVR2 keeps shrinks
what SD can load; with a 16 GB RAM system the cached models (3–4 GB for 3B fp8, up to ~16 GB for 7B fp16) left
almost no free RAM. Keeping models is for repeated
upscales and must be chosen.

### Models are dropped after an error or an interrupt, even with "Keep in RAM"
Why: a failed phase can leave a model on the GPU or half-configured; reusing it is a worse bug than one reload from disk.
The generation context is built fresh every run for the same reason (and it is cheap).

### The cached models are trusted only for the same files (path, size, modification time)
Why: the vendored cache matches by file name; a same-named file in the other model folder or a replaced file would
reuse old weights. Our check lives in `seedvr2_neo/`, the vendored cache stays unpatched.

### BlockSwap is a manual slider with a note for 7B on small GPUs, not an automatic setting
Why: the owner's choice. The right block count depends on the GPU, the model, the resolution and even the PyTorch
allocator (7B fp8 at 1080p on 8 GB: 8 blocks take 21 s with `--cuda-malloc` and 122 s without); a wrong automatic
value either wastes speed or still overflows. The note recommends 16 blocks, the count measured to fit with both
allocators (34–42 s). BlockSwap on also moves the VAE to RAM during the DiT step: when VRAM is short, the idle VAE
should not hold it. The note's threshold (11 GiB) is the 7B fp8 weights (~7.9 GB) plus activations at 1080p.

### VAE tile 512 is the default, on a slider labelled "VAE Tile Size"
Why: with 1024 and `--cuda-malloc` a VAE phase on an 8 GB GPU could run for many minutes; 512/128 is measured clean
and finishes (07-RISKS, R6). Forge keeps a slider's value in `ui-config.json` under its label and prefers it to the code
default, so the old label would have left existing users on 1024; the new label resets the tile size once for
everyone, a value set on purpose included (owner's choice over editing Forge's `ui-config.json` from the extension).
A fixed value, not a VRAM-based one: the same reasoning as BlockSwap; per-GPU presets are a roadmap item.

### No 6K / 8K output sizes until the DiT works in tiles
Why: the DiT sees the whole image; 8K is 33 MP, eight times an output that already takes minutes on an 8 GB GPU, and
the resolution slider would clamp it silently. Offering it would mislead users. Agreed with the owner; DiT tiling is a
separate block, after which the larger sizes are added.

### Fixed pipeline settings: batch 1, LAB color correction, `sdpa`, no `torch.compile`
Why (*reconstructed from code*): the simplest configuration that works without optional packages.

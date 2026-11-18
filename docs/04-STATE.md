# PROJECT STATE — seedvr2-webui-neo-extension

## Where we stopped

**B06 closed** (VRAM/RAM lifecycle) on branch `feature/b06-vram-lifecycle`; **not merged into `main` yet** — waits for
the owner's consent to merge and push (users get `main` on their next "Check for updates").

What B06 changes for users: SeedVR2 is unloaded after every run by default ("After run": Unload / Keep in RAM), and
always after an error or interrupt; Force Reload really reloads; cached models are trusted only for the same files;
the pipeline is imported before any work; Forge's device is used; VRAM is returned between phases; BlockSwap controls
and a 7B note for GPUs below 11 GiB ("start with 16 blocks"); README tips for 8–11 GB GPUs (tile 512, `--cuda-malloc`
trade-off). One local patch in `seedvr2_vendor/optimization/memory_manager.py` (no CPU copy before deleting an
uncached model), listed in `external-apis/seedvr2-upstream.md`.

Live-checked on an 8 GB GPU without bf16: 3B fp8, 3B GGUF (also 1024→2048px), 7B fp8 with BlockSwap 16 — DiT step
780 s → 34–42 s. REQ-3 and REQ-4 done (`01-VISION.md`).

The owner's local Forge has the extension on the test branch `b06-test` (fetched from the local repo, not GitHub).

`seedvr2_vendor/` and the `.pt` embeddings match upstream numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1`, plus the local
patches in `external-apis/seedvr2-upstream.md`. Known problems: `07-RISKS.md` (top: batch upscales only the first
image, VAE decode spills with `--cuda-malloc`, infotext lost).

Tests: 1004 passed, 1 skipped (POSIX-only), 2 xfailed (known bugs, `07-RISKS.md` §1). pytest is not installed in the
system Python; run `uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q`.
CI: none.

## Completion percentage

**65%** (15 of 23) — counted by `plan-drift-checker` on closing B06. 3 blocks open.

## Next steps

1. Merge `feature/b06-vram-lifecycle` into `main` and push — after the owner's consent.
2. B07 — correct results (every image of a batch, infotext, save settings, RGB conversion, empty-result guard).
3. B09 — Forge stubs for testing `run()` + CI; B08 — settings and feedback.

Agreed with the owner: REQ-1…REQ-6, simplified Git Flow, ComfyUI files kept as in upstream (Forge Neo only),
hardware target — 8 GB GPU without bf16, the whole repository under Apache-2.0. The fork is public: `main` is what
users get, pushes only with consent.

## Open questions for the owner

- Merge B06 into `main` and push?

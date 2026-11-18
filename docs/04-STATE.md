# PROJECT STATE — seedvr2-webui-neo-extension

<!-- No longer than 60 lines. Only the current state: what is done but not reflected in the roadmap,
     what is in progress, what is next. Remove what is finished and no longer matters (history is in git). -->

## Where we stopped

**B05 closed** (model discovery) and merged into `main` with the owner's consent; `main` is pushed, users get it on
their next "Check for updates". No open feature branches. No releases or tags.

What B05 changed for users: one model folder — `<WebUI>/models/SeedVR2` (any letter case), else the extension's own
`models/SeedVR2`; the first one with both a DiT and a VAE wins. 🔄 button rescans without a restart; an empty folder
shows the exact path; 3B/7B is read from the file header. A foreign `src` package (another SeedVR2 extension) now
stops the run with a one-line error. Live-checked in Forge Neo on an 8 GB GPU without bf16: 3B GGUF and 3B fp8.

The owner's local Forge still has the extension on the test branch `b05-test` (same code as `main` now).

`src/` and the `.pt` embeddings match upstream numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1`, plus the local patches
listed in `external-apis/seedvr2-upstream.md`. Known problems: `07-RISKS.md` (top: 7B on 8 GB, Force Reload, batch
upscales only the first image, reserved VRAM growth).

Tests: 159 passed, 1 skipped (POSIX-only), 2 xfailed (known bugs, `07-RISKS.md` §1). pytest is not installed in the
system Python; run `uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q`.
CI: none.

## Completion percentage

**39%** (9 of 23) — counted by `plan-drift-checker` on closing B05. 5 blocks open.

## Next steps

1. B04 — repo hygiene (`git rm -r --cached` for `.pyc`, dead code, LICENSE with change notice, rename the `src` package).
2. B06 — VRAM/RAM lifecycle and BlockSwap for 7B on 8 GB GPUs.

Agreed with the owner: REQ-1…REQ-6, simplified Git Flow, ComfyUI files kept untouched (Forge Neo only),
hardware target — 8 GB GPU without bf16. The fork is public: `main` is what users get, pushes only with consent.

## Open questions for the owner

None.

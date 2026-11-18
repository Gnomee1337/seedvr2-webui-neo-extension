# PROJECT STATE — seedvr2-webui-neo-extension

<!-- No longer than 60 lines. Only the current state: what is done but not reflected in the roadmap,
     what is in progress, what is next. Remove what is finished and no longer matters (history is in git). -->

## Where we stopped

**B05 closed** (model discovery) on `feature/b05-model-discovery`, pushed to `origin`. **Not merged into `main`**:
the merge (and the push of `main`, which is also 2 docs commits ahead of `origin/main`) waits for the owner's consent,
because `main` is what users install. No releases or tags.

What B05 changed for users: one model folder — `<WebUI>/models/SeedVR2` (any letter case), else the extension's own
`models/SeedVR2`; the first one with both a DiT and a VAE wins. 🔄 button rescans without a restart; an empty folder
shows the exact path; 3B/7B is read from the file header. A foreign `src` package (another SeedVR2 extension) now
stops the run with a one-line error. Live-checked in Forge Neo on an 8 GB GPU without bf16: 3B GGUF and 3B fp8.

The owner's local Forge still has the extension switched to the test branch `b05-test`; switch back after the merge.

`src/` and the `.pt` embeddings match upstream numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1`, plus the local patches
listed in `external-apis/seedvr2-upstream.md`. Known problems: `07-RISKS.md` (top: 7B on 8 GB, Force Reload, batch
upscales only the first image, reserved VRAM growth).

Tests: 159 passed, 1 skipped (POSIX-only), 2 xfailed (known bugs, `07-RISKS.md` §1). pytest is not installed in the
system Python; run `uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q`.
CI: none.

## Completion percentage

**39%** (9 of 23) — counted by `plan-drift-checker` on closing B05. 5 blocks open.

## Next steps

1. Owner: consent to merge `feature/b05-model-discovery` into `main` and push (or not yet).
2. B04 — repo hygiene (`git rm -r --cached` for `.pyc`, dead code, LICENSE with change notice, rename the `src` package).
3. B06 — VRAM/RAM lifecycle and BlockSwap for 7B on 8 GB GPUs.

Agreed with the owner: REQ-1…REQ-6, simplified Git Flow, ComfyUI files kept untouched (Forge Neo only),
hardware target — 8 GB GPU without bf16. The fork is public: `main` is what users get, pushes only with consent.

## Open questions for the owner

- Merge B05 into `main` and push now?
- REQ-1 ("clean install from URL, `.safetensors` and `.gguf` without manual pip") is still ⬜: mark it done after a
  fresh-install check, or now?

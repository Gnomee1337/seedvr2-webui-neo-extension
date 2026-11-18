# PROJECT STATE — seedvr2-webui-neo-extension

<!-- No longer than 60 lines. Only the current state: what is done but not reflected in the roadmap,
     what is in progress, what is next. Remove what is finished and no longer matters (history is in git). -->

## Where we stopped

**B04 closed** (repo hygiene) on branch `feature/b04-repo-hygiene`. **Not merged into `main` and not pushed** — waiting
for the owner's consent. `main` = B05, already pushed. No releases or tags.

What B04 changes for users: the vendored package is `seedvr2_vendor` (was `src`), so the extension no longer clashes
with another SeedVR2 extension's `src`; the foreign-`src` check is gone. Apache-2.0 `LICENSE` and `NOTICE` added;
tracked `.pyc` removed (Forge's own update handles that; a manual `git pull` may need `git reset --hard`, see README).
Live-checked in Forge Neo on an 8 GB GPU without bf16 on this branch: 3B fp8 (GGUF not re-run; its loading code is unchanged).

The owner's local Forge has the extension on the test branch `b04-test` (fetched from the local repo, not GitHub).

`seedvr2_vendor/` (upstream `src/`) and the `.pt` embeddings match upstream numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1`,
plus the local patches listed in `external-apis/seedvr2-upstream.md`. Known problems: `07-RISKS.md` (top: 7B on 8 GB,
Force Reload, batch upscales only the first image, reserved VRAM growth).

Tests: 274 passed, 1 skipped (POSIX-only), 2 xfailed (known bugs, `07-RISKS.md` §1). pytest is not installed in the
system Python; run `uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q`.
CI: none.

## Completion percentage

**47%** (11 of 23) — counted by `plan-drift-checker` on closing B04. 4 blocks open.

## Next steps

1. Owner's consent → merge `feature/b04-repo-hygiene` into `main` (`--no-ff`) and push.
2. B06 — VRAM/RAM lifecycle and BlockSwap for 7B on 8 GB GPUs (also: pipeline import before any work).

Agreed with the owner: REQ-1…REQ-6, simplified Git Flow, ComfyUI files kept as in upstream (Forge Neo only),
hardware target — 8 GB GPU without bf16, the whole repository under Apache-2.0. The fork is public: `main` is what
users get, pushes only with consent.

## Open questions for the owner

1. Merge B04 into `main` and push (users get the rename on their next update)?

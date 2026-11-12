# PROJECT STATE — seedvr2-webui-neo-extension

<!-- No longer than 60 lines. Only the current state: what is done but not reflected in the roadmap,
     what is in progress, what is next. Remove what is finished and no longer matters (history is in git). -->

## Where we stopped

**Branches.** `main` (fork of the original Chinese repository) holds the move onto the protocol and the live-test
results; local `main` is 2 commits ahead of `origin/main` — **not pushed**, waiting for the owner's consent.
Current work: `feature/b05-model-discovery`. No releases or tags.

The project runs on the protocol: English docs in `docs/`, README and the script's comments translated,
`.gitignore` rewritten in UTF-8. Code behaviour is unchanged. Known problems are in `07-RISKS.md`,
split into blocks B04–B09 in `03-ROADMAP.md`.

`src/` and the `.pt` embeddings match upstream numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1` (no local patches).
What to borrow from wdhameli0/sd-webui-forge-neo-seedvr2 is in `external-apis/seedvr2-upstream.md`.

**Live test done (B02 closed)** on an 8 GB GPU without bf16, current Forge Neo, txt2img: 3B GGUF Q6_K, 3B fp8 and
7B fp8 all upscale cleanly. `gguf` works out of the box (Neo's bundled copy). 7B on 8 GB takes ~13 min per image
(VRAM spill); reserved VRAM grows between phases; models stay in system RAM after the run — details in `07-RISKS.md` §3.
VRAM is back to ~1 GB after a run.

Tests: none. CI: none.

## Completion percentage

**30%** (7 of 23) — see `03-ROADMAP.md`. 5 blocks open. Not yet recounted by `plan-drift-checker`; confirm with `/progress`.

## Next steps

1. B05 — model discovery: one folder resolved in one place, refresh button, empty-folder message, size not by file name.
2. B04 — repo hygiene (`git rm -r --cached` for `.pyc`, dead code, LICENSE).
3. B06 — VRAM/RAM lifecycle and BlockSwap for 7B on 8 GB GPUs.

Agreed with the owner: REQ-1…REQ-6, simplified Git Flow, ComfyUI files kept untouched (Forge Neo only),
hardware target — 8 GB GPU without bf16. The fork is public: `main` is what users get, pushes only with consent.

## Open questions for the owner

None.

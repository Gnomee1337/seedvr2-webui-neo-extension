# PROJECT STATE — seedvr2-webui-neo-extension

<!-- No longer than 60 lines. Only the current state: what is done but not reflected in the roadmap,
     what is in progress, what is next. Remove what is finished and no longer matters (history is in git). -->

## Where we stopped

**Branches.** Only `main` (fork of the original Chinese repository). No releases or tags.

The project was moved onto the protocol: English docs in `docs/`, README and the script's comments translated,
`.gitignore` rewritten in UTF-8. Code behaviour is unchanged. The audit findings are in `07-RISKS.md`
and split into blocks B04–B09 in `03-ROADMAP.md`.

`src/` and the `.pt` embeddings were checked against upstream: identical to numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1`
(no local patches). The related project wdhameli0/sd-webui-forge-neo-seedvr2 was reviewed (read only) — what to borrow is in
`external-apis/seedvr2-upstream.md`.

Not committed yet: the migration changes (waiting for the owner's review).

Tests: none. CI: none.

## Completion percentage

**17%** (4 of 23) — see `03-ROADMAP.md`. 6 blocks open. Initial estimate; confirm with `/progress`.

## Next steps

1. Owner: run the extension on the current Forge Neo once (txt2img, a `.gguf` and a `.safetensors` model) — closes B02 or adds bugs.
2. B05 — install and model discovery (`gguf`, one model folder, refresh button).
3. B04 — repo hygiene (`git rm -r --cached` for `.pyc`, dead code, LICENSE).

Agreed with the owner: REQ-1…REQ-6, simplified Git Flow, ComfyUI files kept untouched (Forge Neo only),
hardware target — 8 GB GPU without bf16. The fork is public: `main` is what users get, pushes only with consent.

## Open questions for the owner

None.

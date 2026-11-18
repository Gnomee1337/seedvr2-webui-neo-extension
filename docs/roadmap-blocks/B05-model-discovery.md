# B05 — Install and model discovery

**Status:** 🔄 in progress
**Weight:** 2 (from `docs/03-ROADMAP.md`)
**Decisions this block rests on:** "Fix in `scripts/` first, patch `src/` only when unavoidable";
"SeedVR2 code is vendored in `src/`, not installed as a package"; "Tests mirror the code folders".

## Why this block
A user who puts models in a slightly different place gets empty dropdowns and no hint why — the most common
first-run failure (R12). After the block: one folder, shown to the user, found on any OS, refreshable without a
restart, and a renamed 7B file still loads the 7B config (R6, R4 in `07-RISKS.md` §1).

## What we do (steps)
1. **One folder, resolved in one place** (`seedvr2_neo/model_discovery.py`, no Forge imports): the absolute
   model folder is `<Forge models_path>/SeedVR2` if it holds models, else the extension's `models/SeedVR2`; the folder
   name is matched case-insensitively so Linux behaves like Windows. Resolved again on every refresh and every run.
2. **The pipeline loads from that folder.** Local patch in `src/utils/constants.py::find_model_file`: when the caller
   passes a folder and the file exists there, that path wins over the cwd-relative `./models/SEEDVR2` scan.
   The run log prints the full path of the DiT and VAE files.
3. **Empty-folder message.** If no DiT or no VAE is found, the UI shows a visible note with the exact folder path
   to put files in. `run()` with an empty selection returns a clear error instead of a traceback.
4. **Refresh button** next to the dropdowns: rescans the folder, updates both dropdowns and the note, keeps the
   current selection if the file is still there.
5. **Model size from the file, not the name.** Read only the header (safetensors JSON header / GGUF tensor list),
   count transformer blocks (`blocks.N.*`): 32 → 3B, 36 → 7B; unknown → fall back to `"7b" in name.lower()`.
   Local patch at the config choice in `src/core/model_configuration.py` (`create_new_runner`).
6. **Tests** in `tests/` mirroring the code folders (`pytest`, no Forge or GPU needed): folder resolution, listing,
   size detection, hostile cases. Both `src/` patches listed in `docs/external-apis/seedvr2-upstream.md`.
7. README "Models" section and `09-PROCESSES.md` updated to the single rule.

## Acceptance criteria
- [ ] Models only in `<WebUI>/models/SeedVR2` → dropdowns filled; models only in `<extension>/models/SeedVR2` →
      dropdowns filled. The run log shows the full path of the loaded DiT/VAE, inside the folder the UI shows.
- [ ] A test shows a folder named `seedvr2` or `SEEDVR2` is found on a case-sensitive file system.
- [ ] Empty or missing folder → the UI shows the expected path; pressing Generate gives a one-line error, no traceback.
- [ ] Live: a model file copied into the folder appears after pressing Refresh, without restarting the WebUI.
- [ ] A test: a 7B header under a name without `7b` is detected as 7B; a 3B header named `...7b...` as 3B;
      a truncated / zero-byte / non-model file gives "unknown", never an exception.
- [ ] `pytest` passes; both `src/` patches are listed in "Local patches" of `seedvr2-upstream.md`.
- [ ] Owner's live run: 3B GGUF and 3B fp8 still upscale as in B02.

## Which checks are needed on `/block-done`
Always: `code-auditor`, `fresh-eyes-reviewer`, `plan-drift-checker`. Additionally (tick what applies):
- [x] `integration-checker` — `src/` patches, the block must be runnable inside Forge
- [ ] `security-auditor` — reads only local files the user put there; header length is bounded
- [x] `stress-tester` — file input: empty, truncated, huge header length, wrong extension, odd folder names

## Dependencies
None. Live check needs the owner's Forge Neo with at least one 3B model.

## Out of scope for this block
- Downloading models from the UI (upstream `download_weight`): not added.
- `extra_model_paths` / several model folders; Forge `--models-dir` beyond what `models_path` already reflects.
- 7B warning on small GPUs, BlockSwap, real Force Reload — B06.
- Renaming the `src` package, removing the unused `download_weight` import — B04.
- CI — B09 (this block only adds the first local tests).

## Hidden assumptions
- Forge Neo (`external-apis/forge-neo.md`): `shared.models_path` defaults to `<WebUI root>/models`, is moved by
  `--model-ref` / `--data-dir` and may lie outside the WebUI root — so the vendored cwd-relative `./models/SEEDVR2`
  is wrong for those users and for anyone who starts the launcher from another folder (no `chdir` anywhere).
- Refresh: `modules.ui_common.create_refresh_button(component, refresh_method, refreshed_args, elem_id)` with
  `refreshed_args` a callable returning `{"choices": ...}`; Gradio 4.40 → handlers return `gr.update(...)`.
  The built-in helper updates choices only; to also update the note and keep the selection, a plain `ToolButton`
  with our own handler may be simpler — decide while coding.
- Neo has no built-in SeedVR2 and no reserved `models/SeedVR2` folder — no clash.
- The dev code is not what the owner's Forge loads by default; the live check needs the extension pointed at this branch.

- DiT tensor names start with `blocks.<N>.`: confirmed on the official 3B fp16, 3B fp8 and 7B sharp fp8 safetensors
  (32 / 32 / 36 blocks). GGUF files not checked on real files (only on generated test files).

## Progress
Steps 1–7 done in code and docs; `python -m pytest tests` — 46 passed; the `find_model_file` test fails without the patch.
Left: the owner's live run (criteria 1, 3, 4, 7), then `/block-done`.

## Open questions
None. (Answered: when both folders have models, Forge's folder wins — `05-DECISIONS.md`.)

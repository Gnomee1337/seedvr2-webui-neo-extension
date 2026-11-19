# B09 — Tests with Forge stubs + GitHub Actions (pytest)

**Status:** 🔄 in progress
**Weight:** 3 (from `docs/03-ROADMAP.md`)
**Decisions this block rests on:** Tests mirror the code folders; `pytest.ini` lives in `tests/`; Branches: simplified
Git Flow; txt2img calls `process_images` from inside `run()`; A result has exactly one image and one infotext per input;
Saving follows "Always save…"; Interruption checks only `shared.state.interrupted`; Models are dropped after an error or
an interrupt.

## Why this block
`scripts/seedvr2_script.py` (~530 lines) is the only Forge entry point and has no tests: every change to `run()` is
checked only by a live run. After this block the run flow is covered by tests that run on every push (REQ-6), and the
file is split so each part fits the size guideline (risks R2, R3).

## What we do (steps)
1. Forge stubs in `tests/`: minimal fake `modules.*` and `gradio` packages whose names and signatures match Forge Neo
   (from `docs/external-apis/forge-neo.md`), installed into `sys.modules` by a fixture; a fake pipeline namespace
   instead of `import_pipeline()`.
2. Tests of the current `run()` flow before any refactoring (safety net): txt2img batch, img2img, Batch-tab repeats,
   saving on/off, error → inputs returned with `SeedVR2 Error`, interrupt, unload SD / Keep in RAM paths, empty input.
3. Split `scripts/seedvr2_script.py` without changing behavior: Forge-free parts move to `seedvr2_neo/`, the script keeps
   UI + Forge glue; the safety-net tests stay green unchanged.
4. `.github/workflows/tests.yml`: on push and pull request — pytest on Python 3.10 with CPU torch. No linter, no
   branch protection (owner's choice: minimal CI).
5. Docs: architecture (new modules, stubs), STATE, RISKS R2/R3, decisions (minimal CI, stub approach).

## Acceptance criteria
- [ ] `python -m pytest tests` runs `Script.run()` for txt2img and img2img through the stubs, without Forge installed.
- [ ] The listed run scenarios (step 2) each have a test; the same tests pass before and after the split.
- [ ] `scripts/seedvr2_script.py` ≤ ~300 lines; no file in `seedvr2_neo/` imports Forge or Gradio.
- [ ] The GitHub Actions run on the block branch is green (link to the run).
- [ ] Full test run green; live smoke run in Forge (one txt2img + one img2img through the API) unchanged after the split.

## Which checks are needed on `/block-done`
Always: `code-auditor`, `fresh-eyes-reviewer`, `plan-drift-checker`. Additionally (tick what applies):
- [x] `integration-checker` — the script is split into modules; Forge must still load it
- [ ] `security-auditor` — user data, authorization, external requests, files, money
- [ ] `stress-tester` — no new user input

## Dependencies
B07 merged (done). Pushing the block branch to GitHub so Actions can run — needs the owner's consent (a branch, not `main`).

## Out of scope for this block
- B08 features (progress bar, interrupt inside the DiT step, title, settings).
- Tests of `seedvr2_vendor/` beyond what exists; running the real pipeline or a GPU in CI.
- Branch protection rules on GitHub (owner's setting; can be suggested after CI is green).
- Any behavior change of the script — only a refactor under tests.
- A linter (`ruff`) in CI — the owner chose minimal CI.

## Hidden assumptions
- The stubs only stay honest if their signatures follow Forge Neo: the names, signatures and `p.*` fields the script
  uses are in `docs/external-apis/forge-neo.md` ("Names `scripts/seedvr2_script.py` uses"). Pitfalls from there:
  stub components must accept `tooltip=` (Neo patches Gradio for it); `p.scripts` is a property over `scripts_value`;
  `sd_models.unload_model_weights` is reached both through `sys.modules` and the `modules` package attribute;
  `images.save_image` returns `(fullfn, txt_fullfn)` and sets `image.already_saved_as`; a simple stub `Processed`
  (the real one reads ~25 `p.*` fields). `FormRow` is imported but unused.
- CI facts (`docs/external-apis/github-actions-ruff.md`): `actions/checkout@v7`, `actions/setup-python@v7`,
  CPU torch from `https://download.pytorch.org/whl/cpu`. Not verified: a cp310 Linux CPU wheel for the current torch
  (else pin an older torch or run CI on a newer Python).
- Existing tests need no torch; importing the script does (torch, torchvision at module level). `numpy` comes only
  through `gguf` — list it in `requirements-dev.txt`.

## Open questions
None. Agreed: minimal CI (pytest only), real CPU torch in CI, the block branch is pushed to GitHub for the Actions run.

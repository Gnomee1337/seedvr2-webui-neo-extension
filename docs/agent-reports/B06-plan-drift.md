# B06 plan-drift report

## Percentage (assuming B06 becomes done)
Done blocks: B01 3 + B02 3 + B03 1 + B04 2 + B05 2 + B06 4 = **15**.
All blocks: B01 3 + B02 3 + B03 1 + B04 2 + B05 2 + B06 4 + B07 3 + B08 2 + B09 3 = **23** (matches the roadmap table).
15 / 23 = 65.2% -> **65%** (was 47%, 11/23).

## Can B06 be marked done?
Yes, with the notes below. Tests: 329 passed, 1 skipped, 2 xfailed (run just now). All 12 steps are implemented; live results 1-8 are in the card.

Acceptance criteria and proof:
1. Pipeline import before base image / SD unload: `scripts/seedvr2_script.py:195` (before the txt2img base image at :209). Proof is code review only. `run()` has no test (B09), and no simulated broken import was run. Weakest criterion; the card allows "a test or a console log" and neither exists.
2. No `cuda:0` / "0" literal: grep finds none in the script. Only `devices.device` at :73 and :76. Met.
3. Unload frees memory: live run 1 shows the log line, GPU back to ~0.6 GB and RAM did not grow. The card does not record the pre-run and post-run RAM numbers for the "± 1 GB" check. Partial; add the numbers or accept as observed.
4. Keep in RAM, second run uses the cache: live 2 ("Reusing cached runner / DiT / VAE"). Met.
5. Force Reload: live 3. Met.
6. Same-name file in the other folder is picked up: covered by `tests/seedvr2_neo/test_loaded_models.py` and `test_model_lifecycle.py` (identity = path, size, mtime). Met.
7. Fresh context after an exception and live interrupt: `test_model_lifecycle.py` and live 4. Met.
8. 7B fp8 with BlockSwap at least 3x faster than 780 s: 34.5 s with 16 blocks, 21 s with 8. Met. The criterion says the block count is "recorded in `07-RISKS.md`"; check that it is there (see gap 7).
9. 7B warning shown, 3B not: live 5 confirms the 7B warning. The result line does not say that the warning disappears for 3B. Pure-helper tests may cover it; confirm.
10. 3B fp8 and GGUF clean on default settings: 3B fp8 defaults clean (live 1). 3B GGUF was checked only with BlockSwap 16 (live 7), not on defaults. Minor gap.
11. All tests pass, helpers covered: met.

## Gaps (docs vs code)
1. `docs/03-ROADMAP.md:33` the B06 status is still "not started" with "all" remaining; flip it and clear "What remains". `:25` Summary says 47%, "Closed: 11" -> 65%, "Closed: 15", counted on closing B06. `:54` the closed sum must read ... + B06 4 = **15**.
2. `docs/04-STATE.md:8-33` still describes B04 as the last closed block; "Next steps" starts with B06 (:32). Rewrite for B06 closed, next B07; percentage at :28; keep it to 60 lines (currently 41).
3. `docs/roadmap-blocks/B06-vram-ram-lifecycle.md`: all acceptance checkboxes are unticked (:39-49) and status is "awaiting live check" (:3). On close the card must be deleted. Move the live-check results (:96-110) into 07-RISKS / external-apis first, then delete. `docs/agent-reports/B06-code-audit.md` and `B06-fresh-eyes.md` are to be deleted after their conclusions are moved.
4. `docs/02-ARCHITECTURE.md:129`: the dead-code row about `finally` keys `images`, `encoded_latents`, `upscaled_latents` ("fix in B06") is stale. The script no longer has those keys (only `ctx['final_video']` at `scripts/seedvr2_script.py:399`). Remove the row. The new modules (`memory_policy`-style helpers `memory_plan.py`, `loaded_models.py`, `model_lifecycle.py`) are described at :74-78, which is fine. Check that the new UI controls (After run, Blocks to swap, I/O layers) are listed among the entry-point inputs.
5. `docs/01-VISION.md:35-36`: REQ-3 and REQ-4 are still marked not done. Update the status after the close (REQ-4's 8 GB 3B GGUF and 7B with BlockSwap are live-checked).
6. `docs/external-apis/seedvr2-upstream.md:29` documents the `memory_manager.py` local patch. This matches the diff (one vendored file changed). OK.
7. Check that `docs/07-RISKS.md` records the final BlockSwap count (16), the `--cuda-malloc` VAE-decode spill (live 8) and the open risk of 4 blocks spilling (297 s), as the card requires. The README and 07-RISKS were updated in the last commits, so only verify.
8. Work outside the plan: none found. The `memory_manager.py` patch is covered by the card (step 7 allowed a vendored patch).

No plan-drift against the 05-DECISIONS entries named in the card was found beyond the checks above; `09-PROCESSES.md` was updated in this branch (43 lines changed).

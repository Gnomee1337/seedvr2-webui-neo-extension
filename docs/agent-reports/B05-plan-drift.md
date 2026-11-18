# B05 plan drift report

## Percentage
Sum of all weights (Delivery + UX + Foundation tables): B04 2 + B05 2 + B06 4 + B07 3 + B09 3 + B08 2 + B01 3 + B02 3 + B03 1 = 23.

- Now (B05 still 🔄): closed = B01 3 + B02 3 + B03 1 = 7. 7/23 = 30.43 -> **30%**. Matches roadmap Summary and `04-STATE.md`.
- After B05 gets ✅: closed = 3 + 3 + 1 + 2 = 9. 9/23 = 39.13 -> **39%**. Sum of weights stays 23.

## Acceptance criteria of B05 (all boxes in the card are still `[ ]`; tick them on close)
1. Both folders -> dropdowns: Forge folder live-checked; extension folder NOT live-checked, covered by tests only (card says so). Log shows full path: live.
2. Case-insensitive folder name: `tests/seedvr2_neo/test_model_discovery.py` (46 pytest pass).
3. Empty folder -> note + one-line error: live (card "Live run results").
4. Refresh without restart: live.
5. Header detection incl. hostile files: `tests/src/utils/test_dit_size.py`, `test_dit_size_stress.py`.
6. pytest passes; both `src/` patches listed in `seedvr2-upstream.md`: yes. Note: upstream doc lists THREE entries (constants, new dit_size.py, model_configuration); the card says "both patches". Fine, but wording in card is stale (it is deleted anyway).
7. Live 3B GGUF and 3B fp8 still upscale: live.
Reviews (code-auditor, fresh-eyes, integration-checker, stress-tester) are not evidenced by any report yet: the percentage counts only after they pass (protocol §5).

## Gaps
1. `docs/03-ROADMAP.md:32` B05 row sits in the Delivery table as ⬜ "all". Mark ✅ (move to Foundation, like B01-B03, or keep with status), update Summary line 25 to "Sum 23. Closed 9. PROGRESS 39%" and "How to check the sum" line 54 (B01 3 + B02 3 + B03 1 + B05 2 = 9). Remove "Not yet recounted" note once this report is accepted.
2. `docs/04-STATE.md:7-9` says waiting for the owner's live run; live run is done. `:23` "Tests ... (B05 branch)". `:27` percentage 30% -> 39% (9 of 23), "5 blocks open" (B04, B06, B07, B08, B09) -- the current text already says 5, which was wrong while B05 was open (6 open). `:30-34` Next steps still list B05 as step 1; drop it, B04 becomes first. File is 43 lines (OK, <60).
3. `docs/07-RISKS.md:37` R12, `:43` "No refresh for the model list (B05)", `:44` "Model config chosen by file name" and Top risks `:103` (#1 R12 "code still inconsistent") are fixed by B05: delete or rewrite as resolved (docs keep only the current state). Check `:36-45` for the residual risk "extension-folder fallback not live-checked" (MINOR) if you want to keep it.
4. `docs/external-apis/seedvr2-upstream.md:39` says "GGUF files not checked yet"; the live run detected the real 3B Q6_K GGUF as 3B from the header. Update the pitfall (the card's "Hidden assumptions" last bullet is the source). Also lines 15-21 should say "three" local patches, consistent with the card/architecture.
5. `docs/roadmap-blocks/B05-model-discovery.md` must be deleted on close (protocol §11). What to carry over before deleting: header-based size fact (done, item 4); Gradio refresh/`create_refresh_button` assumption (check it is in `forge-neo.md`, +41 lines in the diff suggest yes); `shared.models_path` moved by `--model-ref`/`--data-dir` (same); "extension folder fallback not live-checked" -> `07-RISKS.md` MINOR; "dev code is not what the owner's Forge loads" is a process note, not for permanent docs. Decisions (`05-DECISIONS.md:59,64`) already exist, nothing to add. Out-of-scope items (downloads, `extra_model_paths`) -> `06-IDEAS.md` if not already there.
6. `docs/01-VISION.md:33` REQ-1 status ⬜. B05 is the only block mapped to REQ-1 in the roadmap, and its live run proved .gguf and .safetensors discovery; the owner should decide whether REQ-1 becomes ✅/🔄 (the clean-install-from-URL part was never separately checked).
7. `docs/08-GLOSSARY.md`: no entry for "DiT", "VAE", "checkpoint header". Terms now appear in user-facing docs/UI; add (MINOR).
8. Architecture `02-ARCHITECTURE.md`: consistent with the diff (`seedvr2_neo/model_discovery.py`, `src/utils/dit_size.py`, patches in constants/model_configuration, `tests/pytest.ini`, `requirements-dev.txt`, refresh button). No dead or missing items found. Not covered by tests: `scripts/seedvr2_script.py` refresh/run guard (known, B09).

Work outside the plan: none found; every file in `git diff 4dc896e --stat` maps to steps 1-7 (CLAUDE.md/README edits are step 7).

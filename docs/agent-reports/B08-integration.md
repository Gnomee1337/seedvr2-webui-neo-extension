# B08 — integration-checker report (relayed by the main chat; the agent could not write files)

No BLOCKER. The parts fit: `ui()` returns 21 controls in `run()`'s order (the GPU profile not among them); 13/16/19-
argument API calls get `lab` / `sdpa`; the new `seedvr2_neo` modules have no Forge/Gradio imports; keywords match the
vendored signatures; the six color corrections and the four `*_AVAILABLE` flags exist; hook targets (`runner.dit`,
`CompatibleDiT.dit_model`, `NaDiT.blocks`, `runner.vae`) exist and blocks are called through `__call__`; Gradio 4.40
APIs and `shared.state` fields are used as `forge-neo.md` describes; the stubs match.

- IMPORTANT 🟡 R6 (S2×L3): the hook list kept the dropped DiT alive through decode and post-processing (with Unload +
  BlockSwap: the RAM blocks and I/O layers). Fixed in the working tree (`modules = None` before `yield`, no local at
  the call site; `_GeneratorContextManager.__enter__` drops its args), test
  `test_a_dit_dropped_after_its_phase_is_freed_before_the_decode`.
- MINOR M1: img2img Batch never advanced `job_no` — fixed by `finish_forge_progress`; leftover: after "Stop
  generating" mid-txt2img-batch the upscale counts as Forge's job (cosmetic).
- MINOR M2: NaN progress → `int(round(nan))` fails the run (not reachable today).
- MINOR M3: `seedvr2-upstream.md` named `FP8CompatibleDiT`; the class is `CompatibleDiT`.
- MINOR M4: an ERROR line from the vendored phases before "Process Interrupted by User".
- MINOR M5: a shifted API string "None" turns color correction off.
- MINOR M6: the card's test count is out of date.

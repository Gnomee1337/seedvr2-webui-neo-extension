# B07 — integration-checker report

(Recorded by the main chat from the agent's returned report; the agent did not write files. Scope: `main...efcf507`.)

No BLOCKER, no IMPORTANT. Boundaries hold:
- `save_image(..., suffix, save_to_dirs=None)` matches Neo's signature; `save_to_dirs` override passing is needed and
  correct (owner's Batch re-run confirms).
- Forge attributes exist: `save_samples(*, is_video)`, `do_not_save_samples`, `override_settings`, `outpath_samples`,
  `all_seeds` / `all_prompts` (None in img2img), `Processed(..., infotexts=)`, `index_of_first_image`.
- Pipeline: N frames, batch 1, overlap 0 → N frames in order (one write position per frame in `final_video`); RGBA per frame.
- Lifecycle across groups: Keep in RAM reuses cached models; Unload reloads per group (in `07-RISKS`).
- Batch tab: `[img] * batch_size` per file; results extended by `proc.images` / `proc.infotexts`.

Minor findings:
1. 🟡 R4 Results not saved to disk carry no embedded `parameters` (`image.info`) — gallery download / API PNG lack metadata.
2. 🟢 R2 A batched image does not reproduce exactly from its "SeedVR2 Seed" (encode sample drawn from one generator
   per pass, `generation_phases.py:329-330`, `infer.py:178`). Document, no vendor patch.
3. 🟢 R2 Only `save_to_dirs` and `samples_filename_pattern` overrides reach the extension's saves; `samples_format`
   and others do not.
4. 🟢 R2 Only `OSError`/`ValueError` caught around `save_image`; another extension's save callback error loses the result.

New facts for `forge-neo.md`: `save_image` reads `p.override_settings["samples_filename_pattern"]` itself; gallery
temp files and API base64 embed `image.info`.

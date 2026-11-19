# code-auditor — B10 (VAE tile 512 as the default)

Scope: `git diff main...feature/b10-vae-tile-512` (commits 020e374, 39ddd52). During the review the working tree gained
uncommitted changes (overlap cap at half the tile, stubs and `make_request` moved to tile 512, forge-neo key example);
they are covered where they touch a finding and marked "working tree".

Test runs: committed HEAD — 1668 passed, 1 skipped, 3 xfailed; working tree — 1673 passed, 1 skipped, 3 xfailed.

Summary: BLOCKER 0, IMPORTANT 2, MINOR 5.

## IMPORTANT

### I1. 🟡 R6 (S3×L2) Tile 512 plus a saved or API overlap of up to 512 gives a VAE stride of one latent pixel
- `seedvr2_neo/upscale_settings.py:29-30` (committed), `scripts/seedvr2_script.py:199-200`
- What is wrong: the tile default drops to 512, but the "Tile Overlap" slider still allows 64–512 and keeps its label,
  so its saved `ui-config.json` value carries over. The vendored VAE (`attn_video_vae.py` `tiled_encode`/`tiled_decode`)
  clamps the latent overlap to `latent_tile - 1` and steps by `max(1, tile - overlap)`.
- Scenario: a user had tile 1024 / overlap 512 (stride 512, fine). After the update the tile resets to 512 and the
  overlap stays 512 → latent tile 64, overlap 63, stride 1. A 1920×1080 output (latent 240×135) decodes in
  (240-63)×(135-63) ≈ 12 700 tiles instead of ~8: the phase looks frozen for tens of minutes, the same symptom the block
  set out to remove. A new user or an API caller reaches it by moving only the overlap slider to its maximum;
  overlap 448 already gives ~230 tiles.
- Fix: cap the overlap at half the tile in `parse_upscale_settings` (the working tree does exactly this, with tests);
  print one console line when the cap applies (see M4).

### I2. 🟡 R4 (S2×L2) No test checks that a non-default tile size reaches the pipeline
- `tests/scripts/test_seedvr2_script.py:374-378`, `tests/seedvr2_neo/test_upscale_settings.py:7-15`;
  working tree also `tests/scripts/forge_stubs.py:461`, `tests/seedvr2_neo/test_upscale_run.py:84`
- What is wrong: the only assertion on `encode_tile_size` expects `(512, 512)`, which is now also the default and the
  minimum; `decode_tile_size` is not asserted at all. `test_ui_values_pass_unchanged` now passes 512 (= default = minimum),
  so it no longer shows an in-range tile size passing through. The working tree changes the stubs' `run_args` and
  `make_request` from 1024 to 512, so no test anywhere sends a non-512 tile size through the run.
- Proof (mutation, on a scratch copy of HEAD): `seedvr2_neo/upscale_run.py:190,193` changed to
  `encode_tile_size=(512, 512)` / `decode_tile_size=(512, 512)`, which ignores the user's slider. Result: 1668 passed, the same as unmutated.
  Before B10 the mutation "use `TILE_SIZE.default` instead of the setting" was caught (expected 512, default 1024);
  now it is not.
- Scenario: a refactor that wires the default instead of `settings.tile_size` ships green; users with 24 GB who set
  1024/2048 silently get 512.
- Fix: in `tests/scripts/` assert that `tile_size=1024, tile_overlap=256` arrive in both `encode_*` and `decode_*`
  runner kwargs; use 1024 (not the default) in `test_ui_values_pass_unchanged`.

## MINOR

### M1. 🟢 R3 (S1×L3) "512 keeps VAE encoding and decoding inside 8 GB of VRAM" contradicts R6
- `seedvr2_neo/upscale_settings.py:27`, `README.md:74`, `tests/scripts/test_seedvr2_script.py:67-68`,
  `tests/seedvr2_neo/test_upscale_settings.py:41`
- What is wrong: `07-RISKS.md` R6 (section 3) records that with tile 512 and `--cuda-malloc` the decode reserves 8.7 GB,
  pages 0.7–2.9 GB to shared memory and takes 68–77 s at 1024 px. The README's next bullet says the same.
- Scenario: a user with `--cuda-malloc` sees shared-memory spill at the default 512 and is told by the first bullet
  that it cannot happen.
- Fix: state what was measured: "512 finishes on 8 GB GPUs; 1024 with `--cuda-malloc` could stall for many minutes".

### M2. 🟢 R2 (S1×L2) The decision refers to a risk by its rating
- `docs/05-DECISIONS.md:174` — "(07-RISKS, R6)". R6 is a rating, not an id: two entries carry it (sections 3 and 4),
  and the number changes when the risk is re-rated, so the reference goes stale.
- Fix: name the risk ("`--cuda-malloc` VAE decoding spills", `07-RISKS.md` section 3).

### M3. 🟡 R4 (S1×L4) State documents still describe R9
- `docs/04-STATE.md:14,19` name 🟠 R9 as the top risk, but it is no longer in `07-RISKS.md`. `docs/03-ROADMAP.md:37` shows B10 ⬜.
  The card's last acceptance box is unticked, but the full run is green. Expected to be fixed at `/block-done`. Listed so it is
  not missed.

### M4. 🟢 R2 (S1×L2) Working tree: the overlap cap is silent
- `seedvr2_neo/upscale_settings.py:66-68` (uncommitted)
- What is wrong: the UI still shows "Tile Overlap" up to 512 with tile 512. The run uses 256 and prints nothing. The vendored
  debug log shows the capped value only with "Show Debug Logs".
- Scenario: a user sets 512/512, then compares runs with 384 and 512 and sees identical results with no explanation.
- Fix: one console line when the cap changes the value, or a slider `info` text "at most half the tile size".

### M5. 🟢 R2 (S1×L2) README upgrade note has no anchor
- `README.md:76` — "Versions before this one had a 'Tile Size' slider". The repository has no version numbers or
  changelog, so after the next update "this one" refers to nothing.
- Fix: "Before the slider was renamed to 'VAE Tile Size' it was 'Tile Size' with default 1024; …".

## Checked, no finding
- Vendored defaults are never used: `upscale_run.py:189-194` always passes tile size and overlap for encode and decode.
- No code or test still uses the label "Tile Size". The old key appears only in docs that explain the rename.
  `02-ARCHITECTURE.md` does not name the label.
- `slider_range(TILE_SIZE)` builds `value=512, minimum=512, maximum=2048`. If someone renames the label back, the new
  slider test fails with `StopIteration` from its `next(...)` call.
- New tests are in the mirrored folders (`tests/scripts/`, `tests/seedvr2_neo/`). The comments explain why and contain
  no block numbers.
- `forge-neo.md` "Saved UI defaults" agrees with the code paths it cites. The new 🟢 risk "short `script_args` → HTTP 500"
  is consistent with `test_run_accepts_the_arguments_of_an_older_ui`, where 13 arguments cover all required parameters.

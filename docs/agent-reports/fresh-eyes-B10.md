# Fresh-eyes review of B10 (`main...feature/b10-vae-tile-512`)

Saved by the main chat from the agent's reply.

1. **IMPORTANT** — the rename resets "Tile Size" but not "Tile Overlap": a saved overlap of 512 now pairs with tile
   512; the vendored tiler (`attn_video_vae.py`, latent overlap clamped to `latent_tile - 1`) then strides by 1 latent
   pixel, ~7000 tiles per VAE pass for 1364×1024 — the R9 symptom again. Overlap 384 → ~40 tiles vs ~12. Fix: clamp
   the overlap below the tile (e.g. ≤ tile/2) in `parse_upscale_settings`, or rename the overlap label too.
2. MINOR — README reset note sits under "GPUs with 8–11 GB" (12–24 GB users who set 1024 won't see it); "versions
   before this one" has no anchor; "starts at 512 once" is ambiguous.
3. MINOR — "keeps VAE encoding and decoding inside VRAM" contradicts R6 (`--cuda-malloc` decode still spills).
4. MINOR — "API callers are unaffected" holds for positional callers; label-mapping clients of `/sdapi/v1/script-info`
   lose "Tile Size".
5. MINOR — test fixtures still use 1024 as the typical tile (`tests/scripts/forge_stubs.py` `run_args`,
   `tests/seedvr2_neo/test_upscale_run.py`).
6. MINOR — STATE still names R9, ROADMAP shows B10 ⬜ (closing work).
7. MINOR — the ui-config trap applies to every future default change (B11 resolution, B08 profiles); the comment
   should say not to rename the label back.

Not verified by the agent: Forge source claims (fact-checker area), the 8× spatial factor of the 3B VAE.

# B13 — fresh-eyes-reviewer

No blockers. Native allocator path unchanged (only `id in frozenset()`); decode errors re-raised; Interrupt checked
before the trim; encode / DiT never trim; vendored Decoder3D / UpDecoderBlock3D call mid_block, up_blocks, resnets,
upsamplers as submodules, so the hooks fire.

IMPORTANT
- I-1 (R4, S2×L2) `TRIM_BELOW_FREE_BYTES` (3 GiB) is tuned for tile 512 on 8 GB, but the 12 / 24 GB profiles set tile
  768 / 1024 (slider up to 2048). A stage grows with the tile area; a 12 GB user with `--cuda-malloc` may have 3–4 GiB
  free, no trim runs, the pool spills again. Fix: scale with tile area, or drop the threshold under cudaMallocAsync, or
  at least document the coupling.

MINOR
- M-1 (R2) The trim also runs on Linux, where no sysmem fallback exists: cost without benefit (unmeasured).
- M-2 (R2) The trim set silently depends on the hooked set; `decoder_stages` includes leaf children
  (`temporal_modules`, nn.Identity) that are never hooked; the test count `len(decoder_stages) × tiles` would be wrong
  for the real VAE. Fix: keep only containers; test `set(decoder_stages) <= set(container_modules)` and a leaf child.
- M-3 (R1) Redundant trims: `decoder` right before `mid_block`, each `up_block` right before its `resnets[0]`.
- M-4 (R1) Docstrings not updated: `module_called`, `_upscale_tensor` ("after each phase"), `_trim_before` (why ids).
- M-5 (R1) Wording: "one tile's stages outgrow the pool"; "3 GB" vs GiB; "tiled decode" though it also runs untiled.
- M-6 (R2) README: 4K with the flag only has the pre-fix number; the "try launching without it" advice was dropped
  though the flag still decodes ~24% slower at 2048².
- M-7 (R1) 05-DECISIONS DiT-tiling wording ("does not need outputs above 2K") differs from ROADMAP/STATE; 170-char line.
- M-8 (R1) The decision does not record the remaining cost (~24% slower than native at 2048²).

## Re-check after fixes
I-1 fixed (threshold scales with the decoded tile area; extrapolation recorded). M-1, M-3 recorded in 07-RISKS
(M-3 also in the `decoder_stages` docstring). M-2, M-4, M-6, M-8 fixed. M-5: README / decision heading / 07-RISKS
still said a flat "3 GB" — fixed in the next commit. M-7: line reflowed in the next commit. New MINOR: no test for
the untiled threshold, `_trim_before` set outside the `try` — fixed in the next commit. No BLOCKER / IMPORTANT.

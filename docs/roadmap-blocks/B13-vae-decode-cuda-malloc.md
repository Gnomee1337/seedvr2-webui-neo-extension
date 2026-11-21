# B13 — VAE decode with `--cuda-malloc` without the shared-memory spill

**Status:** 🔄 in progress — implemented and measured live; `/block-done` next
**Weight:** 3 (from `docs/03-ROADMAP.md`)
**Decisions this block rests on:** "VAE tile 512 is the default, on a slider labelled 'VAE Tile Size'";
"Interrupt and progress inside a phase come from hooks on the models' container modules"; "Fix in `scripts/` first,
patch `seedvr2_vendor/` only when unavoidable"; "torch is imported in `seedvr2_neo/` only by `upscale_run.py`".

## Why this block
With Forge's `--cuda-malloc` (common on 8 GB setups) the tiled VAE decode reserves far more VRAM than it uses, spills
into shared system memory and runs 4–10× slower: a ×2 of 512×384 → 1024×768 spends 2.5–3 of its 3–3.5 minutes in the
decode (R6). Every run of that setup pays it; users cannot tell why.

## What we do (steps)
0. Baseline live on the 8 GB GPU (3B GGUF Q6_K, tile 512/128), with and without `--cuda-malloc`: decode time, peak
   allocated / reserved / used by the driver, shared memory in use; 1024×768, 1364×1024, 2048×2048 outputs.
1. Prototype lever 1 (`empty_cache()` between decode tiles from the decoder pre-hook) and measure; lever 2 or a
   smaller tile under this allocator only if lever 1 is not enough.
2. Implement the chosen lever in our code (`seedvr2_neo/` / `scripts/`), only for the allocator that needs it, so the
   native allocator's runs stay as they are; tests with stand-ins.
3. Measure again the same cases; record the numbers in `07-RISKS.md` (close or lower R6) and the facts in
   `external-apis/forge-neo.md`.
4. README: update the `--cuda-malloc` trade-off.

## Measurements (step 0 and the prototypes)
8 GB GPU without bf16, PyTorch 2.13, 3B GGUF Q6_K, VAE tile 512/128, img2img ×2. Decode time; "spill" = growth of
the GPU's shared memory during the decode (Windows counter).

| Output | native, `main` | native + trim before each tile | `--cuda-malloc`, `main` | trim before each tile | trim before 44 decoder stages | stages, only under 3 GB free |
|---|---|---|---|---|---|---|
| 1024×768 | 7.9–8.0 s | 8.6–9.8 s | 17.5–27.8 s, spill 1.6 GB | 10.1–10.5 s, spill 2 s | 10.8–11.0 s, none | 9.0–9.5 s, none |
| 1364×1024 | 15.6 s | 16.9 s | 28.8 s, spill 1.8 GB | 19.6 s, spill 2 s | 21.7 s, none | 18.3–18.8 s, none |
| 2048×2048 | 42.1 s | 45.7 s | 111.9 s, spill 1.6 GB | 73.1 s, spill 32 s | 59.0 s, none | 54.3 s, none |

Under `--cuda-malloc` the decode reserves 8.7 GB for a 5.2 GB peak on `main`. Trimming costs 8–20% with the native
allocator (no spill there), so it runs only under `cudaMallocAsync` (`torch.cuda.get_allocator_backend()` returns
`cudaMallocAsync` in Forge's venv with the env var, `native` without). A 2 GB free threshold still spilled for 2–6 s
in 2 of 4 runs; 3 GB: none in 5 runs, 13–96 trims per decode. Each trim is `empty_cache()` before the decoder, its mid
block, each up block and their resnets / upsamplers.

Final code (`65b8095`), same setup: with the flag 9.0–9.1 s / 18.3 s / 52.2 s, no shared growth; without it
8.0 / 7.9 s and 42.3 s (unchanged). Interrupt 5 s into a 2048² decode: the run ended within ~1 s, the input came back
with `SeedVR2 Error: Interrupted`, VRAM 0.8 GB after.

## Acceptance criteria
- [x] With `--cuda-malloc`, ×2 of 512×384 → 1024×768 decodes in at most twice the time without the flag (now
      2.5–3 min vs ~15 s), shared memory not growing during the decode; numbers in `07-RISKS.md`.
- [x] Without `--cuda-malloc` decode time and peak VRAM are not worse than the baseline (±10%).
- [x] 2048×2048 output with `--cuda-malloc` measured before and after.
- [x] Interrupt during the decode still stops the run within seconds; VRAM back to the idle level after a run.
- [ ] Tests for the new code in the mirrored folder; all tests green (3132 passed locally), CI green (after a push).

## Which checks are needed on `/block-done`
Always: `code-auditor`, `fresh-eyes-reviewer`, `plan-drift-checker`. Additionally:
- [ ] `integration-checker` — only if a new module or script argument appears
- [ ] `security-auditor` — not needed (no input, files or network)
- [ ] `stress-tester` — only if a new user setting appears

## Dependencies
B10 (tile 512 default), B08 (hooks on container modules) — closed. The owner's 8 GB GPU; Claude launches Forge with
`--api` itself (agreed earlier), once with and once without `--cuda-malloc`.

## Out of scope for this block
- DiT tiling, 6K / 8K (deferred, `06-IDEAS.md`).
- Changing the user's launch flags or the allocator backend (set by Forge before `import torch`).
- Speeding up the decode with the native allocator.

## Hidden assumptions
From the fact-check (`external-apis/forge-neo.md`, allocator section; PyTorch source v2.8.0 — Forge runs 2.13, the
behaviour is assumed unchanged and confirmed live in step 0):
- Under `cudaMallocAsync` PyTorch sets the pool's release threshold to "never", so the pool keeps every freed block;
  `torch.cuda.empty_cache()` synchronises and trims the pool to the live set (`cudaMemPoolTrimTo(pool, 0)`).
  `memory_reserved()` / `memory_allocated()` report the pool's reserved / used bytes.
- So lever 1 — `empty_cache()` between decode tiles, from the forward pre-hook on the decoder we already install for
  Interrupt — should keep reserved near the 5.2 GB peak instead of 8.7 GB. Cost: one sync per tile.
- Lever 2 — lowering the release threshold through `cudaMemPoolSetAttribute` (ctypes) — is unverified, affects Forge's
  own use of the pool and needs a restore; only if lever 1 is not enough.
- `set_per_process_memory_fraction` caps allocated bytes, not the pool's reserved size: no help against the spill.
- The spill itself is the Windows driver's "Sysmem Fallback Policy" (NVIDIA Control Panel); no app API reads or sets
  it. README can tell users to switch it to "Prefer No Sysmem Fallback" (then an OOM instead of a slowdown).
- The allocator backend can be read at run time (`torch.cuda.get_allocator_backend()`), so the extra trims can be
  limited to `cudaMallocAsync` runs. Not in the fact-check: verify in step 0 inside Forge.

## Open questions
- If the only working lever is a smaller VAE tile under `--cuda-malloc`: set it automatically or only advise it in the UI.

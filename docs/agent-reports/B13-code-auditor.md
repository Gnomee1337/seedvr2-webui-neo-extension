# B13 code audit: VAE decode with `--cuda-malloc`

Scope: `git diff main...HEAD -- seedvr2_neo tests scripts` (final state; the prototype commits are judged only by
what is left in HEAD). The diff touches only `seedvr2_neo/upscale_run.py` and `tests/seedvr2_neo/test_upscale_run.py`.
It does not change `scripts/` or `seedvr2_vendor/`.

Test run: full suite `3132 passed, 1 skipped, 3 xfailed` (CPU torch 2.14.1, Python 3.10). `test_upscale_run.py`: 46 passed.

**Result: 0 BLOCKER, 0 IMPORTANT, 5 MINOR.** The logic holds up. The two most useful fixes are tests:
- the device wiring is not covered by any test (proved with a mutation);
- the test stand-in decoder does not have the shape of the vendored one.

## Checked and found correct (no finding)

- **Native allocator unchanged:** `_trim_before` stays empty unless `keeps_freed_vram` is true. `id(module) in
  self._trim_before` is checked before `free_vram_bytes`, so with the native allocator `module_called` makes no extra
  driver call. The DiT and encode phases never trim because the set is filled only around `decode_all_batches`.
- **Errors and Interrupt:** `check_interruption()` runs before the trim. `_trim_before` is cleared in a `finally`, so
  an `UpscaleInterrupted` or any other error out of the decode leaves no trims behind. The test
  `test_a_failed_decode_stops_returning_the_pool` covers this, and removing the `finally` makes it fail.
- **Non-CUDA devices:**
  - `keeps_freed_vram` returns False for a CPU, MPS or `None` device.
  - It also returns False for a torch with no `get_allocator_backend`: `torch.device(None)` raises `TypeError`, a bad
    string raises `RuntimeError`, a CPU build raises `AttributeError`, and all of these are caught.
  - The `except` names specific exception types and only probes for a feature, so it is not a silent broad catch.
- **`mem_get_info` with an index-less device:**
  - Read in torch 2.14 source: `mem_get_info` calls `_get_device_index(device, optional=True)` with the comment
    "allows `device = torch.device('cuda')`".
  - So `devices.device` with or without an index works and does not raise inside the pre-hook.
  - `cuda:N` reads the right GPU.
- **`empty_cache()` under `cudaMallocAsync`:** it trims every initialised device after a device sync (documented in
  `forge-neo.md`), so a non-zero device index is still covered.
- **Module ids:**
  - The set is built from `runner.vae` at decode time.
  - Hooks were installed on the same objects at the start of `_upscale_tensor`. The vendored code keeps these objects:
    `materialize_model` uses `assign=True`, and a cached VAE is the same object.
  - The ids stay unique while the VAE is alive, and the set is cleared before anything can free it.
  - The only other watched modules are the DiT containers, which are not called during the decode.
- **Leftovers:** no prototype code is left (`PROTOTYPE`, `_trims`, `_trim_module_ids`, prints), no TODO, no
  `seedvr2_vendor/` change.
- **Real decoder shape:**
  - I built the shipped VAE (`VideoAutoencoderKLWrapper` with `s8_c16_t4_inflation_sd3.yaml`) on the meta device.
  - `decoder_stages` lists `Decoder3D`, `UNetMidBlock3D`, 4 `UpDecoderBlock3D`, 12 `ResnetBlock3D` and 3 `Upsample3D`.
  - All 21 of these are containers, so they are hooked and do trim.

## Findings

### MINOR 🟢 R2 (S2×L1): no test checks which device reaches `keeps_freed_vram` / `free_vram_bytes`
`tests/seedvr2_neo/test_upscale_run.py:498-500` (code: `seedvr2_neo/upscale_run.py:238`, `:337`, `:167-169`).

**What is wrong:**
- `run_staged` replaces both functions with `lambda device: ...`, which ignores the argument.
- No test calls `free_vram_bytes` itself.

**Proof (mutation):** in a scratch copy of the repo I changed line 238 to `free_vram_bytes(None)` and line 337 to
`keeps_freed_vram(plan.vae_offload_device)`. Result: `test_upscale_run.py` still has 46 passed.

**Scenario:**
1. A refactor passes the offload device by mistake.
2. With a memory plan that offloads the VAE, `vae_offload_device` is `"cpu"` (`memory_plan.py:45,71`).
3. `keeps_freed_vram("cpu")` returns False.
4. Every `--cuda-malloc` run silently loses the fix: the 2048² decode goes back from ~52 s to ~112 s with the spill.
5. CI stays green.

**Fix:**
- Make the stand-ins record their `device` argument and assert that it equals `hooks.device` (`"gpu-0"` in
  `make_tracked_run`).
- Add a unit test for `free_vram_bytes`: monkeypatch `torch.cuda.mem_get_info` to return `(free, total)`, then assert
  the device passed in and that element `[0]` is the one returned.

### MINOR 🟢 R2 (S2×L1): `decoder_stages` lists modules that are never hooked, and the stand-in hides it
`seedvr2_neo/upscale_run.py:162-163`, `tests/seedvr2_neo/test_upscale_run.py:400-404`.

**What is wrong:**
- `decoder_stages` adds every child of an up block.
- On the shipped VAE that includes the `temporal_modules` `nn.Identity` layers: 33 ids in the set, 12 of them
  `Identity`.
- `container_modules` never hooks a leaf, so these 12 never trim. A stage only works if it also happens to be a
  container, but nothing in the code or the docstring says so. The docstring says "resnets, upsamplers".
- The test `UpBlock` has no `temporal_modules`, and its `Resnet` is built with a child on purpose ("so it is a watched
  container"). So the test cannot catch a stage that is listed but not hooked.

**Scenario:**
1. An upstream sync turns `Upsample3D` into a single conv layer (a leaf), or adds a leaf stage to an up block.
2. `decoder_stages` still lists it and the docstring promises a trim there.
3. No hook fires on it, so the largest allocation of each tile (the upsample) runs without a trim before it.
4. The spill can come back, and the tests stay green.

**Fix:**
- In `decoder_stages`, keep only modules that have children. Share one `has_children(module)` predicate with
  `container_modules`, or check `id in hooked` at the call site.
- Add `temporal_modules = ModuleList([Identity(), ...])` to the `UpBlock` stand-in so it mirrors `UpDecoderBlock3D`.

### MINOR 🟢 R1 (S1×L1): the `module_called` docstring no longer describes what it does
`seedvr2_neo/upscale_run.py:234`.

**What is wrong:** the docstring says "Interrupt, and a DiT block moves the bar". It now also returns the VRAM pool
before decoder stages when free memory is low.

**Scenario:** a reader of `UpscaleRun` misses that the hook synchronises the GPU. For example, they add a new watched
phase and do not expect `empty_cache()` calls from it.

**Fix:** add "and, during a decode under a pool-keeping allocator, returns the unused pool before a decoder stage when
VRAM is short".

### MINOR 🟢 R1 (S1×L1): a parametrised test computes its expected value with the code under test
`tests/seedvr2_neo/test_upscale_run.py:447-449`.

**What is wrong:** `expected = [decoder] if isinstance(decoder, torch.nn.Module) else []` repeats the guard in
`decoder_stages`. A mistake in the shared idea (for example, treating any object with `forward` as a module) would be
copied into both and pass.

**Fix:** give each case its literal expected value: `None` → `[]`, `SimpleNamespace()` → `[]`, `decoder="decoder"` →
`[]`, `Vae()` → `[vae.decoder]`. Also add docstrings to the new stand-in classes `UpBlock` and `StagedVae`
(code-style: "a docstring on every class").

### MINOR 🟢 R1 (S1×L1): the block card's "44 decoder stages" does not match the 21 trim points
`docs/roadmap-blocks/B13-vae-decode-cuda-malloc.md:29` (and the text at lines 33-34).

**What is wrong:**
- Prototype 2 printed `len(_trim_module_ids)` = 44. That count includes the `ModuleList` containers (11) and the
  `Identity` layers (12), which are never called through a hook or never hooked.
- The real number of trim points is 21 per decoder call (per tile), both in that prototype and in the final code.

**Scenario:** if the number is moved into `07-RISKS.md` or `05-DECISIONS.md` at `/block-done`, a later tuning session
reasons from a cost that is twice the real one.

**Fix:** write "21 decoder stages (per tile)" in the card and in anything taken from it.

## Re-check after fixes
Device wiring asserted in tests; `decoder_stages` lists only hooked containers (21 stages on the real VAE: 1
Decoder3D, 1 UNetMidBlock3D, 4 UpDecoderBlock3D, 12 ResnetBlock3D, 3 Upsample3D); docstring and tautological test
fixed. Still open then, fixed in the next commit: `free_vram_bytes` untested (`[1]` mutation survived), untiled
threshold untested (mutation survived), docstrings of test stand-ins. Card stage count "44" — the card is deleted on
close. No BLOCKER / IMPORTANT.

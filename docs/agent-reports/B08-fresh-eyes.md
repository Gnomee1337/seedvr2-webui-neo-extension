# B08 — fresh-eyes review

Reviewed: `git diff main...feature/b08-settings-feedback` (code, tests, docs). Vendored code read only to check the
assumptions the change relies on (`seedvr2_vendor/core/generation_phases.py`, `optimization/memory_manager.py`,
`optimization/compatibility.py`, `core/model_configuration.py`). The Forge Neo source is not in the repository, so
statements about Forge below are marked "unverified" where they rest on the A1111 lineage, not on a read of Neo.

## How I understood the result

- Forge's progress bar: before the upscale the script makes it job 1 of 1 (unless Forge is already inside a job,
  which is meant for img2img Batch), then `RunProgress` turns passes, weighted phases, per-batch callbacks and
  per-DiT-block events into `shared.state.sampling_step` (of 1000) and `textinfo`.
- Interrupt: besides the old between-pass check, `ctx['interrupt_fn']` and the progress callback check it per batch,
  and forward pre-hooks on every "container" module of the VAE and the DiT check it inside a batch. The hooks are
  always removed.
- Color correction (6 methods, default `lab`) and attention (`sdpa` plus installed backends, fallback to `sdpa` with a
  console line) are two new trailing `run()` arguments and go into the infotext.
- A GPU profile dropdown (Manual / 8 / 12 / 24 GB) fills BlockSwap, tile size and overlap; it is not a run argument and
  is not saved; moving a slider by hand sets it back to Manual.
- The title became "SeedVR2 Upscaler". Old API callers with 13/16/19 arguments still work, but only with the new
  `script_name`: the old name gets HTTP 422 (a recorded decision).

Divergence from the task: the task says "older API callers must keep working". Their argument lists do, but every old
caller still sends the old `script_name`, and those calls now fail (finding 1).

## Findings

### IMPORTANT

**1. The old `script_name` breaks every existing API caller, which the task says must keep working.**
`scripts/seedvr2_script.py:65` (`SCRIPT_TITLE`), `README.md:137`, `docs/05-DECISIONS.md:106`, `docs/07-RISKS.md:163`.
What I saw: the rename is deliberate and documented (owner's choice, 🟡 R4), and Forge has no title alias.
Why it matters: an automation that worked yesterday gets HTTP 422 after "Check for updates", with no change on its
side. The task wording ("older API callers must keep working") and the result disagree. The argument-count
compatibility, which the tests cover, does not help a caller that never gets past the script lookup.
Scenario: a user's batch script posts `"script_name": "SeedVR2 Native Upscaler (24G)"` with 19 args → 422 after the update.
Rating: S2×L2 = 🟡 R4 (as in `07-RISKS.md`). Action: make sure the owner accepted this as the exception to "old callers
keep working", and note it in the release note or commit message. Otherwise the block's own claim reads as a contradiction.

**2. img2img Batch: the progress bar probably never gets past the first file's share, and the test assumes otherwise.**
`scripts/seedvr2_script.py:43-52` (`start_forge_progress`), `tests/scripts/test_seedvr2_script_settings.py:46-52`,
`docs/05-DECISIONS.md` ("becomes a job of its own only when Forge has none left"), `docs/09-PROCESSES.md:127+`.
What I saw: the code keeps Forge's `job_no` / `job_count` when `0 <= job_no < job_count`, on the assumption that Forge
advances `job_no` per file in img2img Batch. The test sets `job_no = 1` by hand to simulate the second file.
`external-apis/forge-neo.md` (img2img.py:46, 62) records only that Batch sets `job_count` and `state.job`. `job_no` moves
in `state.nextjob()`, which `process_images_inner` calls, and this script replaces `process_images` in img2img. In the
A1111 lineage the Batch loop never calls `nextjob()` itself (unverified for Neo).
Why it matters: if this holds, every file's upscale moves the bar from 0 to 1/N and back to 0, and Forge's ETA is
meaningless for the whole batch. The test then passes on a state Forge never produces, so it gives false assurance.
Rating: S1×L3 = R3 (cosmetic, but the docs and test state a fact that is unverified). Action: check whether
`job_no` moves per file in Neo's `modules/img2img.py`. If it does not, the script should advance it itself (for
example from `state.job` "i out of N") or the docs should say the bar restarts per file.

### MINOR

**3. Interrupt in img2img Batch is likely soft on the first click; README promises "within a few seconds".**
`README.md:25-27`, `scripts/seedvr2_script.py:47-48`.
What I saw: `forge-neo.md:153` records that with `job_count > 1` and the option `interrupt_after_current`, the Interrupt
button calls `stop_generating()`. This script deliberately ignores `stopping_generation`. In Batch, `job_count` is the
file count, so the first click lets the current file's upscale (minutes at 4K) run to the end.
Why: a user reads "Interrupt stops the upscale within a few seconds" and sees it keep running. The behaviour is
reasonable ("don't interrupt in the middle"), but the README does not mention it, and the option's default in Neo is
not recorded. Rating: S1×L2 = R2.

**4. The `watched` list holds the DiT and VAE module trees alive until post-processing ends.**
`seedvr2_neo/upscale_run.py:799-800` (`watched = container_modules(...)`, used for all four phases).
What I saw: `container_modules` returns the model root itself (it has children) plus every sub-container, held in a
local list across all phases. With "Unload", `cleanup_dit` sets `runner.dit = None` after phase 2 and expects the DiT
to go away. Today VRAM is still freed, because the vendored `release_model_memory` empties every CUDA parameter and
buffer in place (`param.data.set_()`), and the live check shows no change. That is a silent dependency: the comment
and docs only say "the hooks are always removed".
Scenario: an upstream sync drops the in-place `set_()` (or a GGUF / BlockSwap attribute holds a CUDA tensor that is
not a parameter or buffer). The DiT then stays in VRAM through the VAE decode, which is exactly the spill R6 is about,
and no test catches it.
Rating: S3×L1 = R3. Action: either hook per phase (VAE modules for phases 1 and 3, DiT modules for phase 2) and drop
the list after each phase, or add this dependency to the "depends on the vendored model structure" risk in
`07-RISKS.md` and to `external-apis/seedvr2-upstream.md`.

**5. An attention backend that imports can still fail on the user's GPU; the UI does not say so.**
`scripts/seedvr2_script.py:176-179`, `seedvr2_neo/pipeline_options.py:375-391`, `upscale_run.py:628-631`.
What I saw: availability is "the package is installed" (UI) or "the module imported" (run). FlashAttention 2 needs
Ampere or newer. On the project's own target class (8 GB, no bf16: Turing / Pascal), an installed `flash_attn` is
listed, imports, and fails inside the DiT. The run then ends with `SeedVR2 Error: Runtime Error: RuntimeError` and
the models are unloaded. Nothing in the infotext or the UI points to the attention choice. The ui-config also
remembers the choice, so every later run fails the same way.
Rating: S2×L1 = R2. Action: mention GPU requirements in the dropdown's `info`, or record it in `07-RISKS.md`.

**6. GPU profiles assume VAE tiling is on, and treat every 7B the same.**
`seedvr2_neo/gpu_profiles.py:271-278`, `scripts/seedvr2_script.py:83-91`.
What I saw: a profile sets tile size and overlap but not "Use Tile VAE". A user who turned tiling off earlier (saved in
ui-config) picks "8 GB" and keeps an untiled decode. BlockSwap 16 (8 GB) and 8 (12 GB) for 7B are the fp8 numbers;
a 7B fp16 (~16 GB) does not fit with 8 blocks in RAM on 12 GB, or with 16 on 8 GB. The 8 GB profile is marked
`measured=True` as a whole, which a reader takes to cover 7B fp16 too.
Rating: S2×L2 = R4 (🟡). Action: let the profile also turn tiling on, or say in the hint that it assumes tiling;
state "7B fp8" where the BlockSwap values come from.

**7. The profile changes sliders the user cannot see.**
`README.md:44-46` ("the sliders stay as you can see them"), `scripts/seedvr2_script.py:148-152, 361-371`.
What I saw: tile size and overlap are inside the collapsed "Advanced Settings" accordion. Picking "24 GB" silently
changes the tile to 1024, and the user sees only BlockSwap change. The README sentence is also unclear English.
Rating: S1×L2 = R2.

**8. In txt2img the bar jumps from 100% back to about 0%, and Forge's ETA spikes.**
`scripts/seedvr2_script.py:50-51`.
What I saw: after `process_images` (bar at 100%) the script resets `job_count/job_no` to 1/0. Forge's ETA is
`elapsed / progress - elapsed`, where `elapsed` counts from `begin()` and includes the base generation. Right after the
reset `progress` is about 0.01, so the ETA shows hours for the first seconds.
Rating: S1×L3 = R3. Recording it in R3 (the "ETA is approximate" entry) would be enough.

**9. Phase indexes live in two modules.**
`seedvr2_neo/upscale_run.py:604-605` (`ENCODE_PHASE, DECODE_PHASE, POSTPROCESS_PHASE = 0, 2, 3`) vs
`seedvr2_neo/run_progress.py:442-443` (`PHASES`, `UPSCALE_PHASE = 1`).
What I saw: three indexes are bare numbers in one module, and the fourth is defined in the other. A reorder of `PHASES`
or a new phase silently shifts the text and the weights. A reader also wonders why only `UPSCALE_PHASE` is imported.
Rating: S1×L1 = R1. Action: define all four next to `PHASES` (or an enum).

**10. Unclear wording and leftovers in the code.**
- `upscale_run.py:663` `watching_modules` docstring: "while the block runs". "Block" here means the `with` block, but
  in this module "block" means a DiT block, so I first read it as "during a DiT block".
- `upscale_run.py:714` `check_interruption(self, *args, **kwargs)`: the `*args, **kwargs` were there because it used to
  be the progress callback. Now `phase_progress` is the callback, so a reader looks for callers that pass arguments
  and finds none.
- `scripts/seedvr2_script.py:83, 94`: `apply_gpu_profile` and `resolve_pipeline_options` have no type hints, unlike
  their neighbours.
Rating: S1×L1 = R1.

**11. `docs/09-PROCESSES.md:43-47`: the UI list goes 6, 8, 7.**
The new item "8. GPU profile" was inserted before "7. Output size line". A reader looks for the missing 7 or thinks
an item was lost. Rating: S1×L1 = R1.

**12. The test count differs between documents.**
`docs/roadmap-blocks/B08-settings-and-feedback.md:44` says 2583 passed; `docs/04-STATE.md` still says 2471. This is
expected until `/block-done`, but STATE must be updated then. Rating: S1×L1 = R1.

## Checked and fine

- The vendored phase `except Exception` blocks re-raise, so `UpscaleInterrupted` raised from a hook is not swallowed.
  `causal_inflation_lib` catches only `RuntimeError`, and BlockSwap's RoPE retry catches only device or memory errors.
- `progress_callback(current, total, frames, text)` matches `phase_progress(current, total, *args)` in all four phases,
  including the alpha steps of post-processing (monotonic through `max`).
- `color_correction="none"` follows the vendored `!= "none"` guards in encode and post-process.
- `attention_availability` flag names match `optimization/compatibility.py`.
- Hooks are removed in `finally`, and the exception path releases models after the `except` block, so the traceback no
  longer pins the `watched` frame.
- Old `run()` signatures (13/16/19 args) get `lab` / `sdpa` through the defaults, which the tests cover.

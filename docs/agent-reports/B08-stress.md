# B08 stress test report

Targets: the B08 diff (`git diff main...feature/b08-settings-feedback`), tested against the working tree as it was
while the main chat was still editing it (`read_state_count`, `finish_forge_progress`, the FA3/SA3 step-down and
`parse_choice` landed during the run, and the tests cover them). No product code was changed.

## Tests added

| File | What it hits |
|---|---|
| `tests/seedvr2_neo/test_pipeline_options_stress.py` (new) | parse_color_correction / parse_attention_mode with ~45 junk values (None, bytes, memoryview, NaN, 10**400, 1 MB strings, injection `"lab\nSteps: 999"`, fullwidth/zero-width look-alikes, lists, dicts, other enums: str and plain); case/padding of every name; JSON `null` vs the string `"None"`; installed_attention_modes with find_spec returning falsy specs, raising ImportError/ValueError/RuntimeError, a module in `sys.modules` with no `__spec__`; resolve_attention_mode with empty, all-False, string-keyed and odd-flag maps; the v3→v2 step-down over all 2^5 availability masks |
| `tests/seedvr2_neo/test_run_progress_stress.py` (new) | RunProgress with odd pass sizes ([], [0], [-5], 10**12, "3", Decimal, Fraction, None/NaN/inf), odd batch values (None, 0 totals, strings, complex, inf, NaN, 10**400), clamped indexes, odd block counts, 1M blocks for one image, 200 random call sequences (fraction always finite in 0…1, text sane), 300 random real-order runs (never moves back beyond float noise), a report that raises on each call type |
| `tests/seedvr2_neo/test_gpu_profiles_stress.py` (new) | find_profile with junk labels (near-matches, NBSP, fullwidth, bytes, lists, NaN, 400 KB); every profile inside the slider bounds and steps, overlap ≤ tile/2, survives parse_upscale_settings and build_memory_plan unchanged; bigger GPU → no smaller tile / no more swapped blocks |
| `tests/seedvr2_neo/test_upscale_run_stress.py` (new section) | a report raising at every one of its calls (1 and 2 passes); is_interrupted / release_memory raising; Interrupt from the hook of a block and of a container inside a BlockSwap-style wrapped block (wrapper with no try/finally, as vendored); hooks firing through a replaced `forward`; runners whose dit/vae are None, strings, bytes, dicts, lists, generators, fake `modules()`; a bare layer; the same module as DiT and VAE; a VAE nested in the DiT; the DiT swapped mid-phase; hook registration failing half-way; Interrupt between passes. Each checks: outcome returned, no hook left, models dropped even with "Keep in RAM" |
| `tests/scripts/test_seedvr2_script_settings_stress.py` (new) | Script.run with junk color_correction / attention_mode (infotext gets only enum values, never caller text), every backend with and without its import flag, 20/21 positional args, foreign str enum; shared.state with odd job_count/job_no (-1, 0, beyond, 10**9, float, bool, None, strings, NaN, inf, 10**400, object), img2img Batch with junk `n_iter`; report_forge_progress clamping (±inf, -5, 7, rounding edges); a NaN from the pipeline's progress callback; Interrupt mid-run; apply_gpu_profile with junk labels and junk DiT names (path traversal, NUL, 100 KB, lists, bytes) and a deleted model folder; profile values fit the built sliders and reach prepare_runner unchanged |

Run: 5 files, about 560 tests, all pass except the strict xfails below. Full suite: see the summary.

## Findings

All are MINOR: none is reachable through Forge or the vendored pipeline as they are today (the vendored callback
passes ints; Forge's finders and `shared.state` are well-behaved). Rating: S = severity 1–3, L = likelihood 1–3.

### 1. NaN batch value turns the bar into NaN and fails the upscale (MINOR, S2 × L1 = 2)
`seedvr2_neo/run_progress.py:64` — `min(max(fraction, self._batch_fraction), 1.0)` keeps NaN (NaN comes first in
`max`, so even NaN after real progress wins). The fraction is NaN; `scripts/seedvr2_script.py:159`
`int(round(nan))` raises ValueError inside the pipeline's callback, the run's boundary catches it and the user gets
the un-upscaled images with `SeedVR2 Error: "Runtime Error: ValueError"`.
Repro: `RunProgress([1], r).batch_done(float("nan"), 1)` → report(nan, …); through Script.run: the pipeline callback
with `(nan, 1)` → error infotext. Also `(inf, inf)`, `Decimal("NaN")`.
Tests: `test_nan_batch_values_never_reach_the_report`, `test_a_nan_batch_after_a_real_one_does_not_wipe_the_progress`
(run_progress stress), `test_a_nan_from_the_pipeline_callback_never_fails_the_upscale` (script stress) — xfail.
Fix idea: skip non-finite fractions in `batch_done` (`math.isfinite`), and/or guard `report_forge_progress`.

### 2. Huge int batch value raises OverflowError (MINOR, S2 × L1 = 2)
`seedvr2_neo/run_progress.py:61-62` — `float(10**400)` raises OverflowError, not in the caught tuple
(TypeError, ValueError, ZeroDivisionError). Same consequence as 1 (run fails with an error infotext).
Test: `test_a_huge_int_batch_value_is_ignored_like_other_unreadable_values` — xfail.

### 3. installed_attention_modes lets non-import errors escape Script.ui() (MINOR, S2 × L1 = 2)
`seedvr2_neo/pipeline_options.py:106` catches only ImportError/ValueError. A broken meta-path finder of another
package (RuntimeError, AttributeError, TypeError from `find_spec`) propagates out of `_advanced_controls` →
`Script.ui()`, which the code elsewhere explicitly avoids (`memory_plan.is_7b_checkpoint` catches everything for this
reason: a failure there drops the whole script from Forge's UI).
Test: `test_any_find_spec_failure_never_breaks_the_ui_list` (3 cases) — xfail.

### 4. `_dit_blocks` left set when the report raises at the DiT phase start (MINOR, S1 × L1 = 1)
`seedvr2_neo/upscale_run.py:280` sets `self._dit_blocks` before the `try/finally` (line 287) that clears it; a
report raising in `start_phase(UPSCALE_PHASE, …)` skips the reset. No user effect today (one UpscaleRun per
Generate, and the exception ends the run); the attribute's comment "set only while the DiT phase runs" is false then.
Test: `test_a_report_raising_at_the_dit_phase_start_clears_the_block_filter` — xfail.

## Confirmed robust (no finding)

- Any API value for color correction / attention gives a known member; no caller text reaches the infotext; other
  enums' members never cross over; the step-down never lands on an unavailable backend.
- A raising report / is_interrupted / release_memory, a failing hook registration, Interrupt inside a BlockSwap-
  wrapped block: the run always returns an outcome, removes every hook and drops the models even with "Keep in RAM"
  (so a block the vendored wrapper leaves on the GPU goes with the dropped model).
- Runners whose DiT/VAE are not modules run without hooks; duplicated or nested modules are cleaned.
- Odd `shared.state` counters (None, strings, NaN, inf, 10**400, beyond, negative) and junk `n_iter` never fail the
  run; the job counter stays inside 0…job_count.
- GPU profiles: only exact labels match; all values fit the built sliders and reach the pipeline unchanged; junk DiT
  names give the 3B values without reading any file.

## Notes (not bugs)

- A strictly real-order run can step the fraction back by float noise (~1e-17, e.g. 0.2 → 0.19999999999999998);
  invisible in Forge's 1000-step bar. The test allows 1e-9.
- `RunProgress` propagates a raising report to its caller by design; UpscaleRun turns it into a run failure. If the
  bar should never fail a run, `_publish` (or `report_forge_progress`) would have to swallow errors — an owner call.
- Values the script leaves as Forge's own (`job_count = "3"` when Forge "counts jobs") would break Forge's progress
  endpoint, not the run; Forge never sets such values.
- Unicode casefolding lets `"ſdpa"` (long s) pick `sdpa`; harmless.

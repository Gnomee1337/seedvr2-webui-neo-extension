# B10 — VAE tile 512 as the default

**Status:** 🔄 in progress
**Weight:** 1 (from `docs/03-ROADMAP.md`)
**Decisions this block rests on:** "Fix in `scripts/` first, patch `seedvr2_vendor/` only when unavoidable";
"API values of the numeric settings are clamped to the slider ranges"; "BlockSwap is a manual slider with a note for
7B on small GPUs, not an automatic setting" (why the default is fixed, not VRAM-based).

## Why this block
With the default VAE tile 1024 and `--cuda-malloc`, a run on an 8 GB GPU can stall for many minutes in the VAE phase
(🟠 R9). 512/128 is already live-checked clean on that class of GPU; making it the default — for existing users too —
removes the top risk with one setting.

## What we do (steps)
1. Fact check (before code): how Forge Neo saves slider values in `ui-config.json`, the key format, and where API
   script args take their defaults → `external-apis/forge-neo.md`, conclusions into "Hidden assumptions".
2. `seedvr2_neo/upscale_settings.py`: `TILE_SIZE` default 1024 → 512 (overlap stays 128, minimum stays 512).
3. `scripts/seedvr2_script.py`: rename the slider label "Tile Size" → "VAE Tile Size" (new `ui-config.json` key, so a
   saved 1024 no longer applies). Script args stay positional; API callers are unaffected.
4. Tests: parse default, slider default and label (`tests/seedvr2_neo/`, `tests/scripts/` label map).
5. Docs: README (tile advice, label), `07-RISKS.md` (R9 closed, the leftover folded into R6), `02-ARCHITECTURE.md` if
   the label is named there, `05-DECISIONS.md` (default 512 and the label rename that resets it once).
6. Live re-measure (Claude through Forge `--api`, 8 GB GPU without bf16, 3B GGUF): the R9 case — img2img 512×384 →
   1024 shortest edge — at 512/128, with and without `--cuda-malloc`: no stall, encode/decode times, image clean.
   Check that a `ui-config.json` holding the old "Tile Size" = 1024 does not change the new slider.
7. Full test run, `/block-done`.

## Acceptance criteria
- [ ] `parse_upscale_settings()` with no tile size gives 512; the slider is built with `value=512`, `minimum=512`.
- [ ] The slider label is "VAE Tile Size"; the tests' label map and the README use it.
- [ ] Live, `--cuda-malloc` on: the R9 case finishes without a stall; encode and decode times recorded in `07-RISKS.md`.
- [ ] Live, without `--cuda-malloc`: the same case finishes, times recorded.
- [ ] Live: with an old `ui-config.json` entry "Tile Size" = 1024, the UI shows "VAE Tile Size" = 512.
- [ ] R9 removed from `07-RISKS.md`; what remains (decode spill with `--cuda-malloc`) is in R6.
- [ ] Full pytest run green; CI green on the branch.

## Which checks are needed on `/block-done`
Always: `code-auditor`, `fresh-eyes-reviewer`, `plan-drift-checker`. Additionally (tick what applies):
- [ ] `integration-checker` — new module, data format changed, the block must be runnable
- [ ] `security-auditor` — user data, authorization, external requests, files, money
- [ ] `stress-tester` — user input, money, external services (API clamping is unchanged and already covered)

## Dependencies
Owner's local Forge Neo with the 3B GGUF model, for the live re-measure (agreed: Claude launches it with `--api`).

## Out of scope for this block
- A minimum below 512, a VRAM-based default, GPU profiles (B08).
- Interrupt inside a running phase (B08).
- Overlap changes (stays 128 — the measured configuration).
- Output size controls (B11), DiT tiling (B12).

## Hidden assumptions
Verified in Neo's source (`external-apis/forge-neo.md`, "Saved UI defaults"):
- A saved `ui-config.json` value beats the code default; slider `value`, `minimum`, `maximum`, `step` are all frozen
  per key. So keeping the label "Tile Size" would leave existing users on 1024.
- Key: `customscript/seedvr2_script.py/<tab>/<label>/<field>`; accordions are not part of it. A new label means a new
  key: the code default applies and is written; the old key stays in the file, ignored, with no warning.
- API script args omitted or short take the **code** value, not ui-config's, so API callers get 512 either way
  (order inferred from `ui.py`, checked live in step 6).
- Unverified: whether a saved value outside a changed range is clamped (not relevant: the range does not change).

## Open questions
None.

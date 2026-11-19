# B07 — stress-tester report

Scope: `seedvr2_neo/run_inputs.py`, `seedvr2_neo/run_results.py` (pure helpers of B07). Application code not changed.

## Tests added
- `tests/seedvr2_neo/test_run_inputs_stress.py`: 291 cases (286 pass, 5 xfail strict).
- `tests/seedvr2_neo/test_run_results_stress.py`: 183 cases (178 pass, 5 xfail strict).

Full run: `uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q`
→ 1533 passed, 1 skipped, 12 xfailed (2 earlier + 10 new). Pillow 12.3.0.

## Scenarios
- **Image modes:** all 20 Pillow modes (`1 L P RGB RGBA CMYK YCbCr LAB HSV I F LA PA RGBa La RGBX I;16 I;16L I;16B I;16N`),
  each also with `info["transparency"]`, at 1×1, 4096×1, 1×4096, 20000×2; the result is always RGB/RGBA and keeps its size.
  Pixel values: 16-bit layouts (0/257/32768/65535), `F` with ±1e30, ±inf, NaN, `I` with -5 and 2**31-1, `1` black/white,
  half-transparent LA/La/PA/RGBa, premultiplied La unpremultiplied. PNG round trips the way an API caller's file decodes:
  16-bit gray, 16-bit gray + tRNS, L + tRNS, P with per-index alpha bytes, RGB + tRNS colour key. Odd `info` keys,
  RGB/RGBA returned as the same object, the input not mutated.
- **Indices (`select_generated_images`):** NaN, ±inf, `"1.0"`, bytes, lists, objects, bool, ±10**100, 2**63, floats,
  Decimal, Fraction, strings with spaces; huge count; grid-only result; tuples; the returned list is a copy; images and
  infotexts stay paired.
- **Distinct images:** 5000 repeats, tuples, generators, unhashable items.
- **Grouping:** 300 random shapes (every index once, groups ordered, one shape per group), transposed sizes, the same object
  repeated, tuples/generators.
- **Seeds:** None, -1, NaN, ±inf, `"nan"`, `"0x10"`, emoji, bool, lists/dicts, 10**400, `"1e400"`, 2**32, complex; valid
  forms `"1e3"`, `"+42"`, `b"42"`, Decimal, Fraction, 42.9, -0.5, MAX_SEED; image seeds as generator/iterator/int/str/dict;
  random fallback in range, deterministic with an rng, global `random` untouched; result always a plain `int`.
- **Infotexts:** a model of Forge's `parse_generation_parameters` (A1111 `re_param_code`, last line = params only with 3+
  matches, `json.loads` for quoted values). Error text and model names with commas, colons, quotes, backslashes, `\n`,
  `\r\n`, `\r`, U+2028, NUL, unicode/emoji, 100 000 chars, text that looks like params or "Negative prompt:"; non-string
  values (None, NaN, 2**64, lists, bytes, enum). Checked: the params line is the last line with no break inside, every
  SeedVR2 key and Forge's own keys read back, values read back exactly, prompt and negative unchanged. Base infotext
  ending with line breaks, blank, `\r\n` endings, trailing comma, already holding SeedVR2 keys. `from_prompt`: empty
  prompt + negative only, None/blank prompts, prompts that look like params, multi-line negatives, originals saved
  with `with_params({})`.
- **Saving:** all 16 flag combinations against the invariants (results = samples_allowed; never an original when the
  caller disabled saving or when Forge already saved the base); truthy non-bool values.
- **Merge:** negative, huge, string, None, tuple, float indices; 5000 inputs; generators; empty inputs; the same object
  twice; arguments not mutated.

## Findings (xfail strict — remove the mark when fixed)

Risk `R = S × L` on the 1–3 scale of `docs/07-RISKS.md`.

1. **16-bit grayscale PNG with a tRNS chunk comes out white and fully opaque** — S2 × L1 = 2, MINOR.
   `seedvr2_neo/run_inputs.py:61` checks `"transparency" in image.info` before the wide-grayscale branch (`:65`), so
   `I;16` + `transparency` goes through `convert("RGBA")`, which clips 16-bit values at 255. Input: `I;16` PNG
   saved with `transparency=1000`, pixel 32768 → `(255, 255, 255, 255)`; the transparent pixel also gets alpha 255.
   Test: `test_16_bit_png_with_transparency_is_scaled_and_keeps_alpha`.
2. **`I;16N` image comes out black** — S2 × L1 = 2, MINOR. `seedvr2_neo/run_inputs.py:67`: Pillow 12.x converts
   `I;16N` → `I` clipped at 255, then the `/257` scaling gives 0. Input: `Image.new("I;16N", (1, 1), 32768)` → `(0, 0, 0)`.
   No common decoder seems to produce `I;16N` (PNG gives `I;16`), so likelihood is low. Tests:
   `test_every_16_bit_layout_is_scaled_not_clipped[*-I;16N]` (3 cases).
3. **RGB PNG with a tRNS colour key loses its transparency** — S1 × L1 = 1, MINOR. `seedvr2_neo/run_inputs.py:59`
   returns RGB unchanged before looking at `info["transparency"]`, while the docstring promises RGBA "when it has
   transparency" and L/P with tRNS do get RGBA. Input: RGB PNG saved with `transparency=(1, 2, 3)` → mode `RGB`.
   Test: `test_rgb_png_with_transparent_colour_key_keeps_alpha`.
4. **Base infotext ending with a line break: Forge's own params are read as prompt** — S2 × L1 = 2, MINOR.
   `seedvr2_neo/run_results.py:92` (`rpartition("\n")` without stripping) gives an empty `params_line`, so
   `with_params` puts the SeedVR2 keys on a new line; Forge then takes `Steps/Sampler/Seed` as part of the prompt.
   Input: `"a cat\nSteps: 20, Sampler: Euler, Seed: 1\n"` (also `\r\n`, `" \n"`, `"\n  "`) → prompt
   `"a cat\nSteps: 20, Sampler: Euler, Seed: 1"`, no `Seed` key. Forge's `create_infotext` strips its result, so this
   needs another extension that changes `processed.infotexts`. Tests:
   `test_forge_infotext_ending_with_a_line_break_still_extends_forge_params_line` (5 cases).

## Observations (no test failure)
- `_quote` (`run_results.py:124`) quotes only `,`, `:`, `\n`, `"`; `\r`, U+2028 and NUL stay unquoted. Forge splits on
  `"\n"` only, so it parses fine; a third-party reader that uses `str.splitlines()` would split the line.
- `to_pipeline_image` raises `TypeError` on `info["transparency"] = None`; Pillow decoders never set that, and the run's
  `except` in `_upscale_images` would return all inputs unupscaled. Not tested as a bug.
- Pillow's own `RGBA.convert("La")` drops alpha (not our code); tests build La images with `Image.new`.
- `merge_run_images` treats key `True` as index 1 (`True == 1`); the script only passes int indices.
- Values with leading/trailing spaces lose them on Forge's read-back (Forge's regex); accepted.

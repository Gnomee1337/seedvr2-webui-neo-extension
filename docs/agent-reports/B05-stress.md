# B05 stress test — model discovery, DiT size from the header, `find_model_file` patch

Run: `uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q -p no:cacheprovider`
Result against the committed code: **147 passed, 1 skipped, 2 xfailed** (was 46 passed); against the working
tree as last seen: 145 passed, 2 failed (pre-existing tests, old function name — see Observations). 104 new collected cases from 49 new test functions.
The skip is the trailing-space folder name test (POSIX only: Windows strips trailing spaces).
Symlink and junction tests ran (not skipped) on the Windows test machine.
`gguf` in the test environment: 0.19.0 (Forge Neo's bundled copy may be a different version).

## Tests added

| File | New | Scenarios |
|---|---|---|
| `tests/seedvr2_neo/test_model_discovery_stress.py` | 23 functions | directories named `*.safetensors` / `*.gguf`; wrong / partial-download / upper-case extensions; zero-byte models; folder holding only wrong extensions does not hide the fallback; 1500 models + 500 other files; file vanishing between `listdir` and `isfile`; unreadable model folder (fallback used) and unreadable parent (no folder, note names the expected path); symlinked file listed, broken symlink not; model folder reached through a junction/symlink; near-miss folder names (`Seed VR2`, `SeedVR2.bak`, Cyrillic look-alikes, full-width letters); trailing space (POSIX); case variants; `models_path` that is a file; a *file* named `SeedVR2` in the Forge folder; Unicode/emoji/spaces in the parent path; selections `None`, `""`, blank, `../` traversal, upper-cased name, VAE as DiT, swapped, trailing space, 5000-char name, emoji; file / whole folder deleted after the scan; `keep_or_default_*` with `None`, `""`, list, `0`, empty scan |
| `tests/src/utils/test_dit_size_stress.py` | 21 functions | safetensors: zero-length header, max u64 length, length equal to file size, one byte past EOF, `{}`, `null`, string, number, metadata only, BOM, whitespace, truncated JSON; declared length past EOF does not allocate (tracemalloc < 1 MB); length over the 64 MB cap inside a 64 MB+ file rejected unread; 200 000-deep nested JSON; block index with 5000 digits; Unicode tensor names; 50 000 non-block tensors; `double_blocks.` / `single_blocks.` not counted; GGUF content named `.safetensors` and vice versa; `.GGUF` upper case; directories named like models; checkpoint path `None`, `""`, NUL char; empty model name; folder part in the name. GGUF: absurd tensor count, KV count, key length (2^63, 2^64−1), tensor name length, dims count, non-UTF-8 name, unknown value type, unknown version, missing counts — all fall back fast with < 4 MB peak; absurd array length (xfail); file can be deleted after a successful / failed GGUF read and after a failed safetensors read (Windows locks); foreign-writer JSON layout (indent, padding, prefix) |
| `tests/src/utils/test_constants_stress.py` (new) | 5 functions | file deleted from the chosen folder → its path returned, no exception; directory named like the model is not returned as the file; `fallback_dir` `None` / `""`; Unicode and spaces in the folder; `./models/SEEDVR2` being a file (xfail) |

All other hostile cases behave as required: no exception, the name fallback or "unknown" is returned.

## Findings

### 🟡 R4 (S4×L1) — corrupt GGUF array length makes the header read grow without bound
- Where: `src/utils/dit_size.py:87` (`gguf.GGUFReader(path)`); root cause in the `gguf` package,
  `gguf_reader.py` `_get_field_parts`, the loop `for idx in range(alen[0])` (line 247 in gguf 0.19.0).
  The same reader is used at load time, `src/core/model_loader.py:170`, so `dit_size` does not add new exposure,
  and a fix only in `dit_size` would not protect the load.
- Input: a 61-byte `.gguf` with one key/value of type ARRAY(UINT8) whose declared length is N.
  Past EOF each element read returns a zero-size view and the offset stops moving, so the loop runs N times
  and keeps a view per element.
- Wrong result: time and memory grow with the declared N, not the file size: N = 10^4 → ~15 MB / 0.7 s,
  10^5 → ~73 MB / 2.8 s, 10^6 → ~650 MB / 23 s. A corrupt value near 2^63 hangs the WebUI and exhausts RAM
  instead of falling back to the file name.
- Likelihood is low: a truncated download keeps the header intact; this needs corruption inside the KV section.
- Test: `test_gguf_with_absurd_array_length_does_not_grow_memory` (xfail, strict).
- Possible fix: before `GGUFReader`, walk the KV section with bounds checks (array length × minimal item size must
  fit in the remaining bytes) and raise early; use the same pre-check before the loader's `GGUFReader`.

### 🟢 R2 (S2×L1) — `find_model_file` raises when `./models/SEEDVR2` in the working directory is a file
- Where: `src/utils/constants.py:100-101` (`get_all_model_files`: `os.path.exists(path)` then `os.listdir(path)`),
  vendored code reached from `find_model_file` when the file is missing from the caller's folder.
- Input: the chosen file disappeared from the model folder between `run()`'s scan and the load, and the launcher's
  working directory contains a *file* (or an unreadable folder) at `models/SEEDVR2`.
- Wrong result: `NotADirectoryError` (or `PermissionError`) traceback in the run instead of the one-line error.
- Test: `test_cwd_models_entry_that_is_a_file_does_not_raise` (xfail, strict, `raises=NotADirectoryError`).

## Observations (not bugs, no failing test)
- Race between `run()`'s rescan and the pipeline load: if the file vanishes after the scan, `find_model_file`
  silently loads a same-named copy from `<cwd>/models/SEEDVR2` (the upstream lookup, asserted by an existing test).
  Same file name, so practically the same model. S1×L1.
- `seedvr2_neo/model_discovery.py` was being changed in the working tree while this ran ("first complete folder
  wins", `find_subfolder_ignoring_case` → `find_subfolders_ignoring_case`). All new tests pass against both versions.
  Against the new version two **pre-existing** tests fail only because they call the old name:
  `tests/seedvr2_neo/test_model_discovery.py::test_exact_spelling_wins_over_other_case` and
  `tests/seedvr2_neo/test_model_discovery_stress.py::test_file_named_like_the_folder_is_ignored`
  (left for the main chat; a VAE-only Forge folder vs a complete extension folder is also worth a test there).
- `describe_missing_models` puts the folder path in Markdown backticks; a path containing a backtick breaks the
  formatting. Cosmetic, S1×L1.
- Zero-byte or non-model files with a model extension are listed and pass `find_selection_error`; they fail later
  at load. `choose_dit_size` itself falls back to the name for them without raising.
- The safetensors length check compares with the whole file size, not file size − 8; harmless: the short read then
  fails the JSON parse or yields the real header.

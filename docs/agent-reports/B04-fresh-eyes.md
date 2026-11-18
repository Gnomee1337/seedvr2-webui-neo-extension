# Fresh-eyes review — B04 repo hygiene (`git diff -M main...HEAD`, branch `feature/b04-repo-hygiene`)

## How I understood the result
The vendored pipeline folder `src/` is moved to `seedvr2_vendor/` (pure renames; inside imports were already relative).
Two path fixes go with it: the VAE yaml path in `model_configuration.py` is now built from the package's own folder, and the
ComfyUI-only root `__init__.py` imports `.seedvr2_vendor`. The pre-run "foreign `src` package" check (`seedvr2_neo/package_check.py`,
its tests, the README warning) is deleted. All tracked `.pyc` are untracked, the unused `download_weight` import is gone,
`requirements.txt` is an informational list of the packages actually imported, and `LICENSE` (stock Apache-2.0) and `NOTICE`
(credits + list of changes) are added. Docs follow the rename. A new test greps for old-name references.
This matches the task; nothing extra found in code. Note: the working tree has uncommitted edits (`.gitignore`, two
`seedvr2_neo` docstrings) made while I was reviewing; this review covers the committed `HEAD` (`b1a5b39`).

## Findings

### IMPORTANT

**1. The "don't install another SeedVR2 extension" warning is gone, but the main clash with the original extension is still there.**
`README.md` (removed lines after line 27); `scripts/seedvr2_script.py:42`.
Our Script title is `"SeedVR2 Native Upscaler (24G)"`, the same as in akoyaki's original extension (the first commits in this repo's
history). This fork's likeliest neighbor is that original extension. The rename fixes the `src` clash, but two Scripts with one title
remain in the Script dropdown. In A1111-derived code the dropdown is index-based, so one of the two is probably unreachable or
mixed up (not checked against Neo). Each copy also keeps its own `GlobalModelCache`, so models are held twice in VRAM/RAM.
Scenario: a user who had the original installed, then added this fork, picks "SeedVR2 Native Upscaler (24G)" and silently runs the
old code. Before, they were told not to do this. Now nothing tells them.
Fix: keep a shorter README warning (one SeedVR2 Script extension at a time), or give our Script a distinct title. Note that a new
title resets saved `ui-config.json` values for the script.
R = S2 × L2 = 4 🟡

**2. A manual `git pull` update can fail on the deleted `.pyc` files.**
All `cpython-311.pyc` files that are deleted (e.g. `src/core/__pycache__/*.pyc`, `scripts/__pycache__/seedvr2_script.cpython-311.pyc`).
After a clone, every source file has a new mtime. So Python 3.11 finds each tracked `.pyc` stale and rewrites it in place, and
the tracked file now shows as modified. A merge that deletes a locally modified tracked file is refused:
`error: Your local changes to the following files would be overwritten by merge ... Please commit your changes or stash them`.
Forge's own "Check for updates → Apply" does fetch + `reset --hard` in A1111-derived code, so it is not affected (not verified
for Neo). Users who update by hand with `git pull` (common for people who installed by `git clone`) are stuck with a scary error
that names dozens of files.
Fix: a one-line note in the README / release text: "if `git pull` complains about `.pyc` files: `git reset --hard origin/main`".
Alternatively confirm Neo's updater and accept the risk. Add it to `07-RISKS.md` either way.
R = S2 × L2 = 4 🟡

**3. The LICENSE covers the whole repository, including code derived from an extension that the block card itself says has no license.**
`NOTICE:22`, `LICENSE`, `docs/05-DECISIONS.md` ("The whole repository is Apache-2.0").
The B04 card lists as a hidden assumption: "The original Forge extension (first commits in history) has no license file".
`scripts/seedvr2_script.py` grew out of that code. Code with no license is all-rights-reserved by default, so this repo cannot grant
Apache-2.0 on akoyaki's part. The NOTICE line "based on the original SeedVR2 extension for Forge by akoyaki ayagi" credits the
author but says nothing about the terms. A reader will take it that akoyaki's code is Apache-2.0 too. This is a public statement,
and once it is on `main` it is hard to take back. It does not break anything for users.
Fix: an explicit owner decision before the push. Options: check the original repo for a license; ask the author; or state in
NOTICE that the original extension was published without a license. Record whichever is chosen in `05-DECISIONS.md`.
R = S2 × L2 = 4 🟡 (legal, not a runtime risk; rated by how hard it is to retract)

### MINOR

**4. Updated installs keep an orphan `src/` folder.** After the update (by `reset --hard` or `pull`), git deletes the tracked
files but not untracked ones. These are `.pyc` for other Python versions (e.g. `cpython-310/313`) and for modules added after the
`.pyc` were committed (e.g. `src/utils/__pycache__/dit_size.cpython-311.pyc`). So `extensions/<ext>/src/**/__pycache__/` survives.
It has no `__init__.py`, so at most it becomes a namespace package. A regular `src` from another extension still wins over it,
and our code no longer imports `src`, so it is harmless. Still, a user (or a later bug report) will see a "src" folder that the docs
say no longer exists. Fix: a README line saying it can be deleted. R = S1 × L3 = 3 🟢

**5. A failed pipeline import is now noticed only after the base image is generated and the SD checkpoint unloaded.**
`scripts/seedvr2_script.py:166-208`. The removed `find_shadowed_package_error()` was the only import done before generation.
`except ImportError` at line 207 returns `Error: <e>` with no console line (every other error path prints `❌ [SeedVR2] ...`).
In txt2img it also returns an empty result instead of the base image (which is still saved to disk by `process_images`).
For the package itself, losing the early check changes almost nothing (it can hardly be missing now). But the realistic case,
`rotary_embedding_torch` missing because `install.py` failed (the README mentions it), always took this late path, and it is
now the only path. Non-`ImportError` exceptions at import time escape `run()` entirely. Most of this is older than B04; the gap
is now plain to see. Fix (later block): import the pipeline modules at the step-0 check, print the error.
R = S2 × L1 = 2 🟢

**6. Docs still say the root `__init__.py` is untouched.** `docs/01-VISION.md:24` ("kept untouched"),
`docs/02-ARCHITECTURE.md:118` ("keep untouched"). It now carries a local patch (`__init__.py:6`). `05-DECISIONS.md` was updated
("only follows the package rename"), so the three docs now disagree. `docs/04-STATE.md:13,18` still describe the `src` check and
`src/` (expected to change on `/block-done`, listed so it is not forgotten). R = S1 × L2 = 2 🟢

**7. `gguf` origin reads as a contradiction.** `requirements.txt:2` says "Everything except rotary-embedding-torch comes with Forge
Neo itself". `README.md:13-18` tells the user to `pip install ... gguf` and says "Forge Neo does not install it as a separate package".
Both are true (Neo bundles it in `modules_forge/packages`, not as a pip package), but a reader without context sees two opposite
statements. Fix: say "bundled with Forge Neo (not as a pip package)" in one of the two places. R = S1 × L2 = 2 🟢

**8. NOTICE says `requirements.txt` is a vendored file, while `docs/external-apis/seedvr2-upstream.md:12` says it is "Ours".**
`NOTICE:17` lists "requirements.txt lists only the packages the code uses" under "Changes to the files above". `requirements.txt`
is not in the "files above", and the upstream reference calls it ours. Pick one; the simplest is to drop the line from NOTICE.
R = S1 × L1 = 1 🟢

**9. The old-name guard test is narrower than its docstring.** `tests/seedvr2_vendor/test_package_paths.py:12`.
The docstring says "nothing may still reach it by the old name". The regex catches `from/import [.]src` at line start and string
literals starting with `src/` or `src\`. It misses `os.path.join(root, 'src', 'models')`, `importlib.import_module("src.x")`,
`from ..src`, and non-`.py` files (yaml `__object__.path`). The likeliest way an old name comes back is a future upstream sync,
and upstream's own path style was exactly `'src/models/...'` (caught). So the gap is small. Fix: narrow the docstring or add
`['"]src['"]` and `["']src\.` alternatives. R = S1 × L1 = 1 🟢

**10. The clash moved to new names rather than going away, and nothing detects it now.** `seedvr2_neo` and `seedvr2_vendor` are
still top-level packages on a shared `sys.path`. Two copies of this extension, or a fork of this fork, in one Forge would silently
share the first-loaded copy. That is far less likely than the old `src` clash, so dropping the check is reasonable. Worth one line
in `07-RISKS.md` so the trade-off is recorded. R = S2 × L1 = 2 🟢

**11. (unverified) In-process UI reload after the update orphans the old cache.** If Forge is launched in a way that is not
"restartable", "Apply and restart UI" re-executes scripts in the same process. The old `src.core.model_cache` stays in
`sys.modules` with any DiT/VAE it cached, while the new code uses `seedvr2_vendor`'s own cache. Before the rename the same module
was reused. The effect is several GB of RAM/VRAM held until a full restart, once per update. This depends on how Neo implements
the restart, which I did not check. R = S2 × L1 = 2 🟢

### Already being fixed (uncommitted in the working tree during review)
- `.gitignore:23` comment still said `src/models/`; `seedvr2_neo/__init__.py` and `seedvr2_neo/model_discovery.py` docstrings
  still said `src/`. Committed `HEAD` has them; the working tree has the fixes. Make sure they land in the commit.

## Checked and fine
- No absolute `src.` imports remain in `.py` files; YAML configs use registry keys (`dit_3b.nadit`, `video_vae_v3...`), not package paths.
- `get_script_directory()` (three levels up) still resolves to the repo root: `seedvr2_vendor/utils/constants.py` is at the same depth.
  `pos_emb.pt`, `neg_emb.pt`, `configs_*` paths are unchanged.
- The VAE yaml path patch resolves to `seedvr2_vendor/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml`, and the file exists (a test covers it).
- The removed `seedvr2_vendor.utils.downloads` import had no needed side effects (the module only defines functions).
- `requirements.txt` matches the third-party top-level imports (plus `Pillow` for `PIL`). `peft` and `matplotlib` are imported nowhere.
- `LICENSE` is the stock Apache-2.0 text.

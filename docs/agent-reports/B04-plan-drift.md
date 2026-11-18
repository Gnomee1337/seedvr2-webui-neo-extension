# B04 plan-drift report

## Percentage
Total weights: B04 2 + B05 2 + B06 4 + B07 3 + B09 3 + B08 2 + B01 3 + B02 3 + B03 1 = 23 (matches roadmap).
Before: B01 3 + B02 3 + B03 1 + B05 2 = 9 -> 9/23 = 39.1% -> 39% (matches 03-ROADMAP.md:25 and 04-STATE.md:27).
After B04 done: 9 + 2 = 11 -> 11/23 = 47.8% -> 47%. If B04 is not closed: stays 39%.

## REQ-5 (01-VISION.md:37)
Criteria: English docs, README model paths match code, LICENSE + attribution. LICENSE and NOTICE exist, README paths
(README.md:22) match B05 code, `git ls-files | grep -c .pyc` = 0. Verdict: complete -> change 🔄 to ✅.
Caveat: B04 criterion "every modified upstream file carries a change notice" is met via "Local patch (Forge Neo)" markers
(07-RISKS/upstream doc line 16); the rename itself changes no file content (documented in seedvr2-upstream.md:23).

## Gaps
1. docs/03-ROADMAP.md:31 B04 row is ⬜ "everything except UTF-8 .gitignore": set ✅ / "—"; :25 Summary -> 11 of 23, 47%;
   :54 closed sum -> add B04 2 = 11.
2. docs/01-VISION.md:37 REQ-5 🔄 -> ✅.
3. docs/04-STATE.md: "Where we stopped" still says B05 closed; :18 says `src/` matches upstream (now `seedvr2_vendor/`);
   :27 39% (9 of 23) -> 47% (11 of 23), "4 blocks open"; :32 Next steps still lists B04 -> remove; next is B06.
   Tests count (159 passed) -> 254 passed.
4. docs/roadmap-blocks/B04-repo-hygiene.md must be deleted on close (protocol §11); pitfalls already in 05-DECISIONS.
5. Stale `src/` in code docstrings: seedvr2_neo/__init__.py:3 and seedvr2_neo/model_discovery.py:5 ("vendored pipeline (`src/`)") -> `seedvr2_vendor/`.
6. Live check gap: card names 3B GGUF in img2img; only 3B fp8 ran live after the rename. Record as known deviation
   (GGUF load path is unchanged by the rename; the owner accepted fp8) or run GGUF before ✅.
7. 01-VISION.md:23 and 05-DECISIONS already use `seedvr2_vendor` - consistent. 02-ARCHITECTURE.md:25 correct. No other
   stale `src` references found (remaining ones are intentional "upstream `src/`").
8. Roadmap "Why this order" (03-ROADMAP.md:57) still says "B04 is small and removes noise before real changes" - fine, but update to past tense if desired.

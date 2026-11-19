# Idea: VAE tile size 512 as the default (instead of 1024)

## Recommendation
**Take it into the roadmap as its own small block (B10, weight 1), before B08.** It closes the only 🟠 risk (R9)
with a one-constant change whose target value (512 / overlap 128) is already live-checked clean on the target GPU.
Its effect has a limit: unless Forge Neo behaves differently from A1111 (needs a fact check, see below), the new
default only reaches **new installs**. Existing users keep 1024 unless we also change the slider's saved-settings
key, and the owner has to choose whether we do that.

## 1. Already there?
- **Code:** `seedvr2_neo/upscale_settings.py`, `TILE_SIZE = Bounds(512, 2048, 1024, 128)`. The same bounds build the
  slider (`scripts/seedvr2_script.py`, `gr.Slider(label="Tile Size", **slider_range(TILE_SIZE))`) and clamp API
  values. Changing the default is one number, and 512 is also the slider's **minimum**.
- **README.md:74** already recommends 512 for 8–11 GB GPUs. `07-RISKS.md` R9 names "tile 512 as the default, or a
  VRAM-based default" as the way to close it.
- **Upstream:** the ComfyUI node defaults to 1024/128 (`seedvr2_vendor/interfaces/vae_model_loader.py`), but the
  node's Python function signature defaults to 512/64. 512 is a value upstream itself uses.
- **B08 (settings and feedback, ⬜, weight 2):** this is related but does not overlap. B08 covers color correction,
  attention mode, the progress bar, interrupt inside a phase and the title. Interrupt inside a phase is what would
  let a user stop an R9 stall. That complements this idea; it does not replace it.
- **Idea "Settings in Forge's Settings tab"** (06-IDEAS, mentions "default tile size as Forge options"): this one is
  complementary. A `shared.opts` default would let a user pick their own default, but a slider value saved in
  `ui-config.json` would still override it. Leave that idea open.
- **Rejected before:** no. The "Rejected by the owner" list is empty, and no decision in 05-DECISIONS fixes 1024. The
  closest is "BlockSwap is a manual slider…, not an automatic setting", which argues against a VRAM-based default
  (see the options below).
- Side note: `04-STATE.md` on `main` still describes B07 as the last block (78%), while the roadmap says B09 is
  closed (91%). STATE needs updating.

## 2. Fit with the vision
It fits well. "In scope: VRAM management … VAE tiling" and REQ-4 ("fits 8 GB GPUs without bf16"). REQ-4 was accepted
**with tile 512** (01-VISION footnote 3), so today's default is not the configuration that met REQ-4. This idea
lines the default up with what was accepted.

## 3. Weight and options
| Option | What | Weight | Reaches existing users? |
|---|---|---|---|
| A. Note only | Already done (README) | 0 | no; R9 stays 🟠 |
| B. Default 512 | `TILE_SIZE.default = 512`, overlap stays 128; tests in `tests/seedvr2_neo/` (parse default, slider default); README and 07-RISKS rewritten; one live re-measure | 1 | **new installs only** (if the ui-config fact holds) |
| C. B + new saved-settings key | B, plus a slider label change (e.g. "VAE Tile Size"), so Forge's stored `…/Tile Size/value` no longer applies and everyone gets 512 once | 1 | yes, once; a value a user set on purpose is reset once |
| D. VRAM-based default | Default 512 if total VRAM ≤ ~12 GB, else 1024; read the device when the UI is built | 2 | same ui-config problem as B; the first value seen gets frozen |

**Recommended: C**, if the owner accepts a one-time reset of the tile slider; otherwise B. Reasons:
- D adds device probing at UI build time and harder tests, and it goes against the "manual, not automatic" stance
  of the BlockSwap decision.
- On large GPUs 512 costs only some tile overhead. Even on 8 GB, 512 *encoded faster* (5.7 s vs 16 s).
- Overlap: keep **128**. 512/128 is the configuration already measured clean (07-RISKS §3 header, REQ-4 footnote:
  1080px and 2048px outputs). 64 (upstream's function default) saves little and adds seam risk without being measured.
- Do not lower the minimum below 512 in this block. Default = minimum is acceptable. A lower minimum is a separate
  question (and stored slider min/max may also be frozen by ui-config, same fact check).

## 4. Dependencies
- **Fact check before code (fact-checker, Neo `modules/ui_loadsave.py` + API default script args):**
  1. Does Neo write every component's current value (and slider minimum/maximum/step) into `ui-config.json` on startup?
     A1111 does this, so code defaults stop applying after the first launch.
  2. Is the key `customscript/seedvr2_script.py/<tab>/<label>/value`?
  3. When an API call omits the script args, does it fill them from the components' values after ui-config was applied?
  Unverified, from knowledge of A1111; `external-apis/forge-neo.md` only covers dropdowns.
- **Live measurement** (Claude through Forge `--api`, owner's 8 GB no-bf16 GPU, 3B GGUF Q6_K): the R9 case
  (img2img 512×384 → 1024 shortest edge) at 512/128, **with and without `--cuda-malloc`**: no stall, times for encode
  and decode, image clean. Also confirm that a fresh `ui-config.json` gives 512 and, for option C, that an old
  ui-config with `Tile Size = 1024` no longer applies.
- No block has to come first. It does not break B08. If B08 later adds "interrupt inside a phase", R9's leftover
  (the R6 spill) becomes stoppable.

## 5. Risks (R = S × L)
- 🔴 R12 (S3×L4) **For option B: the fix does not reach existing users.** Anyone who launched the extension once has
  1024 in `ui-config.json` (if fact 1 holds), so the R9 stall stays for them. Option C mitigates it, as does a README
  line: "delete the Tile Size entry in ui-config.json or set 512".
- 🟡 R4 (S2×L2) **R9 is not fully closed.** With `--cuda-malloc` the decode still spills at 512 (R6: 25–137 s; a batch
  multiplies it). 512 turns a "stall for many minutes" into a "slow decode". After the change R9 should be rewritten
  into R6, not just deleted.
- 🟢 R3 (S1×L3) **Slower on large GPUs** (more tiles, more overlap work). Not measured on 16–24 GB, expected to be
  small; a user can raise the slider.
- 🟢 R2 (S2×L1) **Seams from more tile boundaries.** 512/128 was already live-checked clean at 1080 and 2048 px.
- 🟢 R2 (S1×L2) **Option C resets a value a user set on purpose**, once. This cannot be told apart from the auto-saved
  1024. The API is not affected: script args are positional, so a label change does not touch them.
- 🟢 R1 **API callers who pass explicit values** are unaffected by design (their value wins, clamped as before).
- No data risk: no files, no checkpoint. Only VRAM and time.

## 6. Percentage
Now 21 / 23 = **91%**.
- As its own block (weight 1): 21 / 24 = **87%** when added, **91%** (22/24) when closed.
- Folded into B08 (2 → 3): also 87% now, but it is counted only when all of B08 closes. That delays a 🟠 fix behind
  polish, so I do not recommend it.

## 7. Questions for the owner
1. Should the new default reach **existing** users too (option C: a slider label change, which resets one saved
   value once) or only new installs plus a README note (option B)?
2. Is 512 as both the default and the minimum fine, or do you want room below 512 (a separate question)?
3. Should this be a separate block B10 before B08 (recommended) or part of B08?
4. Can Claude re-measure the R9 case on your Forge, with and without `--cuda-malloc`, before merging?

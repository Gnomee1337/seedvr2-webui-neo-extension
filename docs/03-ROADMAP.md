# Roadmap

Blocks of work, their weight (how much work, 1–5) and status. How each process works —
`docs/09-PROCESSES.md`; why we decided so — `docs/05-DECISIONS.md`; block cards of open
blocks — `docs/roadmap-blocks/`.

## How the percentage is calculated

```
PROGRESS = (sum of weights of ✅ blocks) / (sum of weights of ALL blocks) × 100, rounded down
```

- A block counts **only as a whole**: all acceptance criteria met and agent checks passed.
- "Code written and tests green" is not ✅ yet. A block that works but that the owner has not
  confirmed with a live example stays 🔄.
- A new idea = a new block with a weight; the percentage honestly drops (the project grew, it did not regress).
- The percentage is recalculated by the `plan-drift-checker` agent, not the main chat.

**Statuses:** ✅ closed · 🔄 in progress or awaiting live check · ⏸ done but switched off · ⬜ not started.

**Weight:** 1 — up to an hour or two · 2 — half a day · 3 — a day · 4 — a couple of days · 5 — a week or more / many unknowns.

## Summary

**Sum of weights: 23. Closed: 18. PROGRESS: 78%.** (Counted by `plan-drift-checker` on closing B07.)

## Delivery requirements (REQ-1…REQ-6, details — `docs/01-VISION.md`)

| # | Block | Req. | Weight | Status | What remains |
|---|---|---|---|---|---|
| B04 | Repo hygiene: `.pyc` out of git, unused import, `requirements.txt`, LICENSE + attribution, `src` package name | REQ-5 | 2 | ✅ | — |
| B05 | Install and model discovery: one model folder resolved in one place, refresh button, empty-folder message, model size not by file name | REQ-1 | 2 | ✅ | — |
| B06 | VRAM and RAM lifecycle: unload SeedVR2 after run, real Force Reload, release reserved VRAM between phases, Forge device, context reset on error, pipeline import before any work (before the base image and the SD unload), BlockSwap/offload in the UI and a 7B warning on 8 GB GPUs | REQ-3, REQ-4 | 4 | ✅ | — |
| B07 | Correct results: every image of a batch (not the grid), RGB conversion, infotext, respect save settings, empty-result guard | REQ-2 | 3 | ✅ | — |
| B09 | Tests with Forge stubs + GitHub Actions (ruff + pytest) | REQ-6 | 3 | ⬜ | all |

## UX

| # | Block | Weight | Status | What remains |
|---|---|---|---|---|
| B08 | Settings and feedback: color-correction and attention-mode choice, Forge progress bar, interrupt inside the DiT step, title without "(24G)" | 2 | ⬜ | all |

## Foundation

| # | Block | Weight | Status |
|---|---|---|---|
| B01 | Vendored SeedVR2 pipeline (ComfyUI port v2.5.24) runnable outside ComfyUI | 3 | ✅ |
| B02 | Forge Script: UI, txt2img/img2img flow, model caching | 3 | ✅ live-checked by the owner on current Forge Neo: 3B GGUF, 3B fp8, 7B fp8 |
| B03 | Project moved onto the protocol: English docs, translated README and comments | 1 | ✅ |

## How to check the sum
Add up the "Weight" column across all tables — it must match "Sum of weights" in "Summary":
B04 2 + B05 2 + B06 4 + B07 3 + B09 3 + B08 2 + B01 3 + B02 3 + B03 1 = **23**.
Add up the weights of ✅ rows — that is "Closed": B01 3 + B02 3 + B03 1 + B04 2 + B05 2 + B06 4 + B07 3 = **18**.

## Why this order
B05 first: a wrong model folder leaves users with empty dropdowns, the most common first-run failure. B04 removed noise
before real changes. B06 and B07 are the main user-visible problems (7B unusable on 8 GB and memory held after a run —
B06; wrong image in batches — B07).
B09 can start in parallel with B06/B07 — the script logic is easier to fix with tests around it. B08 is polish.

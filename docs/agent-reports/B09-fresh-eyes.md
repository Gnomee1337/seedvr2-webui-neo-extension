# B09 — fresh-eyes-reviewer (saved by the main chat; the agent returned it as its reply)

No user-visible behavior change: the safety-net tests (`tests/scripts`, unchanged since 40f7216) pass on `main`'s
script and on HEAD; every branch of old and new code compared.

Silently changed: result images are all built first, then saved (`build_result`); if building image k raised, the old
code had saved files 0..k-1, the new one saves none — 🟢 R1. `enable_pnginfo` read once per result — not observable.

Tests that stayed green on broken code (mutation probes): `ui()` order swapped (only the count was checked) — 🟡 R4;
Forge wiring (`release_memory` no-op, no `torch_gc` after SD unload, wrong device) — 🟡 R6; `-original` infotext — 🟢 R2;
failing-save test failed every save — 🟢 R2.

Unclear: a comment pointing to a decision instead of the reason (`upscale_run.py`); "stopping … removed on purpose"
without a reason and with a wrong attribute name (Forge's is `stopping_generation`) — 🟢 R3.

Hidden assumptions: `from conftest import` relies on one conftest under `tests/` — 🟢 R2; real `seedvr2_vendor`
parent packages still imported by the stand-ins — 🟢 R1; unpinned CI packages, push + pull_request double runs — 🟢 R2;
"Reload UI" after a manual `git pull` keeps old `seedvr2_neo` modules → ImportError of new names until a restart — 🟢 R3.

Missed: `09-PROCESSES.md` process 5 steps 5, 8, 10 name no module — 🟢 R3; one blank line before `forge_hooks`.

Incident: the agent's scratch folder for mutation probes collided with another agent's; the project folder was not
touched (checked by the main chat: the diff holds only intended changes).

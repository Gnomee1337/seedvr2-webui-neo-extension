# Project: seedvr2-webui-neo-extension — a Forge Neo extension that upscales images with SeedVR2

Follow the protocol strictly: @docs/process/protocol.md

Current project state (loaded automatically at every start):
@docs/04-STATE.md

Documentation map — where to look (read files from it ONLY when actually needed):
@docs/README.md

## First action in every new session
Report briefly (no more than 8 lines): where we stopped, what is next, whether you have questions for me.
Then wait for a command.

## Project specifics

- **Status:** in development, **published**: users install it from GitHub into their own Forge Neo, so every push
  to `main` reaches people on their next "Check for updates". No server, no deploy.
  An error breaks their txt2img/img2img run or leaves VRAM occupied — never their files or the SD checkpoint.
  **Nothing is pushed or merged into `main` without the owner's explicit consent**; work goes to `feature/*` / `fix/*`
  branches. Before any push — `git status` and a look at the file list and the diff: no personal data, no local paths,
  no models, no `__pycache__`.

- **Structure:** `scripts/seedvr2_script.py` — the only Forge entry point (the Script class, UI and the
  txt2img/img2img flow). `seedvr2_neo/` — our helpers without Forge imports (model discovery, memory plan, model lifecycle).
  `tests/` — pytest (`python -m pytest tests`). `seedvr2_vendor/` — the vendored SeedVR2 pipeline (port of ComfyUI-SeedVR2_VideoUpscaler v2.5.24).
  `configs_3b/`, `configs_7b/` — model configs; `pos_emb.pt`, `neg_emb.pt` — precomputed text embeddings;
  `install.py` — dependencies installed by Forge at launch. Code paths in the docs are relative to the repository root.

- **Knowledge about Forge Neo** — in `docs/external-apis/forge-neo.md`; about the vendored upstream —
  `docs/external-apis/seedvr2-upstream.md`. Before "figuring out" anything about them — check there.
  A new verified fact — add it there. The Forge Neo source is not in this repository: check facts against
  the `neo` branch of `Haoming02/sd-webui-forge-classic`.

- **`seedvr2_vendor/` is vendored code.** Change it only when the fix cannot live in `scripts/`; every change to `seedvr2_vendor/` is
  listed in `docs/external-apis/seedvr2-upstream.md` ("Local patches"), otherwise it is lost on the next upstream sync.

- **Dangerous modules:** none act on other people's data. What hurts users: VRAM left occupied after a run,
  unloading their SD checkpoint, files saved into their output folders. No kill-switch flags, no `.env`.

- **Stack:** Python 3.10+ inside the Forge Neo process; PyTorch, Gradio (Forge's), OmegaConf, einops, diffusers,
  `rotary-embedding-torch`, `gguf`. Tests: `pytest` outside Forge (`requirements-dev.txt`), `tests/` mirroring the
  code folders; the script itself is not covered yet (Forge stubs planned). **Changed code — update or add tests in the mirrored
  folder in the same commit** (details — `docs/process/code-style.md`).

- **Language:** documentation, comments, docstrings, commit messages — English; communication with
  the owner — Russian. UI labels are English.

- **Branches:** simplified Git Flow (`main` + short `feature/*` / `fix/*`) — `docs/process/git-rules.md`,
  recorded in `docs/05-DECISIONS.md`.

- **No personal data of the owner** (hardware, names, contacts) in docs, code or commits — the repository is public.
  Hardware targets are described as a class ("8 GB GPU without bf16").

- **Secrets:** none in the project. `.env` and `_secrets/` are still in `.gitignore` and blocked in `.claude/settings.json`.
  Model weights (`models/`) are never committed.

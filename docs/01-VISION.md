# What we build — seedvr2-webui-neo-extension

## In one sentence
An extension that brings the SeedVR2 diffusion upscaler into Forge Neo's txt2img and img2img, so a user can
generate and upscale an image in one click without switching to ComfyUI.

## For whom
| Who | What they get |
|---|---|
| **Forge Neo user** | A SeedVR2 upscale of the generated or uploaded image right in the WebUI, with the result saved next to their other outputs. |
| **Users with 8–16 GB GPUs, including older ones without bf16** | The same, without out-of-memory errors: tiling, offloading and quantized (GGUF / fp8) models. |
| **Maintainer (owner)** | A codebase that is documented in English, testable and can be synced with the upstream ComfyUI port. |

## In scope
- SeedVR2 3B and 7B models, `.safetensors` and `.gguf`, on a single GPU.
- txt2img (generate, then upscale) and img2img (upscale the input image) via the Script dropdown.
- VRAM management: unloading the SD checkpoint, VAE tiling, model caching / unloading between runs.
- Installation through Forge's extension installer (`install.py`).

## Out of scope
- **Video upscaling** — SeedVR2 supports it, but Forge Neo is an image UI; the pipeline runs with batch size 1 and no temporal overlap.
- **Multi-GPU** — Forge runs on one device.
- **ComfyUI support** — ComfyUI has its own SeedVR2 node pack (the upstream). `seedvr2_vendor/interfaces/` and the root `__init__.py`
  (ComfyUI entry points) are kept untouched only to make diffing with upstream easy; they are not maintained or tested here.
- **Training / fine-tuning SeedVR2.**
- **Automatic model download** — under discussion, see `06-IDEAS.md`.

## Signs of completion
Confirmed by the owner. The project is delivered when all requirements are met:

| # | Requirement | Now |
|---|---|---|
| REQ-1 | **Clean install** — on a fresh Forge Neo the extension installs from URL, and both a `.safetensors` and a `.gguf` model upscale without manual `pip` | ✅ ¹ |
| REQ-2 | **Correct results for every image** — batch size / batch count > 1 in txt2img upscales every image (not the grid); grayscale/RGBA inputs work; the infotext of the result contains the generation parameters | ⬜ |
| REQ-3 | **VRAM is given back** — after an upscale the next SD generation on an 8 GB GPU runs without OOM; "Force Reload" really reloads the models | ⬜ |
| REQ-4 | **Fits 8 GB GPUs, including ones without bf16** (RTX 20xx / GTX 16xx class) — a 3B GGUF model upscales a 1024px image 2× without OOM and without visible artifacts; if 7B cannot work there, the UI says so instead of failing | ⬜ |
| REQ-5 | **English docs and correct instructions** — README model paths match the code; LICENSE and upstream attribution present | 🔄 |
| REQ-6 | **Automated tests in CI** — the script's logic is covered by tests that run on every push | ⬜ |

Statuses: ✅ done · 🔄 partial / awaiting confirmation · ⬜ not started.

¹ Accepted by the owner on a dependency check, not a fresh install: every runtime package comes from Forge Neo's pinned
`requirements.txt`, its bundled `modules_forge/packages/gguf`, or our `install.py` (`rotary-embedding-torch`), and
`.safetensors` + `.gguf` upscale on a Forge Neo with those pinned versions. Not run: `install.py` on a fresh install.
Completion percentage and blocks — `docs/03-ROADMAP.md`.

# Forge Neo — host API reference

Facts about the Forge Neo internals this extension calls. Each fact has a source.
Source used below: `Haoming02/sd-webui-forge-classic`, branch `neo`, commit `a31a8ba` — "verified in source".
Forge Neo changes often; re-check a fact against the current `neo` branch before relying on it in new code.

## Access
Public GitHub repository, no keys.

## Common pitfalls
- **A plain `import gguf` works inside Forge Neo without the pip package.** Neo ships its own copy in
  `modules_forge/packages/gguf` (used by `backend/loader_gguf.py`; `requirements.txt` has no `gguf`) and puts
  `modules_forge/packages` first on `sys.path` at startup (`modules_forge/initialization.py`). So extensions get the bundled
  copy; the venv's own Python, run outside Forge, does not see it. Source: verified in a Neo install's source and in a live
  run (a Q6_K `.gguf` loaded); re-check against the current `neo` branch before relying on it.
- **`requirements.txt` of an extension is not read.** The launcher runs only `<extension>/install.py`
  (`modules/launch_utils.py`, `run_extension_installer`); `preload.py` is also supported (`modules/script_loading.py`). Source: verified in source.
- **The extension root `__init__.py` is never imported**; only `scripts/*.py` are loaded as scripts. Source: verified in source (`modules/scripts.py`, `list_scripts`).
- **`process_images` returns the grid first.** With more than one image and `opts.return_grid`, `output_images.insert(0, grid)`
  and `index_of_first_image = 1` (`modules/processing.py`). Source: verified in source.
- Neo's requirements do not include `peft`, `rotary-embedding-torch`, `matplotlib`. Source: `requirements.txt`, verified in source.

## Limits and load
Single process, single device. The device Forge uses: `modules.devices.device = memory_management.get_torch_device()`.

## Scripts API
### `scripts.Script`
`title()`, `show(is_img2img)`, `ui(is_img2img)` → list of Gradio components, `run(p, *args)` → `Processed`.
The selected script is called by `ScriptRunner.run`: `script.run(p, *args[script.args_from:script.args_to])`. Source: `modules/scripts.py`.

### `processing.Processed(p, images_list, seed=-1, info="", ..., infotexts=None, ...)`
Without `infotexts`, every image gets `info` as its infotext. Source: `modules/processing.py`.

### `processing.create_infotext(p, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0, use_main_prompt=False, index=None, all_negative_prompts=None)`
Source: `modules/processing.py`.

### Saving
`StableDiffusionProcessing.save_samples()` is `opts.samples_save and not do_not_save_samples and (opts.save_incomplete_images or not interrupted/skipped)`. Source: `modules/processing.py`.

## Memory
- `modules.sd_models.unload_model_weights(*args, **kwargs)` → `memory_management.unload_all_models()`, replaces the model with
  `FakeInitialModel`, `soft_empty_cache()`, `gc.collect()`. The checkpoint is reloaded on the next generation. Source: `modules/sd_models.py`.
- `modules.devices.torch_gc()` → `memory_management.soft_empty_cache()`. Source: `modules/devices.py`.
- `backend.memory_management.free_memory(memory_required, device, keep_loaded=[])`, `unload_all_models()`,
  `get_torch_device()`. Source: `backend/memory_management.py`.

## State (interrupt and progress)
`shared.state.interrupted` (Interrupt button), `shared.state.stopping_generation` (Stop generating), both reset at job start;
`state.job`, `state.job_no`, `state.job_count` drive the progress text. Source: `modules/shared_state.py`.

## Paths and working directory
Base URL for sources: `https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/`.
- **`models_path`** is defined in `modules/paths_internal.py` as
  `models_path = cmd_opts_pre.model_ref or os.path.join(data_path, "models")`, with `data_path = cmd_opts_pre.data_dir`
  (defaults to the WebUI root). `modules/paths.py` re-exports it (`from modules.paths_internal import models_path, script_path, data_path, ...`).
  Both `modules.paths.models_path` and `modules.paths_internal.models_path` work.
  Source: `modules/paths_internal.py`, `modules/paths.py` (read via raw GitHub; the line was relayed by the page summariser, not read in full).
- **Override flag: `--model-ref`** ("Points to a central `models` folder that contains all your models"); there is **no `--models-dir`**
  in `modules/cmd_args.py`. Source: `README.md` (Commandline section, "by. Neo"). A `--data-dir` also moves `models`.
  So `models_path` may point outside the WebUI root; a hard-coded `./models/...` is wrong in that case.
- **Absolute or not:** `models_path` is absolute only when `data_dir` / `--model-ref` are absolute; the code shown does not
  call `abspath`. `--data-dir` and `--model-ref` are declared in `modules/paths_internal.py` itself;
  `--data-dir` defaults to `os.path.dirname(modules_path)`, i.e. the WebUI root, so by default `models_path` is
  `<WebUI root>/models` (checked in a local `neo` checkout).
- **Working directory:** `webui.bat` and `webui.sh` do not `cd`/`pushd`; they run `launch.py` relative to the caller's
  directory (`%PYTHON% launch.py %*`, `"$PYTHON" launch.py "$@"`). `modules/launch_utils.py` has no `os.chdir` and anchors
  paths on `script_path`. So `os.getcwd()` equals the WebUI root only when the user starts the launcher from there
  (double-clicking `webui-user.bat` normally does; a shortcut or a call from elsewhere does not). Do not rely on cwd; use `models_path`.
  `modules/paths_internal.py` also exposes `cwd = os.getcwd()` captured at import. `launch.py` has no `os.chdir` either
  (local `neo` checkout).

## UI helpers (refresh button)
- `modules.ui_components.ToolButton` exists: `class ToolButton(gr.Button, FormComponent)`, adds CSS class `"tool"`. Source: `modules/ui_components.py`.
- `modules.ui_common.refresh_symbol = "\U0001f504"` and
  `create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id)` returning the `ToolButton`. Source: `modules/ui_common.py`.
  Behaviour: `refresh_component` is one component or a list; click runs `refresh_method()` first, then
  `args = refreshed_args() if callable(refreshed_args) else refreshed_args`, sets each key as an attribute on every component
  (`setattr(comp, k, v)`) and returns `gr.update(**args)` (a list of them for several components). So
  `refreshed_args=lambda: {"choices": [...]}` works. The tooltip is `"<label>: refresh"` using the component's `label`.
  `elem_id` is required (positional). Source: `modules/ui_common.py` (signature and `refresh` body confirmed in two reads).

## Gradio
Neo's default install pin is `gradio==4.40.0 gradio_rangeslider==0.0.8` (overridable by env `GRADIO_PACKAGE`), in
`prepare_environment()` of `modules/launch_utils.py`; it is not in `requirements.txt`. The README says Neo continues Forge built on
Gradio `4.40.0`. Gradio 4: an event handler may return `gr.update(choices=..., value=...)` for a Dropdown. Source: `modules/launch_utils.py`, `README.md`.

## Built-in SeedVR2
`README.md` of the `neo` branch does not mention SeedVR in any casing, and a web search found no built-in SeedVR2 support or
`models/SeedVR2` folder in Neo. A case-insensitive search for "seedvr" over `modules/`, `modules_forge/` and `backend/`
of a local `neo` checkout finds nothing: Neo neither uses nor reserves such a folder.

## Script loading, results and saved UI values
- **`sys.path`:** `load_scripts` prepends every extension root (`ScriptFile.basedir`) to `sys.path` for good, and while
  executing each script puts its root first on a temporary list that is replaced by the saved one afterwards. So a
  script's own `sys.path` edits at import time are discarded, and at run time the extension root is on the path but not
  necessarily first — a top-level package name shared by two extensions resolves to whichever is imported first.
  Source: `modules/scripts.py`, `load_scripts`.
- **Empty `Processed`:** `Processed(p, images_list, seed, info, ..., infotexts=None, ...)`. Normal txt2img shows
  `processed.info`; only the gallery ✨ (hires upscale) handler reads `processed.infotexts[0]` and splices
  `processed.images` / `processed.infotexts` into the gallery in place of the selected image. So an empty result needs
  `infotexts=[message]` to avoid an `IndexError` there, at the cost of one infotext more than images. Source:
  `modules/processing.py` (`class Processed`), `modules/txt2img.py` (hires button handler).
- **Saved UI defaults (`ui-config.json`):** a saved dropdown value that is not among the current choices is ignored
  (`check_dropdown`), so stale model names left by older versions do not break the dropdowns. Source: `modules/ui_loadsave.py`.

## Not verified
- Whether `shared.state.skipped` still exists in Neo (used by another extension's interrupt check).
- Behaviour of the "Extras" postprocessing API (`scripts_postprocessing.ScriptPostprocessing`) in Neo — needed for the Extras idea.

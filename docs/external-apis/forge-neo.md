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
- **"Apply and restart UI" updates an extension with `git fetch --all` + `git reset --hard <remote>/<branch>`**
  (`Extension.fetch_and_reset_hard` in `modules/extensions.py`, called from `modules/ui_extensions.py`). Locally modified
  tracked files are overwritten, files deleted upstream are removed; untracked files (e.g. `__pycache__`) stay. A manual
  `git pull` has no such guarantee. The restart that follows ends the Python process (`os._exit(0)` in `modules/restart.py`,
  after writing `tmp/restart`) and the launcher script starts it again, so nothing loaded by the old code stays in memory;
  without a launcher watching, the button is "Apply and quit". Source: verified in a Neo install's source (commit `4f4284b`).
- **The extension root `__init__.py` is never imported**; only `scripts/*.py` are loaded as scripts. Source: verified in source (`modules/scripts.py`, `list_scripts`).
- **`process_images` returns the grid first.** With more than one image and `opts.return_grid`, `output_images.insert(0, grid)`
  and `index_of_first_image = 1` (`modules/processing.py`). Source: verified in source.
- **`create_infotext` needs a loaded SD checkpoint.** With none loaded (right after start Neo holds a `FakeInitialModel`)
  it raises `AttributeError: 'FakeInitialModel' object has no attribute 'use_distilled_cfg_scale'`, so a script that
  unloads the checkpoint cannot build infotexts afterwards. Source: live run (img2img through the API, Neo `neo-2.31`).
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
- Facts below: `neo` branch, `backend/memory_management.py` (file credits ComfyUI), read via raw GitHub on the day the `neo`
  head was `89e2e09`; the page summariser relayed the code, bodies were not read line by line.
- **Device choice:** `get_torch_device() -> torch.device`: CUDA → `torch.device(torch.cuda.current_device())` (index of PyTorch's
  current device, no flag read in this file); XPU → `torch.device("xpu", torch.xpu.current_device())`; also CPU and MPS branches.
  `modules.devices.device: torch.device = memory_management.get_torch_device()`, a module-level variable (`modules/devices.py`).
  The GPU flag is applied earlier, in `initialize_forge()` (`modules_forge/initialization.py`): before `import torch`,
  `if args.gpu_device_id is not None:` sets `os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device_id)`. `args` comes from
  `backend/args.py`: `--gpu-device-id` (int, default `None`, "Set the id of device to use (all other devices will not be
  visible)"). With the flag, the chosen GPU is the only visible one, i.e. index 0 inside the process. So an extension should use
  `memory_management.get_torch_device()` or `modules.devices.device`, not a hard-coded `cuda:0`.
- **Memory numbers (bytes):** `get_total_memory(dev=None, torch_total_too=False)`; `get_free_memory(dev=None, torch_free_too=False)`
  returning an int, or `(mem_free_total, mem_free_torch)` with `torch_free_too`. CUDA branch: `stats = torch.cuda.memory_stats(dev)`,
  `mem_free_cuda, _ = torch.cuda.mem_get_info(dev)`, `mem_free_torch = reserved - active`,
  `mem_free_total = mem_free_cuda + mem_free_torch`. So free memory **includes** the torch allocator's reserved-but-unused blocks.
  Units: bytes (all values are raw torch byte counts).
- **Memory held by foreign torch models:** the driver-side free value comes from `mem_get_info`, so VRAM held by an extension's
  tensors is already missing from it. Only with `args.pynvml` and a successful `pynvml.nvmlInit()` (`PYNVML_IS_AVAILABLE`),
  `get_free_memory` additionally subtracts `nvml_mem.used - torch_reserved` (other processes); that handle is
  `nvmlDeviceGetHandleByIndex(0)`, hard-coded.
- **`soft_empty_cache(force=False)`:** on CUDA calls `torch.cuda.synchronize()`, `torch.cuda.empty_cache()`,
  `torch.cuda.ipc_collect()` unconditionally; `force` is not used. MPS and XPU have their own branches (synchronize + empty cache),
  CPU does nothing; it ends by setting the global `signal_empty_cache = False`.
- **`free_memory(memory_required, device, keep_loaded=[])`:** synchronizes, collects dead models, unloads loaded models on `device`
  (not in `keep_loaded`) until `memory_required` fits (with `--disable-smart-memory` it unloads candidates without checking free
  memory), calls `soft_empty_cache()` if anything was unloaded (or if torch-reserved is above 25% of free memory outside
  HIGH_VRAM), returns the list of unloaded models. It sees only models registered in `current_loaded_models`, never foreign tensors.
- **`unload_all_models()`** = `free_memory(1e30, get_torch_device())`: every Forge-managed model on the device goes through
  `model_unload` → `self.model.detach(...)`. Where the weights go (CPU RAM or dropped) is decided in `backend/patcher/base` — not read.
  `unet_offload_device()` returns CPU unless HIGH_VRAM.
- **Reserve:** `--reserve-vram GB` (float, default `None`); otherwise `EXTRA_RESERVED_VRAM` defaults to about 400 MB (600 MB on
  Windows, +100 MB on Windows GPUs above 15 GB); `minimum_inference_memory()` = 0.8 GB + that reserve. Source: same file.
- **Allocator config is opt-in:** `--cuda-malloc` (default `False`) → `try_cuda_malloc()` (`modules_forge/cuda_malloc.py`) sets
  `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` (appended with a comma to an existing value) unless `nvidia-smi -L` lists a
  blacklisted old GPU; `--expandable-segments` (default `False`) → `try_expandable_segments()` sets
  `PYTORCH_ALLOC_CONF=expandable_segments:True`. Both are called from `initialize_forge()` before `import torch`.
  Nothing sets either by default, so the default allocator is PyTorch's native caching allocator (unless the user's
  environment sets it). Other flags: `--cuda-stream` (int, optional value, const 2), `--pin-shared-memory`,
  `--disable-smart-memory`, `--cpu`. Source: `backend/args.py`, `modules_forge/initialization.py`.
- **`torch.cuda.empty_cache()`:** PyTorch docs: "releases all unused cached memory from PyTorch"; it does not free tensor-occupied
  memory. Source: <https://docs.pytorch.org/docs/2.15/notes/cuda.html> (Memory management) and
  `torch.cuda.memory.empty_cache` page. The docs do not say how it behaves under `backend:cudaMallocAsync`; they only say some
  memory stats are "specific to backend:native" and not meaningful with `cudaMallocAsync`.
- **Callbacks:** `modules/script_callbacks.py` has 21 `on_*` registrars (names): `on_app_started`, `on_before_reload`,
  `on_model_loaded`, `on_ui_tabs`, `on_ui_train_tabs`, `on_ui_settings`, `on_before_image_saved`, `on_image_saved`,
  `on_extra_noise`, `on_cfg_denoiser`, `on_cfg_denoised`, `on_cfg_after_cfg`, `on_before_component`, `on_after_component`,
  `on_image_grid`, `on_infotext_pasted`, `on_script_unloaded`, `on_before_ui`, `on_list_optimizers`, `on_list_unets`,
  `on_before_token_counter`. Model-related: `on_model_loaded` only (fired when the SD model is created); there is no
  "model about to load/unload" hook and no registry for extensions to release VRAM on request in `memory_management.py`
  (only a `weakref.finalize` that cleans up Forge's own patched models).

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
  - **Frozen after the first launch.** `UiLoadsave.add_component` → `apply_field` reads `saved_value = self.ui_settings.get(key)`;
    if it is `None` the component's current code value is written into `ui_settings[key]`, otherwise the saved value is
    `setattr`-ed onto the component (`setattr(obj, field, saved_value)`), so a saved value beats a changed code default.
    The file is rewritten at the end of `create_ui` (`modules/ui.py`) only when `ui_settings` differ from the file
    (`loadsave.dump_defaults()`). Applied to `Slider` (`value`, `minimum`, `maximum`, `step`), `Radio`, `Checkbox`, `Textbox`,
    `Number`, `Dropdown` (and `visible`); so a changed slider `minimum`/`maximum`/`step` in code is also ignored for existing
    users. Whether a saved `value` outside a new code range is clamped: not verified.
  - **Key.** `customscript/<basename of the script file>/<tab>/<label>/<field>`, e.g.
    `customscript/seedvr2_script.py/txt2img/VAE Tile Size/value`. `<tab>` is the interface id passed by `loadsave.add_block(interface, ifid)`
    in `modules/ui.py` (`txt2img`, `img2img`); the script prefix comes from `control.custom_script_source =
    os.path.basename(script.filename)` (`ScriptRunner.create_script_ui_inner`, `modules/scripts.py`). `add_block` recurses into
    containers with the same `path`, so an Accordion/Row/Group label is not part of the key. Components with `label=None` are not
    tracked (Buttons use their `value` text). Two components with the same label on one tab share one key (all get the
    saved value; only the first is in `component_mapping`).
  - **Renaming a label** gives a new key, so the new code default applies and is written to the file. The old key is never removed
    (`ui_settings` starts as the loaded file and only gains keys) and no warning is printed (no such code in `ui_loadsave.py`).
  - **API defaults come from the code, not from `ui-config.json`.** `create_script_ui_inner` builds `script.api_info` from
    `getattr(control, "value")` at script-UI creation, and `ui.py` creates the script UI (`scripts_txt2img.setup_ui()`) before
    `UiLoadsave.add_block` runs; `init_default_script_args` (`modules/api/api.py`) fills each script's slice with
    `[arg.value for arg in script.api_info.args]` (`/sdapi/v1/script-info` shows them). `init_script_args` copies those
    defaults and then **replaces** the selected script's slice with `request.script_args`
    (`script_args[args_from:args_to] = request.script_args`), so a short list is not padded with defaults: `run()` gets
    only the values sent, and a required parameter left out fails the request with a `TypeError` (HTTP 500; checked live).
    The code defaults therefore reach `run()` only for the always-on scripts, never for a selected script's omitted args.
  - **Reset by the user:** Settings → "Defaults" tab (`loadsave.create_ui()`, `modules/ui_settings.py`) edits and writes values
    ("View changes", "Apply", applied after a UI restart), or delete the key / the file `ui-config.json` and press "Reload UI"
    (`shared.state.request_restart`). Deleting the file is inferred from the code (a missing file is ignored, `FileNotFoundError`).

## Results, infotexts and saving
Source for all of this section: `neo` branch, files named per item (read via raw GitHub; the page summariser relayed the code,
quotes are short fragments).
- **`Processed` fields** (`modules/processing.py`, `Processed.__init__`): `self.images = images_list`,
  `self.extra_images = extra_images_list`, `self.index_of_first_image = index_of_first_image` (default 0),
  `self.all_prompts = all_prompts or p.all_prompts or [self.prompt]`, `self.all_seeds = all_seeds or p.all_seeds or [self.seed]`,
  `self.infotexts = infotexts or [info] * len(images_list)`, `self.comments` from `p.comments`.
  `js()` dumps `prompt`, `all_prompts`, `all_seeds`, `infotexts`, `index_of_first_image`, ... (not `info`, not `extra_images`).
- **Gallery:** txt2img and img2img show `processed.images + processed.extra_images`; the generation-info JSON is `processed.js()`
  (`modules/txt2img.py`, `txt2img_function`; `modules/img2img.py`, `img2img_function`). `opts.do_not_show_images` empties `processed.images`.
- **Infotext per image:** `update_generation_info` (`modules/ui_common.py`) does `json.loads(generation_info)`, takes the selected
  gallery index from the browser (`selected_gallery_index()` in the button's `_js`), and shows `generation_info["infotexts"][img_index]`;
  out of range (`img_index < 0 or >= len(infotexts)`) or any exception keeps the old HTML. The index is the raw gallery index
  (extra images count), with no `index_of_first_image` subtraction there. `javascript/generationParams.js` only clicks
  `{tab}_generation_info_button` on gallery click, arrow keys and lightbox close.
  So each returned image shows its own parameters when `processed.infotexts[i]` matches `processed.images[i]` for every gallery index.
- **Save button / zip** (`save_files` in `modules/ui_common.py`): uses `data["infotexts"][image_index]` as the saved metadata;
  `IndexError` there saves that image without metadata and returns early (skips the rest and the zip). `index_of_first_image` decides
  grid vs single image (`index >= data["index_of_first_image"]` with `opts.save_selected_only`).
  An `infotexts` list shorter than `images` therefore breaks Save/zip.
- **`create_infotext(p, ...)`** (`modules/processing.py`) reads from `p`: `steps`, `sampler_name`, `scheduler`, `cfg_scale`,
  `distilled_cfg_scale`, `width`, `height`, `sd_model` (flags and `model_config`), `all_seeds`/`all_subseeds` (only with
  `use_main_prompt`), `subseed_strength`, `seed_resize_from_w/h`, `restore_faces`, `main_prompt`, `main_negative_prompt`,
  `all_negative_prompts` (only as default when the argument is None), `batch_size`, `is_using_inpainting_conditioning`,
  `sd_model_name`, `sd_model_hash`, `get_token_merging_ratio()`, plus `getattr` for `enable_hr`, `clip_skip`, `init_img_hash`, `user`.
  The list arguments are used directly: `prompt_text = p.main_prompt if use_main_prompt else all_prompts[index]`.
  **`p.extra_generation_params` is merged in** (`**p.extra_generation_params`), so a script can add its own keys; callable values
  are invoked with `value(**locals())`.
- **Prompt/seed lists:** `p.setup_prompts()` (method of the base class) sets `all_prompts`, `all_negative_prompts`, `main_prompt`,
  `main_negative_prompt` (`batch_size * n_iter * [prompt]`). `all_seeds` and `all_subseeds` are set only in `process_images_inner`
  (after `setup_prompts()`, before `p.scripts.process(p)`). `StableDiffusionProcessingImg2Img.init` sets none of these.
  So in img2img without `process_images`, these lists are `None` (the dataclass default).
- **Img2img Batch tab** (`modules/img2img.py`, `process_batch`): calls `processing.fix_seed(p)` once; per file builds
  `p.init_images = [img] * p.batch_size`; with an output dir sets `p.outpath_samples = output_dir` and
  `p.override_settings["save_to_dirs"] = False`; runs `proc = modules.scripts.scripts_img2img.run(p, *args)` and falls back to
  `process_images(p)` when it returns None; then `batch_results.images.extend(proc.images)` and
  `batch_results.infotexts.extend(proc.infotexts)` (capped by `opts.img2img_batch_show_results_limit`). It does not call
  `save_image` and does not set `do_not_save_samples` (checked in the excerpt). Files are saved only by `process_images`
  via `p.outpath_samples`; a script that returns without `process_images` and saves nothing writes nothing in batch mode, and one that
  saves itself does not duplicate anything from `process_batch`.
  `state.job_count = len(batch_images) * p.n_iter` is set there.
- **Saving inside `process_images_inner`:** the base image is saved with
  `images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(i), p=p)`
  when `p.save_samples()` is true; so saving the same base image again from the script duplicates it. Also saved when enabled:
  `-before-face-restoration`, `-before-color-correction`, `-mask`, `-mask-composite` (suffixes). Grid: saved when `opts.grid_save`
  (`p.outpath_grids`, basename `grid`); returned when `opts.return_grid`, suppressed when `len(output_images) < 2 and opts.grid_only_if_multiple`.
- **`images.save_image` signature** (`modules/images.py`): `save_image(image, path, basename, seed=None, prompt=None, extension="png",
  info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name="parameters", p=None, existing_info=None,
  forced_filename=None, suffix="", save_to_dirs=None)`. It does not check `opts.samples_save` (the caller does, via `p.save_samples()`).
  `suffix` is appended to the generated name (`namegen.apply(file_decoration) + suffix`) and is ignored when `forced_filename` is set.
  Low-level writer: `save_image_with_geninfo(image, geninfo, filename, extension=None, existing_pnginfo=None, pnginfo_section_name="parameters")`.
  I found no user option for a filename suffix; only the `suffix` argument.
- **Extra images:** `Processed(..., extra_images_list=p.extra_result_images)`; `p.extra_result_images = []` is created in
  `StableDiffusionProcessing.__post_init__`. They are shown after `processed.images` in the gallery (and the API adds them to the
  base64 list), not inside `processed.images`. Mask images (`-mask`, `-mask-composite`) are appended to `output_images`
  (`opts.save_mask*`-gated) and so do land in `processed.images`. Generated images count: `len(p.all_seeds)` (= `batch_size * n_iter`);
  `index_of_first_image` is 1 only when a grid was inserted at 0.
- **API** (`modules/api/api.py`, `text2imgapi` / `img2imgapi`): `do_not_save_samples = not save_images` and
  `do_not_save_grid = not save_images` set on `p` (then `save_images` popped); with `send_images` false no images are returned, else
  `encode_pil_to_base64` over `processed.images + processed.extra_images`; `info` in the response is `processed.js()`
  (carries `infotexts`; there is no separate `infotexts` field). A selectable script runs through the script runner, else `process_images(p)`.
- **`p.save_samples(*, is_video=False)`** = `opts.samples_save and not do_not_save_samples and (opts.save_incomplete_images
  or not state.interrupted and not state.skipped) and not (is_video and not opts.video_save_frames)`. `state.skipped` is reset
  only at the top of each batch, so after a Skip on the last txt2img batch it is still set when a script runs afterwards.
  Source: local Neo checkout, `modules/processing.py:495-497`.
- **`override_settings` are applied only inside `process_images`** and restored afterwards when
  `p.override_settings_restore_afterwards` (default True). `save_image` itself reads only
  `p.override_settings["samples_filename_pattern"]`; `save_to_dirs=None` falls back to the global `opts.save_to_dirs`.
  Source: local Neo checkout, `modules/processing.py:182, 836-840`; `modules/images.py:704-722`.
- **API defaults:** `save_images` defaults to `False`, `send_images` to `True` (txt2img and img2img request models).
  Source: local Neo checkout, `modules/api/models.py:91-92, 110-111`.
- **Comments under the result:** `p.comment(text)` adds to `p.comments`; `Processed` joins them into `processed.comments`
  (shown in the UI under the generation info). Source: local Neo checkout, `modules/processing.py:317-318, 510`.
- **Unsaved images carry `image.info`:** the gallery's temp-file writer (`modules/ui_tempdir.py`, `save_pil_to_file`) and the
  API's `encode_pil_to_base64` copy `pil_image.info` into the PNG; an image with `already_saved_as` is served from that file.
  Source: local Neo checkout, `modules/ui_tempdir.py:27-48`, `modules/api/api.py:106`.
- **State** (`modules/shared_state.py`): `skipped` exists (class attribute `False`; `skip()` sets it, `begin()` resets it).
  `job_count` (`begin()` sets -1, `end()` sets 0), `job_no` (`begin()` sets 0), `nextjob()` increments `job_no` and resets step
  counters. `process_images_inner` sets `state.job_count = p.n_iter` only `if state.job_count == -1`, calls `state.nextjob()` after each
  batch, resets `state.skipped` at the top of each batch, and breaks on `state.interrupted or state.stopping_generation`.
  `state.job` is a free text ("Batch n out of m"). A script that upscales N images can set `state.job_count` and call `state.nextjob()`
  per image (follows from the fields; not seen in an existing script).

## Names `scripts/seedvr2_script.py` uses (reference for test stubs)
Source for every line: local `neo` checkout (read in full at the cited file), file named per item. Base URL for the same files:
`https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/`. A stub must match these shapes.
- **`modules.shared`** (`modules/shared.py`): `models_path` is re-exported from `modules.paths_internal`
  (`from modules.paths_internal import data_path, extensions_builtin_dir, extensions_dir, models_path, script_path  # noqa: F401`),
  so `shared.models_path` works (a string path). `opts` and `state` are declared as `None` placeholders in `shared.py` and assigned at
  startup; `state` is `modules.shared_state.State`. Options read by the script, with defaults (`modules/shared_options.py`):
  `samples_save` (`True`), `samples_format` (`"png"`), `save_to_dirs` (`True`), `enable_pnginfo` (`True`).
- **`shared.state`** (`modules/shared_state.py`, class attributes): `skipped=False`, `interrupted=False`, `stopping_generation=False`,
  `job=""`, `job_no=0`, `job_count=0`, `job_timestamp="0"`. The script reads only `interrupted`.
- **`modules.devices`** (`modules/devices.py`): `device: torch.device = memory_management.get_torch_device()` (module variable;
  the script reads `devices.device.type` and passes it to `torch.cuda.get_device_properties`); `torch_gc()` takes no arguments and
  returns nothing (calls `memory_management.soft_empty_cache()`).
- **`modules.sd_models.unload_model_weights(*args, **kwargs)`**: no arguments needed, returns `None` (body: `unload_all_models()`,
  `del model_data.sd_model`, `FakeInitialModel()`, `soft_empty_cache()`, `gc.collect()`). Imported by the script as
  `import modules.sd_models` and called as `modules.sd_models.unload_model_weights()`, so a stub needs the submodule both in
  `sys.modules` and as an attribute of the `modules` package. The real module imports `torch` at top level.
- **`modules.images.save_image`** (`modules/images.py:664`): `save_image(image, path, basename, seed=None, prompt=None,
  extension="png", info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name="parameters", p=None,
  existing_info=None, forced_filename=None, suffix="", save_to_dirs=None)`. **Returns `(fullfn, txt_fullfn)`**: the full path of the
  saved image and the path of the text file or `None` (docstring "Returns: (fullfn, txt_fullfn)"; the function ends `return fullfn, txt_fullfn`).
  Side effects the script relies on: `image.already_saved_as = fullfn` is set on the image passed in; with `save_to_dirs=None`
  the global `opts.save_to_dirs` decides; `info` is written as `pnginfo[pnginfo_section_name]`; it creates `path` (`os.makedirs`).
  The script does not use the return value.
- **`modules.processing.process_images(p) -> Processed`** (`modules/processing.py:797`): one argument; calls `p.scripts.before_process(p)`
  when `p.scripts is not None`, applies `p.override_settings`.
- **`modules.processing.Processed.__init__(self, p, images_list, seed=-1, info="", subseed=None, all_prompts=None,
  all_negative_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None, comments="",
  extra_images_list=[])`** (`modules/processing.py:500-501`). It reads from `p`: `prompt`, `negative_prompt`, `subseed_strength`,
  `comments` (a dict; `"".join(f"{c}\n" for c in p.comments)` iterates its keys), `width`, `height`, `sampler_name`, `cfg_scale`,
  `steps`, `batch_size`, `restore_faces`, `sd_model_name`, `sd_model_hash`, `sd_vae_name`, `sd_vae_hash`, `seed_resize_from_w/h`,
  `extra_generation_params`, `styles`, `token_merging_ratio`, `token_merging_ratio_hr`, `eta`, `ddim_discretize`, `s_churn`,
  `s_tmin`, `s_tmax` (the rest of the constructor was not read), and `opts.CLIP_stop_at_last_layers`, `opts.face_restoration_model`,
  `state.job_timestamp`. A stub `p` for the real `Processed` needs all of these; a stub `Processed` needs only the
  fields the script and tests read (`images`, `infotexts`, `info`, `seed`, `index_of_first_image`, `comments`).
- **`p` (`StableDiffusionProcessing` dataclass, `modules/processing.py:142-234`)** attributes the script touches, with defaults:
  `outpath_samples=None`, `prompt=""`, `negative_prompt=""`, `seed=-1`, `do_not_save_samples=False`,
  `override_settings` (set to `{}` in `__post_init__` when empty, so never `None` on a real `p`; the script still guards with `getattr`),
  `all_seeds=None`, `all_prompts=None` (both `init=False`, set only by `process_images`; `None` in img2img),
  `comments={}` (a dict, `p.comment(text)` does `self.comments[text] = 1`), `scripts` (a property over `scripts_value`, default `None`;
  when set it is a `ScriptRunner`, whose `scripts` attribute is a plain `list` that the script removes itself from and appends to).
  `StableDiffusionProcessingTxt2Img` and `StableDiffusionProcessingImg2Img` both subclass it; img2img adds `init_images: list = None`.
- **`modules.scripts.Script`** (`modules/scripts.py:58-93`): class attributes `name`, `section`, `filename`, `args_from`, `args_to`,
  `alwayson=False`, `is_img2img=False`, `tabname=None`, `group=None`. `elem_id(self, item_id)` (line 343) builds the
  id from `is_img2img`, `self.show(True) == self.show(False)` (tab prefix `txt2img_`/`img2img_` only when equal) and a sanitised
  `self.title()` (lines 343-349; the final `return` line was not read), so it calls `show` and `title`; a stub must do the same only if a test checks the id. `ScriptRunner.__init__` sets `self.scripts = []` (line 559).
- **`ToolButton`** (`modules/ui_components.py:17-26`): `ToolButton(value="", *args, elem_classes=None, **kwargs)`, passes
  `elem_classes=["tool", *elem_classes]` and `value=value` to `gr.Button`. **`FormRow(gr.Row, FormComponent)`** has no `__init__` (lines 43-47); the script does not use it.
- **`tooltip=` keyword on Gradio components:** Gradio 4.40's own constructors do not take it; Neo patches every component class
  (`repair` in `modules/gradio_extensions.py:129-153`: `__repaired_init__(self, *args, tooltip=None, source=None, ...)` stores it as
  `self.webui_tooltip`, and unknown kwargs are dropped with a `GradioDeprecationWarning`). So `ToolButton(..., tooltip="...")`
  is valid inside Forge Neo and a stub of `gradio` components must accept `tooltip` (and `elem_classes`, `elem_id`, `info`,
  `label`, `choices`, `value`, `visible`, ...), or it will reject a call that works for real.
- **`gr.update(...)`**, `.click(fn=, inputs=, outputs=)`, `.change(...)` are used by the script on Gradio's components: stubs need
  `update(**kwargs)` returning a dict-like and event methods that accept `fn`, `inputs`, `outputs`.

- **`torchvision` is present in a Neo install:** `modules/launch_utils.py` installs `torch` and `torchvision` together
  (`torch_command`, line 288; reinstall check at line 313); `requirements.txt` lists `torch` only. Source: local Neo checkout.

## Not verified
- The launch flag name: `backend/args.py` defines `--gpu-device-id`, while a read of `modules/cmd_args.py` showed `--device-id`
  (str); which one Neo's launcher actually accepts, and whether both exist, was not settled.
- Where `unload_all_models()` puts the weights (CPU RAM or dropped): `backend/patcher/base` (`detach`, `partially_unload`) not read.
- Behaviour of `torch.cuda.empty_cache()` and `memory_reserved()` under `cudaMallocAsync`: PyTorch docs are silent; check with
  a live run (`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` via `--cuda-malloc`: allocate, free, `empty_cache()`, compare
  `nvidia-smi` with `torch.cuda.memory_reserved()`).
- Whether any `modules/` code (not `script_callbacks`) lets an extension ask Forge to release memory before loading the SD
  model: none found in `memory_management.py`; a full search of the tree was not done.
- `ScriptRunner.run` was only summarised (no state or `do_not_save_samples` handling in it); it was not read line by line.
- Whether `p.seed == -1` is resolved before `run()` in single-image img2img and txt2img: `fix_seed` is defined in `modules/processing.py`
  but not called in `img2img_function` / `txt2img_function` or `process_images*` (they call `get_fixed_seed`); only `process_batch`
  calls it. A script that reads `p.seed` before `process_images` should call `processing.fix_seed(p)` itself (not confirmed as needed).
- The handler of `{tab}_generation_info_button` is read (see "Results, infotexts and saving"), but the JS `selected_gallery_index()` was not.
- Behaviour of the "Extras" postprocessing API (`scripts_postprocessing.ScriptPostprocessing`) in Neo — needed for the Extras idea.

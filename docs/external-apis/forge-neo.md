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

## Not verified
- Whether `shared.state.skipped` still exists in Neo (used by another extension's interrupt check).
- Behaviour of the "Extras" postprocessing API (`scripts_postprocessing.ScriptPostprocessing`) in Neo — needed for the Extras idea.

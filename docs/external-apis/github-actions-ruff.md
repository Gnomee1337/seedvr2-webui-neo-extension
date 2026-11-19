# GitHub Actions, ruff and CPU PyTorch — reference

Facts for the CI workflow (ruff + pytest on a GitHub-hosted Ubuntu runner). Each fact has a source.
Facts about actions come from the action READMEs / GitHub release pages as read on the day of writing; action majors move,
so re-check before pinning.

## Access
Public repository: GitHub-hosted runners are free for public repositories; no keys needed for checkout, setup-python or ruff.
(`GITHUB_TOKEN` is provided automatically; the workflow should declare `permissions: contents: read`.)

## Common pitfalls
- **`actions/checkout@v7` blocks checkouts of fork PRs under `pull_request_target` / `workflow_run`** (breaking change of v7.0.0).
  A plain `on: push` / `on: pull_request` workflow is not affected. Source: <https://github.com/actions/checkout/releases/tag/v7.0.0>
  (relayed by a search summary, not read in full).
- **Quote the Python version in YAML** (`'3.10'`): unquoted `3.10` parses as the float `3.1`. This is a YAML fact; the setup-python
  docs only show quoted values (`python-version: '3.13'`) and do not explain it. Source: examples in
  <https://github.com/actions/setup-python/blob/main/docs/advanced-usage.md>.
- **The default PyPI `torch` wheel on Linux is the CUDA build** (about 2 GB or more with its CUDA dependencies, against about 200 MB
  for the CPU wheel; the numbers are for older releases, 2.0 and 1.7.1, from forum posts, not for the current 2.14). Use the CPU index
  below. Current size not confirmed: check `pip download` output or the index page.

## Versions
| Item | Value | Source |
|---|---|---|
| `actions/checkout` | **v7** (README example `actions/checkout@v7`; latest release v7.0.1; v7.0.0 published 18 June) | <https://github.com/actions/checkout>, <https://github.com/actions/checkout/releases> |
| `actions/setup-python` | **v7** (README example `actions/setup-python@v7`; the page links V6 release notes as the previous major) | <https://github.com/actions/setup-python> |
| `astral-sh/ruff-action` | README example `astral-sh/ruff-action@v4.1.0`; the repository is in Astral's organisation (the maker of ruff), "Made by Astral" badge; a floating `@v4` tag was not seen on the page, only the full tag | <https://github.com/astral-sh/ruff-action> |
| `ubuntu-latest` | Ubuntu 24.04; standard public-repository Linux runner: 4 vCPU, 16 GB RAM, 14 GB SSD | <https://docs.github.com/en/actions/reference/runners/github-hosted-runners> |
| PyTorch current stable | 2.14.1 with torchvision 0.29.1 | <https://pytorch.org/get-started/previous-versions/> |

## Steps
### `actions/setup-python`
`uses: actions/setup-python@v7` with `python-version: '3.13'`-style input (semver ranges supported, e.g. `"3.11.0 - 3.14.0"`);
pip cache: `cache: 'pip'`, and `cache-dependency-path` takes a file, a list or wildcards (example `'**/requirements-dev.txt'`).
The docs do not say how the cache key is built from those files. Source: <https://github.com/actions/setup-python>,
<https://github.com/actions/setup-python/blob/main/docs/advanced-usage.md>.

### CPU-only PyTorch on Linux
Official command (previous-versions page, newest listed release):
`pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu`
Source: <https://pytorch.org/get-started/previous-versions/>. The index `https://download.pytorch.org/whl/cpu/` carries the
`+cpu` wheels. Size and install time of the current 2.14.x CPU wheels were not measured (see "Not verified").

### Ruff in CI
Two ways, both fine:
- `astral-sh/ruff-action`: installs ruff and by default runs `ruff check`. Inputs: `args` (arguments to `ruff`, default `check`),
  `src` (paths, default the whole workspace), `version` (ruff version; if unset it is read from `pyproject.toml`, falling back
  to `latest`). Source: <https://github.com/astral-sh/ruff-action>.
- `pip install ruff` then `ruff check .` (nothing in the Astral docs forbids it; the pip route is the one that pins a version
  through `requirements-dev.txt`).

### Ruff configuration
Source: <https://docs.astral.sh/ruff/configuration/>.
- Ruff reads `pyproject.toml`, `ruff.toml` or `.ruff.toml`. In one directory the priority is `.ruff.toml` > `ruff.toml` > `pyproject.toml`.
  It uses the closest config file and does **not** merge settings across files. A `pyproject.toml` without a `[tool.ruff]`
  section is ignored when ruff looks for the closest config.
- In `pyproject.toml` the settings sit under `[tool.ruff]` (and `[tool.ruff.lint]`); in `ruff.toml` the `tool.ruff` prefix is dropped
  (`[lint]`, top-level keys).
- `target-version = "py310"` is a key of the ruff table. If no config sets it, ruff infers it from `requires-python` in a
  nearby `pyproject.toml`.
- `exclude` **replaces** the default exclusion list (`.git`, `.venv`, `build`, `dist`, `node_modules`, ...); `extend-exclude`
  **adds** to the defaults. Documented example:
  ```toml
  [tool.ruff]
  target-version = "py310"
  extend-exclude = ["*.ipynb"]
  ```
  A folder is excluded with a path or glob in `extend-exclude` (the docs example shows a glob; folder syntax such as
  `"seedvr2_vendor"` was not shown on the page read, only globs).

## Not verified
- Size and install time of the CPU wheels `torch` 2.14.x on Linux: not found in official docs. Check with a live CI run
  (`pip install ... --index-url https://download.pytorch.org/whl/cpu`, read the log: download size and elapsed time).
- Whether a `cp310` Linux CPU wheel exists for 2.14.x: the index page was truncated and showed only 2.1.x; check with
  `pip download torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu --python-version 3.10 --only-binary=:all: --no-deps`.
  If not, run CI on a newer Python than 3.10 or pin an older torch.
- The existence of a floating `@v4` tag for `astral-sh/ruff-action` and its latest release number (the page shows `v4.1.0` only).
- The exact `exclude` syntax for a folder (`extend-exclude = ["seedvr2_vendor"]`): follows from the documented glob/path
  patterns, not shown as an example; a first CI run confirms it.

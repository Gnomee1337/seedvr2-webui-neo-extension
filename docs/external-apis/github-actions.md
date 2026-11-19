# GitHub Actions and CPU PyTorch — reference

Facts for the CI workflow `.github/workflows/tests.yml` (pytest on a GitHub-hosted Ubuntu runner). Each fact has a source.
Facts about actions come from the action READMEs / GitHub release pages; action majors move, so re-check before changing
the pinned major.

## Access
Public repository: GitHub-hosted runners are free for public repositories; no keys needed for checkout or setup-python.
(`GITHUB_TOKEN` is provided automatically; the workflow declares `permissions: contents: read`.)

## Common pitfalls
- **`actions/checkout@v7` blocks checkouts of fork PRs under `pull_request_target` / `workflow_run`** (breaking change of v7.0.0).
  A plain `on: push` / `on: pull_request` workflow is not affected. Source: <https://github.com/actions/checkout/releases/tag/v7.0.0>
  (relayed by a search summary, not read in full).
- **Quote the Python version in YAML** (`"3.10"`): unquoted `3.10` parses as the float `3.1`. This is a YAML fact; the setup-python
  docs only show quoted values and do not explain it. Source: examples in
  <https://github.com/actions/setup-python/blob/main/docs/advanced-usage.md>.
- **The default PyPI `torch` wheel on Linux is the CUDA build** (several GB with its CUDA dependencies). The workflow installs
  the CPU build from the PyTorch index first, so `pip install -r requirements-dev.txt` finds torch already satisfied.

## Versions
| Item | Value | Source |
|---|---|---|
| `actions/checkout` | **v7** (README example `actions/checkout@v7`; latest release v7.0.1) | <https://github.com/actions/checkout>, <https://github.com/actions/checkout/releases> |
| `actions/setup-python` | **v7** (README example `actions/setup-python@v7`) | <https://github.com/actions/setup-python> |
| `ubuntu-latest` | Ubuntu 24.04; standard public-repository Linux runner: 4 vCPU, 16 GB RAM, 14 GB SSD | <https://docs.github.com/en/actions/reference/runners/github-hosted-runners> |
| PyTorch | 2.14.1 with torchvision 0.29.x; a `cp310` Linux CPU wheel exists (`torch-2.14.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl`, 196 MB) | <https://pytorch.org/get-started/previous-versions/>; the first CI run's log |

## Steps
### `actions/setup-python`
`uses: actions/setup-python@v7` with `python-version: "3.10"` (semver ranges supported, e.g. `"3.11.0 - 3.14.0"`);
pip cache: `cache: 'pip'`, and `cache-dependency-path` takes a file, a list or wildcards. Not used: installing takes ~35 s
without it. Source: <https://github.com/actions/setup-python>, <https://github.com/actions/setup-python/blob/main/docs/advanced-usage.md>.

### CPU-only PyTorch on Linux
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` (official form, with versions, on
<https://pytorch.org/get-started/previous-versions/>). The index carries the `+cpu` wheels. Measured in CI: the whole
"Install dependencies" step takes ~33 s, the test run ~30 s.

## Not verified
- How the setup-python pip cache key is built from `cache-dependency-path` (the docs do not say); the cache is not used.

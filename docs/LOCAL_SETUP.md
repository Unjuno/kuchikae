# Local Setup and Nix Workflow

This repository is managed through Nix and uv.

The local development environment must be entered through `nix develop`. Do not rely on global Python, global pip, Homebrew-only setup, or manually created virtual environments.

## 1. Clone the repository

First clone the remote repository locally:

```bash
git clone git@github.com:Unjuno/kuchikae.git
cd kuchikae
```

If SSH is not configured, use HTTPS:

```bash
git clone https://github.com/Unjuno/kuchikae.git
cd kuchikae
```

## 2. Enter the Nix development shell

```bash
nix develop
```

The dev shell must provide at least:

- Python 3.11
- uv
- ffmpeg
- sox
- libsndfile
- portaudio
- git

The shell hook must set:

```bash
export UV_PROJECT_ENVIRONMENT=.venv
export PYTHONPATH=$PWD
```

## 3. Install Python dependencies through uv

Inside the Nix shell:

```bash
uv sync
```

Do not run global `pip install`.
Do not manually create `.venv` outside the configured uv environment.

## 4. Run tests

```bash
uv run pytest
```

## 5. Run the app

```bash
uv run python app.py
```

## 6. Required first implementation behavior

The first implementation must run with dummy backends only.

The command flow must be:

```bash
git clone git@github.com:Unjuno/kuchikae.git
cd kuchikae
nix develop
uv sync
uv run pytest
uv run python app.py
```

## 7. Repository hygiene

The following must not be committed:

- `.venv/`
- generated audio in `outputs/`
- model weights
- external model repositories
- run directories
- cache directories

External model repositories, when added later, should live outside this repository and be referenced by path, for example:

```yaml
external_models:
  openvoice_path: "../OpenVoice"
  gpt_sovits_path: "../GPT-SoVITS"
  cosyvoice_path: "../CosyVoice"
```

## 8. Nix boundary

Nix owns the system-level development environment.
uv owns Python package resolution.

The repository must not depend on undeclared system packages installed elsewhere.

Reject any implementation that only works because of local machine state outside the Nix shell.

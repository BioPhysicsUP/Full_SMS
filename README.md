# Full SMS

Single-molecule spectroscopy data analysis application.

## Installation

This project uses [mise](https://mise.jdx.dev/) for tool management and [uv](https://docs.astral.sh/uv/) for Python package management.

```bash
# Install mise (if not already installed)
curl https://mise.run | sh

# Initialize the project
mise run init
```

## Development

```bash
# Run the application
mise run run

# Run tests
mise run test
```

## Manual Setup

If you prefer not to use mise:

```bash
# Ensure Python 3.12+ is installed
python --version

# Install uv
pip install uv

# Install dependencies
uv sync

# Run the application
uv run python -m full_sms.app
```

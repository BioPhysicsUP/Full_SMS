#MISE description="Run the test suite"
$ErrorActionPreference = "Stop"

uv run pytest @args

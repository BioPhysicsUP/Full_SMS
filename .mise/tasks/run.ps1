#MISE description="Run the Full SMS application"
$ErrorActionPreference = "Stop"

uv run python -m full_sms.app @args

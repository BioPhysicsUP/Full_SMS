#MISE description="Initialize the project after cloning"
$ErrorActionPreference = "Stop"

Write-Host "==> Installing mise tools (Python, uv)..."
mise install

Write-Host "==> Installing Python dependencies..."
uv sync

Write-Host "==> Project initialized successfully!"
Write-Host ""
Write-Host "You can now run the application with:"
Write-Host "  uv run python -m full_sms.app"
Write-Host ""
Write-Host "Or run tests with:"
Write-Host "  uv run pytest"

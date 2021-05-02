@echo off

CALL "python" "-m" "pip" "install" "--upgrade" "pip"
CALL "pip" "install" "poetry"
CALL "poetry" "install"

pause

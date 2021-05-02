@echo off

cd ..
pause
CALL "ssh-agent"
CALL "git" "reset" "--hard" "HEAD"
CALL "git" "fetch" "origin"
CALL "git" "pull"
CALL "poetry" "install"

pause
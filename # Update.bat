@echo off

CALL "ssh-agent"
CALL "git" "reset" "--hard" "HEAD"
CALL "git" "fetch" "origin"
CALL "git" "pull"

pause
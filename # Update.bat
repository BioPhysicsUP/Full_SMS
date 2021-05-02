@echo off

CALL "ssh-agent"
CALL "git" "fetch" "origin"
CALL "git" "pull"

pause
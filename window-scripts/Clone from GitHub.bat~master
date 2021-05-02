ECHO off

echo.
echo ###############################################
echo    THIS SCRIPT MUST BE RUN AS ADMINISTRATOR!
echo ###############################################
echo.
echo.

echo Before running this script, please ensure that both Git and Python are installed.
echo.
echo Testing for git:
CALL "git" "--version"
echo.
echo Testing for Python:
CALL "python" "--version"
echo.
echo Git can be downloaded here: https://git-scm.com/downloads
echo Python can be downloaded here: https://www.python.org/downloads/
echo.

pause

powershell -Command "Get-Service -Name ssh-agent | Set-Service -StartupType Manual"
CALL "ssh-agent"

SET file_key_name=full_sms_deploy_key

CALL "xcopy" "%~dp0%file_key_name%" "%homedrive%%homepath%\.ssh\"

(
echo.
echo Host github.com
echo HostName github.com
echo User git
echo IdentityFile ~/.ssh/%file_key_name%
) > %homedrive%%homepath%\.ssh\config

CALL "ssh-add" "%homedrive%%homepath%\.ssh\%file_key_name%"
CALL "ssh-keyscan" "github.com" >> "%homedrive%%homepath%\.ssh\known_hosts"

setlocal
set "psCommand="(new-object -COM 'Shell.Application')^
.BrowseForFolder(0,'Choose folder to clone Full_SMS into. A new folder called Full_SMS will be created.',0,0).self.path""
for /f "usebackq delims=" %%I in (`powershell %psCommand%`) do set "folder=%%I"
setlocal enabledelayedexpansion

CALL "git" "clone" "ssh://github.com/BioPhysicsUP/Full_SMS.git" "!folder!\Full_SMS"

pause
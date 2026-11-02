@echo off
title Strata Flint 40B
cd /d "%~dp0"
set "PYTHONPATH=%~dp0;%~dp0\tools"
python "serve\server.py" --engine strata --config "strata-flint.json" --port 8081 --open
if errorlevel 1 pause

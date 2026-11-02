@echo off
setlocal
set "ROOT=%~dp0.."
set "PYTHONPATH=%ROOT%;%ROOT%\tools"
python "%ROOT%\tools\strata_flint_engine.py" %*

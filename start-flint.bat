@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title Flint 40B MoE (Strata Tiered Engine)

set "PORT=8081"
set "FORK_DIR=E:\AI-models\strata-fork"

echo.
echo ============================================================
echo   Flint 40B MoE (256 экспертов) - Strata Engine
echo ============================================================
echo   Модель:        flint-UD-Q3_K_XL.gguf (18.5 ГБ)
echo   GPU:           NVIDIA GeForce RTX 5060 Ti 16 GB VRAM
echo   CPU:           AMD Ryzen 5 9600X (12 потоков)
echo   Платформа:     Strata Server (OpenAI API / Web Chat)
echo   Порт API:      %PORT%
echo ------------------------------------------------------------
echo   [1] 256K Context (262 144 токена) - Максимальный контекст
echo   [2] 200K Context (204 800 токенов) - Оптимальный длинный контекст (~68 т/с)
echo   [3] 32K Context  (32 768 токенов)  - Турбо-скорость (~88 т/с) [По умолчанию]
echo ============================================================
echo.

set "MODE=%~1"
if "%MODE%"=="" set /p MODE="  Выберите режим [1-3, Enter=3]: "
if "%MODE%"=="" set "MODE=3"

set "CFG=strata-flint.json"
set "DESC=32K Context (32 768 токенов)"

if "%MODE%"=="1" (
    set "CFG=strata-flint-256k.json"
    set "DESC=256K Context (262 144 токена)"
)
if "%MODE%"=="2" (
    set "CFG=strata-flint-200k.json"
    set "DESC=200K Context (204 800 токенов)"
)
if "%MODE%"=="3" (
    set "CFG=strata-flint.json"
    set "DESC=32K Context (32 768 токенов)"
)

echo.
echo   ^>^> Запуск режима: %DESC%
echo   ^>^> Конфиг: %CFG%
echo   ^>^> OpenAI API: http://127.0.0.1:%PORT%/v1
echo   ^>^> Web Chat:   http://127.0.0.1:%PORT%/
echo   ^>^> Нажмите Ctrl+C для остановки.
echo.

cd /d "%FORK_DIR%"
set "PYTHONPATH=%FORK_DIR%;%FORK_DIR%\tools"
python "serve\server.py" --engine strata --config "%CFG%" --port %PORT% --open

if errorlevel 1 pause

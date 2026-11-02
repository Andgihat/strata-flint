Strata Terminal 0.1.52 (windows-x86_64)
======================================

This is the terminal distribution. It carries its own Python 3.12.15 and its own packages, so you do not
need Python, pip, Node, a compiler or a development checkout. Serving and watching (`strata serve`, `strata runtime`,
`strata monitor`) never install or download anything; if something is missing, the error names the command that
installs it. Only two commands fetch files, and only when you run them: `strata launch` (the engine and the model you
choose, after it tells you the size) and `strata update` (the engine and the Strata components).

Install
-------
    Expand-Archive strata-terminal-0.1.52-windows-x86_64.zip .
    cd strata-terminal-0.1.52-windows-x86_64
    .\bin\strata.cmd --help

Put `bin\` on your PATH to get the `strata` command everywhere (cmd.exe and PowerShell):

    setx PATH "%PATH%;%CD%\bin"

Before you start a big model
----------------------------
A model of tens of GB needs free memory. Check how much is free first:

    Task Manager > Performance > Memory > Available, or in PowerShell:
    [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)   # GB free

Close other programs (browsers use a lot) if it is less than the model needs; `strata launch` shows what each model needs.

What it does NOT include
------------------------
The **engine** is a separate, independently versioned component and is not bundled: the right engine package
depends on your OS, CPU and GPU. `strata launch` (or `strata update`) fetches the one for your card from the signed
release feed; or point Strata at one with STRATA_ENGINE. Model weights are likewise not included: `strata launch`
downloads the one you choose (the installer files setup.py and tools/ are in this archive for that).

Useful commands
---------------
    strata launch                     choose a model, tune it, serve it and watch it (guided screens; docs/TERMINAL_LAUNCH.md)
    strata connect opencode           the setup for a coding app on the running server (also codex, claude-code, omp, pi, hermes, openclaw)
    strata --version                  the version and the interpreter it runs on
    strata doctor                     read-only component inventory
    strata profile list               saved serving profiles
    strata status                     one read-only telemetry snapshot
    strata monitor                    the terminal monitor
    strata serve --profile NAME       start the inference server in the foreground
    strata serve --engine mock ...    a scripted engine for client tests (no GPU, no model)
    strata runtime start|status|stop  run a profile as a managed background instance

Existing installs are found in place: the run configs (strata-<model>.json) and the data folder remembered in
settings.json are used where they are; nothing is copied or moved.

Verification
------------
The manifest next to the archive records the SHA-256 of the archive, of the pinned interpreter and of the package
lock it was built from. The launcher prefers the bundled interpreter and ignores the user's site-packages.

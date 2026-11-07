@echo off
cd /d "%~dp0"
where py >nul 2>&1
if errorlevel 1 goto usepython
py -3 start.py %*
goto done
:usepython
python start.py %*
:done
if errorlevel 1 (
    echo.
    pause
)

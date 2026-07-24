@echo off
title PDH Breakout Trader - KEEP THIS WINDOW OPEN (minimize it)
REM ============================================================================
REM  PDH breakout forward-test - auto-restarting background runner (Windows)
REM  Restarts the Python trader if it ever exits, and can relaunch the MT5
REM  terminal itself (needs saved login in the terminal).
REM  Trade record : data\processed\pdh_trader_log.csv
REM  Console log  : data\processed\pdh_trader_console.log
REM  CLOSING THIS WINDOW STOPS THE TRADER. Minimize it instead.
REM ============================================================================
cd /d "%~dp0"
set "TERMINAL=C:\Program Files\MetaTrader 5\terminal64.exe"
set "CONSOLE=..\data\processed\pdh_trader_console.log"

echo ============================================================
echo  PDH breakout trader is RUNNING in this window.
echo  Do NOT close it - minimize it. Closing = trading stops.
echo.
echo  Watch trades:   data\processed\pdh_trader_log.csv
echo  Full output:    data\processed\pdh_trader_console.log
echo ============================================================
echo.

:loop
echo [%date% %time%] launching pdh_breakout_trader  (window stays quiet; see logs)
echo [%date% %time%] launching pdh_breakout_trader >> "%CONSOLE%"
python -u pdh_breakout_trader.py --terminal-path "%TERMINAL%" %* >> "%CONSOLE%" 2>&1
echo [%date% %time%] trader exited (code %errorlevel%) - restarting in 30s
echo [%date% %time%] trader exited (code %errorlevel%) - restarting in 30s >> "%CONSOLE%"
ping -n 31 127.0.0.1 >nul
goto loop

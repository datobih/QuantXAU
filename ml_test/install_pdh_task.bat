@echo off
REM ============================================================================
REM  Register the PDH trader to auto-start at logon and keep running.
REM  Run this ONCE (double-click). Creates a Task Scheduler job "PDHTrader".
REM  To remove:  schtasks /delete /tn PDHTrader /f
REM ============================================================================
set "BAT=%~dp0run_pdh_trader.bat"
schtasks /create /tn "PDHTrader" /tr "\"%BAT%\"" /sc onlogon /rl LIMITED /f
if %errorlevel%==0 (
  echo.
  echo Installed. "PDHTrader" will start at every logon.
  echo Start it now without rebooting:   schtasks /run /tn PDHTrader
  echo Check status:                      schtasks /query /tn PDHTrader
  echo Remove:                            schtasks /delete /tn PDHTrader /f
) else (
  echo.
  echo Install failed - try running this file as Administrator.
)
pause

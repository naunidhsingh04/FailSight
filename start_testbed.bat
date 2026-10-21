@echo off
rem FailSight testbed: starts the monitor, the demo shop and the load generator and opens the dashboard (Live demo tab). No Docker.
rem Needs Python 3.11 with the project requirements; for the dashboard build it once: cd app\frontend ^&^& npm install ^&^& npm run build
cd /d "%~dp0"
python -m testbed.launcher %*
pause

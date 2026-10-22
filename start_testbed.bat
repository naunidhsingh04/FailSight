@echo off
rem Starts the shop, its storefront (http://127.0.0.1:5000), the FailSight monitor (http://127.0.0.1:8000) and the load generator, then opens two browser tabs: the storefront and the monitor.
rem Anything that is already running on its port is reused. Presenter controls: http://127.0.0.1:5000/ops . No Docker.
rem For the monitor page build it once: cd app\frontend ^&^& npm install ^&^& npm run build
cd /d "%~dp0"
python -m testbed.launcher %*
pause

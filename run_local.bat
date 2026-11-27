@echo off
chcp 65001 >nul
cd /d "%~dp0"
python stock_model.py
start "" "docs\index.html"
pause

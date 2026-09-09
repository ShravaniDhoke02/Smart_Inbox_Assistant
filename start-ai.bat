@echo off
setlocal
cd /d "%~dp0"
call "%~dp0\ai-service\.venv\Scripts\activate.bat"
cd /d "%~dp0\ai-service"
python main.py

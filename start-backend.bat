@echo off
setlocal
cd /d "%~dp0"
set JAVA_HOME=%~dp0jdk21\temurin-21\jdk-21.0.12.1+1
set PATH=%JAVA_HOME%\bin;%PATH%
cd /d "%~dp0\backend"
call gradlew.bat bootRun --no-daemon

@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo [오류] 아직 설치되지 않았습니다.
    echo install.bat 을 먼저 더블클릭해서 실행하세요.
    pause
    exit /b 1
)

title FinHOLLY 서버 - 이 창을 닫으면 서버가 종료됩니다
venv\Scripts\python.exe demo\server.py

echo.
echo 서버가 종료되었습니다.
pause

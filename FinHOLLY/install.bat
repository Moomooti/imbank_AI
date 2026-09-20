@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   FinHOLLY 설치 (최초 1회만 실행하면 됩니다)
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo.
    echo   https://www.python.org/downloads/ 에서 Python 3.11을 설치한 뒤
    echo   이 install.bat을 다시 실행하세요.
    echo   ^(설치 화면에서 "Add Python to PATH"를 반드시 체크하세요^)
    echo.
    pause
    exit /b 1
)

echo [1/3] 가상환경 생성 중...
if exist venv (
    echo   이미 venv 폴더가 있어 건너뜁니다.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [오류] 가상환경 생성 실패
        pause
        exit /b 1
    )
)

echo [2/3] pip 업그레이드 중...
venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo [3/3] 필요 라이브러리 설치 중...
if exist offline_packages (
    echo   offline_packages 폴더 발견 -^> 인터넷 없이 로컬 설치합니다.
    venv\Scripts\python.exe -m pip install --no-index --find-links=offline_packages -r requirements.txt
) else (
    echo   인터넷에서 설치합니다. ^(폐쇄망 PC라면 make_offline_packages.bat 안내를 참고하세요^)
    venv\Scripts\python.exe -m pip install -r requirements.txt
)
if errorlevel 1 (
    echo.
    echo [오류] 라이브러리 설치 실패. 위 오류 메시지를 확인하세요.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   설치 완료!
echo   이제부터는 "실행.bat"을 더블클릭해서 실행하세요.
echo ============================================
pause

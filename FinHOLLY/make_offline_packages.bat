@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   폐쇄망 설치용 패키지 내려받기
echo   (인터넷 되는 PC에서 이 스크립트를 실행한 뒤,
echo    생성된 offline_packages 폴더를 통째로
echo    폐쇄망 PC의 FinHOLLY 폴더 밑에 복사하세요)
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    pause
    exit /b 1
)

python -m pip download -r requirements.txt -d offline_packages
if errorlevel 1 (
    echo [오류] 다운로드 실패
    pause
    exit /b 1
)

echo.
echo 완료: offline_packages 폴더가 생성되었습니다.
echo 이 폴더를 폐쇄망 PC의 FinHOLLY 폴더 밑에 복사한 뒤 install.bat을 실행하면
echo 인터넷 연결 없이 설치됩니다.
pause

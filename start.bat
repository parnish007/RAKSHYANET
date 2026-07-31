@echo off
echo ========================================
echo   RAKSHYANET FULLSTACK LAUNCHER
echo ========================================
echo.

echo [1/2] Starting Backend Server...
start "RakshyaNet Backend" cmd /k "cd /d %~dp0 && python backend/main.py"

timeout /t 3 /nobreak > nul

echo [2/2] Starting Frontend Dev Server...
start "RakshyaNet Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================
echo   BOTH SERVERS STARTED!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to stop both servers...
pause > nul

taskkill /FI "WINDOWTITLE eq RakshyaNet Backend*" /F
taskkill /FI "WINDOWTITLE eq RakshyaNet Frontend*" /F

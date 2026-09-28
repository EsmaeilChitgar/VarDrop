@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0.."
if errorlevel 1 exit /b 1

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"

REM ============================================================
REM LPRA minimal matched rank screen - Traffic 96 -> 96
REM NO backbone training.
REM One shared calibration pass for rank16 and rank32.
REM Existing rank32 checkpoint is read-only reference.
REM ============================================================

echo ==========================================================
echo LPRA MATCHED RANK SCREEN - TRAFFIC 96 TO 96
echo NO BACKBONE TRAINING
echo Rank16 + Rank32 in ONE shared calibration pass
echo ==========================================================
echo.

for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "START_TIME=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "START_MS=%%T"

echo Start Time: !START_TIME!
echo.

%PYTHON% -u script\lpra_rank16_matched_screen.py
set "EXIT_CODE=!ERRORLEVEL!"

for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "END_TIME=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "END_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "$s=[double]'!START_MS!'; $e=[double]'!END_MS!'; [Math]::Round(($e-$s)/1000.0,3)"') do set "DURATION_SEC=%%T"

echo.
echo ==========================================================
echo MATCHED RANK SCREEN FINISHED
echo Start        : !START_TIME!
echo End          : !END_TIME!
echo Duration sec : !DURATION_SEC!
echo Exit Code    : !EXIT_CODE!
echo.
echo COPY BACK:
echo   all [RANK SCREEN] and [RANK SCREEN RESULT] lines
echo ==========================================================
echo.

popd
exit /b !EXIT_CODE!

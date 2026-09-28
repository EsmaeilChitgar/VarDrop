@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0.."
if errorlevel 1 exit /b 1

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"

echo ==========================================================
echo LPRA ADDITIVE CONFOUND SCREEN - TRAFFIC 96 TO 96
echo NO BACKBONE TRAINING - VALIDATION ONLY
echo Channel-only + Channel+Phase in ONE shared calibration pass
echo ==========================================================
echo.

for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "START_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "START_TIME=%%T"

echo Start Time: !START_TIME!
echo.

%PYTHON% -u diagnostics\lpra_additive_confound_screen.py

set "EXIT_CODE=!ERRORLEVEL!"

for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "END_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "END_TIME=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "$s=[double]'!START_MS!'; $e=[double]'!END_MS!'; [Math]::Round(($e-$s)/1000.0,3)"') do set "DURATION_SEC=%%T"

echo.
echo ==========================================================
echo CONFOUND SCREEN FINISHED
echo Start        : !START_TIME!
echo End          : !END_TIME!
echo Duration sec : !DURATION_SEC!
echo Exit Code    : !EXIT_CODE!
echo.
echo COPY BACK:
echo   all [CONFOUND SCREEN] and [CONFOUND SCREEN RESULT] lines
echo ==========================================================

popd
exit /b !EXIT_CODE!

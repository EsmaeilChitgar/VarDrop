@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0.."
if errorlevel 1 exit /b 1

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"

REM ============================================================
REM Minimal LPRA structural screen - Traffic 96 -> 96
REM
REM NO 10-epoch backbone training.
REM Reuses the existing GPT3d Traffic96 checkpoint.
REM Trains ONLY two controls in ONE shared calibration pass:
REM   1) Phase-only weekly residual: 168 params
REM   2) Full channel x phase table: 862*168 = 144,816 params
REM Existing LPRA rank-32 checkpoint is read-only reference.
REM ============================================================

echo ==========================================================
echo LPRA MINIMAL STRUCTURAL SCREEN - TRAFFIC 96 TO 96
echo NO BACKBONE TRAINING
ECHO Phase-only + Full-table in one shared calibration pass
ECHO ==========================================================
echo.

for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "START_TIME=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "START_MS=%%T"

echo Start Time: !START_TIME!
echo.

%PYTHON% -u script\lpra_structural_screen.py
set "EXIT_CODE=!ERRORLEVEL!"

for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "END_TIME=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "END_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "$s=[double]'!START_MS!'; $e=[double]'!END_MS!'; [Math]::Round(($e-$s)/1000.0,3)"') do set "DURATION_SEC=%%T"

echo.
echo ==========================================================
echo STRUCTURAL SCREEN FINISHED
echo Start        : !START_TIME!
echo End          : !END_TIME!
echo Duration sec : !DURATION_SEC!
echo Exit Code    : !EXIT_CODE!
echo.
echo COPY BACK:
echo   all [STRUCT SCREEN] and [STRUCT SCREEN RESULT] lines
echo ==========================================================
echo.

popd
exit /b !EXIT_CODE!

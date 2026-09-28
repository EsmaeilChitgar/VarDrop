@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0.."
if errorlevel 1 exit /b 1

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"

echo ==========================================================
echo MATCHED VarDrop vs Exact-Fast TIMING BENCHMARK
echo Traffic96 k=4 + ECL96 k=3
echo NO FULL TRAINING - NO VALIDATION - NO TEST
echo ==========================================================
echo.

for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "START_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "START_TIME=%%T"

echo Start Time: !START_TIME!
echo.

%PYTHON% -u diagnostics\vardrop_exactfast_timing_benchmark.py ^
  --warmup 50 ^
  --timed 300 ^
  --seed 2023 ^
  --num_workers 0 ^
  --device cuda:0

set "EXIT_CODE=!ERRORLEVEL!"

for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "END_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "END_TIME=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "$s=[double]'!START_MS!'; $e=[double]'!END_MS!'; [Math]::Round(($e-$s)/1000.0,3)"') do set "DURATION_SEC=%%T"

echo.
echo ==========================================================
echo TIMING BENCHMARK SUMMARY
echo Start        : !START_TIME!
echo End          : !END_TIME!
echo Duration sec : !DURATION_SEC!
echo Exit Code    : !EXIT_CODE!
echo ==========================================================

popd
exit /b !EXIT_CODE!

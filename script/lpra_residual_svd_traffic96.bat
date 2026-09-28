@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0.."
if errorlevel 1 exit /b 1

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"

set "CHECKPOINT=./checkpoints/traffic_96_96_gpt3d_lpra_r32_OURS_custom_M_ft96_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dttest_k4_gs10_projection_0_fastdfh_lpra_r32_p168/checkpoint_lpra.pth"

echo ==========================================================
echo LPRA RESIDUAL SVD DIAGNOSTIC - TRAFFIC 96 TO 96
echo NO TRAINING - NO CALIBRATION - VALIDATION ONLY
echo ==========================================================
echo.

if not exist "%CHECKPOINT%" (
    echo ERROR: checkpoint not found:
    echo %CHECKPOINT%
    popd
    exit /b 1
)

for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "START_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "START_TIME=%%T"

echo Start Time: !START_TIME!
echo.

%PYTHON% -u diagnostics\lpra_residual_svd.py ^
  --checkpoint_path "%CHECKPOINT%" ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic.csv ^
  --split val ^
  --batch_size 32 ^
  --num_workers 0 ^
  --device cuda:0 ^
  --ranks 1,4,8,16,32,64 ^
  --output_dir ./diagnostics/results/lpra_residual_svd_traffic96

set "EXIT_CODE=!ERRORLEVEL!"

for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "END_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "END_TIME=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "$s=[double]'!START_MS!'; $e=[double]'!END_MS!'; [Math]::Round(($e-$s)/1000.0,3)"') do set "DURATION_SEC=%%T"

echo.
echo ==========================================================
echo RESIDUAL SVD DIAGNOSTIC FINISHED
echo Start        : !START_TIME!
echo End          : !END_TIME!
echo Duration sec : !DURATION_SEC!
echo Exit Code    : !EXIT_CODE!
echo.
echo COPY BACK:
echo   all [RESIDUAL SVD] lines
echo ==========================================================

popd
exit /b !EXIT_CODE!

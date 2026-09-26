@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0.."
if errorlevel 1 exit /b 1

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"
set "FAILED=0"

REM ============================================================
REM GPT3d - GPT3b Exact Fast VarDrop + LPRA
REM Traffic 96 -> 96
REM Backbone training is exactly GPT3b; after the best checkpoint is loaded,
REM a tiny rank-32 periodic residual adapter is calibrated for one epoch.
REM ============================================================

echo ==========================================================
echo GPT3d LPRA - Traffic 96 to 96
echo GPT3b EXACT FAST k-DFH + LOW-RANK PERIODIC RESIDUAL ADAPTER
echo ==========================================================
echo.

echo [1/2] Running LPRA sanity test...
%PYTHON% -u script\lpra_sanity.py
if errorlevel 1 (
    echo.
    echo LPRA sanity test FAILED. Training will not start.
    popd
    exit /b 1
)

echo.
echo [2/2] Starting Traffic training...

set DATA_ROOT=./dataset/traffic/
set DATA_FILE=traffic.csv
set K=4
set GROUP_SIZE=10
set SEQ_LEN=96
set LABEL_LEN=48
set PRED_LEN=96
set ENC_IN=862
set DEC_IN=862
set C_OUT=862
set D_MODEL=512
set N_HEADS=8
set E_LAYERS=4
set D_LAYERS=1
set D_FF=512
set BATCH_SIZE=32
set LEARNING_RATE=0.001
set TRAIN_EPOCHS=10
set PATIENCE=3
set NUM_WORKERS=0
set ITR=1
set LOG_EVERY=100

set LPRA_RANK=32
set LPRA_PERIOD=168
set LPRA_CAL_EPOCHS=1
set LPRA_LR=0.005
set LPRA_ALPHA_MAX=1.25

for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "START_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "START_TIME=%%T"

echo Start Time: !START_TIME!
echo.

%PYTHON% -u run.py ^
  --is_training 1 ^
  --model_id traffic_96_96_gpt3d_lpra_r32 ^
  --model OURS ^
  --data custom ^
  --root_path %DATA_ROOT% ^
  --data_path %DATA_FILE% ^
  --features M ^
  --target OT ^
  --freq h ^
  --seq_len %SEQ_LEN% ^
  --label_len %LABEL_LEN% ^
  --pred_len %PRED_LEN% ^
  --enc_in %ENC_IN% ^
  --dec_in %DEC_IN% ^
  --c_out %C_OUT% ^
  --d_model %D_MODEL% ^
  --n_heads %N_HEADS% ^
  --e_layers %E_LAYERS% ^
  --d_layers %D_LAYERS% ^
  --d_ff %D_FF% ^
  --factor 1 ^
  --embed timeF ^
  --batch_size %BATCH_SIZE% ^
  --learning_rate %LEARNING_RATE% ^
  --train_epochs %TRAIN_EPOCHS% ^
  --patience %PATIENCE% ^
  --k %K% ^
  --group_size %GROUP_SIZE% ^
  --itr %ITR% ^
  --num_workers %NUM_WORKERS% ^
  --exact_fast_vardrop ^
  --fast_log_every %LOG_EVERY% ^
  --use_lpra ^
  --lpra_rank %LPRA_RANK% ^
  --lpra_period %LPRA_PERIOD% ^
  --lpra_cal_epochs %LPRA_CAL_EPOCHS% ^
  --lpra_lr %LPRA_LR% ^
  --lpra_alpha_max %LPRA_ALPHA_MAX%

set "EXIT_CODE=!ERRORLEVEL!"
for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "END_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "END_TIME=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "$s=[double]'!START_MS!'; $e=[double]'!END_MS!'; [Math]::Round(($e-$s)/1000.0,3)"') do set "DURATION_SEC=%%T"

echo.
echo ==========================================================
echo GPT3d LPRA FINAL SUMMARY
echo ==========================================================
echo Dataset               : Traffic
echo Forecast              : 96 -^> 96
echo k                     : %K%
echo group_size            : %GROUP_SIZE%
echo exact_fast_vardrop    : ON
echo LPRA rank             : %LPRA_RANK%
echo LPRA period           : %LPRA_PERIOD%
echo LPRA calibration ep   : %LPRA_CAL_EPOCHS%
echo LPRA lr               : %LPRA_LR%
echo num_workers           : %NUM_WORKERS%
echo.
echo VarDrop/GPT3b ref MSE : 0.39686113595962524
echo VarDrop/GPT3b ref MAE : 0.2727734446525574
echo VarDrop ref time sec  : 1332
echo GPT3b ref time sec    : about 305
echo.
echo Start                 : !START_TIME!
echo End                   : !END_TIME!
echo Duration sec          : !DURATION_SEC!
echo Exit Code             : !EXIT_CODE!
echo ==========================================================
echo.
echo IMPORTANT - COPY THESE LINES BACK:
echo   1. All [LPRA] lines including alpha and validation gain
echo   2. All [FastVarDrop] epoch lines
echo   3. Final mse/mae
echo   4. Epoch cost times
echo   5. This FINAL SUMMARY
echo.

popd
exit /b !EXIT_CODE!

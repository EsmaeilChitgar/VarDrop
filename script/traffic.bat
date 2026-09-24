@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM GPT3b - Exact Fast k-DFH VarDrop
REM Traffic 96 -> 96
REM Pure ablation: ORIGINAL VarDrop vs exact fast implementation.
REM No cache. No probe. No stability threshold. No approximation.
REM ============================================================

echo ==========================================================
echo GPT3b Exact Fast VarDrop - Traffic 96 to 96
echo ==========================================================
echo.

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

REM ============================================================
REM Experiment 1 - Original VarDrop baseline
REM ============================================================

echo.
echo ==========================================================
echo Experiment 1: ORIGINAL VarDrop
echo ==========================================================
for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "BASE_START_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "BASE_START=%%T"
echo Start Time: !BASE_START!
echo.

python -u run.py ^
  --is_training 1 ^
  --model_id traffic_96_96_gpt3b_baseline ^
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
  --num_workers %NUM_WORKERS%

set "BASE_EXIT=!ERRORLEVEL!"
for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "BASE_END_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "BASE_END=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "$s=[double]'!BASE_START_MS!'; $e=[double]'!BASE_END_MS!'; [Math]::Round(($e-$s)/1000.0,3)"') do set "BASE_DURATION_SEC=%%T"

echo.
echo Original Start        : !BASE_START!
echo Original End          : !BASE_END!
echo Original Duration sec : !BASE_DURATION_SEC!
echo Original Exit         : !BASE_EXIT!
echo.

if not "!BASE_EXIT!"=="0" (
    echo Original baseline failed. GPT3b will not run.
    pause
    exit /b !BASE_EXIT!
)

REM ============================================================
REM Experiment 2 - GPT3b Exact Fast k-DFH
REM ============================================================

echo.
echo ==========================================================
echo Experiment 2: GPT3b Exact Fast k-DFH VarDrop
echo NO CACHE - NO PROBE - NO APPROXIMATION
echo ==========================================================
for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "FAST_START_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "FAST_START=%%T"
echo Start Time: !FAST_START!
echo.

python -u run.py ^
  --is_training 1 ^
  --model_id traffic_96_96_gpt3b_fast ^
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
  --fast_log_every %LOG_EVERY%

set "FAST_EXIT=!ERRORLEVEL!"
for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "FAST_END_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "FAST_END=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "$s=[double]'!FAST_START_MS!'; $e=[double]'!FAST_END_MS!'; [Math]::Round(($e-$s)/1000.0,3)"') do set "FAST_DURATION_SEC=%%T"

echo.
echo ==========================================================
echo FINAL SUMMARY
echo ==========================================================
echo Dataset               : Traffic
echo Forecast              : 96 -^> 96
echo k                     : %K%
echo group_size            : %GROUP_SIZE%
echo num_workers           : %NUM_WORKERS%
echo.
echo Original Start        : !BASE_START!
echo Original End          : !BASE_END!
echo Original Duration sec : !BASE_DURATION_SEC!
echo Original Exit         : !BASE_EXIT!
echo.
echo GPT3b Start           : !FAST_START!
echo GPT3b End             : !FAST_END!
echo GPT3b Duration sec    : !FAST_DURATION_SEC!
echo GPT3b Exit            : !FAST_EXIT!
echo ==========================================================
echo.
echo IMPORTANT:
echo Copy the [FastVarDrop] lines, epoch times, final MSE/MAE,
echo and this FINAL SUMMARY.
echo.
pause

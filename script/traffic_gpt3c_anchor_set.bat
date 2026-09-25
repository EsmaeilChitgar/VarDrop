@echo off
setlocal EnableDelayedExpansion
set "CUDA_VISIBLE_DEVICES=0"

REM ============================================================
REM GPT3c - Anchor-Set k-DFH on top of GPT3b Exact Fast VarDrop
REM Traffic 96 -> 96
REM
REM Scientific change:
REM   GPT3b ranked hash : [f1, f2, f3, f4]
REM   GPT3c anchor-set  : [f1, sort({f2, f3, f4})]
REM
REM The strongest dominant frequency is preserved.
REM Only the order of weaker dominant frequencies is canonicalized.
REM GPT3b's single batched GPU->CPU transfer optimization is retained.
REM ============================================================

set "GPT3C_HASH_MODE=anchor_set"

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

REM Known matched GPT3b reference
set REF_MSE=0.396861
set REF_MAE=0.272773
set REF_DURATION_SEC=305

REM ============================================================
REM Step 0 - Real-data hash probe BEFORE training
REM ============================================================

echo ==========================================================
echo GPT3c Anchor-Set VarDrop - Traffic 96 to 96
echo ==========================================================
echo.
echo Hash mode: %GPT3C_HASH_MODE%
echo.
echo ==========================================================
echo Step 0: REAL TRAFFIC HASH PROBE
ECHO This does NOT train the model.
echo ==========================================================

python -u script/probe_gpt3c_hash.py ^
  --csv "%DATA_ROOT%%DATA_FILE%" ^
  --seq_len %SEQ_LEN% ^
  --batch_size %BATCH_SIZE% ^
  --k %K% ^
  --group_size %GROUP_SIZE% ^
  --batches 32 ^
  --max_new_pairs_per_batch 128 ^
  --seed 2023 ^
  --device auto

set "PROBE_EXIT=!ERRORLEVEL!"
if not "!PROBE_EXIT!"=="0" (
    echo.
    echo Hash probe failed. Training will NOT start.
    echo Probe Exit Code: !PROBE_EXIT!
    pause
    exit /b !PROBE_EXIT!
)

REM ============================================================
REM Step 1 - GPT3c training
REM ============================================================

echo.
echo ==========================================================
echo Step 1: GPT3c Anchor-Set k-DFH TRAINING
ECHO GPT3b fast transfer path is still active.
echo ==========================================================

for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "START_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "START_TIME=%%T"

echo Start Time: !START_TIME!
echo.

python -u run.py ^
  --is_training 1 ^
  --model_id traffic_96_96_gpt3c_anchor_set ^
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

set "EXIT_CODE=!ERRORLEVEL!"

for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "END_MS=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'"') do set "END_TIME=%%T"
for /f "delims=" %%T in ('powershell -NoProfile -Command "$s=[double]'!START_MS!'; $e=[double]'!END_MS!'; [Math]::Round(($e-$s)/1000.0,3)"') do set "DURATION_SEC=%%T"

echo.
echo ==========================================================
echo GPT3c TRAFFIC FINAL SUMMARY
ECHO ==========================================================
echo Dataset               : Traffic
echo Forecast              : 96 -^> 96
echo Method                : GPT3b + Anchor-Set k-DFH
echo Hash mode             : %GPT3C_HASH_MODE%
echo k                     : %K%
echo group_size            : %GROUP_SIZE%
echo num_workers           : %NUM_WORKERS%
echo.
echo GPT3b reference MSE    : %REF_MSE%
echo GPT3b reference MAE    : %REF_MAE%
echo GPT3b reference sec    : %REF_DURATION_SEC%
echo.
echo GPT3c Start            : !START_TIME!
echo GPT3c End              : !END_TIME!
echo GPT3c Duration sec     : !DURATION_SEC!
echo GPT3c Exit             : !EXIT_CODE!
echo ==========================================================
echo.
echo IMPORTANT - COPY ALL OF THESE BACK TO CHAT:
echo   1. The full Step-0 PROBE RESULT block
echo   2. Every [FastVarDrop] / [FastVarDrop Epoch] line
echo   3. Epoch times
echo   4. Final MSE and MAE
echo   5. This FINAL SUMMARY
echo.
echo Sanity check:
echo   FastVarDrop lines MUST show: hash=anchor_set
echo   If they show hash=ranked, stop: GPT3c is not active.
echo.
pause

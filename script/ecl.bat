@echo off
setlocal EnableDelayedExpansion

echo ==========================================================
echo GPT3b Exact Fast VarDrop - ECL 96 to 96
echo NO CACHE - NO PROBE - NO APPROXIMATION
echo ==========================================================
echo.

REM ============================================================
REM Configuration
REM ============================================================

set DATA_ROOT=./dataset/electricity/
set DATA_FILE=electricity.csv

set K=4
set GROUP_SIZE=10

set SEQ_LEN=96
set LABEL_LEN=48
set PRED_LEN=96

set ENC_IN=321
set DEC_IN=321
set C_OUT=321

set D_MODEL=512
set N_HEADS=8
set E_LAYERS=3
set D_LAYERS=1
set D_FF=512

set BATCH_SIZE=32
set LEARNING_RATE=0.0005
set TRAIN_EPOCHS=10
set PATIENCE=3
set NUM_WORKERS=0
set ITR=1

REM ============================================================
REM Start
REM ============================================================

for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format \"yyyy-MM-dd HH:mm:ss.fff\""') do set "START_TIME=%%i"

echo Start Time: !START_TIME!
echo.

python -u run.py ^
  --is_training 1 ^
  --model_id ECL_96_96_gpt3b_fast ^
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
  --fast_log_every 100

set "EXIT_CODE=!ERRORLEVEL!"

for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format \"yyyy-MM-dd HH:mm:ss.fff\""') do set "END_TIME=%%i"

for /f "delims=" %%i in ('powershell -NoProfile -Command "$s=[datetime]::ParseExact('!START_TIME!','yyyy-MM-dd HH:mm:ss.fff',$null); $e=[datetime]::ParseExact('!END_TIME!','yyyy-MM-dd HH:mm:ss.fff',$null); [math]::Round(($e-$s).TotalSeconds,3)"') do set "DURATION_SEC=%%i"

echo.
echo ==========================================================
echo GPT3b ECL FINAL SUMMARY
echo ==========================================================
echo Method       : Exact Fast VarDrop
echo Dataset      : ECL
echo Forecast     : 96 -^> 96
echo k            : %K%
echo group_size   : %GROUP_SIZE%
echo num_workers  : %NUM_WORKERS%
echo.
echo Start        : !START_TIME!
echo End          : !END_TIME!
echo Duration sec : !DURATION_SEC!
echo Exit Code    : !EXIT_CODE!
echo ==========================================================
echo.

pause
@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM GPT3 - Stability-Gated VarDrop
REM Traffic 96 -> 96
REM Compares ORIGINAL VarDrop vs GPT3 under identical settings.
REM ============================================================

echo ==========================================================
echo GPT3 Stability-Gated VarDrop - Traffic 96 to 96
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
set NUM_WORKERS=1
set ITR=1

REM Conservative defaults: accuracy preservation first.
set PROBE_SIZE=4
set STABILITY_THRESHOLD=0.98
set MAX_STALE=64
set LOG_EVERY=100

REM ============================================================
REM Experiment 1 - Original VarDrop baseline
REM ============================================================

echo.
echo ==========================================================
echo Experiment 1: ORIGINAL VarDrop
echo ==========================================================
set "BASE_START=%TIME%"
echo Start Time: !BASE_START!
echo.

python -u run.py ^
  --is_training 1 ^
  --model_id traffic_96_96_gpt3_baseline ^
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
set "BASE_END=%TIME%"
call :CalculateDuration "!BASE_START!" "!BASE_END!" BASE_DURATION

echo.
echo Original Start:    !BASE_START!
echo Original End:      !BASE_END!
echo Original Duration: !BASE_DURATION!
echo Original Exit:     !BASE_EXIT!
echo.

if not "!BASE_EXIT!"=="0" (
    echo Original baseline failed. GPT3 will not run.
    pause
    exit /b !BASE_EXIT!
)

REM ============================================================
REM Experiment 2 - GPT3 Stability-Gated VarDrop
REM ============================================================

echo.
echo ==========================================================
echo Experiment 2: GPT3 Stability-Gated VarDrop
echo probe=%PROBE_SIZE% threshold=%STABILITY_THRESHOLD% max_stale=%MAX_STALE%
echo ==========================================================
set "GPT3_START=%TIME%"
echo Start Time: !GPT3_START!
echo.

python -u run.py ^
  --is_training 1 ^
  --model_id traffic_96_96_gpt3_stability ^
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
  --stability_vardrop ^
  --stability_probe_size %PROBE_SIZE% ^
  --stability_threshold %STABILITY_THRESHOLD% ^
  --stability_max_stale %MAX_STALE% ^
  --stability_log_every %LOG_EVERY%

set "GPT3_EXIT=!ERRORLEVEL!"
set "GPT3_END=%TIME%"
call :CalculateDuration "!GPT3_START!" "!GPT3_END!" GPT3_DURATION
call :CalculateDuration "!BASE_START!" "!GPT3_END!" TOTAL_DURATION

echo.
echo ==========================================================
echo FINAL SUMMARY

echo ==========================================================
echo Dataset             : Traffic
echo Forecast            : 96 -^> 96
echo k                   : %K%
echo group_size          : %GROUP_SIZE%
echo probe_size          : %PROBE_SIZE%
echo stability_threshold : %STABILITY_THRESHOLD%
echo max_stale           : %MAX_STALE%
echo.
echo Original Start      : !BASE_START!
echo Original End        : !BASE_END!
echo Original Duration   : !BASE_DURATION!
echo Original Exit       : !BASE_EXIT!
echo.
echo GPT3 Start          : !GPT3_START!
echo GPT3 End            : !GPT3_END!
echo GPT3 Duration       : !GPT3_DURATION!
echo GPT3 Exit           : !GPT3_EXIT!
echo.
echo Total Time          : !TOTAL_DURATION!
echo ==========================================================
echo.
echo IMPORTANT: Copy the [SG-VarDrop] lines plus final MSE/MAE.
echo.
pause
goto :eof

:CalculateDuration
set "START=%~1"
set "END=%~2"
for /f "tokens=1-4 delims=:., " %%a in ("%START%") do (
    set /a START_CS=(((1%%a %% 100)*60 + (1%%b %% 100))*60 + (1%%c %% 100))*100 + (1%%d %% 100)
)
for /f "tokens=1-4 delims=:., " %%a in ("%END%") do (
    set /a END_CS=(((1%%a %% 100)*60 + (1%%b %% 100))*60 + (1%%c %% 100))*100 + (1%%d %% 100)
)
set /a DIFF_CS=END_CS-START_CS
if !DIFF_CS! LSS 0 set /a DIFF_CS+=24*60*60*100
set /a TOTAL_SEC=DIFF_CS/100
set /a HOURS=TOTAL_SEC/3600
set /a MINUTES=(TOTAL_SEC%%3600)/60
set /a SECONDS=TOTAL_SEC%%60
set /a CENTISEC=DIFF_CS%%100
if !HOURS! LSS 10 set "HOURS=0!HOURS!"
if !MINUTES! LSS 10 set "MINUTES=0!MINUTES!"
if !SECONDS! LSS 10 set "SECONDS=0!SECONDS!"
if !CENTISEC! LSS 10 set "CENTISEC=0!CENTISEC!"
set "%~3=!HOURS!:!MINUTES!:!SECONDS!.!CENTISEC!"
goto :eof

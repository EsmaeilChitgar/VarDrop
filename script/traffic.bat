@echo off
setlocal EnableDelayedExpansion

set CUDA_VISIBLE_DEVICES=0

echo ==========================================================
echo Mass-Preserving VarDrop - Traffic 96 to 96
echo ==========================================================
echo.

REM ============================================================
REM Common settings
REM ============================================================

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


REM ============================================================
REM Experiment 1
REM Mass VarDrop - alpha = 0
REM ============================================================

echo.
echo ==========================================================
echo Experiment 1: Mass-VarDrop alpha=0
echo Sanity Control
echo Expected: Original VarDrop attention behavior
echo ==========================================================

set "ALPHA0_START=%TIME%"

echo Start Time: !ALPHA0_START!
echo.

python -u run.py ^
  --is_training 1 ^
  --model_id traffic_96_96_mass_alpha0 ^
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
  --mass_vardrop ^
  --mass_alpha 0

set "ALPHA0_EXIT_CODE=!ERRORLEVEL!"
set "ALPHA0_END=%TIME%"

call :CalculateDuration "!ALPHA0_START!" "!ALPHA0_END!" ALPHA0_DURATION

echo.
echo ----------------------------------------------------------
echo Alpha = 0 finished
echo Start Time: !ALPHA0_START!
echo End Time:   !ALPHA0_END!
echo Duration:   !ALPHA0_DURATION!
echo Exit Code:  !ALPHA0_EXIT_CODE!
echo ----------------------------------------------------------
echo.


REM ============================================================
REM Stop if alpha=0 failed
REM ============================================================

if not "!ALPHA0_EXIT_CODE!"=="0" (
    echo.
    echo ==========================================================
    echo ERROR:
    echo Alpha=0 experiment failed.
    echo Alpha=1 will NOT be executed.
    echo ==========================================================
    echo.
    pause
    exit /b !ALPHA0_EXIT_CODE!
)


REM ============================================================
REM Experiment 2
REM Mass VarDrop - alpha = 1
REM ============================================================

echo.
echo ==========================================================
echo Experiment 2: Mass-VarDrop alpha=1
echo Full Mass-Preserving Attention Correction
echo ==========================================================

set "ALPHA1_START=%TIME%"

echo Start Time: !ALPHA1_START!
echo.

python -u run.py ^
  --is_training 1 ^
  --model_id traffic_96_96_mass_alpha1 ^
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
  --mass_vardrop ^
  --mass_alpha 1

set "ALPHA1_EXIT_CODE=!ERRORLEVEL!"
set "ALPHA1_END=%TIME%"

call :CalculateDuration "!ALPHA1_START!" "!ALPHA1_END!" ALPHA1_DURATION

echo.
echo ----------------------------------------------------------
echo Alpha = 1 finished
echo Start Time: !ALPHA1_START!
echo End Time:   !ALPHA1_END!
echo Duration:   !ALPHA1_DURATION!
echo Exit Code:  !ALPHA1_EXIT_CODE!
echo ----------------------------------------------------------
echo.


REM ============================================================
REM Total duration
REM ============================================================

call :CalculateDuration "!ALPHA0_START!" "!ALPHA1_END!" TOTAL_DURATION


REM ============================================================
REM Final Summary
REM ============================================================

echo.
echo.
echo ==========================================================
echo                  FINAL SUMMARY
echo ==========================================================
echo.
echo Dataset      : Traffic
echo Forecast     : 96 -^> 96
echo k            : %K%
echo group_size   : %GROUP_SIZE%
echo.
echo ----------------------------------------------------------
echo Alpha = 0
echo ----------------------------------------------------------
echo Model ID     : traffic_96_96_mass_alpha0
echo Start Time   : !ALPHA0_START!
echo End Time     : !ALPHA0_END!
echo Duration     : !ALPHA0_DURATION!
echo Exit Code    : !ALPHA0_EXIT_CODE!
echo.
echo ----------------------------------------------------------
echo Alpha = 1
echo ----------------------------------------------------------
echo Model ID     : traffic_96_96_mass_alpha1
echo Start Time   : !ALPHA1_START!
echo End Time     : !ALPHA1_END!
echo Duration     : !ALPHA1_DURATION!
echo Exit Code    : !ALPHA1_EXIT_CODE!
echo.
echo ----------------------------------------------------------
echo TOTAL
echo ----------------------------------------------------------
echo Total Start  : !ALPHA0_START!
echo Total End    : !ALPHA1_END!
echo Total Time   : !TOTAL_DURATION!
echo.
echo ==========================================================
echo ALL MASS-VARDROP EXPERIMENTS COMPLETED
echo ==========================================================
echo.

pause
goto :eof


REM ============================================================
REM Calculate Duration
REM Input format:
REM HH:MM:SS.cc
REM ============================================================

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

REM Handle crossing midnight
if !DIFF_CS! LSS 0 (
    set /a DIFF_CS+=24*60*60*100
)

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
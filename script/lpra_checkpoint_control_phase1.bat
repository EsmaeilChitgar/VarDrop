@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0.."
if errorlevel 1 exit /b 1

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"
set "FAILED=0"

REM ============================================================
REM LPRA SAME-CHECKPOINT / SAME-FORWARD TEST CONTROL
REM Reuses the five completed Phase-1 checkpoints.
REM NO TRAINING and NO LPRA calibration are performed here.
REM ============================================================

echo ==========================================================
echo LPRA PHASE-1 CHECKPOINT CONTROL - 5 COMPLETED RUNS
echo NO TRAINING - TEST ONLY
echo ==========================================================
echo.

call :RUN_ECL 96 "ECL_96_96_gpt3d_lpra_r32_phase1"
if errorlevel 1 goto :FAIL
call :RUN_TRAFFIC 192 "traffic_96_192_gpt3d_lpra_r32_phase1"
if errorlevel 1 goto :FAIL
call :RUN_ECL 336 "ECL_96_336_gpt3d_lpra_r32_phase1"
if errorlevel 1 goto :FAIL
call :RUN_TRAFFIC 336 "traffic_96_336_gpt3d_lpra_r32_phase1"
if errorlevel 1 goto :FAIL
call :RUN_TRAFFIC 720 "traffic_96_720_gpt3d_lpra_r32_phase1"
if errorlevel 1 goto :FAIL

goto :FINAL

:FAIL
set "FAILED=1"
echo.
echo ==========================================================
echo CONTROL BATCH STOPPED: A RUN FAILED
echo ==========================================================
echo.

:FINAL
echo.
echo ==========================================================
echo LPRA CHECKPOINT CONTROL FINISHED
echo Failed: !FAILED!
echo.
echo COPY BACK ALL [LPRA TEST CONTROL] LINES AND FINAL mse/mae.
echo ==========================================================
echo.
popd
if "!FAILED!"=="1" exit /b 1
exit /b 0

:RUN_TRAFFIC
set "PRED_LEN=%~1"
set "MODEL_ID=%~2"
echo.
echo ==========================================================
echo CONTROL TRAFFIC 96 -^> !PRED_LEN!
echo model_id: !MODEL_ID!
echo ==========================================================
%PYTHON% -u run.py ^
  --is_training 0 ^
  --model_id !MODEL_ID! ^
  --model OURS ^
  --data custom ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic.csv ^
  --features M ^
  --target OT ^
  --freq h ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len !PRED_LEN! ^
  --enc_in 862 ^
  --dec_in 862 ^
  --c_out 862 ^
  --d_model 512 ^
  --n_heads 8 ^
  --e_layers 4 ^
  --d_layers 1 ^
  --d_ff 512 ^
  --factor 1 ^
  --dropout 0.1 ^
  --embed timeF ^
  --activation gelu ^
  --batch_size 32 ^
  --learning_rate 0.001 ^
  --train_epochs 10 ^
  --patience 3 ^
  --k 4 ^
  --group_size 10 ^
  --itr 1 ^
  --num_workers 0 ^
  --use_norm 1 ^
  --exact_fast_vardrop ^
  --fast_log_every 100 ^
  --use_lpra ^
  --lpra_rank 32 ^
  --lpra_period 168 ^
  --lpra_cal_epochs 1 ^
  --lpra_lr 0.005 ^
  --lpra_alpha_max 1.25
exit /b !ERRORLEVEL!

:RUN_ECL
set "PRED_LEN=%~1"
set "MODEL_ID=%~2"
echo.
echo ==========================================================
echo CONTROL ECL 96 -^> !PRED_LEN!
echo model_id: !MODEL_ID!
echo ==========================================================
%PYTHON% -u run.py ^
  --is_training 0 ^
  --model_id !MODEL_ID! ^
  --model OURS ^
  --data custom ^
  --root_path ./dataset/electricity/ ^
  --data_path electricity.csv ^
  --features M ^
  --target OT ^
  --freq h ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len !PRED_LEN! ^
  --enc_in 321 ^
  --dec_in 321 ^
  --c_out 321 ^
  --d_model 512 ^
  --n_heads 8 ^
  --e_layers 3 ^
  --d_layers 1 ^
  --d_ff 512 ^
  --factor 1 ^
  --dropout 0.1 ^
  --embed timeF ^
  --activation gelu ^
  --batch_size 32 ^
  --learning_rate 0.0005 ^
  --train_epochs 10 ^
  --patience 3 ^
  --k 4 ^
  --group_size 10 ^
  --itr 1 ^
  --num_workers 0 ^
  --use_norm 1 ^
  --exact_fast_vardrop ^
  --fast_log_every 100 ^
  --use_lpra ^
  --lpra_rank 32 ^
  --lpra_period 168 ^
  --lpra_cal_epochs 1 ^
  --lpra_lr 0.005 ^
  --lpra_alpha_max 1.25
exit /b !ERRORLEVEL!

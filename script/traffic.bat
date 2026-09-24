@echo off
setlocal EnableExtensions EnableDelayedExpansion


echo ========================================================================================
echo GPT4 Diagnostic 2 - Traffic 96 to 96
echo Batch Spectral Redundancy vs Low-Rank Sensitivity
echo NO TRAINING - FIXED VARDROP SUBSETS - CHECKPOINT SCREEN ONLY
echo ========================================================================================
echo.

set "CHECKPOINT=.\checkpoints\traffic_96_96_gpt3b_fast_OURS_custom_M_ft96_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dttest_k4_gs10_projection_0_fastdfh\checkpoint.pth"
set "OUTPUT=.\diagnostics\results\traffic_gpt4_redundancy_rank"

if not exist "%CHECKPOINT%" (
    echo ERROR: checkpoint not found:
    echo %CHECKPOINT%
    exit /b 2
)

echo [1/2] Running helper self-test...
python -u diagnostics\gpt4_redundancy_rank_diag.py --self_test
if errorlevel 1 (
    echo ERROR: helper self-test failed.
    exit /b 3
)

echo.
echo [2/2] Running batch redundancy-rank diagnostic...
echo Checkpoint: %CHECKPOINT%
echo Output    : %OUTPUT%
echo.

python -u diagnostics\gpt4_redundancy_rank_diag.py ^
  --checkpoint_path "%CHECKPOINT%" ^
  --output_dir "%OUTPUT%" ^
  --diag_batches 48 ^
  --permutations 1999 ^
  --safe_gap_pct 3.0 ^
  --k 4 ^
  --group_size 10 ^
  --freq_start 1 ^
  --freq_end 25 ^
  --data custom ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic.csv ^
  --features M ^
  --target OT ^
  --freq h ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 96 ^
  --enc_in 862 ^
  --dec_in 862 ^
  --c_out 862 ^
  --d_model 512 ^
  --n_heads 8 ^
  --e_layers 4 ^
  --d_layers 1 ^
  --d_ff 512 ^
  --factor 1 ^
  --embed timeF ^
  --batch_size 32 ^
  --num_workers 0

set "EXIT_CODE=!ERRORLEVEL!"
echo.
echo ========================================================================================
echo GPT4 redundancy-rank diagnostic finished. Exit Code: !EXIT_CODE!
echo Results folder: %OUTPUT%
echo Please copy the REDUNDANCY, CONTROL, RANK SUMMARY, CORRELATION, QUARTILE,
echo and DECISION sections, or send summary.json + batch_metrics.csv.
echo ========================================================================================
exit /b !EXIT_CODE!

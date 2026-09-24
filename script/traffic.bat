@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0\.."

echo ======================================================================
echo GPT4 Diagnostic - Traffic 96 to 96
echo Weight Spectrum + Functional Low-Rank Ablation
echo NO TRAINING - CHECKPOINT SCREEN ONLY
echo ======================================================================
echo.

REM Use the exact FastVarDrop Traffic checkpoint. Its learned trajectory/results
REM matched Original VarDrop exactly in our previous control.
set "CHECKPOINT=.\checkpoints\traffic_96_96_gpt3b_fast_OURS_custom_M_ft96_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dttest_k4_gs10_projection_0_fastdfh\checkpoint.pth"
set "OUTPUT_DIR=.\diagnostics\results\traffic_gpt4_weight_rank"

if not exist "%CHECKPOINT%" (
    echo ERROR: checkpoint not found:
    echo   %CHECKPOINT%
    echo.
    echo Edit CHECKPOINT at the top of this BAT file if your checkpoint folder has a different name.
    exit /b 2
)

echo [1/2] Running diagnostic helper self-test...
python -u diagnostics\gpt4_weight_rank_diag.py --self_test
if errorlevel 1 (
    echo SELF-TEST FAILED. Real diagnostic will NOT run.
    exit /b 3
)

echo.
echo [2/2] Running checkpoint diagnostic...
echo Checkpoint: %CHECKPOINT%
echo Output    : %OUTPUT_DIR%
echo.

python -u diagnostics\gpt4_weight_rank_diag.py ^
  --checkpoint_path "%CHECKPOINT%" ^
  --output_dir "%OUTPUT_DIR%" ^
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
  --dropout 0.1 ^
  --embed timeF ^
  --activation gelu ^
  --batch_size 16 ^
  --num_workers 0 ^
  --diag_batches 16 ^
  --ranks 64,128,256 ^
  --modes attention,ffn,both ^
  --seed 2023 ^
  --svd_cpu

set "EXIT_CODE=!ERRORLEVEL!"
echo.
echo ======================================================================
echo GPT4 Diagnostic finished. Exit Code: !EXIT_CODE!
echo Results folder: %OUTPUT_DIR%
echo Please copy the CONTROL, SVD SANITY, FUNCTIONAL, SPECTRAL SUMMARY,
echo DECISION lines (or send summary.json + functional.csv).
echo ======================================================================
exit /b !EXIT_CODE!

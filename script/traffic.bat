@echo off
setlocal EnableDelayedExpansion

echo ==========================================================
echo GPT4-A Asymmetric Interaction Bottleneck - Traffic 96 to 96
echo Residual d_model = 512 ^| Attention internal = 256 ^| FFN = 128
echo ORIGINAL VarDrop sampler - NO GPT3b fast path
echo ==========================================================
echo.

echo [1/2] Running GPT4-A architecture sanity test...
python -u diagnostics\gpt4a_sanity.py --tokens 229
if errorlevel 1 (
  echo GPT4-A sanity test FAILED. Training aborted.
  exit /b 1
)
echo.

echo [2/2] Starting Traffic training...
set START_TIME=%TIME%

echo Start Time: %DATE% %TIME%
echo.

python -u run.py ^
  --is_training 1 ^
  --model_id traffic_96_96_gpt4a_a256_f128 ^
  --model OURS ^
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
  --gpt4_attn_dim 256 ^
  --n_heads 8 ^
  --e_layers 4 ^
  --d_layers 1 ^
  --d_ff 128 ^
  --factor 1 ^
  --embed timeF ^
  --batch_size 32 ^
  --learning_rate 0.001 ^
  --train_epochs 10 ^
  --patience 3 ^
  --k 4 ^
  --group_size 10 ^
  --itr 1 ^
  --num_workers 0

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ==========================================================
echo GPT4-A Traffic finished

echo End Time : %DATE% %TIME%
echo Exit Code: %EXIT_CODE%
echo ==========================================================
echo.
echo IMPORTANT:
echo Copy all Epoch cost times and final mse/mae.
echo Do NOT enable exact_fast_vardrop in this first isolation run.
echo.
pause
exit /b %EXIT_CODE%

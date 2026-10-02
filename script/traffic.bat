@echo off
setlocal
pushd "%~dp0.."
if errorlevel 1 exit /b 1

echo ==========================================
echo GPT5 - Traffic 96 Regression Test
echo ==========================================

git switch gpt5
if errorlevel 1 goto :ERROR

python -u run.py ^
 --is_training 0 ^
 --model_id traffic_96_96_gpt3b_fast ^
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
 --num_workers 0 ^
 --k 4 ^
 --group_size 10 ^
 --use_norm 1 ^
 --class_strategy projection ^
 --des test ^
 --itr 1 ^
 --exact_fast_vardrop ^
 --fast_log_every 100 > gpt5_traffic96_regression.txt 2>&1

if errorlevel 1 goto :ERROR

echo.
echo ==========================================
echo GPT5 RESULT
echo ==========================================
findstr /i "mse:" gpt5_traffic96_regression.txt

goto :END

:ERROR
echo.
echo ERROR:
type gpt5_traffic96_regression.txt

:END
popd
pause
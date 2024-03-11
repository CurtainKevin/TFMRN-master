CUDA_VISIBLE_DEVICES=0 nohup python -u run_long.py --model_name 'LightTS' --data WTH --itr True --pred_len 24  --seq_len 96  --hiddim 128 --lr 1e-4 --chunk_size 24 > weather24.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u run_long.py --model_name 'LightTS' --data WTH --itr True --pred_len 48  --seq_len 192 --hiddim 128 --lr 1e-4 --chunk_size 48 > weather48.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u run_long.py --model_name 'LightTS' --data WTH --itr True --pred_len 168 --seq_len 720 --hiddim 128 --lr 1e-4 --chunk_size 72 > weather168.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u run_long.py --model_name 'LightTS' --data WTH --itr True --pred_len 336 --seq_len 720 --hiddim 128 --lr 1e-4 --chunk_size 72 > weather336.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u run_long.py --model_name 'LightTS' --data WTH --itr True --pred_len 720 --seq_len 720 --hiddim 128 --lr 1e-4 --chunk_size 72 > weather720.log 2>&1 &

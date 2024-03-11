CUDA_VISIBLE_DEVICES=0 nohup python run_short.py --dataset_name solar_AL --window_size 160 --horizon  3 --chunk_size 40 --hiddim 256 --lr 9e-5 --batch_size 16 --model_name 'LightTS' --single_step 1 --epochs 200 > solar3s.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python run_short.py --dataset_name solar_AL --window_size 160 --horizon  6 --chunk_size 40 --hiddim 256 --lr 8e-5 --batch_size 16 --model_name 'LightTS' --single_step 1 --epochs 200 > solar6s.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python run_short.py --dataset_name solar_AL --window_size 160 --horizon 12 --chunk_size 40 --hiddim 256 --lr 6e-5 --batch_size 16 --model_name 'LightTS' --single_step 1 --epochs 200 > solar12s.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python run_short.py --dataset_name solar_AL --window_size 160 --horizon 24 --chunk_size 40 --hiddim 256 --lr 4e-5 --batch_size 16 --model_name 'LightTS' --single_step 1 --epochs 200 > solar24s.log 2>&1 &
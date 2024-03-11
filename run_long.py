import argparse
import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import ast

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from experiments.exp_long import Exp_long

parser = argparse.ArgumentParser(description='TFMRN on ETT dataset')

# -------  dataset settings --------------
parser.add_argument('--data', type=str, required=False, default='ETTh1',
                    choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'WTH', 'electricity', 'ILI', 'traffic',
                             'exchange'],
                    help='name of dataset')
parser.add_argument('--root_path', type=str, default='../datasets/long/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='location of the data file')
parser.add_argument('--features', type=str, default='M', choices=['S', 'M'],
                    help='features S is univariate, M is multivariate')
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='exp/ETT_checkpoints/', help='location of model checkpoints')
parser.add_argument('--inverse', type=bool, default=False, help='denorm the output data')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')

# -------  device settings --------------
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

# -------  input/output length settings --------------
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of TFMRN, look back window')
parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length, horizon')
parser.add_argument('--lastWeight', type=float, default=1.0)

# -------  training settings --------------
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=bool, default=False, help='multiple seeds or not')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=int, default=1, help='adjust learning rate')
parser.add_argument('--save', type=bool, default=False, help='save the output results')
parser.add_argument('--enc_num', type=int, default=1, help='')
parser.add_argument('--dec_num', type=int, default=1, help='')
parser.add_argument('--model_name', type=str, default='TFMRN')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=False)

# -------  model settings --------------
parser.add_argument('--hiddim', default=512, type=int, help='hidden channel of module')
parser.add_argument('--num_layers', default=1, type=int, help='hidden channel of module')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--chunk_size', type=int, default=20)
parser.add_argument('--topk', type=int, default=8)

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'WTH': {'data': 'WTH.csv', 'T': 'OT', 'M': [21, 21, 21], 'S': [1, 1, 1], 'MS': [21, 21, 1]},
    'electricity': {'data': 'electricity.csv', 'T': 'OT', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
    'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
    'ILI': {'data': 'ILI.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'traffic': {'data': 'traffic.csv', 'T': 'OT', 'M': [862, 862, 862], 'S': [1, 1, 1], 'MS': [862, 862, 1]},
    'exchange': {'data': 'exchange_rate.csv', 'T': 'OT', 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [8, 8, 1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]
# args.label_len = args.seq_len
args.label_len = 0
print('Args in experiment:')
print(args)

torch.manual_seed(4321)  # reproducible
torch.cuda.manual_seed_all(4321)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
torch.backends.cudnn.enabled = True

Exp = Exp_long

mae_ = []
maes_ = []
mse_ = []
mses_ = []

seeds = [4321, 1234, 2021, 255, 1023]

if args.evaluate:
    setting = '{}_{}_ft{}_sl{}_pl{}_lr{}_bs{}_dim{}_itr0'.format(args.model_name, args.data, args.features,
                                                                 args.seq_len,
                                                                 args.pred_len, args.lr, args.batch_size, args.hiddim)
    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, maes, mse, mses = exp.test(setting, evaluate=True)
    print('Final mean normed mse:{:.4f},mae:{:.4f},denormed mse:{:.4f},mae:{:.4f}'.format(mse, mae, mses, maes))
else:
    if args.itr:
        for ii in range(len(seeds)):
            seed = seeds[ii]
            torch.manual_seed(seed)  # reproducible
            torch.cuda.manual_seed_all(seed)
            # setting record of experiments
            setting = '{}_{}_ft{}_sl{}_pl{}_lr{}_bs{}_dim{}_enum{}_dnum{}_seed{}'.format(args.model_name, args.data,
                                                                                         args.features,
                                                                                         args.seq_len, args.pred_len,
                                                                                         args.lr,
                                                                                         args.batch_size, args.hiddim,
                                                                                         args.enc_num, args.dec_num,
                                                                                         seed)

            exp = Exp(args)  # set experiments
            before_train = datetime.now().timestamp()
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

            exp.train(setting)
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mae, mse = exp.test(setting)
            mae_.append(mae)
            mse_.append(mse)

            torch.cuda.empty_cache()
        setting = '{}_{}_ft{}_sl{}_pl{}_lr{}_bs{}_dim{}_enum{}_dnum{}'.format(args.model_name, args.data, args.features,
                                                                              args.seq_len,
                                                                              args.pred_len, args.lr, args.batch_size,
                                                                              args.hiddim,
                                                                              args.enc_num, args.dec_num)
        print('{} random seeds finished with {}:'.format(len(seeds), setting))
        print('Final mean normed mse:{:.4f}, std mse:{:.4f}, mae:{:.4f}, std mae:{:.4f}'.format(np.mean(mse_),
                                                                                                np.std(mse_),
                                                                                                np.mean(mae_),
                                                                                                np.std(mae_)))
        # print('Final mean denormed mse:{:.4f}, std mse:{:.4f}, mae:{:.4f}, std mae:{:.4f}'.format(np.mean(mses_),np.std(mses_), np.mean(maes_), np.std(maes_)))
        print('Final min normed mse:{:.4f}, mae:{:.4f}'.format(min(mse_), min(mae_)))
        # print('Final min denormed mse:{:.4f}, mae:{:.4f}'.format(min(mses_), min(maes_)))
    else:
        seed = seeds[0]
        torch.manual_seed(seed)  # reproducible
        torch.cuda.manual_seed_all(seed)
        setting = '{}_{}_ft{}_sl{}_pl{}_lr{}_bs{}_dim{}_itr0'.format(args.model_name, args.data, args.features,
                                                                     args.seq_len,
                                                                     args.pred_len, args.lr, args.hiddim,
                                                                     args.batch_size)
        exp = Exp(args)  # set experiments
        before_train = datetime.now().timestamp()
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

        exp.train(setting)
        after_train = datetime.now().timestamp()
        print(f'Training took {(after_train - before_train) / 60} minutes')

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mae, mse = exp.test(setting)
        print('\033[1;31;40m Final mean normed mse:{:.4f},mae:{:.4f} \033[0m'.format(mse, mae))

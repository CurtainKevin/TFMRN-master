from metrics.ETTh_metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, save_model_ETT, load_model_ETT, plot_line
from experiments.exp_basic import Exp_Basic
from data_provider.long_dataloader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
import os
import time

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
from models.Layers import decomp_k

warnings.filterwarnings('ignore')

from models.Layers import *
from ptflops import get_model_complexity_info
from models.tfmrn import TFMRN
from thop import profile

class Exp_long(Exp_Basic):
    def __init__(self, args):
        super(Exp_long, self).__init__(args)

    def _build_model(self):

        if self.args.features == 'S':
            in_dim = 1
        elif self.args.features == 'M':
            in_dim = self.args.enc_in
        else:
            print('Error!')


        model = TFMRN(lookback=self.args.seq_len, lookahead=self.args.pred_len, hid_dim=self.args.hiddim,
                        num_node=self.args.enc_in, dropout=self.args.dropout, chunk_size=self.args.chunk_size)
        print(model)

        # macs, params = get_model_complexity_info(
        #     model.double(), (self.args.seq_len, self.args.enc_in), as_strings=False, verbose=False
        # )
        # print('Mac: {}, Params: {}'.format(macs, params))

        # input1 = torch.randn(self.args.batch_size, self.args.seq_len, self.args.enc_in, dtype=torch.float)
        # flops, params = profile(model, inputs=(input1,))
        # print('FLOPs = ' + str(flops / 1000 ** 2) + 'M')
        # print('Params = ' + str(params / 1000 ** 2) + 'M')


        return model.double().cuda()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            # 'ILI': Dataset_Custom,
            # 'traffic': Dataset_Custom,
            # 'exchange_rate': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data if self.args.data in data_dict else 'custom']
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim

    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def valid(self, valid_data, valid_loader, criterion):
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        pbar = tqdm(valid_loader,
                    total=len(valid_loader),
                    leave=True,
                    ncols=100,
                    unit="it",
                    unit_scale=False)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pbar):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                valid_data, batch_x, batch_y, batch_x_mark)

            loss = criterion(pred.detach().cpu(), true.detach().cpu())

            total_loss.append(loss)
            pbar.set_description(f"Iter.{i}")
            pbar.set_postfix({"loss": loss.item()})
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        total_loss = np.average(total_loss)
        mae, mse = metric(preds, trues)

        print('normed mse:{:.4f}, mae:{:.4f}'.format(mse, mae))

        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        valid_data, valid_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        writer = SummaryWriter(
            'event/run_ETTh/{}'.format(self.args.model_name))

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        # criterion = self._select_criterion("mae")

        epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            pbar = tqdm(train_loader,
                        total=len(train_loader),
                        leave=True,
                        ncols=100,
                        unit="it",
                        unit_scale=False)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pbar):
                iter_count += 1

                model_optim.zero_grad()
                pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                    train_data, batch_x, batch_y, batch_x_mark)

                loss = criterion(pred, true)
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                train_loss.append(loss.item())

                pbar.set_description(f"Iter.{i}")
                pbar.set_postfix({"loss": train_loss[i]})

                loss.backward()
                model_optim.step()


            print("Epoch: {} cost time: {}".format(
                epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print('--------start to validate-----------')
            valid_loss = self.valid(valid_data, valid_loader, criterion)
            print('--------start to test-----------')
            test_loss = self.valid(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss, test_loss))

            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
            writer.add_scalar('test_loss', test_loss, global_step=epoch)

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping with setting{} and epoch{}".format(setting, epoch + 1))
                break

            lr = adjust_learning_rate(model_optim, epoch + 1, self.args)

        save_model_ETT(epoch, lr, self.model, path, model_name=self.args.data, horizon=self.args.pred_len)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, evaluate=False):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []

        if evaluate:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        pbar = tqdm(test_loader,
                    total=len(test_loader),
                    leave=True,
                    ncols=100,
                    unit="it",
                    unit_scale=False)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pbar):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                test_data, batch_x, batch_y, batch_x_mark)

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())

            pbar.set_description(f"Iter.{i}")
        preds = np.array(preds)
        trues = np.array(trues)

        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        true_scales = true_scales.reshape(-1,
                                          true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1,
                                          pred_scales.shape[-2], pred_scales.shape[-1])
        mae, mse = metric(preds, trues)

        print('normed mse:{:.4f}, mae:{:.4f}'.format(mse, mae))

        # result save
        if self.args.save:
            folder_path = 'exp/ett_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            print('Test:mse:{:.4f}, mae:{:.4f}'.format(mse, mae))

            np.save(folder_path + 'metrics.npy',
                    np.array([mae, mse]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            np.save(folder_path + 'pred_scales.npy', pred_scales)
            np.save(folder_path + 'true_scales.npy', true_scales)

            np.savetxt(f'{folder_path}/pred.csv', preds[0], delimiter=",")
            np.savetxt(f'{folder_path}/true.csv', trues[0], delimiter=",")
            np.savetxt(f'{folder_path}/pred_scales.csv',
                       pred_scales[0], delimiter=",")
            np.savetxt(f'{folder_path}/true_scales.csv',
                       true_scales[0], delimiter=",")

        return mae, mse

    def _process_one_batch_SCINet(self, dataset_object, batch_x, batch_y, batch_x_mark):
        batch_x = batch_x.double().cuda()
        # batch_x_mark = batch_x_mark.double().cuda()
        batch_y = batch_y.double().cuda()

        outputs = self.model(batch_x)

        # if self.args.inverse:
        outputs_scaled = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
        batch_y_scaled = dataset_object.inverse_transform(batch_y)

        return outputs, outputs_scaled, 0, 0, batch_y, batch_y_scaled

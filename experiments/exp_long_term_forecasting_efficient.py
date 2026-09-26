from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import pdb
import numpy as np
import random

warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from VarDrop import efficient_sampler, ExactFastVarDropSampler


class Exp_Long_Term_Forecast_Efficient(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Efficient, self).__init__(args)

        # GPT3b: exact fast k-DFH. No caching or approximation.
        self.exact_fast_vardrop = bool(
            getattr(args, 'exact_fast_vardrop', False)
        )
        self.fast_vardrop_sampler = None

        if self.exact_fast_vardrop:
            self.fast_vardrop_sampler = ExactFastVarDropSampler(
                k=args.k,
                group_size=args.group_size,
                freq_list=range(1, 25),
                log_every=getattr(args, 'fast_log_every', 100)
            )

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _model_core(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _lpra_enabled(self):
        return bool(getattr(self.args, 'use_lpra', False))

    def _future_week_phase(self, batch_y_mark):
        """Decode hour-of-week from iTransformer's hourly timeF features.

        For freq='h', time_features are [HourOfDay, DayOfWeek, ...], normalized
        respectively as hour/23-0.5 and weekday/6-0.5.  We only use future
        marks, so no target values are involved and there is no leakage.
        """
        if batch_y_mark is None:
            raise ValueError('LPRA currently requires timeF marks (Traffic/ECL style data).')
        if batch_y_mark.shape[-1] < 2:
            raise ValueError('LPRA requires HourOfDay and DayOfWeek timeF features.')
        period = int(getattr(self.args, 'lpra_period', 168))
        if period != 168:
            raise ValueError('The first LPRA experiment supports weekly hourly period=168 only.')

        future_mark = batch_y_mark[:, -self.args.pred_len:, :]
        hour = torch.round((future_mark[..., 0] + 0.5) * 23.0).long().clamp_(0, 23)
        weekday = torch.round((future_mark[..., 1] + 0.5) * 6.0).long().clamp_(0, 6)
        return weekday * 24 + hour

    def _lpra_channel_ids(self, indices=None, n_channels=None):
        if indices is None:
            if n_channels is None:
                raise ValueError('n_channels is required for full-channel LPRA inference')
            return torch.arange(n_channels, device=self.device, dtype=torch.long)
        return torch.as_tensor(indices, device=self.device, dtype=torch.long)

    def _apply_lpra(self, outputs, channel_ids, batch_y_mark):
        if not self._lpra_enabled():
            return outputs
        core = self._model_core()
        phase = self._future_week_phase(batch_y_mark)
        correction = core.lpra_correction(channel_ids, phase)
        return outputs + core.lpra_alpha * correction

    def _calibrate_lpra(self, train_loader, vali_loader, criterion, path):
        """Freeze GPT3b, fit only LPRA on sparse training batches, then choose
        a validation-only shrinkage alpha. alpha=0 is always an allowed fallback.
        """
        if not self._lpra_enabled():
            return

        core = self._model_core()
        if core.lpra is None:
            raise RuntimeError('use_lpra=True but model has no LPRA module')

        cal_epochs = int(getattr(self.args, 'lpra_cal_epochs', 1))
        cal_lr = float(getattr(self.args, 'lpra_lr', 5e-3))
        alpha_max = float(getattr(self.args, 'lpra_alpha_max', 1.25))

        # Freeze the complete GPT3b backbone; only the adapter is optimized.
        for p in core.parameters():
            p.requires_grad_(False)
        for p in core.lpra.parameters():
            p.requires_grad_(True)
        core.set_lpra_alpha(0.0)

        adapter_optim = optim.Adam(core.lpra.parameters(), lr=cal_lr)
        print('[LPRA] calibration start: epochs={} lr={} params={}'.format(
            cal_epochs, cal_lr, core.lpra.num_parameters))

        # Keep the frozen backbone deterministic during calibration.
        self.model.eval()
        cal_start = time.time()
        for epoch in range(cal_epochs):
            losses = []
            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.fast_vardrop_sampler is not None:
                    sparse_indices = self.fast_vardrop_sampler(batch_x)
                else:
                    sparse_indices = efficient_sampler(
                        batch_x,
                        k=self.args.k,
                        group_size=self.args.group_size,
                        freq_list=range(1, 25)
                    )
                sparse_indices = np.unique(sparse_indices)
                channel_ids = self._lpra_channel_ids(indices=sparse_indices)

                batch_x_sparse = batch_x[:, :, sparse_indices]
                batch_y_sparse = batch_y[:, :, sparse_indices]
                dec_inp = torch.zeros_like(batch_y_sparse[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y_sparse[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                with torch.no_grad():
                    base = self.model(batch_x_sparse, batch_x_mark, dec_inp, batch_y_mark)
                    base = base[:, -self.args.pred_len:, :]
                    target = batch_y_sparse[:, -self.args.pred_len:, :]

                phase = self._future_week_phase(batch_y_mark)
                correction = core.lpra_correction(channel_ids, phase)
                loss = criterion(base.detach() + correction, target)

                adapter_optim.zero_grad()
                loss.backward()
                adapter_optim.step()
                losses.append(loss.item())

            print('[LPRA] calibration epoch {}/{} loss={:.7f}'.format(
                epoch + 1, cal_epochs, float(np.mean(losses))))

        print('[LPRA] calibration time: {:.3f}s'.format(time.time() - cal_start))

        # Validation-only scalar shrinkage.  Closed-form MSE alpha is tried first;
        # if validation MAE regresses, alpha is halved until both metrics are no
        # worse than GPT3b.  alpha=0 is the exact GPT3b fallback.
        num = 0.0
        den = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)
                base = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                base = base[:, -self.args.pred_len:, :]
                target = batch_y[:, -self.args.pred_len:, :]
                ids = self._lpra_channel_ids(n_channels=base.shape[-1])
                phase = self._future_week_phase(batch_y_mark)
                corr = core.lpra_correction(ids, phase)

                residual = target - base
                num += torch.sum(residual * corr).item()
                den += torch.sum(corr * corr).item()

        alpha_closed = 0.0 if den <= 1e-20 else num / den
        alpha_closed = float(np.clip(alpha_closed, 0.0, alpha_max))

        # Evaluate alpha_closed and progressively safer halvings in one streaming
        # validation pass.  This avoids caching the large [B,96,862] tensors.
        candidates = [alpha_closed * (0.5 ** i) for i in range(9)]
        candidates.append(0.0)
        candidates = list(dict.fromkeys(float(a) for a in candidates))
        stats = {
            a: {'se': 0.0, 'ae': 0.0, 'count': 0}
            for a in candidates
        }

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)
                base = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                base = base[:, -self.args.pred_len:, :]
                target = batch_y[:, -self.args.pred_len:, :]
                ids = self._lpra_channel_ids(n_channels=base.shape[-1])
                phase = self._future_week_phase(batch_y_mark)
                corr = core.lpra_correction(ids, phase)

                for a in candidates:
                    err = base + a * corr - target
                    stats[a]['se'] += torch.sum(err * err).item()
                    stats[a]['ae'] += torch.sum(torch.abs(err)).item()
                    stats[a]['count'] += err.numel()

        def _metric_for(a):
            st = stats[a]
            return st['se'] / st['count'], st['ae'] / st['count']

        base_mse, base_mae = _metric_for(0.0)
        alpha = 0.0
        final_mse, final_mae = base_mse, base_mae
        for a in candidates:
            mse_a, mae_a = _metric_for(a)
            if mse_a <= base_mse + 1e-12 and mae_a <= base_mae + 1e-12:
                alpha = a
                final_mse, final_mae = mse_a, mae_a
                break

        core.set_lpra_alpha(alpha)
        mse_gain = 100.0 * (base_mse - final_mse) / max(base_mse, 1e-12)
        mae_gain = 100.0 * (base_mae - final_mae) / max(base_mae, 1e-12)
        print('[LPRA] alpha_closed={:.6f} alpha={:.6f}'.format(alpha_closed, alpha))
        print('[LPRA] validation base MSE={:.7f} MAE={:.7f}'.format(base_mse, base_mae))
        print('[LPRA] validation final MSE={:.7f} MAE={:.7f}'.format(final_mse, final_mae))
        print('[LPRA] validation gain MSE={:.3f}% MAE={:.3f}%'.format(mse_gain, mae_gain))

        torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint_lpra.pth'))
        print('[LPRA] saved {}'.format(os.path.join(path, 'checkpoint_lpra.pth')))

    def vali(self, vali_data, vali_loader, criterion, partial_train=False):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    elif self.args.channel_independence:
                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape
                        if batch_x_mark == None:
                            outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1), batch_x_mark, \
                                                 dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1), batch_y_mark).reshape(
                                B, N, -1).permute(0, 2, 1)
                        else:
                            outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1),
                                                 batch_x_mark.repeat(N, 1, 1), \
                                                 dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1),
                                                 batch_y_mark.repeat(N, 1, 1)) \
                                .reshape(B, N, -1).permute(0, 2, 1)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # VarDrop ----------------------------
                if self.fast_vardrop_sampler is not None:
                    sparse_indices = self.fast_vardrop_sampler(batch_x)
                else:
                    sparse_indices = efficient_sampler(
                        batch_x,
                        k=self.args.k,
                        group_size=self.args.group_size,
                        freq_list=range(1,25)
                    )

                # Preserve original experiment behavior exactly.
                sparse_indices = np.unique(sparse_indices)

                batch_x = batch_x[:, :, sparse_indices]
                batch_y = batch_y[:, :, sparse_indices]
                # ------------------------------------
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    elif self.args.channel_independence:
                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape
                        if batch_x_mark == None:
                            outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1), batch_x_mark, \
                                                 dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1), batch_y_mark).reshape(
                                B, N, -1).permute(0, 2, 1)
                        else:
                            a = batch_x.permute(0, 2, 1)
                            b = batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1)
                            outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1),
                                                 batch_x_mark.repeat(N, 1, 1), \
                                                 dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1),
                                                 batch_y_mark.repeat(N, 1, 1)) \
                                .reshape(B, N, -1).permute(0, 2, 1)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.fast_vardrop_sampler is not None:
                print(
                    self.fast_vardrop_sampler.format_status(
                        prefix=f"[FastVarDrop Epoch {epoch + 1}]"
                    )
                )
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, partial_train=False)
            test_loss = self.vali(test_data, test_loader, criterion, partial_train=False)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        if self._lpra_enabled():
            self._calibrate_lpra(train_loader, vali_loader, criterion, path)

        return self.model

    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            checkpoint_name = 'checkpoint_lpra.pth' if self._lpra_enabled() else 'checkpoint.pth'
            checkpoint_path = os.path.join('./checkpoints/' + setting, checkpoint_name)
            if self._lpra_enabled() and not os.path.exists(checkpoint_path):
                raise FileNotFoundError('LPRA checkpoint not found: ' + checkpoint_path)
            self.model.load_state_dict(torch.load(checkpoint_path))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # During model inference, test the obtained model directly on all variates.
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    elif self.args.channel_independence:  # compare the result with channel_independence
                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape
                        if batch_x_mark == None:
                            outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1), batch_x_mark, \
                                                 dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1), batch_y_mark).reshape(
                                B, N, -1).permute(0, 2, 1)
                        else:
                            outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1),
                                                 batch_x_mark.repeat(N, 1, 1), \
                                                 dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1),
                                                 batch_y_mark.repeat(N, 1, 1)) \
                                .reshape(B, N, -1).permute(0, 2, 1)
                    else:
                        # directly test the trained model on all variates without fine-tuning.
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self._lpra_enabled():
                    full_ids = self._lpra_channel_ids(n_channels=outputs.shape[-1])
                    outputs = self._apply_lpra(outputs, full_ids, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

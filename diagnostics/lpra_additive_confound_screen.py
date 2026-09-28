import argparse
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch import optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.exp_long_term_forecasting_efficient import Exp_Long_Term_Forecast_Efficient


TRAFFIC96_SETTING = (
    'traffic_96_96_gpt3d_lpra_r32_OURS_custom_M_'
    'ft96_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_'
    'fctimeF_ebTrue_dttest_k4_gs10_projection_0_'
    'fastdfh_lpra_r32_p168'
)


class ChannelOnlyResidual(nn.Module):
    """One learned residual scalar per channel, constant over forecast phase."""
    def __init__(self, num_channels):
        super().__init__()
        self.bias = nn.Embedding(int(num_channels), 1)
        nn.init.zeros_(self.bias.weight)

    def forward(self, phase_indices, channel_ids):
        channel_bias = self.bias(channel_ids).squeeze(-1)  # [N]
        return channel_bias.view(1, 1, -1).expand(
            phase_indices.shape[0], phase_indices.shape[1], -1
        )

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class AdditiveChannelPhaseResidual(nn.Module):
    """Additive correction C[n,p] = a[n] + b[p], with no interaction term."""
    def __init__(self, num_channels, period):
        super().__init__()
        self.channel_bias = nn.Embedding(int(num_channels), 1)
        self.phase_bias = nn.Embedding(int(period), 1)
        nn.init.zeros_(self.channel_bias.weight)
        nn.init.zeros_(self.phase_bias.weight)

    def forward(self, phase_indices, channel_ids):
        channel = self.channel_bias(channel_ids).squeeze(-1).view(1, 1, -1)
        phase = self.phase_bias(phase_indices)  # [B,H,1]
        return channel + phase

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_project_args():
    return SimpleNamespace(
        is_training=0,
        model_id='traffic_96_96_gpt3d_lpra_r32',
        model='OURS',
        data='custom',
        root_path='./dataset/traffic/',
        data_path='traffic.csv',
        features='M',
        target='OT',
        freq='h',
        checkpoints='./checkpoints/',
        seq_len=96,
        label_len=48,
        pred_len=96,
        enc_in=862,
        dec_in=862,
        c_out=862,
        d_model=512,
        n_heads=8,
        e_layers=4,
        d_layers=1,
        d_ff=512,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed='timeF',
        activation='gelu',
        output_attention=False,
        do_predict=False,
        num_workers=0,
        itr=1,
        train_epochs=10,
        batch_size=32,
        patience=3,
        learning_rate=0.001,
        des='test',
        loss='MSE',
        lradj='type1',
        use_amp=False,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices='0,1,2,3',
        exp_name='MTSF',
        channel_independence=False,
        inverse=False,
        class_strategy='projection',
        target_root_path='./data/electricity/',
        target_data_path='electricity.csv',
        efficient_training=False,
        use_norm=1,
        partial_start_index=0,
        k=4,
        group_size=10,
        exact_fast_vardrop=True,
        fast_log_every=0,
        use_lpra=True,
        lpra_rank=32,
        lpra_period=168,
        lpra_cal_epochs=1,
        lpra_lr=0.005,
        lpra_alpha_max=1.25,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_batch(exp, batch):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    return (
        batch_x.float().to(exp.device),
        batch_y.float().to(exp.device),
        batch_x_mark.float().to(exp.device),
        batch_y_mark.float().to(exp.device),
    )


def base_forward(exp, batch_x, batch_y, batch_x_mark, batch_y_mark, sparse_indices=None):
    if sparse_indices is not None:
        batch_x = batch_x[:, :, sparse_indices]
        batch_y = batch_y[:, :, sparse_indices]

    dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
    dec_inp = torch.cat(
        [batch_y[:, :exp.args.label_len, :], dec_inp], dim=1
    ).float().to(exp.device)

    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    if exp.args.output_attention:
        outputs = outputs[0]
    return (
        outputs[:, -exp.args.pred_len:, :],
        batch_y[:, -exp.args.pred_len:, :],
    )


def calibrate_shared(exp, adapters, lr, seed):
    set_seed(seed)
    _, train_loader = exp._get_data(flag='train')
    opts = {name: optim.Adam(module.parameters(), lr=lr) for name, module in adapters.items()}
    losses = {name: [] for name in adapters}

    exp.model.eval()
    for module in adapters.values():
        module.train()

    start = time.time()
    for batch in train_loader:
        batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(exp, batch)

        if exp.fast_vardrop_sampler is None:
            raise RuntimeError('This diagnostic expects Exact Fast VarDrop.')
        sparse_indices = np.unique(exp.fast_vardrop_sampler(batch_x))
        channel_ids = torch.as_tensor(
            sparse_indices, device=exp.device, dtype=torch.long
        )

        with torch.no_grad():
            base, target = base_forward(
                exp, batch_x, batch_y, batch_x_mark, batch_y_mark,
                sparse_indices=sparse_indices
            )
        phase = exp._future_week_phase(batch_y_mark)

        for name, module in adapters.items():
            corr = module(phase, channel_ids)
            loss = torch.mean((base.detach() + corr - target) ** 2)
            opts[name].zero_grad()
            loss.backward()
            opts[name].step()
            losses[name].append(float(loss.item()))

    elapsed = time.time() - start
    print('[CONFOUND SCREEN] shared calibration pass complete')
    print('[CONFOUND SCREEN] calibration time={:.3f}s'.format(elapsed))
    for name in adapters:
        print('[CONFOUND SCREEN] {} train loss={:.7f}'.format(
            name, float(np.mean(losses[name]))
        ))


def choose_alpha_and_evaluate_validation(exp, adapters, alpha_max, seed):
    core = exp._model_core()
    lpra_alpha = core.get_lpra_alpha()

    # Pass 1: closed-form alpha for diagnostic controls.
    set_seed(seed)
    _, loader = exp._get_data(flag='val')
    num = {name: 0.0 for name in adapters}
    den = {name: 0.0 for name in adapters}

    exp.model.eval()
    for module in adapters.values():
        module.eval()

    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(exp, batch)
            base, target = base_forward(exp, batch_x, batch_y, batch_x_mark, batch_y_mark)
            ids = torch.arange(base.shape[-1], device=exp.device, dtype=torch.long)
            phase = exp._future_week_phase(batch_y_mark)
            residual = target - base
            for name, module in adapters.items():
                corr = module(phase, ids)
                num[name] += torch.sum(residual * corr).item()
                den[name] += torch.sum(corr * corr).item()

    alpha_closed = {}
    candidates = {}
    for name in adapters:
        a = 0.0 if den[name] <= 1e-20 else num[name] / den[name]
        a = float(np.clip(a, 0.0, alpha_max))
        alpha_closed[name] = a
        candidates[name] = list(dict.fromkeys(
            [float(a * (0.5 ** i)) for i in range(9)] + [0.0]
        ))

    # Pass 2: validation-only selection and matched evaluation.
    set_seed(seed)
    _, loader = exp._get_data(flag='val')
    names = ['base', 'lpra'] + list(adapters.keys())
    stats = {name: {'se': 0.0, 'ae': 0.0, 'count': 0} for name in names}
    control_stats = {
        name: {
            a: {'se': 0.0, 'ae': 0.0, 'count': 0}
            for a in candidates[name]
        }
        for name in adapters
    }

    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(exp, batch)
            base, target = base_forward(exp, batch_x, batch_y, batch_x_mark, batch_y_mark)
            ids = torch.arange(base.shape[-1], device=exp.device, dtype=torch.long)
            phase = exp._future_week_phase(batch_y_mark)

            def update(name, pred):
                err = (pred - target).double()
                stats[name]['se'] += torch.sum(err * err).item()
                stats[name]['ae'] += torch.sum(torch.abs(err)).item()
                stats[name]['count'] += err.numel()

            update('base', base)
            lpra_corr = core.lpra_correction(ids, phase)
            update('lpra', base + lpra_alpha * lpra_corr)

            for name, module in adapters.items():
                corr = module(phase, ids)
                for a in candidates[name]:
                    err = (base + a * corr - target).double()
                    st = control_stats[name][a]
                    st['se'] += torch.sum(err * err).item()
                    st['ae'] += torch.sum(torch.abs(err)).item()
                    st['count'] += err.numel()

    def metric(st):
        return st['se'] / st['count'], st['ae'] / st['count']

    base_mse, base_mae = metric(stats['base'])
    lpra_mse, lpra_mae = metric(stats['lpra'])

    selected = {}
    control_metrics = {}
    for name in adapters:
        chosen = 0.0
        final = (base_mse, base_mae)
        for a in candidates[name]:
            mse, mae = metric(control_stats[name][a])
            if mse <= base_mse + 1e-12 and mae <= base_mae + 1e-12:
                chosen = a
                final = (mse, mae)
                break
        selected[name] = chosen
        control_metrics[name] = final

    print('')
    print('==========================================================')
    print('LPRA ADDITIVE CONFOUND SCREEN - VALIDATION ONLY')
    print('==========================================================')
    print('[CONFOUND SCREEN] parameter counts')
    for name, module in adapters.items():
        print('  {:14s}: {}'.format(name, module.num_parameters))
    print('  LPRA r32      : {}'.format(core.lpra.num_parameters))
    print('[CONFOUND SCREEN] alphas')
    print('  LPRA existing : {:.9f}'.format(lpra_alpha))
    for name in adapters:
        print('  {:14s}: closed={:.9f} selected={:.9f}'.format(
            name, alpha_closed[name], selected[name]
        ))

    def report(label, mse, mae):
        mse_gain = 100.0 * (base_mse - mse) / max(base_mse, 1e-12)
        mae_gain = 100.0 * (base_mae - mae) / max(base_mae, 1e-12)
        print('[CONFOUND SCREEN RESULT] {:14s} MSE={:.9f} MAE={:.9f} gain={:.3f}%/{:.3f}%'.format(
            label, mse, mae, mse_gain, mae_gain
        ))

    report('BASE', base_mse, base_mae)
    report('LPRA', lpra_mse, lpra_mae)
    for name in adapters:
        report(name.upper().replace('-', '_'), *control_metrics[name])

    for name in adapters:
        gap = 100.0 * (control_metrics[name][0] - lpra_mse) / max(lpra_mse, 1e-12)
        print('[CONFOUND SCREEN] {} MSE gap vs LPRA={:+.3f}%'.format(name, gap))
    print('==========================================================')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', default=os.path.join('./checkpoints', TRAFFIC96_SETTING))
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--alpha_max', type=float, default=1.25)
    cli = p.parse_args()

    ckpt = os.path.join(cli.checkpoint_dir, 'checkpoint_lpra.pth')
    if not os.path.exists(ckpt):
        raise FileNotFoundError('checkpoint_lpra.pth not found: ' + ckpt)

    set_seed(cli.seed)
    args = build_project_args()
    exp = Exp_Long_Term_Forecast_Efficient(args)

    print('[CONFOUND SCREEN] loading existing Traffic96 LPRA checkpoint:')
    print('  ' + ckpt)
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    exp.model.eval()

    core = exp._model_core()
    channel = ChannelOnlyResidual(args.enc_in).to(exp.device)
    additive = AdditiveChannelPhaseResidual(args.enc_in, args.lpra_period).to(exp.device)
    adapters = {
        'channel-only': channel,
        'channel+phase': additive,
    }

    print('[CONFOUND SCREEN] NO backbone training')
    print('[CONFOUND SCREEN] NO test-set evaluation')
    print('[CONFOUND SCREEN] controls share ONE train calibration pass')
    print('[CONFOUND SCREEN] params: channel-only={} channel+phase={} LPRA={}'.format(
        channel.num_parameters, additive.num_parameters, core.lpra.num_parameters
    ))

    calibrate_shared(exp, adapters, lr=cli.lr, seed=cli.seed + 17011)
    choose_alpha_and_evaluate_validation(
        exp, adapters, alpha_max=cli.alpha_max, seed=cli.seed + 19001
    )


if __name__ == '__main__':
    main()

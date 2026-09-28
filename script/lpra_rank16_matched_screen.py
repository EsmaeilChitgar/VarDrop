import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.exp_long_term_forecasting_efficient import Exp_Long_Term_Forecast_Efficient
from model.LPRA import LowRankPeriodicResidualAdapter


TRAFFIC96_SETTING = (
    'traffic_96_96_gpt3d_lpra_r32_OURS_custom_M_'
    'ft96_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_'
    'fctimeF_ebTrue_dttest_k4_gs10_projection_0_'
    'fastdfh_lpra_r32_p168'
)


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


def make_adapter(num_channels, period, rank, device, init_seed):
    # Isolate adapter initialization so it does not perturb loader/sampler RNG.
    devices = []
    if device.type == 'cuda':
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(init_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(init_seed)
        adapter = LowRankPeriodicResidualAdapter(
            num_channels=num_channels,
            period=period,
            rank=rank,
        ).to(device)
    return adapter


def calibrate_matched(exp, adapters, lr, seed):
    """Calibrate r16 and r32 in one shared pass over identical backbone outputs."""
    set_seed(seed)
    _, train_loader = exp._get_data(flag='train')

    optimizers = {
        name: optim.Adam(module.parameters(), lr=lr)
        for name, module in adapters.items()
    }
    losses = {name: [] for name in adapters}

    exp.model.eval()
    for p in exp.model.parameters():
        p.requires_grad_(False)
    for module in adapters.values():
        module.train()

    start = time.time()
    for batch in train_loader:
        batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(exp, batch)

        if exp.fast_vardrop_sampler is None:
            raise RuntimeError('Matched rank screen expects exact_fast_vardrop.')
        sparse_indices = np.unique(exp.fast_vardrop_sampler(batch_x))
        channel_ids = torch.as_tensor(
            sparse_indices, device=exp.device, dtype=torch.long
        )

        with torch.no_grad():
            base, target = base_forward(
                exp,
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                sparse_indices=sparse_indices,
            )
        phase = exp._future_week_phase(batch_y_mark)

        # Separate optimizers, identical data/base/targets, one backbone forward.
        for name, module in adapters.items():
            corr = module(channel_ids, phase)
            loss = torch.mean((base.detach() + corr - target) ** 2)
            optimizers[name].zero_grad()
            loss.backward()
            optimizers[name].step()
            losses[name].append(float(loss.item()))

    elapsed = time.time() - start
    print('[RANK SCREEN] shared calibration pass complete')
    print('[RANK SCREEN] calibration time={:.3f}s'.format(elapsed))
    for name in adapters:
        print('[RANK SCREEN] {} train loss={:.7f}'.format(
            name, float(np.mean(losses[name]))
        ))


def choose_alphas(exp, adapters, alpha_max, seed):
    # Closed-form MSE alpha, then same validation safety rule used by LPRA.
    set_seed(seed)
    _, vali_loader = exp._get_data(flag='val')
    num = {name: 0.0 for name in adapters}
    den = {name: 0.0 for name in adapters}

    exp.model.eval()
    for module in adapters.values():
        module.eval()

    with torch.no_grad():
        for batch in vali_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(exp, batch)
            base, target = base_forward(exp, batch_x, batch_y, batch_x_mark, batch_y_mark)
            ids = torch.arange(base.shape[-1], device=exp.device, dtype=torch.long)
            phase = exp._future_week_phase(batch_y_mark)
            residual = target - base

            for name, module in adapters.items():
                corr = module(ids, phase)
                num[name] += torch.sum(residual * corr).item()
                den[name] += torch.sum(corr * corr).item()

    alpha_closed = {}
    candidates = {}
    for name in adapters:
        a = 0.0 if den[name] <= 1e-20 else num[name] / den[name]
        a = float(np.clip(a, 0.0, alpha_max))
        alpha_closed[name] = a
        vals = [a * (0.5 ** i) for i in range(9)] + [0.0]
        candidates[name] = list(dict.fromkeys(float(v) for v in vals))

    set_seed(seed)
    _, vali_loader = exp._get_data(flag='val')
    base_stats = {'se': 0.0, 'ae': 0.0, 'count': 0}
    stats = {
        name: {
            a: {'se': 0.0, 'ae': 0.0, 'count': 0}
            for a in candidates[name]
        }
        for name in adapters
    }

    with torch.no_grad():
        for batch in vali_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(exp, batch)
            base, target = base_forward(exp, batch_x, batch_y, batch_x_mark, batch_y_mark)
            ids = torch.arange(base.shape[-1], device=exp.device, dtype=torch.long)
            phase = exp._future_week_phase(batch_y_mark)

            base_err = base - target
            base_stats['se'] += torch.sum(base_err * base_err).item()
            base_stats['ae'] += torch.sum(torch.abs(base_err)).item()
            base_stats['count'] += base_err.numel()

            for name, module in adapters.items():
                corr = module(ids, phase)
                for a in candidates[name]:
                    err = base + a * corr - target
                    st = stats[name][a]
                    st['se'] += torch.sum(err * err).item()
                    st['ae'] += torch.sum(torch.abs(err)).item()
                    st['count'] += err.numel()

    base_mse = base_stats['se'] / base_stats['count']
    base_mae = base_stats['ae'] / base_stats['count']
    chosen = {}

    print('[RANK SCREEN] base validation MSE={:.7f} MAE={:.7f}'.format(
        base_mse, base_mae
    ))

    for name in adapters:
        selected = 0.0
        final_mse, final_mae = base_mse, base_mae
        for a in candidates[name]:
            st = stats[name][a]
            mse = st['se'] / st['count']
            mae = st['ae'] / st['count']
            if mse <= base_mse + 1e-12 and mae <= base_mae + 1e-12:
                selected = a
                final_mse, final_mae = mse, mae
                break
        chosen[name] = float(selected)
        mse_gain = 100.0 * (base_mse - final_mse) / max(base_mse, 1e-12)
        mae_gain = 100.0 * (base_mae - final_mae) / max(base_mae, 1e-12)
        print('[RANK SCREEN] {} alpha_closed={:.6f} alpha={:.6f}'.format(
            name, alpha_closed[name], selected
        ))
        print('[RANK SCREEN] {} validation MSE={:.7f} MAE={:.7f} gain={:.3f}%/{:.3f}%'.format(
            name, final_mse, final_mae, mse_gain, mae_gain
        ))

    return chosen


def update_stats(stats, name, pred, target):
    err = (pred - target).double()
    stats[name]['se'] += torch.sum(err * err).item()
    stats[name]['ae'] += torch.sum(torch.abs(err)).item()
    stats[name]['count'] += err.numel()


def evaluate(exp, adapters, alphas):
    _, test_loader = exp._get_data(flag='test')
    core = exp._model_core()
    existing_alpha = core.get_lpra_alpha()

    names = ['base', 'existing_r32', 'matched_r16', 'matched_r32']
    stats = {
        name: {'se': 0.0, 'ae': 0.0, 'count': 0}
        for name in names
    }

    exp.model.eval()
    for module in adapters.values():
        module.eval()

    with torch.no_grad():
        for batch in test_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(exp, batch)
            base, target = base_forward(exp, batch_x, batch_y, batch_x_mark, batch_y_mark)
            ids = torch.arange(base.shape[-1], device=exp.device, dtype=torch.long)
            phase = exp._future_week_phase(batch_y_mark)

            existing_corr = core.lpra_correction(ids, phase)
            r16_corr = adapters['r16'](ids, phase)
            r32_corr = adapters['r32'](ids, phase)

            update_stats(stats, 'base', base, target)
            update_stats(stats, 'existing_r32', base + existing_alpha * existing_corr, target)
            update_stats(stats, 'matched_r16', base + alphas['r16'] * r16_corr, target)
            update_stats(stats, 'matched_r32', base + alphas['r32'] * r32_corr, target)

    metrics = {}
    for name in names:
        st = stats[name]
        metrics[name] = (
            st['se'] / st['count'],
            st['ae'] / st['count'],
        )

    base_mse, base_mae = metrics['base']
    print('')
    print('==========================================================')
    print('LPRA MATCHED RANK SCREEN - TRAFFIC 96 -> 96')
    print('==========================================================')
    print('[RANK SCREEN] parameter counts')
    print('  r16 : {}'.format(adapters['r16'].num_parameters))
    print('  r32 : {}'.format(adapters['r32'].num_parameters))
    print('  full-table reference : {}'.format(862 * 168))
    print('[RANK SCREEN] learned alphas')
    print('  existing r32 : {:.9f}'.format(existing_alpha))
    print('  matched r16  : {:.9f}'.format(alphas['r16']))
    print('  matched r32  : {:.9f}'.format(alphas['r32']))
    print('')

    labels = [
        ('base', 'BASE'),
        ('existing_r32', 'EXISTING_R32'),
        ('matched_r16', 'MATCHED_R16'),
        ('matched_r32', 'MATCHED_R32'),
    ]
    for key, label in labels:
        mse, mae = metrics[key]
        mse_gain = 100.0 * (base_mse - mse) / max(base_mse, 1e-12)
        mae_gain = 100.0 * (base_mae - mae) / max(base_mae, 1e-12)
        print('[RANK SCREEN RESULT] {:12s} MSE={:.9f} MAE={:.9f} gain={:.3f}%/{:.3f}%'.format(
            label, mse, mae, mse_gain, mae_gain
        ))

    r16_mse, r16_mae = metrics['matched_r16']
    r32_mse, r32_mae = metrics['matched_r32']
    mse_gap = 100.0 * (r16_mse - r32_mse) / max(r32_mse, 1e-12)
    mae_gap = 100.0 * (r16_mae - r32_mae) / max(r32_mae, 1e-12)
    print('')
    print('[RANK SCREEN] matched r16 gap vs matched r32: MSE={:+.3f}% MAE={:+.3f}%'.format(
        mse_gap, mae_gap
    ))
    print('[RANK SCREEN] compression vs full-table: r16={:.2f}x r32={:.2f}x'.format(
        (862 * 168) / adapters['r16'].num_parameters,
        (862 * 168) / adapters['r32'].num_parameters,
    ))

    existing_known = 0.3819281756877899
    existing_diff = abs(metrics['existing_r32'][0] - existing_known)
    print('[RANK SCREEN] existing-r32 reproduction abs MSE diff={:.9e}'.format(existing_diff))
    if existing_diff > 1e-5:
        print('[RANK SCREEN] WARNING: existing LPRA reference did not reproduce closely.')
    print('==========================================================')


def main():
    project_args = build_project_args()
    checkpoint_dir = os.path.join('./checkpoints', TRAFFIC96_SETTING)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_lpra.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('Existing LPRA checkpoint not found: ' + checkpoint_path)

    set_seed(2023)
    exp = Exp_Long_Term_Forecast_Efficient(project_args)
    print('[RANK SCREEN] loading existing Traffic96 checkpoint:')
    print('  ' + checkpoint_path)
    state = torch.load(checkpoint_path, map_location=exp.device)
    exp.model.load_state_dict(state)
    exp.model.eval()

    # New matched adapters. Existing checkpoint LPRA remains read-only.
    r16 = make_adapter(862, 168, 16, exp.device, init_seed=314159)
    r32 = make_adapter(862, 168, 32, exp.device, init_seed=314159)
    adapters = {'r16': r16, 'r32': r32}

    print('[RANK SCREEN] NO backbone training')
    print('[RANK SCREEN] existing rank32 checkpoint is read-only reference')
    print('[RANK SCREEN] matched r16 + r32 share ONE calibration pass')
    print('[RANK SCREEN] params: r16={} r32={} full-table={}'.format(
        r16.num_parameters, r32.num_parameters, 862 * 168
    ))

    calibrate_matched(
        exp,
        adapters,
        lr=0.005,
        seed=2023 + 9107,
    )
    alphas = choose_alphas(
        exp,
        adapters,
        alpha_max=1.25,
        seed=2023 + 12031,
    )
    evaluate(exp, adapters, alphas)


if __name__ == '__main__':
    main()

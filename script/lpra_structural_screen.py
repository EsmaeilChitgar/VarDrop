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


class PhaseOnlyResidual(nn.Module):
    """One learned residual scalar per periodic phase, shared by all channels."""

    def __init__(self, period):
        super().__init__()
        self.bias = nn.Embedding(int(period), 1)
        nn.init.zeros_(self.bias.weight)

    def forward(self, phase_indices, channel_ids):
        corr = self.bias(phase_indices)  # [B,H,1]
        return corr.expand(-1, -1, int(channel_ids.numel()))

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class FullTableResidual(nn.Module):
    """Unconstrained periodic-phase x channel residual table."""

    def __init__(self, num_channels, period):
        super().__init__()
        self.table = nn.Embedding(int(period), int(num_channels))
        nn.init.zeros_(self.table.weight)

    def forward(self, phase_indices, channel_ids):
        corr = self.table(phase_indices)  # [B,H,N_all]
        return corr.index_select(dim=-1, index=channel_ids)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_project_args():
    # Exact Traffic 96->96 configuration used by the existing GPT3d checkpoint.
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
    batch_x = batch_x.float().to(exp.device)
    batch_y = batch_y.float().to(exp.device)
    batch_x_mark = batch_x_mark.float().to(exp.device)
    batch_y_mark = batch_y_mark.float().to(exp.device)
    return batch_x, batch_y, batch_x_mark, batch_y_mark


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
    outputs = outputs[:, -exp.args.pred_len:, :]
    target = batch_y[:, -exp.args.pred_len:, :]
    return outputs, target


def calibrate_controls(exp, phase_adapter, table_adapter, lr, seed):
    # One shared pass: both controls see exactly the same shuffled batches,
    # VarDrop samples, frozen-backbone predictions, and targets.
    set_seed(seed)
    _, train_loader = exp._get_data(flag='train')

    phase_opt = optim.Adam(phase_adapter.parameters(), lr=lr)
    table_opt = optim.Adam(table_adapter.parameters(), lr=lr)

    exp.model.eval()
    phase_adapter.train()
    table_adapter.train()

    phase_losses = []
    table_losses = []
    start = time.time()

    for batch in train_loader:
        batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(exp, batch)

        if exp.fast_vardrop_sampler is not None:
            sparse_indices = exp.fast_vardrop_sampler(batch_x)
        else:
            raise RuntimeError('This screen expects --exact_fast_vardrop behavior.')
        sparse_indices = np.unique(sparse_indices)
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

        phase_corr = phase_adapter(phase, channel_ids)
        phase_loss = torch.mean((base.detach() + phase_corr - target) ** 2)
        phase_opt.zero_grad()
        phase_loss.backward()
        phase_opt.step()

        table_corr = table_adapter(phase, channel_ids)
        table_loss = torch.mean((base.detach() + table_corr - target) ** 2)
        table_opt.zero_grad()
        table_loss.backward()
        table_opt.step()

        phase_losses.append(float(phase_loss.item()))
        table_losses.append(float(table_loss.item()))

    elapsed = time.time() - start
    print('[STRUCT SCREEN] shared calibration pass complete')
    print('[STRUCT SCREEN] calibration time={:.3f}s'.format(elapsed))
    print('[STRUCT SCREEN] phase-only train loss={:.7f}'.format(float(np.mean(phase_losses))))
    print('[STRUCT SCREEN] full-table train loss={:.7f}'.format(float(np.mean(table_losses))))
    return elapsed


def choose_alpha_for_controls(exp, adapters, alpha_max, seed):
    # Pass 1: closed-form MSE scaling for each correction. Reset RNG before
    # both validation passes so even a shuffled/drop-last validation loader
    # uses the same samples in both passes.
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
            base, target = base_forward(
                exp, batch_x, batch_y, batch_x_mark, batch_y_mark
            )
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
        vals = [a * (0.5 ** i) for i in range(9)] + [0.0]
        candidates[name] = list(dict.fromkeys(float(v) for v in vals))

    # Pass 2: reproduce the LPRA validation rule exactly: first candidate whose
    # MSE and MAE are both no worse than the frozen backbone.
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
            base, target = base_forward(
                exp, batch_x, batch_y, batch_x_mark, batch_y_mark
            )
            ids = torch.arange(base.shape[-1], device=exp.device, dtype=torch.long)
            phase = exp._future_week_phase(batch_y_mark)

            base_err = base - target
            base_stats['se'] += torch.sum(base_err * base_err).item()
            base_stats['ae'] += torch.sum(torch.abs(base_err)).item()
            base_stats['count'] += base_err.numel()

            for name, module in adapters.items():
                corr = module(phase, ids)
                for a in candidates[name]:
                    err = base + a * corr - target
                    st = stats[name][a]
                    st['se'] += torch.sum(err * err).item()
                    st['ae'] += torch.sum(torch.abs(err)).item()
                    st['count'] += err.numel()

    base_mse = base_stats['se'] / base_stats['count']
    base_mae = base_stats['ae'] / base_stats['count']

    chosen = {}
    validation = {}
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
        validation[name] = (float(final_mse), float(final_mae))

        mse_gain = 100.0 * (base_mse - final_mse) / max(base_mse, 1e-12)
        mae_gain = 100.0 * (base_mae - final_mae) / max(base_mae, 1e-12)
        print('[STRUCT SCREEN] {} alpha_closed={:.6f} alpha={:.6f}'.format(
            name, alpha_closed[name], chosen[name]))
        print('[STRUCT SCREEN] {} validation MSE={:.7f} MAE={:.7f} gain={:.3f}%/{:.3f}%'.format(
            name, final_mse, final_mae, mse_gain, mae_gain))

    print('[STRUCT SCREEN] base validation MSE={:.7f} MAE={:.7f}'.format(
        base_mse, base_mae))
    return chosen


def streaming_update(stats, name, pred, target):
    err = (pred - target).double()
    stats[name]['se'] += torch.sum(err * err).item()
    stats[name]['ae'] += torch.sum(torch.abs(err)).item()
    stats[name]['count'] += err.numel()


def evaluate_test(exp, phase_adapter, table_adapter, phase_alpha, table_alpha):
    _, test_loader = exp._get_data(flag='test')
    core = exp._model_core()
    lpra_alpha = core.get_lpra_alpha()

    names = ['base', 'lpra', 'phase_only', 'full_table']
    stats = {
        name: {'se': 0.0, 'ae': 0.0, 'count': 0}
        for name in names
    }

    exp.model.eval()
    phase_adapter.eval()
    table_adapter.eval()

    with torch.no_grad():
        for batch in test_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(exp, batch)
            base, target = base_forward(
                exp, batch_x, batch_y, batch_x_mark, batch_y_mark
            )
            ids = torch.arange(base.shape[-1], device=exp.device, dtype=torch.long)
            phase = exp._future_week_phase(batch_y_mark)

            lpra_corr = core.lpra_correction(ids, phase)
            phase_corr = phase_adapter(phase, ids)
            table_corr = table_adapter(phase, ids)

            streaming_update(stats, 'base', base, target)
            streaming_update(stats, 'lpra', base + lpra_alpha * lpra_corr, target)
            streaming_update(stats, 'phase_only', base + phase_alpha * phase_corr, target)
            streaming_update(stats, 'full_table', base + table_alpha * table_corr, target)

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
    print('LPRA STRUCTURAL SCREEN - TRAFFIC 96 -> 96')
    print('==========================================================')
    print('[STRUCT SCREEN] parameter counts')
    print('  phase-only : {}'.format(phase_adapter.num_parameters))
    print('  LPRA r32   : {}'.format(core.lpra.num_parameters))
    print('  full-table : {}'.format(table_adapter.num_parameters))
    print('[STRUCT SCREEN] learned alphas')
    print('  LPRA       : {:.9f} (from existing checkpoint)'.format(lpra_alpha))
    print('  phase-only : {:.9f}'.format(phase_alpha))
    print('  full-table : {:.9f}'.format(table_alpha))
    print('')

    labels = [
        ('base', 'BASE'),
        ('lpra', 'LPRA'),
        ('phase_only', 'PHASE_ONLY'),
        ('full_table', 'FULL_TABLE'),
    ]
    for key, label in labels:
        mse, mae = metrics[key]
        mse_gain = 100.0 * (base_mse - mse) / max(base_mse, 1e-12)
        mae_gain = 100.0 * (base_mae - mae) / max(base_mae, 1e-12)
        print('[STRUCT SCREEN RESULT] {:10s} MSE={:.9f} MAE={:.9f} gain={:.3f}%/{:.3f}%'.format(
            label, mse, mae, mse_gain, mae_gain))

    known_lpra_mse = 0.3819281756877899
    observed_lpra_mse = metrics['lpra'][0]
    delta = abs(observed_lpra_mse - known_lpra_mse)
    print('')
    print('[STRUCT SCREEN] existing-LPRA reproduction abs MSE diff={:.9e}'.format(delta))
    if delta > 1e-5:
        print('[STRUCT SCREEN] WARNING: existing LPRA result did not reproduce closely; inspect before interpreting controls.')

    phase_gap_vs_lpra = 100.0 * (metrics['phase_only'][0] - observed_lpra_mse) / max(observed_lpra_mse, 1e-12)
    table_gap_vs_lpra = 100.0 * (metrics['full_table'][0] - observed_lpra_mse) / max(observed_lpra_mse, 1e-12)
    print('[STRUCT SCREEN] phase-only MSE gap vs LPRA = {:+.3f}%'.format(phase_gap_vs_lpra))
    print('[STRUCT SCREEN] full-table MSE gap vs LPRA = {:+.3f}%'.format(table_gap_vs_lpra))
    print('==========================================================')


def main():
    parser = argparse.ArgumentParser(
        description='Minimal LPRA structural screen on existing Traffic96 checkpoint'
    )
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join('./checkpoints', TRAFFIC96_SETTING))
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--alpha_max', type=float, default=1.25)
    args_cli = parser.parse_args()

    checkpoint_path = os.path.join(args_cli.checkpoint_dir, 'checkpoint_lpra.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            'Existing LPRA checkpoint not found: {}\n'
            'Run this from the VarDrop project root, or pass --checkpoint_dir.'.format(checkpoint_path)
        )

    set_seed(args_cli.seed)
    project_args = build_project_args()
    exp = Exp_Long_Term_Forecast_Efficient(project_args)

    print('[STRUCT SCREEN] loading existing LPRA checkpoint:')
    print('  {}'.format(checkpoint_path))
    state = torch.load(checkpoint_path, map_location=exp.device)
    exp.model.load_state_dict(state)
    exp.model.eval()

    core = exp._model_core()
    if core.lpra is None:
        raise RuntimeError('Loaded model has no LPRA module.')

    # Existing LPRA stays untouched. Only the two diagnostic controls below are trained.
    phase_adapter = PhaseOnlyResidual(project_args.lpra_period).to(exp.device)
    table_adapter = FullTableResidual(
        project_args.enc_in, project_args.lpra_period
    ).to(exp.device)

    print('[STRUCT SCREEN] NO backbone training')
    print('[STRUCT SCREEN] existing LPRA is read-only reference')
    print('[STRUCT SCREEN] calibrating phase-only + full-table together for ONE shared pass')
    print('[STRUCT SCREEN] params: phase-only={} LPRA={} full-table={}'.format(
        phase_adapter.num_parameters,
        core.lpra.num_parameters,
        table_adapter.num_parameters,
    ))

    calibrate_controls(
        exp,
        phase_adapter,
        table_adapter,
        lr=args_cli.lr,
        seed=args_cli.seed + 9107,
    )

    alphas = choose_alpha_for_controls(
        exp,
        {'phase-only': phase_adapter, 'full-table': table_adapter},
        alpha_max=args_cli.alpha_max,
        seed=args_cli.seed + 12031,
    )

    evaluate_test(
        exp,
        phase_adapter,
        table_adapter,
        phase_alpha=alphas['phase-only'],
        table_alpha=alphas['full-table'],
    )


if __name__ == '__main__':
    main()

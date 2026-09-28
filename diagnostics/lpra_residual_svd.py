import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_factory import data_provider
from model.OURS import Model


def parse_ranks(text):
    ranks = sorted({int(x.strip()) for x in text.split(',') if x.strip()})
    if not ranks or any(r <= 0 for r in ranks):
        raise ValueError('ranks must be positive integers')
    return ranks


def make_args(cli):
    # Minimal config matching the completed Traffic 96->96 GPT3d run.
    return SimpleNamespace(
        model='OURS',
        data='custom',
        root_path=cli.root_path,
        data_path=cli.data_path,
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
        num_workers=cli.num_workers,
        batch_size=cli.batch_size,
        inverse=False,
        class_strategy='projection',
        use_norm=1,
        use_lpra=True,
        lpra_rank=32,
        lpra_period=168,
    )


def load_state(model, path, device):
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError('checkpoint is not a state_dict dictionary')
    if state and all(k.startswith('module.') for k in state.keys()):
        state = {k[len('module.'):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def future_week_phase(batch_y_mark, pred_len):
    if batch_y_mark is None or batch_y_mark.shape[-1] < 2:
        raise RuntimeError('hourly timeF marks with HourOfDay and DayOfWeek are required')
    future = batch_y_mark[:, -pred_len:, :]
    hour = torch.round((future[..., 0] + 0.5) * 23.0).long().clamp_(0, 23)
    weekday = torch.round((future[..., 1] + 0.5) * 6.0).long().clamp_(0, 6)
    return weekday * 24 + hour


def main():
    p = argparse.ArgumentParser(
        description='Diagnostic SVD of frozen GPT3b residual channel x weekly-phase matrix'
    )
    p.add_argument('--checkpoint_path', required=True)
    p.add_argument('--root_path', default='./dataset/traffic/')
    p.add_argument('--data_path', default='traffic.csv')
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ranks', default='1,4,8,16,32,64')
    p.add_argument('--output_dir', default='./diagnostics/results/lpra_residual_svd_traffic96')
    cli = p.parse_args()

    ranks = parse_ranks(cli.ranks)
    device = torch.device(cli.device if torch.cuda.is_available() else 'cpu')
    args = make_args(cli)

    if not os.path.exists(cli.checkpoint_path):
        raise FileNotFoundError(cli.checkpoint_path)

    print('==========================================================')
    print('LPRA RESIDUAL SVD DIAGNOSTIC - TRAFFIC 96 -> 96')
    print('NO TRAINING - NO CALIBRATION - NO TEST SET')
    print('==========================================================')
    print('[RESIDUAL SVD] device={}'.format(device))
    print('[RESIDUAL SVD] split={}'.format(cli.split))
    print('[RESIDUAL SVD] checkpoint={}'.format(cli.checkpoint_path))

    model = Model(args).float().to(device)
    load_state(model, cli.checkpoint_path, device)
    model.eval()

    _, loader = data_provider(args, cli.split)
    period = 168
    n_channels = args.enc_in

    # Aggregate mean residual for each weekly phase and channel:
    # R[n,p] = E[y - base_prediction | channel=n, phase=p].
    sums = torch.zeros(period, n_channels, dtype=torch.float64, device=device)
    counts = torch.zeros(period, dtype=torch.float64, device=device)

    total_sse = 0.0
    total_sae = 0.0
    total_count = 0

    with torch.inference_mode():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat(
                [batch_y[:, :args.label_len, :], dec_inp], dim=1
            ).float().to(device)

            # IMPORTANT: Model.forward() is the frozen GPT3b backbone prediction.
            # LPRA correction is applied externally in the experiment class, so
            # no adapter correction is included here.
            base = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if isinstance(base, (tuple, list)):
                base = base[0]
            base = base[:, -args.pred_len:, :]
            target = batch_y[:, -args.pred_len:, :]
            residual = target - base

            phase = future_week_phase(batch_y_mark, args.pred_len)
            flat_phase = phase.reshape(-1)
            flat_residual = residual.reshape(-1, n_channels).double()

            sums.index_add_(0, flat_phase, flat_residual)
            counts.index_add_(
                0, flat_phase, torch.ones_like(flat_phase, dtype=torch.float64)
            )

            err = (base - target).double()
            total_sse += torch.sum(err * err).item()
            total_sae += torch.sum(torch.abs(err)).item()
            total_count += err.numel()

            if (batch_idx + 1) % 20 == 0:
                print('[RESIDUAL SVD] processed batches={}'.format(batch_idx + 1))

    if torch.any(counts == 0):
        missing = torch.where(counts == 0)[0].detach().cpu().tolist()
        raise RuntimeError('weekly phases with zero observations: {}'.format(missing))

    # [phase, channel] -> [channel, phase]
    R = (sums / counts[:, None]).T.contiguous()
    s = torch.linalg.svdvals(R)
    energy = s * s
    total_energy = torch.sum(energy)
    cumulative = torch.cumsum(energy, dim=0) / torch.clamp(total_energy, min=1e-30)

    base_mse = total_sse / total_count
    base_mae = total_sae / total_count
    fro = torch.linalg.vector_norm(R).item()

    print('')
    print('[RESIDUAL SVD] base MSE={:.9f} MAE={:.9f}'.format(base_mse, base_mae))
    print('[RESIDUAL SVD] residual matrix shape={}x{}'.format(R.shape[0], R.shape[1]))
    print('[RESIDUAL SVD] phase count min={} max={}'.format(
        int(counts.min().item()), int(counts.max().item())
    ))
    print('[RESIDUAL SVD] Frobenius norm={:.9f}'.format(fro))
    print('[RESIDUAL SVD] top singular values={}'.format(
        ', '.join('{:.6f}'.format(float(x)) for x in s[:10].detach().cpu())
    ))

    rows = []
    max_rank = int(s.numel())
    for r in ranks:
        rr = min(r, max_rank)
        e = float(cumulative[rr - 1].item())
        residual_fraction = 1.0 - e
        rows.append({
            'rank': r,
            'effective_rank': rr,
            'energy_fraction': e,
            'energy_percent': 100.0 * e,
            'unexplained_percent': 100.0 * residual_fraction,
        })
        print(
            '[RESIDUAL SVD] rank={:<3d} energy={:7.3f}% unexplained={:7.3f}%'.format(
                r, 100.0 * e, 100.0 * residual_fraction
            )
        )

    os.makedirs(cli.output_dir, exist_ok=True)

    summary = {
        'split': cli.split,
        'checkpoint_path': cli.checkpoint_path,
        'matrix_shape': [int(R.shape[0]), int(R.shape[1])],
        'base_mse': base_mse,
        'base_mae': base_mae,
        'phase_count_min': int(counts.min().item()),
        'phase_count_max': int(counts.max().item()),
        'frobenius_norm': fro,
        'top_10_singular_values': [float(x) for x in s[:10].detach().cpu()],
        'ranks': rows,
    }

    with open(os.path.join(cli.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(cli.output_dir, 'rank_energy.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['rank', 'effective_rank', 'energy_fraction', 'energy_percent', 'unexplained_percent']
        )
        writer.writeheader()
        writer.writerows(rows)

    np.save(
        os.path.join(cli.output_dir, 'singular_values.npy'),
        s.detach().cpu().numpy()
    )

    print('[RESIDUAL SVD] saved {}'.format(cli.output_dir))
    print('==========================================================')


if __name__ == '__main__':
    main()

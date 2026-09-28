import argparse
import os
import random
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch import optim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_provider.data_factory import data_provider
from model.OURS import Model
from VarDrop import efficient_sampler, efficient_sampler_fast_exact


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_args(dataset, num_workers):
    common = dict(
        model='OURS',
        data='custom',
        features='M',
        target='OT',
        freq='h',
        seq_len=96,
        label_len=48,
        pred_len=96,
        d_model=512,
        n_heads=8,
        d_layers=1,
        d_ff=512,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed='timeF',
        activation='gelu',
        output_attention=False,
        num_workers=num_workers,
        batch_size=32,
        inverse=False,
        class_strategy='projection',
        use_norm=1,
        use_lpra=False,
        lpra_rank=32,
        lpra_period=168,
    )
    if dataset == 'traffic':
        common.update(
            root_path='./dataset/traffic/',
            data_path='traffic.csv',
            enc_in=862, dec_in=862, c_out=862,
            e_layers=4,
            learning_rate=0.001,
            k=4, group_size=10,
        )
    elif dataset == 'ecl':
        common.update(
            root_path='./dataset/electricity/',
            data_path='electricity.csv',
            enc_in=321, dec_in=321, c_out=321,
            e_layers=3,
            learning_rate=0.0005,
            k=3, group_size=10,
        )
    else:
        raise ValueError(dataset)
    return SimpleNamespace(**common)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_mode(dataset, mode, warmup, timed, seed, num_workers, device):
    args = make_args(dataset, num_workers)

    # Identical initialization for original and exact-fast.
    set_seed(seed)
    model = Model(args).float().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    # Identical training shuffle/order for both timing modes.
    loader_seed = seed + 11003
    set_seed(loader_seed)
    _, loader = data_provider(args, 'train')
    iterator = iter(loader)

    # Reset numpy sampling RNG after DataLoader construction.
    np.random.seed(seed + 22007)

    total_needed = warmup + timed
    iter_ms = []
    sampler_ms = []

    for step in range(total_needed):
        try:
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(iterator)
        except StopIteration:
            # Normally not reached for requested settings; deterministic restart if needed.
            set_seed(loader_seed)
            iterator = iter(loader)
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(iterator)

        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        sync(device)
        iter_start = time.perf_counter()

        sampler_start = time.perf_counter()
        if mode == 'original':
            indices = efficient_sampler(
                batch_x,
                k=args.k,
                group_size=args.group_size,
                freq_list=range(1, 25),
            )
        elif mode == 'fast':
            indices = efficient_sampler_fast_exact(
                batch_x,
                k=args.k,
                group_size=args.group_size,
                freq_list=range(1, 25),
            )
        else:
            raise ValueError(mode)
        sampler_end = time.perf_counter()

        indices = np.unique(indices)
        x_sparse = batch_x[:, :, indices]
        y_sparse = batch_y[:, :, indices]

        dec_inp = torch.zeros_like(y_sparse[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat(
            [y_sparse[:, :args.label_len, :], dec_inp], dim=1
        ).float().to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x_sparse, batch_x_mark, dec_inp, batch_y_mark)
        outputs = outputs[:, -args.pred_len:, :]
        target = y_sparse[:, -args.pred_len:, :]
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        sync(device)
        iter_end = time.perf_counter()

        if step >= warmup:
            iter_ms.append((iter_end - iter_start) * 1000.0)
            sampler_ms.append((sampler_end - sampler_start) * 1000.0)

    return {
        'iter_mean_ms': float(np.mean(iter_ms)),
        'iter_median_ms': float(np.median(iter_ms)),
        'iter_std_ms': float(np.std(iter_ms, ddof=1)) if len(iter_ms) > 1 else 0.0,
        'sampler_mean_ms': float(np.mean(sampler_ms)),
        'sampler_median_ms': float(np.median(sampler_ms)),
        'n': len(iter_ms),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--timed', type=int, default=300)
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    cli = p.parse_args()

    device = torch.device(cli.device if torch.cuda.is_available() else 'cpu')
    print('==========================================================')
    print('MATCHED VarDrop vs Exact-Fast TRAINING ITERATION TIMING')
    print('Same model / same seed / same k / same batch size')
    print('Timing excludes validation/test and includes sampler+FW+BW+optimizer')
    print('warmup={} timed={} num_workers={} device={}'.format(
        cli.warmup, cli.timed, cli.num_workers, device
    ))
    print('==========================================================')

    for dataset in ('traffic', 'ecl'):
        args = make_args(dataset, cli.num_workers)
        print('')
        print('[TIMING] dataset={} k={} group_size={}'.format(
            dataset.upper(), args.k, args.group_size
        ))
        original = run_mode(
            dataset, 'original', cli.warmup, cli.timed,
            cli.seed, cli.num_workers, device
        )
        fast = run_mode(
            dataset, 'fast', cli.warmup, cli.timed,
            cli.seed, cli.num_workers, device
        )
        speedup = original['iter_mean_ms'] / fast['iter_mean_ms']
        sampler_speedup = original['sampler_mean_ms'] / max(fast['sampler_mean_ms'], 1e-12)

        print('[TIMING RESULT] {} ORIGINAL iter_mean={:.3f}ms median={:.3f}ms std={:.3f}ms sampler_mean={:.3f}ms n={}'.format(
            dataset.upper(), original['iter_mean_ms'], original['iter_median_ms'],
            original['iter_std_ms'], original['sampler_mean_ms'], original['n']
        ))
        print('[TIMING RESULT] {} EXACT_FAST iter_mean={:.3f}ms median={:.3f}ms std={:.3f}ms sampler_mean={:.3f}ms n={}'.format(
            dataset.upper(), fast['iter_mean_ms'], fast['iter_median_ms'],
            fast['iter_std_ms'], fast['sampler_mean_ms'], fast['n']
        ))
        print('[TIMING RESULT] {} ITER_SPEEDUP={:.3f}x SAMPLER_SPEEDUP={:.3f}x'.format(
            dataset.upper(), speedup, sampler_speedup
        ))

    print('')
    print('==========================================================')
    print('TIMING BENCHMARK FINISHED')
    print('Copy all [TIMING] and [TIMING RESULT] lines.')
    print('==========================================================')


if __name__ == '__main__':
    main()

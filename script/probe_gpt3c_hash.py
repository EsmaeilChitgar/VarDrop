import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

# Run from repository root (same convention as the other scripts).
sys.path.insert(0, os.path.abspath('.'))
from VarDrop import (  # noqa: E402
    k_dominant_frequency_hashing_fast_exact,
    efficient_sampler_fast_exact,
)


def _spectral_cosine(amp, i, j):
    a = amp[:, i]
    b = amp[:, j]
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-12
    return float(torch.dot(a, b) / den)


def _time_abs_corr(flat, i, j):
    a = flat[:, i]
    b = flat[:, j]
    a = a - a.mean()
    b = b - b.mean()
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-12
    return float(torch.abs(torch.dot(a, b) / den))


def main():
    parser = argparse.ArgumentParser(description='GPT3c hash diagnostic on real Traffic data')
    parser.add_argument('--csv', type=str, default='./dataset/traffic/traffic.csv')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--group_size', type=int, default=10)
    parser.add_argument('--batches', type=int, default=64)
    parser.add_argument('--max_new_pairs_per_batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    df = pd.read_csv(args.csv)
    if 'date' in df.columns:
        data_df = df.drop(columns=['date'])
    else:
        data_df = df.iloc[:, 1:]

    data = data_df.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    if np.isnan(data).any():
        # Same spirit as the forecasting loaders: avoid losing a whole probe
        # because of isolated missing values.
        data = pd.DataFrame(data).ffill().bfill().to_numpy(dtype=np.float32)

    n_rows, n_vars = data.shape
    n_train = int(n_rows * 0.7)
    train = data[:n_train]

    if n_train <= args.seq_len:
        raise ValueError(f'Not enough train rows: {n_train} for seq_len={args.seq_len}')

    rng = np.random.default_rng(args.seed)
    max_start = n_train - args.seq_len

    changed_fracs = []
    ranked_groups = []
    anchor_groups = []
    ranked_tokens = []
    anchor_tokens = []
    new_pair_spec = []
    new_pair_corr = []
    new_pair_count = 0

    print('==========================================================')
    print('GPT3c Anchor-Set k-DFH probe')
    print('==========================================================')
    print(f'csv           : {args.csv}')
    print(f'rows          : {n_rows}')
    print(f'variates      : {n_vars}')
    print(f'train rows    : {n_train}')
    print(f'seq_len       : {args.seq_len}')
    print(f'batch_size    : {args.batch_size}')
    print(f'batches       : {args.batches}')
    print(f'device        : {device}')
    print()

    with torch.no_grad():
        for batch_id in range(args.batches):
            starts = rng.integers(0, max_start + 1, size=args.batch_size)
            batch_np = np.stack([train[s:s + args.seq_len] for s in starts], axis=0)
            x = torch.from_numpy(batch_np).to(device)

            h_ranked = k_dominant_frequency_hashing_fast_exact(
                x, args.k, range(1, 25), hash_mode='ranked'
            )
            h_anchor = k_dominant_frequency_hashing_fast_exact(
                x, args.k, range(1, 25), hash_mode='anchor_set'
            )

            changed_fracs.append(float(np.mean(h_ranked != h_anchor)))
            ranked_groups.append(len(np.unique(h_ranked)))
            anchor_groups.append(len(np.unique(h_anchor)))

            # Same seed for the two samplers makes token-count comparison easier
            # to interpret. The actual selected identities are expected to differ.
            np.random.seed(args.seed + batch_id)
            idx_r = efficient_sampler_fast_exact(
                x, args.k, args.group_size, range(1, 25),
                hash_mode='ranked'
            )
            np.random.seed(args.seed + batch_id)
            idx_a = efficient_sampler_fast_exact(
                x, args.k, args.group_size, range(1, 25),
                hash_mode='anchor_set'
            )
            ranked_tokens.append(len(np.unique(idx_r)))
            anchor_tokens.append(len(np.unique(idx_a)))

            # Inspect only pairs newly merged by Anchor-Set: same Anchor-Set hash,
            # different ranked hash. This directly tests the new grouping decision.
            groups = defaultdict(list)
            for i, hv in enumerate(h_anchor):
                groups[hv].append(i)

            candidate_pairs = []
            for inds in groups.values():
                if len(inds) < 2:
                    continue
                for p in range(len(inds)):
                    i = inds[p]
                    for q in range(p + 1, len(inds)):
                        j = inds[q]
                        if h_ranked[i] != h_ranked[j]:
                            candidate_pairs.append((i, j))

            if candidate_pairs:
                if len(candidate_pairs) > args.max_new_pairs_per_batch:
                    take = rng.choice(
                        len(candidate_pairs),
                        size=args.max_new_pairs_per_batch,
                        replace=False,
                    )
                    candidate_pairs = [candidate_pairs[z] for z in take]

                amp = torch.fft.rfft(x, dim=1).abs().mean(dim=0)[1:25]
                flat = x.reshape(-1, n_vars)
                for i, j in candidate_pairs:
                    new_pair_spec.append(_spectral_cosine(amp, i, j))
                    new_pair_corr.append(_time_abs_corr(flat, i, j))
                new_pair_count += len(candidate_pairs)

    def m(v):
        return float(np.mean(v)) if len(v) else float('nan')

    print('===================== PROBE RESULT =======================')
    print(f'mean changed-hash fraction : {m(changed_fracs):.6f}')
    print(f'mean groups - GPT3b ranked : {m(ranked_groups):.2f}')
    print(f'mean groups - GPT3c anchor : {m(anchor_groups):.2f}')
    print(f'mean unique tokens ranked  : {m(ranked_tokens):.2f}')
    print(f'mean unique tokens anchor  : {m(anchor_tokens):.2f}')
    if ranked_tokens:
        reduction = 100.0 * (m(ranked_tokens) - m(anchor_tokens)) / max(m(ranked_tokens), 1e-12)
        print(f'anchor token delta vs ranked: {-reduction:+.3f}%')
    print(f'newly merged pairs sampled : {new_pair_count}')
    if new_pair_spec:
        print(f'new-pair spectral cosine   : {m(new_pair_spec):.6f}')
        print(f'new-pair |time corr|       : {m(new_pair_corr):.6f}')
    print('==========================================================')
    print('Interpretation:')
    print('  - changed-hash > 0 confirms GPT3c is NOT behavior-identical to GPT3b.')
    print('  - fewer groups/tokens means extra training-time reduction is plausible.')
    print('  - high new-pair spectral cosine supports the merge assumption.')
    print('  - final forecasting MSE/MAE still requires the real training run.')


if __name__ == '__main__':
    main()

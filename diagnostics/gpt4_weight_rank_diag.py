import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse the already-tested checkpoint/SVD/weight-ablation helpers from the
# preceding GPT4 diagnostic. This file does not modify training or model code.
from gpt4_weight_rank_diag import (  # noqa: E402
    apply_full_reconstruction,
    apply_rank,
    collect_targets,
    compute_svd,
    full_reconstruction_error,
    load_checkpoint,
    restore_targets,
    set_seed,
)


def _json_float(x):
    x = float(x)
    return x if math.isfinite(x) else None


def pct_gap(value: float, baseline: float) -> float:
    return 100.0 * (value - baseline) / max(abs(baseline), 1e-30)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks, equivalent to scipy.stats.rankdata(method='average')."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + j)  # 1-based average rank for positions i..j-1
        ranks[order[i:j]] = avg
        i = j
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float('nan')
    xc = x - x.mean()
    yc = y - y.mean()
    denom = math.sqrt(float(np.dot(xc, xc) * np.dot(yc, yc)))
    if denom <= 0.0:
        return float('nan')
    return float(np.dot(xc, yc) / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(average_ranks(np.asarray(x)), average_ranks(np.asarray(y)))


def permutation_pvalue_spearman(
    x: np.ndarray,
    y: np.ndarray,
    observed: float,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    if not math.isfinite(observed) or permutations <= 0:
        return float('nan')
    rx = average_ranks(np.asarray(x))
    ry = average_ranks(np.asarray(y))
    extreme = 0
    for _ in range(permutations):
        rho = pearson_corr(rx, rng.permutation(ry))
        if math.isfinite(rho) and abs(rho) >= abs(observed) - 1e-15:
            extreme += 1
    return float((extreme + 1) / (permutations + 1))


def redundancy_features_from_hashes(hash_values: np.ndarray) -> Dict[str, float]:
    hashes = np.asarray(hash_values)
    n = int(hashes.size)
    if n <= 0:
        raise ValueError('Cannot compute redundancy features from zero variates.')
    _, counts = np.unique(hashes, return_counts=True)
    counts = counts.astype(np.float64)
    p = counts / float(n)
    entropy = float(-np.sum(p * np.log(np.clip(p, 1e-30, None))))
    effective_groups = float(np.exp(entropy))
    num_groups = int(counts.size)
    pair_collision = 0.0
    if n > 1:
        pair_collision = float(np.sum(counts * (counts - 1.0)) / (n * (n - 1.0)))
    singleton_variables = float(np.sum(counts[counts == 1.0]))
    return {
        'n_vars': n,
        'num_groups': num_groups,
        'group_ratio': float(num_groups / n),
        'effective_groups': effective_groups,
        'effective_group_ratio': float(effective_groups / n),
        # Primary predictor: 0 when every variate is its own group, near 1 when
        # a batch collapses into very few spectral groups.
        'redundancy_score': float(1.0 - effective_groups / n),
        'largest_group_ratio': float(np.max(counts) / n),
        'mean_group_size': float(n / num_groups),
        'singleton_var_ratio': float(singleton_variables / n),
        'pair_collision': pair_collision,
        'group_entropy': entropy,
    }


def sample_indices_from_hashes(hash_values: np.ndarray, group_size: int) -> np.ndarray:
    """Reproduce original VarDrop group-wise random sampling from precomputed hashes."""
    sparse_indices = []
    for value in np.unique(hash_values):
        group_indices = np.where(hash_values == value)[0].tolist()
        sampled = np.random.choice(
            group_indices,
            min(int(group_size), len(group_indices)),
            replace=True,
        )
        sparse_indices.append(sampled)
    if not sparse_indices:
        return np.asarray([], dtype=np.int64)
    flattened = sorted(int(i) for group in sparse_indices for i in group)
    return np.unique(np.asarray(flattened, dtype=np.int64))


def tensor_batch_to_device(batch, device: torch.device):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = batch_x.float().to(device, non_blocking=False)
    batch_y = batch_y.float().to(device, non_blocking=False)
    if batch_x_mark is not None:
        batch_x_mark = batch_x_mark.float().to(device, non_blocking=False)
    if batch_y_mark is not None:
        batch_y_mark = batch_y_mark.float().to(device, non_blocking=False)
    return batch_x, batch_y, batch_x_mark, batch_y_mark


def prepare_sparse_validation_batches(
    args,
    data_provider_fn,
    kdfh_fn,
    device: torch.device,
) -> List[Dict]:
    """Cache fixed training-like VarDrop subsets from fixed validation batches.

    Redundancy is measured from the *full* batch before sampling. The sampled
    indices are then frozen and reused for baseline and every rank ablation so
    sampling randomness cannot contaminate rank sensitivity.
    """
    _, loader = data_provider_fn(args, 'val')
    records: List[Dict] = []

    # Reset sampling RNG explicitly so this diagnostic is reproducible even if
    # data-provider construction consumed random numbers internally.
    np.random.seed(args.seed + 1701)

    for batch_idx, batch in enumerate(loader):
        if len(records) >= args.diag_batches:
            break
        batch_x, batch_y, batch_x_mark, batch_y_mark = tensor_batch_to_device(batch, device)

        # Keep batch cardinality fixed at the requested batch_size. The final
        # short validation batch would otherwise change the batch-averaged FFT
        # statistic relative to training.
        if int(batch_x.shape[0]) != int(args.batch_size):
            continue

        hashes = kdfh_fn(
            batch_x,
            k=args.k,
            freq_list=range(args.freq_start, args.freq_end),
            min_thres=None,
        )
        feats = redundancy_features_from_hashes(hashes)

        # Sample directly from the *same* hashes used for the redundancy score.
        # This is exactly the original VarDrop np.unique/np.where/np.random.choice
        # rule, but avoids computing the FFT a second time solely for sampling.
        sparse_indices = sample_indices_from_hashes(hashes, args.group_size)
        if sparse_indices.size == 0:
            raise RuntimeError(f'Batch {batch_idx}: VarDrop selected zero variates.')

        feats['selected_unique'] = int(sparse_indices.size)
        feats['selected_ratio'] = float(sparse_indices.size / batch_x.shape[-1])
        feats['drop_ratio'] = float(1.0 - feats['selected_ratio'])

        sparse_x = batch_x[:, :, sparse_indices].detach().cpu().clone()
        sparse_y = batch_y[:, :, sparse_indices].detach().cpu().clone()
        x_mark_cpu = None if batch_x_mark is None else batch_x_mark.detach().cpu().clone()
        y_mark_cpu = None if batch_y_mark is None else batch_y_mark.detach().cpu().clone()

        records.append({
            'batch_index': int(batch_idx),
            'features': feats,
            'sparse_indices': sparse_indices.tolist(),
            'batch': (sparse_x, sparse_y, x_mark_cpu, y_mark_cpu),
        })

    if len(records) < 8:
        raise RuntimeError(f'Only {len(records)} full-size validation batches were cached; need at least 8.')
    return records


def evaluate_batches_per_record(model, records: List[Dict], args, device: torch.device) -> List[Dict]:
    model.eval()
    output = []
    with torch.inference_mode():
        for rec in records:
            batch_x, batch_y, batch_x_mark, batch_y_mark = tensor_batch_to_device(rec['batch'], device)
            if 'PEMS' in args.data or 'Solar' in args.data:
                batch_x_mark = None
                batch_y_mark = None

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if args.output_attention:
                outputs = outputs[0]
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            truth = batch_y[:, -args.pred_len:, f_dim:]
            diff = outputs - truth
            mse = float(torch.mean(diff.double().pow(2)).item())
            mae = float(torch.mean(torch.abs(diff.double())).item())
            output.append({
                'batch_index': rec['batch_index'],
                'mse': mse,
                'mae': mae,
                'elements': int(diff.numel()),
            })
    return output


def attach_metrics(records: List[Dict], label: str, metrics: List[Dict]) -> None:
    if len(records) != len(metrics):
        raise RuntimeError(f'Metric count mismatch for {label}.')
    for rec, met in zip(records, metrics):
        if rec['batch_index'] != met['batch_index']:
            raise RuntimeError(f'Batch order mismatch for {label}.')
        rec[label] = met


def get_metric_gap(rec: Dict, label: str, metric: str = 'mse') -> float:
    return pct_gap(float(rec[label][metric]), float(rec['baseline'][metric]))


def required_rank(rec: Dict, mode: str, threshold_pct: float) -> int:
    if get_metric_gap(rec, f'{mode}_r128') <= threshold_pct:
        return 128
    if get_metric_gap(rec, f'{mode}_r256') <= threshold_pct:
        return 256
    return 512


def correlation_record(
    records: List[Dict],
    feature: str,
    target_values: np.ndarray,
    target: str,
    args,
    rng: np.random.Generator,
) -> Dict:
    x = np.asarray([float(r['features'][feature]) for r in records], dtype=np.float64)
    y = np.asarray(target_values, dtype=np.float64)
    rho = spearman_corr(x, y)
    p = permutation_pvalue_spearman(x, y, rho, args.permutations, rng)
    return {
        'feature': feature,
        'target': target,
        'n': int(x.size),
        'spearman_rho': _json_float(rho),
        'permutation_p_two_sided': _json_float(p),
    }


def quartile_analysis(records: List[Dict], mode: str, threshold_pct: float) -> Dict:
    red = np.asarray([r['features']['redundancy_score'] for r in records], dtype=np.float64)
    q25 = float(np.quantile(red, 0.25))
    q75 = float(np.quantile(red, 0.75))
    low = [r for r in records if r['features']['redundancy_score'] <= q25]
    high = [r for r in records if r['features']['redundancy_score'] >= q75]

    def block(rows):
        gaps128 = np.asarray([get_metric_gap(r, f'{mode}_r128') for r in rows], dtype=np.float64)
        gaps256 = np.asarray([get_metric_gap(r, f'{mode}_r256') for r in rows], dtype=np.float64)
        req = np.asarray([required_rank(r, mode, threshold_pct) for r in rows], dtype=np.int64)
        return {
            'n': len(rows),
            'mean_gap_r128_pct': float(np.mean(gaps128)),
            'median_gap_r128_pct': float(np.median(gaps128)),
            'mean_gap_r256_pct': float(np.mean(gaps256)),
            'safe_r128_rate': float(np.mean(gaps128 <= threshold_pct)),
            'safe_r256_rate': float(np.mean(gaps256 <= threshold_pct)),
            'mean_required_rank': float(np.mean(req)),
        }

    lo = block(low)
    hi = block(high)
    return {
        'mode': mode,
        'threshold_pct': threshold_pct,
        'q25_redundancy': q25,
        'q75_redundancy': q75,
        'low_redundancy': lo,
        'high_redundancy': hi,
        # Expected positive if redundancy helps low-rank tolerance.
        'safe_r128_rate_advantage_high_minus_low': hi['safe_r128_rate'] - lo['safe_r128_rate'],
        # Expected positive if high-redundancy batches have smaller r128 loss.
        'r128_gap_advantage_low_minus_high_pct': lo['mean_gap_r128_pct'] - hi['mean_gap_r128_pct'],
        # Expected positive if low-redundancy batches need larger ranks.
        'required_rank_advantage_low_minus_high': lo['mean_required_rank'] - hi['mean_required_rank'],
    }


def choose_decision(primary_corr: Dict, quartile: Dict, n: int) -> str:
    rho = primary_corr.get('spearman_rho')
    p = primary_corr.get('permutation_p_two_sided')
    gap_adv = float(quartile['r128_gap_advantage_low_minus_high_pct'])
    safe_adv = float(quartile['safe_r128_rate_advantage_high_minus_low'])

    if rho is None:
        return 'INCONCLUSIVE_CONSTANT_OR_INVALID_SIGNAL'
    if rho > 0.15 and gap_adv < 0.0:
        return 'NO_GO_WRONG_DIRECTION'
    if n >= 24 and rho <= -0.30 and p is not None and p <= 0.05 and (gap_adv >= 1.0 or safe_adv >= 0.25):
        return 'STRONG_GO_REDUNDANCY_CONDITIONED_WIDTH'
    if rho <= -0.20 and (gap_adv >= 0.5 or safe_adv >= 0.15):
        return 'PROMISING_REDUNDANCY_CONDITIONED_WIDTH'
    if abs(rho) < 0.15 and abs(gap_adv) < 0.5 and abs(safe_adv) < 0.15:
        return 'NO_GO_WEAK_RELATION'
    return 'INCONCLUSIVE_MORE_EVIDENCE_NEEDED'


def write_outputs(output_dir: str, records: List[Dict], correlations: List[Dict], summary: Dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, 'correlations.json'), 'w', encoding='utf-8') as f:
        json.dump(correlations, f, indent=2)

    flat_rows = []
    for r in records:
        row = {'batch_index': r['batch_index']}
        row.update(r['features'])
        row['baseline_mse'] = r['baseline']['mse']
        row['baseline_mae'] = r['baseline']['mae']
        for mode in ('attention', 'ffn', 'both'):
            for rank in (128, 256):
                key = f'{mode}_r{rank}'
                row[f'{key}_mse'] = r[key]['mse']
                row[f'{key}_mae'] = r[key]['mae']
                row[f'{key}_mse_gap_pct'] = get_metric_gap(r, key, 'mse')
                row[f'{key}_mae_gap_pct'] = get_metric_gap(r, key, 'mae')
            row[f'{mode}_required_rank_1pct'] = required_rank(r, mode, 1.0)
            row[f'{mode}_required_rank_3pct'] = required_rank(r, mode, 3.0)
        flat_rows.append(row)

    with open(os.path.join(output_dir, 'batch_metrics.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    with open(os.path.join(output_dir, 'correlations.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['feature', 'target', 'n', 'spearman_rho', 'permutation_p_two_sided'])
        writer.writeheader()
        writer.writerows(correlations)


def run_self_test() -> None:
    print('[SELF-TEST] Starting GPT4 redundancy-rank diagnostic tests...')

    # Known grouping: counts 3,2,1 over N=6.
    hashes = np.asarray(['a', 'a', 'a', 'b', 'b', 'c'])
    f = redundancy_features_from_hashes(hashes)
    assert f['n_vars'] == 6
    assert f['num_groups'] == 3
    assert abs(f['largest_group_ratio'] - 0.5) < 1e-12
    assert abs(f['pair_collision'] - (8.0 / 30.0)) < 1e-12
    assert 0.0 < f['redundancy_score'] < 1.0

    # All-singleton should have zero redundancy score (up to floating error).
    singleton = redundancy_features_from_hashes(np.asarray(['a', 'b', 'c', 'd']))
    assert abs(singleton['redundancy_score']) < 1e-12

    # Spearman sign/tie handling.
    x = np.asarray([1, 2, 3, 4, 5], dtype=np.float64)
    assert abs(spearman_corr(x, x) - 1.0) < 1e-12
    assert abs(spearman_corr(x, x[::-1]) + 1.0) < 1e-12
    tied = average_ranks(np.asarray([1.0, 1.0, 3.0, 4.0]))
    assert np.allclose(tied, [1.5, 1.5, 3.0, 4.0])

    # Permutation test should identify a perfect monotonic relation in a modest sample.
    xx = np.arange(20, dtype=np.float64)
    yy = -xx.copy()
    rho = spearman_corr(xx, yy)
    p = permutation_pvalue_spearman(xx, yy, rho, 399, np.random.default_rng(123))
    assert rho < -0.999
    assert p <= 0.01, p

    # Exact sampling equivalence against original VarDrop when available.
    try:
        from VarDrop import efficient_sampler as ref_sampler, k_dominant_frequency_hashing as ref_kdfh
        xref = torch.randn(8, 32, 20)
        href = ref_kdfh(xref, k=4, freq_list=range(1, 10))
        np.random.seed(444)
        ours = sample_indices_from_hashes(href, group_size=3)
        np.random.seed(444)
        ref = np.unique(np.asarray(ref_sampler(xref, k=4, group_size=3, freq_list=range(1, 10)), dtype=np.int64))
        assert np.array_equal(ours, ref), (ours, ref)
        sampling_msg = ' exact_sampling=PASS'
    except ImportError:
        sampling_msg = ' exact_sampling=SKIPPED(no VarDrop import)'

    # Decision rule sanity.
    strong = choose_decision(
        {'spearman_rho': -0.50, 'permutation_p_two_sided': 0.01},
        {'r128_gap_advantage_low_minus_high_pct': 1.2, 'safe_r128_rate_advantage_high_minus_low': 0.30},
        32,
    )
    assert strong == 'STRONG_GO_REDUNDANCY_CONDITIONED_WIDTH'
    weak = choose_decision(
        {'spearman_rho': 0.02, 'permutation_p_two_sided': 0.8},
        {'r128_gap_advantage_low_minus_high_pct': 0.1, 'safe_r128_rate_advantage_high_minus_low': 0.05},
        32,
    )
    assert weak == 'NO_GO_WEAK_RELATION'

    print(f'[SELF-TEST] PASS. known_redundancy={f["redundancy_score"]:.6f} rho={rho:.3f} p={p:.4f}{sampling_msg}')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Batch spectral redundancy vs low-rank sensitivity diagnostic')
    p.add_argument('--self_test', action='store_true')
    p.add_argument('--checkpoint_path', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='./diagnostics/results/gpt4_redundancy_rank')
    p.add_argument('--diag_batches', type=int, default=48)
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--permutations', type=int, default=1999)
    p.add_argument('--safe_gap_pct', type=float, default=3.0,
                   help='primary per-batch MSE tolerance used for required-rank/quartile decision')

    p.add_argument('--k', type=int, default=4)
    p.add_argument('--group_size', type=int, default=10)
    p.add_argument('--freq_start', type=int, default=1)
    p.add_argument('--freq_end', type=int, default=25,
                   help='exclusive end; defaults to range(1,25) exactly as original Traffic VarDrop')

    p.add_argument('--data', type=str, default='custom')
    p.add_argument('--root_path', type=str, default='./dataset/traffic/')
    p.add_argument('--data_path', type=str, default='traffic.csv')
    p.add_argument('--features', type=str, default='M')
    p.add_argument('--target', type=str, default='OT')
    p.add_argument('--freq', type=str, default='h')
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--label_len', type=int, default=48)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--enc_in', type=int, default=862)
    p.add_argument('--dec_in', type=int, default=862)
    p.add_argument('--c_out', type=int, default=862)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--e_layers', type=int, default=4)
    p.add_argument('--d_layers', type=int, default=1)
    p.add_argument('--d_ff', type=int, default=512)
    p.add_argument('--moving_avg', type=int, default=25)
    p.add_argument('--factor', type=int, default=1)
    p.add_argument('--distil', action='store_true', default=True)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--embed', type=str, default='timeF')
    p.add_argument('--activation', type=str, default='gelu')
    p.add_argument('--output_attention', action='store_true', default=False)
    p.add_argument('--use_norm', type=int, default=True)
    p.add_argument('--class_strategy', type=str, default='projection')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.checkpoint_path:
        raise SystemExit('--checkpoint_path is required unless --self_test is used.')
    if args.diag_batches <= 0:
        raise SystemExit('--diag_batches must be positive.')
    if args.freq_end <= args.freq_start:
        raise SystemExit('--freq_end must be greater than --freq_start.')

    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('=' * 88)
    print('GPT4 BATCH REDUNDANCY vs LOW-RANK SENSITIVITY DIAGNOSTIC')
    print('=' * 88)
    print(f'checkpoint       : {args.checkpoint_path}')
    print(f'device           : {device}')
    print(f'diagnostic data  : up to {args.diag_batches} full-size validation batches x batch_size {args.batch_size}')
    print(f'VarDrop grouping : k={args.k}, group_size={args.group_size}, freq=range({args.freq_start},{args.freq_end})')
    print('rank tests       : attention/ffn/both x {128,256}; baseline=full rank')
    print('primary question : does higher pre-sampling spectral redundancy predict smaller rank-128 MSE damage?')

    from model import OURS
    from data_provider.data_factory import data_provider
    try:
        from VarDrop import k_dominant_frequency_hashing_fast_exact as kdfh_for_diag
        kdfh_name = 'k_dominant_frequency_hashing_fast_exact'
    except ImportError:
        from VarDrop import k_dominant_frequency_hashing as kdfh_for_diag
        kdfh_name = 'k_dominant_frequency_hashing'
    print(f'[GPT4-CORR] k-DFH implementation for diagnostic: {kdfh_name} (same grouping semantics)')

    model = OURS.Model(args).float().to(device)
    load_checkpoint(model, args.checkpoint_path)
    print('[GPT4-CORR] Checkpoint loaded with strict=True.')

    targets = collect_targets(model)
    print(f'[GPT4-CORR] Found {len(targets)} matrices.')

    records = prepare_sparse_validation_batches(
        args,
        data_provider,
        kdfh_for_diag,
        device,
    )
    print(f'[GPT4-CORR] Cached {len(records)} fixed training-like VarDrop batches.')
    selected = np.asarray([r['features']['selected_unique'] for r in records], dtype=np.float64)
    red = np.asarray([r['features']['redundancy_score'] for r in records], dtype=np.float64)
    print(f'[REDUNDANCY] selected_unique mean={selected.mean():.1f} min={selected.min():.0f} max={selected.max():.0f}')
    print(f'[REDUNDANCY] score mean={red.mean():.4f} std={red.std():.4f} min={red.min():.4f} max={red.max():.4f}')

    # Baseline repeat on the exact same sparse tensors catches any accidental
    # stochasticity in evaluation before rank comparisons are trusted.
    restore_targets(targets)
    baseline = evaluate_batches_per_record(model, records, args, device)
    baseline_repeat = evaluate_batches_per_record(model, records, args, device)
    max_repeat_mse = max(abs(a['mse'] - b['mse']) for a, b in zip(baseline, baseline_repeat))
    max_repeat_mae = max(abs(a['mae'] - b['mae']) for a, b in zip(baseline, baseline_repeat))
    print(f'[CONTROL repeat] max_batch_|ΔMSE|={max_repeat_mse:.3e} max_batch_|ΔMAE|={max_repeat_mae:.3e}')
    if max_repeat_mse > 1e-10 or max_repeat_mae > 1e-10:
        raise RuntimeError('Baseline repeat is not deterministic; aborting.')
    attach_metrics(records, 'baseline', baseline)

    # CPU float64 SVD avoids cuSOLVER float32 full-rank reconstruction noise.
    compute_svd(targets, torch.device('cpu'))
    max_weight_recon_error = max(full_reconstruction_error(t) for t in targets)
    print(f'[SVD SANITY] max_full_rank_weight_rel_error={max_weight_recon_error:.3e}')
    if (not math.isfinite(max_weight_recon_error)) or max_weight_recon_error > 1e-10:
        raise RuntimeError('Full-rank CPU-float64 SVD reconstruction is unexpectedly inaccurate.')

    apply_full_reconstruction(targets)
    full_recon_metrics = evaluate_batches_per_record(model, records, args, device)
    max_full_mse_diff = max(abs(a['mse'] - b['mse']) for a, b in zip(baseline, full_recon_metrics))
    max_full_mae_diff = max(abs(a['mae'] - b['mae']) for a, b in zip(baseline, full_recon_metrics))
    print(f'[SVD FULL CONTROL] max_batch_|ΔMSE|={max_full_mse_diff:.3e} max_batch_|ΔMAE|={max_full_mae_diff:.3e}')
    if max_full_mse_diff > 1e-8 or max_full_mae_diff > 1e-8:
        raise RuntimeError('Full-rank SVD reconstruction changed sparse-batch predictions; aborting.')
    restore_targets(targets)

    for rank in (128, 256):
        for mode in ('attention', 'ffn', 'both'):
            apply_rank(targets, rank, mode)
            metrics = evaluate_batches_per_record(model, records, args, device)
            label = f'{mode}_r{rank}'
            attach_metrics(records, label, metrics)
            mean_gap = float(np.mean([get_metric_gap(r, label) for r in records]))
            median_gap = float(np.median([get_metric_gap(r, label) for r in records]))
            safe1 = float(np.mean([get_metric_gap(r, label) <= 1.0 for r in records]))
            safe3 = float(np.mean([get_metric_gap(r, label) <= 3.0 for r in records]))
            print(f'[RANK SUMMARY] mode={mode:<9} rank={rank} mean_MSE_gap={mean_gap:+.3f}% '
                  f'median={median_gap:+.3f}% safe<=1%={100*safe1:5.1f}% safe<=3%={100*safe3:5.1f}%')

    restore_targets(targets)

    feature_names = [
        'redundancy_score',
        'effective_group_ratio',
        'group_ratio',
        'largest_group_ratio',
        'mean_group_size',
        'singleton_var_ratio',
        'pair_collision',
        'selected_ratio',
    ]
    correlations: List[Dict] = []
    rng = np.random.default_rng(args.seed + 991)

    for mode in ('attention', 'ffn', 'both'):
        for rank in (128, 256):
            label = f'{mode}_r{rank}'
            gaps = np.asarray([get_metric_gap(r, label) for r in records], dtype=np.float64)
            for feature in feature_names:
                correlations.append(correlation_record(
                    records, feature, gaps, f'{label}_mse_gap_pct', args, rng
                ))
        req3 = np.asarray([required_rank(r, mode, args.safe_gap_pct) for r in records], dtype=np.float64)
        correlations.append(correlation_record(
            records, 'redundancy_score', req3, f'{mode}_required_rank_at_{args.safe_gap_pct:.1f}pct', args, rng
        ))

    # Primary pre-registered-ish test: redundancy_score vs BOTH rank-128 gap.
    primary = next(c for c in correlations
                   if c['feature'] == 'redundancy_score' and c['target'] == 'both_r128_mse_gap_pct')

    print('\n' + '=' * 88)
    print('PRIMARY CORRELATION')
    print('=' * 88)
    print(f'[CORRELATION primary] redundancy_score vs both_r128_mse_gap_pct: '
          f'rho={primary["spearman_rho"]} p_perm={primary["permutation_p_two_sided"]} n={primary["n"]}')

    # Secondary mode-specific correlations, useful if redundancy predicts only
    # attention width or only FFN width rather than a joint width.
    for mode in ('attention', 'ffn', 'both'):
        row = next(c for c in correlations
                   if c['feature'] == 'redundancy_score' and c['target'] == f'{mode}_r128_mse_gap_pct')
        print(f'[CORRELATION] mode={mode:<9} rank=128 rho={row["spearman_rho"]} '
              f'p_perm={row["permutation_p_two_sided"]}')

    quartiles = {}
    print('\n' + '=' * 88)
    print('HIGH-vs-LOW REDUNDANCY QUARTILES')
    print('=' * 88)
    for mode in ('attention', 'ffn', 'both'):
        q = quartile_analysis(records, mode, args.safe_gap_pct)
        quartiles[mode] = q
        print(f'[QUARTILE] mode={mode:<9} high-minus-low safe@128={100*q["safe_r128_rate_advantage_high_minus_low"]:+.1f}pp '
              f'low-minus-high mean_gap@128={q["r128_gap_advantage_low_minus_high_pct"]:+.3f}pp '
              f'low-minus-high required_rank={q["required_rank_advantage_low_minus_high"]:+.1f}')

    # Mode-specific decisions use the same conservative rule. BOTH is the main
    # hypothesis; ATTENTION/FFN are diagnostic fallbacks rather than hidden wins.
    decisions = {}
    for mode in ('attention', 'ffn', 'both'):
        corr = next(c for c in correlations
                    if c['feature'] == 'redundancy_score' and c['target'] == f'{mode}_r128_mse_gap_pct')
        decisions[mode] = choose_decision(corr, quartiles[mode], len(records))

    if decisions['both'].startswith('STRONG_GO') or decisions['both'].startswith('PROMISING'):
        overall = 'GO_ADAPTIVE_BOTH_WIDTH' if decisions['both'].startswith('STRONG_GO') else 'PROMISING_ADAPTIVE_BOTH_WIDTH'
    elif decisions['attention'].startswith('STRONG_GO') or decisions['attention'].startswith('PROMISING'):
        overall = 'GO_ADAPTIVE_ATTENTION_WIDTH_ONLY' if decisions['attention'].startswith('STRONG_GO') else 'PROMISING_ADAPTIVE_ATTENTION_WIDTH_ONLY'
    elif decisions['ffn'].startswith('STRONG_GO') or decisions['ffn'].startswith('PROMISING'):
        overall = 'GO_ADAPTIVE_FFN_WIDTH_ONLY' if decisions['ffn'].startswith('STRONG_GO') else 'PROMISING_ADAPTIVE_FFN_WIDTH_ONLY'
    elif all(d.startswith('NO_GO') for d in decisions.values()):
        overall = 'NO_GO_REDUNDANCY_DOES_NOT_PREDICT_REQUIRED_WIDTH'
    else:
        overall = 'INCONCLUSIVE_REDUNDANCY_WIDTH_RELATION'

    print('\n' + '=' * 88)
    print('DECISION')
    print('=' * 88)
    print(f'attention : {decisions["attention"]}')
    print(f'ffn       : {decisions["ffn"]}')
    print(f'both      : {decisions["both"]}')
    print(f'OVERALL   : {overall}')
    print('Interpretation: this is a checkpoint sensitivity/correlation screen, not a trained dynamic-width model.')

    summary = {
        'checkpoint': os.path.abspath(args.checkpoint_path),
        'device': str(device),
        'seed': args.seed,
        'diag_batches': len(records),
        'batch_size': args.batch_size,
        'k': args.k,
        'group_size': args.group_size,
        'freq_range': [args.freq_start, args.freq_end],
        'safe_gap_pct': args.safe_gap_pct,
        'permutations': args.permutations,
        'redundancy_summary': {
            'mean': float(red.mean()),
            'std': float(red.std()),
            'min': float(red.min()),
            'max': float(red.max()),
            'selected_unique_mean': float(selected.mean()),
            'selected_unique_min': int(selected.min()),
            'selected_unique_max': int(selected.max()),
        },
        'baseline_repeat_max_abs_diff': {'mse': max_repeat_mse, 'mae': max_repeat_mae},
        'full_rank_svd_control': {
            'max_weight_rel_error': max_weight_recon_error,
            'max_batch_mse_abs_diff': max_full_mse_diff,
            'max_batch_mae_abs_diff': max_full_mae_diff,
        },
        'primary_correlation': primary,
        'quartiles': quartiles,
        'mode_decisions': decisions,
        'overall_decision': overall,
    }
    write_outputs(args.output_dir, records, correlations, summary)
    print(f'[GPT4-CORR] Results saved to: {os.path.abspath(args.output_dir)}')


if __name__ == '__main__':
    main()

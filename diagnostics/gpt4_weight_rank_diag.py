import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class TargetMatrix:
    name: str
    group: str  # attention or ffn
    param: torch.nn.Parameter
    original: torch.Tensor  # CPU, original parameter shape
    matrix: torch.Tensor  # CPU, 2-D view of original
    u: torch.Tensor = None
    s: torch.Tensor = None
    vh: torch.Tensor = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_ranks(text: str) -> List[int]:
    values = []
    for item in text.split(','):
        item = item.strip()
        if item:
            value = int(item)
            if value <= 0:
                raise ValueError('Ranks must be positive integers.')
            values.append(value)
    if not values:
        raise ValueError('At least one rank is required.')
    return sorted(set(values))


def parse_modes(text: str) -> List[str]:
    allowed = {'attention', 'ffn', 'both'}
    modes = []
    for item in text.split(','):
        item = item.strip().lower()
        if item:
            if item not in allowed:
                raise ValueError(f'Unknown mode {item!r}; allowed: {sorted(allowed)}')
            modes.append(item)
    if not modes:
        raise ValueError('At least one mode is required.')
    return list(dict.fromkeys(modes))


def matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        return tensor[:, :, 0]
    raise ValueError(f'Unsupported target tensor shape: {tuple(tensor.shape)}')


def copy_matrix_to_param(param: torch.nn.Parameter, matrix: torch.Tensor) -> None:
    with torch.no_grad():
        if param.ndim == 2:
            if tuple(param.shape) != tuple(matrix.shape):
                raise ValueError(f'Shape mismatch: param={tuple(param.shape)}, matrix={tuple(matrix.shape)}')
            param.copy_(matrix.to(device=param.device, dtype=param.dtype))
        elif param.ndim == 3 and param.shape[-1] == 1:
            if tuple(param.shape[:2]) != tuple(matrix.shape):
                raise ValueError(f'Shape mismatch: param={tuple(param.shape)}, matrix={tuple(matrix.shape)}')
            param[:, :, 0].copy_(matrix.to(device=param.device, dtype=param.dtype))
        else:
            raise ValueError(f'Unsupported target parameter shape: {tuple(param.shape)}')


def get_base_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def collect_targets(model) -> List[TargetMatrix]:
    base = get_base_model(model)
    if not hasattr(base, 'encoder') or not hasattr(base.encoder, 'attn_layers'):
        raise RuntimeError('Expected model.encoder.attn_layers; incompatible model structure.')

    targets: List[TargetMatrix] = []
    for li, layer in enumerate(base.encoder.attn_layers):
        if not hasattr(layer, 'attention'):
            raise RuntimeError(f'Encoder layer {li} has no attention module.')
        attn = layer.attention
        for attr, short in [
            ('query_projection', 'q'),
            ('key_projection', 'k'),
            ('value_projection', 'v'),
            ('out_projection', 'o'),
        ]:
            module = getattr(attn, attr, None)
            if module is None or not hasattr(module, 'weight'):
                raise RuntimeError(f'Layer {li}: missing attention.{attr}.weight')
            original = module.weight.detach().cpu().float().clone()
            targets.append(TargetMatrix(
                name=f'layer{li}.attn.{short}',
                group='attention',
                param=module.weight,
                original=original,
                matrix=matrix_view(original).clone(),
            ))

        for attr in ('conv1', 'conv2'):
            module = getattr(layer, attr, None)
            if module is None or not hasattr(module, 'weight'):
                raise RuntimeError(f'Layer {li}: missing {attr}.weight')
            original = module.weight.detach().cpu().float().clone()
            targets.append(TargetMatrix(
                name=f'layer{li}.ffn.{attr}',
                group='ffn',
                param=module.weight,
                original=original,
                matrix=matrix_view(original).clone(),
            ))

    if not targets:
        raise RuntimeError('No target matrices found.')
    return targets


def restore_targets(targets: List[TargetMatrix]) -> None:
    for target in targets:
        copy_matrix_to_param(target.param, matrix_view(target.original))


def compute_svd(targets: List[TargetMatrix], svd_device: torch.device) -> None:
    # Functional rank ablation is a diagnostic, not a training-time operation.
    # On CUDA, float32 torch.linalg.svd may reconstruct a full-rank 512x512
    # matrix with ~1e-4 relative error (cuSOLVER numerical behavior). That is
    # large enough to contaminate our sanity control. When SVD is requested on
    # CPU we therefore promote the *already-trained float32 weights* to float64
    # solely for the decomposition/reconstruction. No model weights are changed
    # until a reconstructed matrix is copied back to the original float32 param.
    svd_dtype = torch.float64 if svd_device.type == 'cpu' else torch.float32
    print(f'[GPT4-DIAG] Computing SVDs on {svd_device} with dtype={svd_dtype} ...')
    for i, target in enumerate(targets, start=1):
        mat = target.matrix.to(device=svd_device, dtype=svd_dtype)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        target.u = u.detach().cpu()
        target.s = s.detach().cpu()
        target.vh = vh.detach().cpu()
        print(f'  SVD {i:02d}/{len(targets)} {target.name:<22} shape={tuple(mat.shape)}')


def reconstruct(target: TargetMatrix, rank: int) -> torch.Tensor:
    max_rank = int(target.s.numel())
    r = min(int(rank), max_rank)
    if r <= 0:
        raise ValueError('Reconstruction rank must be positive.')
    return (target.u[:, :r] * target.s[:r].unsqueeze(0)) @ target.vh[:r, :]


def apply_rank(targets: List[TargetMatrix], rank: int, mode: str) -> None:
    restore_targets(targets)
    selected_groups = {'attention', 'ffn'} if mode == 'both' else {mode}
    for target in targets:
        if target.group in selected_groups:
            copy_matrix_to_param(target.param, reconstruct(target, rank))


def apply_full_reconstruction(targets: List[TargetMatrix]) -> None:
    restore_targets(targets)
    for target in targets:
        copy_matrix_to_param(target.param, reconstruct(target, int(target.s.numel())))


def spectral_record(target: TargetMatrix, ranks: List[int]) -> Dict:
    s2 = target.s.double().pow(2)
    total = float(s2.sum().item())
    if total <= 0.0:
        energies = {str(r): 1.0 for r in ranks}
        stable_rank = 0.0
        rank_99 = 0
        rank_999 = 0
    else:
        c = torch.cumsum(s2, dim=0) / total
        energies = {}
        for rank in ranks:
            rr = min(rank, c.numel())
            energies[str(rank)] = float(c[rr - 1].item())
        stable_rank = float(total / max(float(s2[0].item()), 1e-30))
        rank_99 = int(torch.searchsorted(c, torch.tensor(0.99, dtype=c.dtype)).item() + 1)
        rank_999 = int(torch.searchsorted(c, torch.tensor(0.999, dtype=c.dtype)).item() + 1)
    return {
        'name': target.name,
        'group': target.group,
        'rows': int(target.matrix.shape[0]),
        'cols': int(target.matrix.shape[1]),
        'max_rank': int(target.s.numel()),
        'stable_rank': stable_rank,
        'rank_99_energy': rank_99,
        'rank_999_energy': rank_999,
        'energy': energies,
    }


def full_reconstruction_error(target: TargetMatrix) -> float:
    rec = reconstruct(target, int(target.s.numel()))
    denom = torch.linalg.vector_norm(target.matrix).item()
    num = torch.linalg.vector_norm(rec - target.matrix).item()
    return float(num / max(denom, 1e-30))


def load_checkpoint(model, checkpoint_path: str) -> None:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    obj = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        state = obj['state_dict']
    elif isinstance(obj, dict):
        state = obj
    else:
        raise RuntimeError('Unsupported checkpoint format; expected a state_dict dictionary.')

    base = get_base_model(model)
    model_keys = base.state_dict().keys()
    state_keys = list(state.keys())
    if state_keys and all(k.startswith('module.') for k in state_keys) and not any(k.startswith('module.') for k in model_keys):
        state = {k[len('module.'):]: v for k, v in state.items()}
    try:
        base.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f'Checkpoint does not exactly match the model architecture:\n{exc}') from exc


def cache_validation_batches(args, data_provider_fn, max_batches: int):
    _, loader = data_provider_fn(args, 'val')
    cached = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        cached.append(tuple(x.detach().cpu().clone() if torch.is_tensor(x) else x for x in batch))
    if not cached:
        raise RuntimeError('Validation loader produced zero cached batches.')
    return cached


def evaluate_cached(model, batches, args, device: torch.device) -> Tuple[float, float, int]:
    model.eval()
    sum_sq = 0.0
    sum_abs = 0.0
    count = 0
    with torch.inference_mode():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in batches:
            batch_x = batch_x.float().to(device, non_blocking=False)
            batch_y = batch_y.float().to(device, non_blocking=False)
            if 'PEMS' in args.data or 'Solar' in args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(device, non_blocking=False)
                batch_y_mark = batch_y_mark.float().to(device, non_blocking=False)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if args.output_attention:
                outputs = outputs[0]
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            truth = batch_y[:, -args.pred_len:, f_dim:]
            diff = outputs - truth
            sum_sq += float(torch.sum(diff.double() * diff.double()).item())
            sum_abs += float(torch.sum(torch.abs(diff.double())).item())
            count += int(diff.numel())

    return sum_sq / count, sum_abs / count, count


def pct_gap(value: float, baseline: float) -> float:
    return 100.0 * (value - baseline) / baseline


def choose_recommendation(results: List[Dict]) -> str:
    lookup = {(r['mode'], r['rank']): r for r in results if r['rank'] > 0}

    def gap(mode, rank):
        row = lookup.get((mode, rank))
        return None if row is None else row['mse_gap_pct']

    for rank in (64, 128, 256):
        g = gap('both', rank)
        if g is not None and g <= 3.0:
            if rank <= 128:
                return f'STRONG_GO_BOTH_R{rank}'
            return f'GO_BOTH_R{rank}'

    a128 = gap('attention', 128)
    f128 = gap('ffn', 128)
    if a128 is not None and a128 <= 3.0 and (f128 is None or f128 > 3.0):
        return 'GO_ATTENTION_BOTTLENECK_R128'
    if f128 is not None and f128 <= 3.0 and (a128 is None or a128 > 3.0):
        return 'GO_FFN_BOTTLENECK_R128'

    b128 = gap('both', 128)
    b256 = gap('both', 256)
    if (b128 is not None and b128 <= 10.0) or (b256 is not None and b256 <= 5.0):
        return 'PROMISING_RETRAINING_MAY_RECOVER'
    return 'WEAK_LOW_RANK_EVIDENCE'


def write_outputs(output_dir: str, spectral: List[Dict], functional: List[Dict], summary: Dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'spectral.json'), 'w', encoding='utf-8') as f:
        json.dump(spectral, f, indent=2)
    with open(os.path.join(output_dir, 'functional.json'), 'w', encoding='utf-8') as f:
        json.dump(functional, f, indent=2)
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    if functional:
        keys = ['mode', 'rank', 'mse', 'mae', 'mse_gap_pct', 'mae_gap_pct', 'elements']
        with open(os.path.join(output_dir, 'functional.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in functional:
                writer.writerow({k: row.get(k) for k in keys})


def run_self_test() -> None:
    print('[SELF-TEST] Starting GPT4 diagnostic helper tests...')
    set_seed(123)

    # 1) SVD reconstruction and numerical rank.
    m = torch.randn(37, 29)
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    full = (u * s.unsqueeze(0)) @ vh
    rel = torch.linalg.vector_norm(full - m) / torch.linalg.vector_norm(m)
    assert rel.item() < 1e-5, f'Full-rank SVD reconstruction error too large: {rel.item()}'
    r = 7
    low = (u[:, :r] * s[:r].unsqueeze(0)) @ vh[:r, :]
    numerical_rank = int(torch.linalg.matrix_rank(low, tol=1e-4).item())
    assert numerical_rank <= r, (numerical_rank, r)

    # 2) Conv1d kernel-size-one matrix view/copy behavior.
    conv = torch.nn.Conv1d(11, 13, kernel_size=1, bias=False)
    original = conv.weight.detach().clone()
    mat = matrix_view(original)
    assert mat.shape == (13, 11)
    copy_matrix_to_param(conv.weight, mat * 0.5)
    assert torch.allclose(conv.weight[:, :, 0], mat * 0.5)
    copy_matrix_to_param(conv.weight, mat)
    assert torch.equal(conv.weight.detach(), original)

    # 3) Tiny mock of the exact VarDrop encoder attribute layout; collect/apply/restore.
    class MockAttention(torch.nn.Module):
        def __init__(self, d=16):
            super().__init__()
            self.query_projection = torch.nn.Linear(d, d)
            self.key_projection = torch.nn.Linear(d, d)
            self.value_projection = torch.nn.Linear(d, d)
            self.out_projection = torch.nn.Linear(d, d)

    class MockLayer(torch.nn.Module):
        def __init__(self, d=16):
            super().__init__()
            self.attention = MockAttention(d)
            self.conv1 = torch.nn.Conv1d(d, d, 1)
            self.conv2 = torch.nn.Conv1d(d, d, 1)

    class MockEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_layers = torch.nn.ModuleList([MockLayer(), MockLayer()])

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = MockEncoder()

    mock = MockModel()
    targets = collect_targets(mock)
    assert len(targets) == 12, len(targets)
    before = {t.name: t.param.detach().clone() for t in targets}
    compute_svd(targets, torch.device('cpu'))
    max_full_err = max(full_reconstruction_error(t) for t in targets)
    assert max_full_err < 1e-5, max_full_err
    apply_rank(targets, 4, 'both')
    assert any(not torch.equal(t.param.detach(), before[t.name]) for t in targets)
    restore_targets(targets)
    for t in targets:
        assert torch.equal(t.param.detach().cpu(), before[t.name].cpu()), t.name

    # 4) Rank parsing / mode parsing guardrails.
    assert parse_ranks('256,64,128,128') == [64, 128, 256]
    assert parse_modes('attention,ffn,both,attention') == ['attention', 'ffn', 'both']

    print(f'[SELF-TEST] PASS. full_svd_rel_error={rel.item():.3e}, max_mock_reconstruction_error={max_full_err:.3e}')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='GPT4 weight-spectrum + functional low-rank diagnostic for VarDrop/iTransformer')
    p.add_argument('--self_test', action='store_true', help='run internal helper tests and exit')
    p.add_argument('--checkpoint_path', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='./diagnostics/results/gpt4_weight_rank')
    p.add_argument('--ranks', type=str, default='64,128,256')
    p.add_argument('--modes', type=str, default='attention,ffn,both')
    p.add_argument('--diag_batches', type=int, default=16)
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--svd_cpu', action='store_true', help='force SVD computation on CPU')

    # Dataset/model arguments kept compatible with the original VarDrop Model/DataProvider.
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
    p.add_argument('--batch_size', type=int, default=16)
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

    ranks = parse_ranks(args.ranks)
    modes = parse_modes(args.modes)
    set_seed(args.seed)

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    svd_device = torch.device('cpu') if args.svd_cpu or device.type == 'cpu' else device
    print('=' * 78)
    print('GPT4 WEIGHT SPECTRUM + FUNCTIONAL LOW-RANK DIAGNOSTIC')
    print('=' * 78)
    print(f'checkpoint      : {args.checkpoint_path}')
    print(f'device          : {device}')
    print(f'svd_device      : {svd_device}')
    print(f'ranks           : {ranks}')
    print(f'modes           : {modes}')
    print(f'diagnostic data : {args.diag_batches} cached validation batches x batch_size {args.batch_size}')

    from model import OURS
    from data_provider.data_factory import data_provider

    model = OURS.Model(args).float().to(device)
    load_checkpoint(model, args.checkpoint_path)
    print('[GPT4-DIAG] Checkpoint loaded with strict=True.')

    targets = collect_targets(model)
    print(f'[GPT4-DIAG] Found {len(targets)} matrices: '
          f'{sum(t.group == "attention" for t in targets)} attention + '
          f'{sum(t.group == "ffn" for t in targets)} FFN.')

    # Cache once because the original validation DataLoader shuffles validation batches.
    cached_batches = cache_validation_batches(args, data_provider, args.diag_batches)
    cached_examples = sum(int(batch[0].shape[0]) for batch in cached_batches)
    print(f'[GPT4-DIAG] Cached {len(cached_batches)} fixed validation batches ({cached_examples} examples).')

    # Baseline control twice on exactly the same cached tensors.
    restore_targets(targets)
    baseline_mse, baseline_mae, elements = evaluate_cached(model, cached_batches, args, device)
    repeat_mse, repeat_mae, repeat_elements = evaluate_cached(model, cached_batches, args, device)
    if elements != repeat_elements:
        raise RuntimeError('Baseline repeat element count changed unexpectedly.')
    repeat_mse_diff = abs(repeat_mse - baseline_mse)
    repeat_mae_diff = abs(repeat_mae - baseline_mae)
    print(f'[CONTROL rank=0] MSE={baseline_mse:.10f} MAE={baseline_mae:.10f}')
    print(f'[CONTROL repeat] MSE={repeat_mse:.10f} MAE={repeat_mae:.10f} '
          f'|ΔMSE|={repeat_mse_diff:.3e} |ΔMAE|={repeat_mae_diff:.3e}')
    if repeat_mse_diff > 1e-10 or repeat_mae_diff > 1e-10:
        raise RuntimeError('Baseline repeat is not deterministic on cached batches; aborting diagnostic.')

    compute_svd(targets, svd_device)
    spectral = [spectral_record(t, ranks) for t in targets]
    recon_errors = {t.name: full_reconstruction_error(t) for t in targets}
    max_recon_error = max(recon_errors.values())
    print(f'[SVD SANITY] max full-rank relative weight reconstruction error={max_recon_error:.3e}')
    if not math.isfinite(max_recon_error) or max_recon_error > 1e-4:
        raise RuntimeError('Full-rank SVD reconstruction error is too large; aborting.')

    # Functional sanity: reconstruct every targeted matrix at its own exact maximum rank.
    apply_full_reconstruction(targets)
    full_mse, full_mae, _ = evaluate_cached(model, cached_batches, args, device)
    full_gap = pct_gap(full_mse, baseline_mse)
    print(f'[SVD FULL per-matrix-max] MSE={full_mse:.10f} MAE={full_mae:.10f} MSE_gap={full_gap:+.6f}%')
    if abs(full_gap) > 0.05:
        raise RuntimeError('Full-rank SVD reconstruction changed MSE by >0.05%; diagnostic is not trustworthy.')

    functional = [{
        'mode': 'baseline', 'rank': 0,
        'mse': baseline_mse, 'mae': baseline_mae,
        'mse_gap_pct': 0.0, 'mae_gap_pct': 0.0,
        'elements': elements,
    }]

    for rank in ranks:
        for mode in modes:
            apply_rank(targets, rank, mode)
            mse, mae, n = evaluate_cached(model, cached_batches, args, device)
            row = {
                'mode': mode,
                'rank': rank,
                'mse': mse,
                'mae': mae,
                'mse_gap_pct': pct_gap(mse, baseline_mse),
                'mae_gap_pct': pct_gap(mae, baseline_mae),
                'elements': n,
            }
            functional.append(row)
            print(f'[FUNCTIONAL] mode={mode:<9} rank={rank:<4} '
                  f'MSE={mse:.10f} ({row["mse_gap_pct"]:+.3f}%) '
                  f'MAE={mae:.10f} ({row["mae_gap_pct"]:+.3f}%)')

    restore_targets(targets)
    recommendation = choose_recommendation(functional)

    print('\n' + '=' * 78)
    print('SPECTRAL SUMMARY')
    print('=' * 78)
    for group in ('attention', 'ffn'):
        rows = [r for r in spectral if r['group'] == group]
        print(f'{group.upper()}:')
        for rank in ranks:
            mean_energy = float(np.mean([r['energy'][str(rank)] for r in rows]))
            min_energy = float(np.min([r['energy'][str(rank)] for r in rows]))
            print(f'  rank {rank:<4}: mean energy={100*mean_energy:6.2f}%  min matrix energy={100*min_energy:6.2f}%')
        print(f'  mean rank@99%   = {np.mean([r["rank_99_energy"] for r in rows]):.1f}')
        print(f'  mean rank@99.9% = {np.mean([r["rank_999_energy"] for r in rows]):.1f}')

    print('\n' + '=' * 78)
    print('DECISION')
    print('=' * 78)
    print(recommendation)
    print('Interpretation note: this is a checkpoint sensitivity screen, not a runtime speed test.')
    print('A trained bottleneck can recover from larger one-shot rank-ablation errors, so the result is conservative.')

    summary = {
        'checkpoint': os.path.abspath(args.checkpoint_path),
        'seed': args.seed,
        'device': str(device),
        'svd_device': str(svd_device),
        'diag_batches': len(cached_batches),
        'cached_examples': cached_examples,
        'ranks': ranks,
        'modes': modes,
        'baseline': {'mse': baseline_mse, 'mae': baseline_mae, 'elements': elements},
        'baseline_repeat_abs_diff': {'mse': repeat_mse_diff, 'mae': repeat_mae_diff},
        'full_rank_svd_sanity': {
            'rank': 'per_matrix_max',
            'max_weight_rel_error': max_recon_error,
            'mse': full_mse,
            'mae': full_mae,
            'mse_gap_pct': full_gap,
        },
        'recommendation': recommendation,
    }
    write_outputs(args.output_dir, spectral, functional, summary)
    print(f'[GPT4-DIAG] Results saved to: {os.path.abspath(args.output_dir)}')


if __name__ == '__main__':
    main()

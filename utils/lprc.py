import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


LPRC_METHOD = 'LPRC'
LPRC_ARTIFACT_VERSION = 2
LPRC_FIT_SPLIT = 'train'
LPRC_SCALE_SEMANTICS = 'model_output_before_optional_inverse_transform'
LPRC_OVERLAP_WEIGHTING = (
    'each forecast element from each overlapping TRAIN window contributes once'
)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError('Backbone checkpoint not found: {}'.format(path))
    return {
        'path': os.path.abspath(path),
        'sha256': sha256_file(path),
        'size_bytes': int(os.path.getsize(path)),
    }


def _checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError('Checkpoint must contain a state_dict dictionary.')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if not isinstance(state_dict, dict):
            raise RuntimeError('checkpoint["state_dict"] is not a dictionary.')
        return state_dict
    return checkpoint


def load_backbone_checkpoint(
    model,
    path,
    allow_missing_lpra_alpha=False,
    allow_untrained_lpra_module=False,
):
    """Load a backbone while rejecting every non-allowlisted key mismatch."""
    identity_before = checkpoint_identity(path)
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = _checkpoint_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)

    allowed_missing = set()
    if allow_missing_lpra_alpha:
        allowed_missing.update({'lpra_alpha', 'module.lpra_alpha'})
    allowed_unexpected = set()
    if allow_untrained_lpra_module:
        allowed_unexpected.update({
            'lpra.sensor_factor.weight',
            'lpra.phase_factor.weight',
            'module.lpra.sensor_factor.weight',
            'module.lpra.phase_factor.weight',
        })

    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    disallowed_missing = missing - allowed_missing
    disallowed_unexpected = unexpected - allowed_unexpected
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            'Checkpoint does not exactly match the backbone. '
            'missing={} unexpected={}'.format(
                sorted(disallowed_missing), sorted(disallowed_unexpected)
            )
        )
    ignored_lpra_keys = unexpected & allowed_unexpected
    if ignored_lpra_keys:
        alpha_key = (
            'module.lpra_alpha'
            if 'module.lpra_alpha' in state_dict
            else 'lpra_alpha'
        )
        if alpha_key not in state_dict:
            raise RuntimeError(
                'LPRA-wrapped base checkpoint has no alpha=0 evidence.'
            )
        alpha = float(state_dict[alpha_key].detach().cpu().item())
        if alpha != 0.0:
            raise RuntimeError(
                'Refusing to treat a calibrated LPRA checkpoint as a plain '
                'backbone; stored alpha={}.'.format(alpha)
            )
    identity_after = checkpoint_identity(path)
    if identity_before != identity_after:
        raise RuntimeError(
            'Backbone checkpoint changed while it was being loaded: {}'.format(
                path
            )
        )
    return identity_after


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _git_commit_hash():
    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repository_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_provenance(args, device):
    device_type = getattr(device, 'type', str(device))
    gpu_name = None
    if device_type == 'cuda':
        try:
            gpu_name = torch.cuda.get_device_name(device)
        except (AssertionError, RuntimeError, ValueError):
            gpu_name = None
    return {
        'created_utc': datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        'cli_arguments': _json_safe(vars(args)),
        'git_commit': _git_commit_hash(),
        'seed': int(getattr(args, 'seed', getattr(args, 'random_seed', 2023))),
        'python_version': platform.python_version(),
        'pytorch_version': str(torch.__version__),
        'cuda_version': (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        'device': str(device),
        'gpu_name': gpu_name,
        'numpy_version': str(np.__version__),
    }


class HourOfWeekPhaseIndexer:
    mode = 'hour_of_week'
    required_period = 168

    def __init__(self, period, freq, pred_len, embed='timeF'):
        self.period = int(period)
        self.freq = str(freq)
        self.pred_len = int(pred_len)
        self.embed = str(embed)
        if self.period != self.required_period:
            raise ValueError(
                'hour_of_week requires period=168; received {}.'.format(
                    self.period
                )
            )
        if self.freq.lower() not in ('h', '1h'):
            raise ValueError(
                'hour_of_week currently supports hourly timeF data only; '
                'received freq={!r}.'.format(self.freq)
            )
        if self.embed != 'timeF':
            raise ValueError(
                'hour_of_week currently requires normalized timeF marks; '
                'received embed={!r}.'.format(self.embed)
            )
        if self.pred_len <= 0:
            raise ValueError('pred_len must be positive.')

    def __call__(self, batch_y_mark):
        if batch_y_mark is None:
            raise ValueError(
                'hour_of_week requires future hourly timeF marks; received None.'
            )
        if batch_y_mark.ndim != 3 or batch_y_mark.shape[-1] < 2:
            raise ValueError(
                'hour_of_week requires marks shaped [B,T,F] with HourOfDay '
                'and DayOfWeek as the first two features.'
            )
        if batch_y_mark.shape[1] < self.pred_len:
            raise ValueError('batch_y_mark is shorter than pred_len.')

        future_mark = batch_y_mark[:, -self.pred_len:, :]
        hour = torch.round((future_mark[..., 0] + 0.5) * 23.0).long()
        weekday = torch.round((future_mark[..., 1] + 0.5) * 6.0).long()
        hour = hour.clamp(0, 23)
        weekday = weekday.clamp(0, 6)
        phase = weekday * 24 + hour
        if torch.any(phase < 0) or torch.any(phase >= self.period):
            raise ValueError('hour_of_week produced an out-of-range phase.')
        return phase

    def metadata(self):
        return {
            'phase_mode': self.mode,
            'period': self.period,
            'freq': self.freq,
            'embed': self.embed,
            'formula': 'weekday * 24 + hour',
        }


def make_phase_indexer(mode, period, freq, pred_len, embed='timeF'):
    if mode == HourOfWeekPhaseIndexer.mode:
        return HourOfWeekPhaseIndexer(period, freq, pred_len, embed=embed)
    raise ValueError(
        'Unsupported LPRC phase mode {!r}. Implemented modes: hour_of_week.'.format(
            mode
        )
    )


def phase_indexer_from_args(args):
    data_file = os.path.basename(str(args.data_path)).lower()
    supported = {'traffic.csv', 'electricity.csv'}
    if data_file not in supported:
        raise ValueError(
            'hour_of_week LPRC is currently approved only for Traffic '
            '(traffic.csv) and ECL (electricity.csv); received {!r}. No '
            'Solar, Weather, or PEMS phase rule is implemented.'.format(
                args.data_path
            )
        )
    return make_phase_indexer(
        args.lprc_phase_mode,
        args.lprc_period,
        args.freq,
        args.pred_len,
        embed=args.embed,
    )


class ResidualTableAccumulator:
    def __init__(self, channel_count, period, device):
        self.channel_count = int(channel_count)
        self.period = int(period)
        if self.channel_count <= 0 or self.period <= 0:
            raise ValueError('channel_count and period must be positive.')
        self.residual_sum = torch.zeros(
            self.period,
            self.channel_count,
            dtype=torch.float64,
            device=device,
        )
        self.phase_counts = torch.zeros(
            self.period, dtype=torch.int64, device=device
        )
        self.window_count = 0

    def update(self, predictions, targets, phase_indices):
        if predictions.shape != targets.shape:
            raise ValueError('Prediction and target shapes must match.')
        if predictions.ndim != 3:
            raise ValueError('Predictions must have shape [B,H,C].')
        if predictions.shape[-1] != self.channel_count:
            raise ValueError('Prediction channel count changed during fitting.')
        if tuple(phase_indices.shape) != tuple(predictions.shape[:2]):
            raise ValueError('Phase indices must have shape [B,H].')
        if torch.any(phase_indices < 0) or torch.any(phase_indices >= self.period):
            raise ValueError('Phase index is outside the configured period.')

        residual = (targets - predictions).detach().to(torch.float64)
        flat_residual = residual.reshape(-1, self.channel_count)
        flat_phase = phase_indices.detach().reshape(-1).to(
            device=self.residual_sum.device, dtype=torch.long
        )
        self.residual_sum.index_add_(0, flat_phase, flat_residual)
        ones = torch.ones_like(flat_phase, dtype=torch.int64)
        self.phase_counts.index_add_(0, flat_phase, ones)
        self.window_count += int(predictions.shape[0])

    def finalize(self):
        missing = torch.nonzero(self.phase_counts == 0, as_tuple=False).flatten()
        if missing.numel() > 0:
            raise RuntimeError(
                'TRAIN residuals do not cover all configured phases; missing={}. '
                'No phase imputation is implemented.'.format(
                    missing.detach().cpu().tolist()
                )
            )
        table = self.residual_sum / self.phase_counts.to(torch.float64).unsqueeze(1)
        table = table.transpose(0, 1).contiguous().cpu()
        residual_sum = self.residual_sum.transpose(0, 1).contiguous().cpu()
        phase_counts = self.phase_counts.cpu()
        cell_counts = phase_counts.unsqueeze(0).repeat(self.channel_count, 1)
        return table, residual_sum, phase_counts, cell_counts


def truncated_svd(residual_table, requested_rank):
    if residual_table.ndim != 2:
        raise ValueError('Residual table must have shape [channels, period].')
    requested_rank = int(requested_rank)
    if requested_rank <= 0:
        raise ValueError('requested_rank must be positive.')

    max_rank = min(int(residual_table.shape[0]), int(residual_table.shape[1]))
    effective_rank = min(requested_rank, max_rank)
    u, singular_values, vh = torch.linalg.svd(
        residual_table.to(dtype=torch.float64, device='cpu'),
        full_matrices=False,
    )
    energy = singular_values.square()
    total_energy = energy.sum()
    if total_energy.item() == 0.0:
        cumulative_energy = torch.zeros_like(energy)
        retained_energy = 0.0
    else:
        cumulative_energy = torch.cumsum(energy, dim=0) / total_energy
        retained_energy = float(cumulative_energy[effective_rank - 1].item())

    channel_factor = u[:, :effective_rank] * singular_values[:effective_rank]
    phase_factor = vh[:effective_rank, :]
    return {
        'requested_rank': requested_rank,
        'effective_rank': effective_rank,
        'channel_factor': channel_factor.contiguous(),
        'phase_factor': phase_factor.contiguous(),
        'singular_values': singular_values.contiguous(),
        'cumulative_spectral_energy': cumulative_energy.contiguous(),
        'retained_spectral_energy': retained_energy,
    }


def _channel_order_from_args(args, channel_count):
    source_path = os.path.join(str(args.root_path), str(args.data_path))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(
            'Cannot validate LPRC channel order; dataset file not found: '
            '{}'.format(source_path)
        )
    with open(source_path, 'r', encoding='utf-8-sig', newline='') as handle:
        header = next(csv.reader(handle), None)
    if not header or 'date' not in header or str(args.target) not in header:
        raise RuntimeError(
            'Cannot validate LPRC channel order from dataset header: {}'.format(
                source_path
            )
        )

    target = str(args.target)
    features = str(args.features)
    non_target = [name for name in header if name not in ('date', target)]
    if features == 'M':
        ordered = non_target + [target]
    elif features in ('S', 'MS'):
        ordered = [target]
    else:
        raise ValueError('Unsupported feature mode for LPRC: {!r}.'.format(features))
    if len(ordered) != int(channel_count):
        raise RuntimeError(
            'LPRC channel count/order mismatch: dataset header gives {} but '
            'the forecast path gives {}.'.format(len(ordered), channel_count)
        )
    return ordered


def model_config_from_args(args, channel_count):
    return {
        'model_id': str(args.model_id),
        'model': str(args.model),
        'data': str(args.data),
        'root_path': str(args.root_path),
        'data_path': str(args.data_path),
        'data_source': os.path.abspath(
            os.path.join(str(args.root_path), str(args.data_path))
        ),
        'features': str(args.features),
        'target': str(args.target),
        'freq': str(args.freq),
        'embed': str(args.embed),
        'seq_len': int(args.seq_len),
        'label_len': int(args.label_len),
        'pred_len': int(args.pred_len),
        'enc_in': int(args.enc_in),
        'c_out': int(args.c_out),
        'd_model': int(args.d_model),
        'n_heads': int(args.n_heads),
        'e_layers': int(args.e_layers),
        'd_layers': int(args.d_layers),
        'd_ff': int(args.d_ff),
        'factor': int(args.factor),
        'class_strategy': str(args.class_strategy),
        'dropout': float(args.dropout),
        'activation': str(args.activation),
        'use_norm': int(args.use_norm),
        'channel_count': int(channel_count),
        'channel_order': _channel_order_from_args(args, channel_count),
        'feature_slice': 'last target channel' if args.features == 'MS' else 'all channels',
        'scale_semantics': LPRC_SCALE_SEMANTICS,
        'vardrop_k': None if getattr(args, 'k', None) is None else int(args.k),
        'vardrop_group_size': (
            None
            if getattr(args, 'group_size', None) is None
            else int(args.group_size)
        ),
        'exact_fast_vardrop': bool(
            getattr(args, 'exact_fast_vardrop', False)
        ),
    }


def build_lprc_artifact(
    residual_table,
    residual_sum,
    phase_counts,
    cell_counts,
    requested_rank,
    phase_metadata,
    model_config,
    source_checkpoint,
    fit_timing,
    window_count,
    provenance,
):
    svd_start = time.perf_counter()
    decomposition = truncated_svd(residual_table, requested_rank)
    fit_timing = dict(fit_timing)
    fit_timing['svd_sec'] = float(time.perf_counter() - svd_start)
    fit_timing['fit_compute_sec'] = float(
        fit_timing.get('residual_collection_sec', 0.0)
        + fit_timing.get('aggregation_finalize_sec', 0.0)
        + fit_timing['svd_sec']
    )
    fit_timing['fit_total_sec'] = float(
        fit_timing.get('checkpoint_load_sec', 0.0)
        + fit_timing['fit_compute_sec']
    )

    metadata = {
        'method': LPRC_METHOD,
        'artifact_version': LPRC_ARTIFACT_VERSION,
        'fit_split': LPRC_FIT_SPLIT,
        'phase_mode': phase_metadata['phase_mode'],
        'period': int(phase_metadata['period']),
        'phase': dict(phase_metadata),
        'requested_rank': int(decomposition['requested_rank']),
        'effective_rank': int(decomposition['effective_rank']),
        'alpha': 1.0,
        'rank_policy': 'explicit requested rank; structural dimension clamp only',
        'residual_sign': 'target - backbone_forecast',
        'overlapping_window_weighting': LPRC_OVERLAP_WEIGHTING,
        'model_config': dict(model_config),
        'source_checkpoint': dict(source_checkpoint),
        'runtime_provenance': dict(provenance),
        'fit_timing': fit_timing,
        'train_window_count': int(window_count),
        'retained_spectral_energy': float(
            decomposition['retained_spectral_energy']
        ),
    }
    return {
        'method': LPRC_METHOD,
        'artifact_version': LPRC_ARTIFACT_VERSION,
        'metadata': metadata,
        'channel_ids': torch.arange(
            residual_table.shape[0], dtype=torch.int64
        ),
        'channel_factor': decomposition['channel_factor'].cpu(),
        'phase_factor': decomposition['phase_factor'].cpu(),
        'residual_table': residual_table.cpu(),
        'residual_sum': residual_sum.cpu(),
        'phase_counts': phase_counts.cpu(),
        'cell_counts': cell_counts.cpu(),
        'singular_values': decomposition['singular_values'].cpu(),
        'cumulative_spectral_energy': decomposition[
            'cumulative_spectral_energy'
        ].cpu(),
    }


def _json_artifact_summary(artifact):
    return {
        'metadata': artifact['metadata'],
        'singular_values': artifact['singular_values'].tolist(),
        'cumulative_spectral_energy': artifact[
            'cumulative_spectral_energy'
        ].tolist(),
        'phase_counts': artifact['phase_counts'].tolist(),
        'cell_count_shape': list(artifact['cell_counts'].shape),
        'cell_count_min': int(artifact['cell_counts'].min().item()),
        'cell_count_max': int(artifact['cell_counts'].max().item()),
        'tensor_keys': [
            'channel_ids',
            'channel_factor',
            'phase_factor',
            'residual_table',
            'residual_sum',
            'phase_counts',
            'cell_counts',
            'singular_values',
            'cumulative_spectral_energy',
        ],
    }


def save_lprc_artifact(path, artifact):
    json_path = os.path.splitext(path)[0] + '.json'
    if os.path.exists(path):
        raise FileExistsError(
            'Refusing to overwrite existing LPRC artifact: {}'.format(path)
        )
    if os.path.exists(json_path):
        raise FileExistsError(
            'Refusing to overwrite existing LPRC metadata: {}'.format(json_path)
        )
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    torch.save(artifact, path)
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(_json_artifact_summary(artifact), handle, indent=2, sort_keys=True)
    return json_path


def _expected_channel_count(args):
    if args.features == 'MS':
        return 1
    if args.model in ('OURS', 'iTransformer'):
        return int(args.enc_in)
    raise ValueError(
        'LPRC is currently integrated only for OURS and iTransformer; '
        'received model={!r}.'.format(args.model)
    )


def validate_lprc_artifact(artifact, args, checkpoint_path=None):
    if not isinstance(artifact, dict):
        raise RuntimeError('LPRC artifact must be a dictionary.')
    if artifact.get('method') != LPRC_METHOD:
        raise RuntimeError('Artifact method is not LPRC.')
    if artifact.get('artifact_version') != LPRC_ARTIFACT_VERSION:
        raise RuntimeError('Unsupported LPRC artifact version.')
    metadata = artifact.get('metadata')
    if not isinstance(metadata, dict):
        raise RuntimeError('LPRC artifact metadata is missing.')
    if metadata.get('fit_split') != LPRC_FIT_SPLIT:
        raise RuntimeError('LPRC artifact was not fitted on TRAIN.')
    if float(metadata.get('alpha', -1.0)) != 1.0:
        raise RuntimeError('Stored LPRC main-method alpha must be 1.0.')
    if metadata.get('residual_sign') != 'target - backbone_forecast':
        raise RuntimeError('LPRC residual sign metadata is invalid.')

    expected = model_config_from_args(args, _expected_channel_count(args))
    actual = metadata.get('model_config', {})
    keys = (
        'model', 'data', 'data_path', 'features', 'target',
        'freq', 'embed',
        'seq_len', 'label_len', 'pred_len', 'enc_in', 'c_out',
        'd_model', 'n_heads', 'e_layers', 'd_layers', 'd_ff', 'factor',
        'class_strategy', 'dropout', 'activation', 'use_norm',
        'channel_count', 'channel_order', 'feature_slice', 'scale_semantics',
        'vardrop_k', 'vardrop_group_size', 'exact_fast_vardrop',
    )
    mismatches = {
        key: {'expected': expected.get(key), 'actual': actual.get(key)}
        for key in keys
        if expected.get(key) != actual.get(key)
    }
    if mismatches:
        raise RuntimeError(
            'LPRC artifact/configuration mismatch: {}'.format(mismatches)
        )

    if metadata.get('phase_mode') != getattr(args, 'lprc_phase_mode'):
        raise RuntimeError('LPRC phase mode does not match the artifact.')
    if int(metadata.get('period', -1)) != int(getattr(args, 'lprc_period')):
        raise RuntimeError('LPRC period does not match the artifact.')
    if int(metadata.get('requested_rank', -1)) != int(
        getattr(args, 'lprc_rank')
    ):
        raise RuntimeError('LPRC requested rank does not match the artifact.')

    channel_count = int(actual['channel_count'])
    expected_effective_rank = min(
        int(metadata['requested_rank']), channel_count, int(metadata['period'])
    )
    if int(metadata.get('effective_rank', -1)) != expected_effective_rank:
        raise RuntimeError('LPRC effective rank is invalid for the artifact shape.')
    if tuple(artifact['channel_factor'].shape) != (
        channel_count,
        int(metadata['effective_rank']),
    ):
        raise RuntimeError('LPRC channel factor shape is invalid.')
    if tuple(artifact['phase_factor'].shape) != (
        int(metadata['effective_rank']),
        int(metadata['period']),
    ):
        raise RuntimeError('LPRC phase factor shape is invalid.')
    expected_table_shape = (channel_count, int(metadata['period']))
    for key in ('residual_table', 'residual_sum', 'cell_counts'):
        if tuple(artifact[key].shape) != expected_table_shape:
            raise RuntimeError('LPRC {} shape is invalid.'.format(key))
    if tuple(artifact['phase_counts'].shape) != (int(metadata['period']),):
        raise RuntimeError('LPRC phase count shape is invalid.')
    expected_ids = torch.arange(channel_count, dtype=torch.int64)
    if not torch.equal(artifact['channel_ids'].cpu(), expected_ids):
        raise RuntimeError('LPRC channel order is not the expected dataset order.')
    tensor_keys = (
        'channel_ids', 'channel_factor', 'phase_factor', 'residual_table',
        'residual_sum', 'phase_counts', 'cell_counts', 'singular_values',
        'cumulative_spectral_energy',
    )
    if any(artifact[key].device.type != 'cpu' for key in tensor_keys):
        raise RuntimeError('LPRC artifacts must store CPU tensors only.')

    if checkpoint_path is not None:
        current = checkpoint_identity(checkpoint_path)
        source = metadata.get('source_checkpoint', {})
        if current['sha256'] != source.get('sha256'):
            raise RuntimeError(
                'LPRC artifact source checkpoint hash does not match the '
                'selected backbone checkpoint.'
            )
    return artifact


def load_lprc_artifact(path, args, checkpoint_path=None):
    if not os.path.isfile(path):
        raise FileNotFoundError('LPRC artifact not found: {}'.format(path))
    artifact = torch.load(path, map_location='cpu')
    return validate_lprc_artifact(artifact, args, checkpoint_path)


def apply_lprc(outputs, phase_indices, artifact, alpha=1.0):
    if outputs.ndim != 3:
        raise ValueError('LPRC outputs must have shape [B,H,C].')
    if tuple(phase_indices.shape) != tuple(outputs.shape[:2]):
        raise ValueError('LPRC phase indices must have shape [B,H].')
    period = int(artifact['metadata']['period'])
    if torch.any(phase_indices < 0) or torch.any(phase_indices >= period):
        raise ValueError('LPRC phase index is outside the artifact period.')
    if outputs.shape[-1] != artifact['channel_factor'].shape[0]:
        raise ValueError('LPRC output channel count does not match the artifact.')

    channel_factor = artifact['channel_factor'].to(
        device=outputs.device, dtype=outputs.dtype
    )
    phase_factor = artifact['phase_factor'].to(
        device=outputs.device, dtype=outputs.dtype
    )
    phase = phase_indices.to(device=outputs.device, dtype=torch.long)
    selected_phase = phase_factor[:, phase]
    correction = torch.einsum('cr,rbh->bhc', channel_factor, selected_phase)
    return outputs + float(alpha) * correction


class PairedErrorAccumulator:
    def __init__(self):
        self.base_se = 0.0
        self.base_ae = 0.0
        self.corrected_se = 0.0
        self.corrected_ae = 0.0
        self.element_count = 0
        self.per_window_base_mse = []
        self.per_window_base_mae = []
        self.per_window_corrected_mse = []
        self.per_window_corrected_mae = []
        self.channel_base_se = None
        self.channel_base_ae = None
        self.channel_corrected_se = None
        self.channel_corrected_ae = None
        self.channel_count = None
        self.horizon_base_se = None
        self.horizon_base_ae = None
        self.horizon_corrected_se = None
        self.horizon_corrected_ae = None
        self.horizon_count = None

    def update(self, base, corrected, true):
        base = np.asarray(base, dtype=np.float64)
        corrected = np.asarray(corrected, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        if base.shape != corrected.shape or base.shape != true.shape:
            raise ValueError('Paired LPRC evaluation arrays must have equal shapes.')
        if base.ndim != 3:
            raise ValueError('Paired evaluation arrays must have shape [B,H,C].')

        base_error = base - true
        corrected_error = corrected - true
        base_sq = np.square(base_error)
        corrected_sq = np.square(corrected_error)
        base_abs = np.abs(base_error)
        corrected_abs = np.abs(corrected_error)

        self.base_se += float(base_sq.sum())
        self.base_ae += float(base_abs.sum())
        self.corrected_se += float(corrected_sq.sum())
        self.corrected_ae += float(corrected_abs.sum())
        self.element_count += int(base.size)

        self.per_window_base_mse.extend(base_sq.mean(axis=(1, 2)).tolist())
        self.per_window_base_mae.extend(base_abs.mean(axis=(1, 2)).tolist())
        self.per_window_corrected_mse.extend(
            corrected_sq.mean(axis=(1, 2)).tolist()
        )
        self.per_window_corrected_mae.extend(
            corrected_abs.mean(axis=(1, 2)).tolist()
        )

        channel_count = int(base.shape[0] * base.shape[1])
        horizon_count = int(base.shape[0] * base.shape[2])
        if self.channel_base_se is None:
            channels = base.shape[2]
            horizon = base.shape[1]
            self.channel_base_se = np.zeros(channels, dtype=np.float64)
            self.channel_base_ae = np.zeros(channels, dtype=np.float64)
            self.channel_corrected_se = np.zeros(channels, dtype=np.float64)
            self.channel_corrected_ae = np.zeros(channels, dtype=np.float64)
            self.channel_count = np.zeros(channels, dtype=np.int64)
            self.horizon_base_se = np.zeros(horizon, dtype=np.float64)
            self.horizon_base_ae = np.zeros(horizon, dtype=np.float64)
            self.horizon_corrected_se = np.zeros(horizon, dtype=np.float64)
            self.horizon_corrected_ae = np.zeros(horizon, dtype=np.float64)
            self.horizon_count = np.zeros(horizon, dtype=np.int64)

        self.channel_base_se += base_sq.sum(axis=(0, 1))
        self.channel_base_ae += base_abs.sum(axis=(0, 1))
        self.channel_corrected_se += corrected_sq.sum(axis=(0, 1))
        self.channel_corrected_ae += corrected_abs.sum(axis=(0, 1))
        self.channel_count += channel_count
        self.horizon_base_se += base_sq.sum(axis=(0, 2))
        self.horizon_base_ae += base_abs.sum(axis=(0, 2))
        self.horizon_corrected_se += corrected_sq.sum(axis=(0, 2))
        self.horizon_corrected_ae += corrected_abs.sum(axis=(0, 2))
        self.horizon_count += horizon_count

    @staticmethod
    def _gain(base, corrected):
        absolute = base - corrected
        percent = np.divide(
            100.0 * absolute,
            base,
            out=np.zeros_like(absolute),
            where=np.abs(base) > 1e-30,
        )
        return absolute, percent

    def finalize(self):
        if self.element_count == 0:
            raise RuntimeError('No paired LPRC evaluation data were accumulated.')
        channel_base_mse = self.channel_base_se / self.channel_count
        channel_base_mae = self.channel_base_ae / self.channel_count
        channel_corrected_mse = self.channel_corrected_se / self.channel_count
        channel_corrected_mae = self.channel_corrected_ae / self.channel_count
        horizon_base_mse = self.horizon_base_se / self.horizon_count
        horizon_base_mae = self.horizon_base_ae / self.horizon_count
        horizon_corrected_mse = self.horizon_corrected_se / self.horizon_count
        horizon_corrected_mae = self.horizon_corrected_ae / self.horizon_count

        channel_mse_gain, channel_mse_gain_pct = self._gain(
            channel_base_mse, channel_corrected_mse
        )
        channel_mae_gain, channel_mae_gain_pct = self._gain(
            channel_base_mae, channel_corrected_mae
        )
        horizon_mse_gain, horizon_mse_gain_pct = self._gain(
            horizon_base_mse, horizon_corrected_mse
        )
        horizon_mae_gain, horizon_mae_gain_pct = self._gain(
            horizon_base_mae, horizon_corrected_mae
        )

        arrays = {
            'per_window_base_mse': np.asarray(self.per_window_base_mse),
            'per_window_base_mae': np.asarray(self.per_window_base_mae),
            'per_window_corrected_mse': np.asarray(
                self.per_window_corrected_mse
            ),
            'per_window_corrected_mae': np.asarray(
                self.per_window_corrected_mae
            ),
            'per_channel_base_mse': channel_base_mse,
            'per_channel_base_mae': channel_base_mae,
            'per_channel_corrected_mse': channel_corrected_mse,
            'per_channel_corrected_mae': channel_corrected_mae,
            'per_channel_mse_gain': channel_mse_gain,
            'per_channel_mse_gain_pct': channel_mse_gain_pct,
            'per_channel_mae_gain': channel_mae_gain,
            'per_channel_mae_gain_pct': channel_mae_gain_pct,
            'per_horizon_base_mse': horizon_base_mse,
            'per_horizon_base_mae': horizon_base_mae,
            'per_horizon_corrected_mse': horizon_corrected_mse,
            'per_horizon_corrected_mae': horizon_corrected_mae,
            'per_horizon_mse_gain': horizon_mse_gain,
            'per_horizon_mse_gain_pct': horizon_mse_gain_pct,
            'per_horizon_mae_gain': horizon_mae_gain,
            'per_horizon_mae_gain_pct': horizon_mae_gain_pct,
        }
        summary = {
            'element_count': int(self.element_count),
            'window_count': len(self.per_window_base_mse),
            'base_mse': self.base_se / self.element_count,
            'base_mae': self.base_ae / self.element_count,
            'corrected_mse': self.corrected_se / self.element_count,
            'corrected_mae': self.corrected_ae / self.element_count,
        }
        summary['mse_gain'] = summary['base_mse'] - summary['corrected_mse']
        summary['mae_gain'] = summary['base_mae'] - summary['corrected_mae']
        summary['mse_gain_pct'] = (
            100.0 * summary['mse_gain'] / max(summary['base_mse'], 1e-30)
        )
        summary['mae_gain_pct'] = (
            100.0 * summary['mae_gain'] / max(summary['base_mae'], 1e-30)
        )
        return summary, arrays


def write_lprc_evaluation(
    output_dir,
    artifact,
    paired_accumulator,
    alpha,
    artifact_path,
    apply_time_sec,
    saved_base_predictions,
    evaluation_provenance,
):
    summary, arrays = paired_accumulator.finalize()
    arrays.update({
        'fit_phase_counts': artifact['phase_counts'].numpy(),
        'fit_cell_counts': artifact['cell_counts'].numpy(),
        'singular_values': artifact['singular_values'].numpy(),
        'cumulative_spectral_energy': artifact[
            'cumulative_spectral_energy'
        ].numpy(),
    })
    npz_path = os.path.join(output_dir, 'lprc_paired_statistics.npz')
    json_path = os.path.join(output_dir, 'lprc_summary.json')
    for evidence_path in (npz_path, json_path):
        if os.path.exists(evidence_path):
            raise FileExistsError(
                'Refusing to overwrite existing LPRC evaluation evidence: '
                '{}'.format(evidence_path)
            )
    np.savez_compressed(npz_path, **arrays)

    record = {
        'method': LPRC_METHOD,
        'artifact_version': LPRC_ARTIFACT_VERSION,
        'artifact_path': os.path.abspath(artifact_path),
        'artifact_metadata': artifact['metadata'],
        'evaluation_runtime_provenance': dict(evaluation_provenance),
        'runtime_alpha': float(alpha),
        'alpha_role': 'main_method' if float(alpha) == 1.0 else 'ablation',
        'evaluation': summary,
        'correction_dispatch_time_sec': float(apply_time_sec),
        'paired_statistics_file': os.path.basename(npz_path),
        'full_array_storage': {
            'corrected_predictions': 'pred.npy',
            'truth': 'true.npy',
            'base_predictions': (
                'base_pred.npy' if saved_base_predictions else None
            ),
        },
    }
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return record


def guard_lprc_evaluation_output_dir(output_dir):
    evidence_names = (
        'lprc_summary.json',
        'lprc_paired_statistics.npz',
        'base_pred.npy',
        'metrics.npy',
        'pred.npy',
        'true.npy',
        'real_prediction.npy',
    )
    existing = [
        os.path.join(output_dir, name)
        for name in evidence_names
        if os.path.exists(os.path.join(output_dir, name))
    ]
    if existing:
        raise FileExistsError(
            'Refusing to overwrite existing LPRC evaluation evidence: {}'.format(
                existing
            )
        )


def default_lprc_artifact_path(args, setting):
    filename = 'lprc_{}_p{}_r{}.pt'.format(
        args.lprc_phase_mode, args.lprc_period, args.lprc_rank
    )
    return os.path.join(args.checkpoints, setting, filename)


def lprc_result_setting(setting, args):
    alpha_text = '{:.8g}'.format(float(args.lprc_alpha)).replace('.', 'p')
    return '{}_lprc_{}_p{}_r{}_a{}'.format(
        setting,
        args.lprc_phase_mode,
        args.lprc_period,
        args.lprc_rank,
        alpha_text,
    )


def _synchronize(device):
    if getattr(device, 'type', None) == 'cuda':
        torch.cuda.synchronize(device)


def fit_lprc_artifact(
    model,
    train_data,
    args,
    device,
    batch_forward,
    checkpoint_path,
    artifact_path,
):
    allow_legacy = (
        args.model == 'OURS'
        and not bool(getattr(args, 'use_lpra', False))
    )
    checkpoint_load_start = time.perf_counter()
    source = load_backbone_checkpoint(
        model,
        checkpoint_path,
        allow_missing_lpra_alpha=allow_legacy,
        allow_untrained_lpra_module=(
            allow_legacy
            and os.path.basename(checkpoint_path) == 'checkpoint.pth'
        ),
    )
    model.to(device)
    checkpoint_load_sec = time.perf_counter() - checkpoint_load_start
    indexer = phase_indexer_from_args(args)
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    was_training = model.training
    model.eval()
    accumulator = None
    batch_count = 0

    _synchronize(device)
    collection_start = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch in loader:
                predictions, targets, batch_y_mark = batch_forward(batch)
                phases = indexer(batch_y_mark)
                if accumulator is None:
                    accumulator = ResidualTableAccumulator(
                        predictions.shape[-1],
                        args.lprc_period,
                        predictions.device,
                    )
                accumulator.update(predictions, targets, phases)
                batch_count += 1
        _synchronize(device)
        collection_sec = time.perf_counter() - collection_start
    finally:
        if was_training:
            model.train()
    if accumulator is None:
        raise RuntimeError('TRAIN loader produced no batches for LPRC fitting.')

    finalize_start = time.perf_counter()
    residual_table, residual_sum, phase_counts, cell_counts = accumulator.finalize()
    finalize_sec = time.perf_counter() - finalize_start
    artifact = build_lprc_artifact(
        residual_table=residual_table,
        residual_sum=residual_sum,
        phase_counts=phase_counts,
        cell_counts=cell_counts,
        requested_rank=args.lprc_rank,
        phase_metadata=indexer.metadata(),
        model_config=model_config_from_args(
            args, residual_table.shape[0]
        ),
        source_checkpoint=source,
        fit_timing={
            'checkpoint_load_sec': float(checkpoint_load_sec),
            'residual_collection_sec': float(collection_sec),
            'aggregation_finalize_sec': float(finalize_sec),
            'train_batch_count': int(batch_count),
            'data_loader_num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        },
        window_count=accumulator.window_count,
        provenance=runtime_provenance(args, device),
    )
    save_lprc_artifact(artifact_path, artifact)
    return artifact


class LPRCExperimentMixin:
    def _initialize_lprc(self):
        self.lprc_artifact = None
        self.lprc_artifact_path = None

    def _lprc_enabled(self):
        return bool(getattr(self.args, 'use_lprc', False))

    def _lprc_checkpoint_path(self, setting):
        explicit = getattr(self.args, 'lprc_checkpoint', None)
        if explicit:
            return explicit
        return os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')

    def _resolve_lprc_artifact_path(self, setting):
        explicit = getattr(self.args, 'lprc_artifact', None)
        return explicit or default_lprc_artifact_path(self.args, setting)

    def _load_lprc_backbone(self, setting):
        checkpoint_path = self._lprc_checkpoint_path(setting)
        allow_legacy = (
            self.args.model == 'OURS'
            and not bool(getattr(self.args, 'use_lpra', False))
        )
        identity = load_backbone_checkpoint(
            self.model,
            checkpoint_path,
            allow_missing_lpra_alpha=allow_legacy,
            allow_untrained_lpra_module=(
                allow_legacy
                and os.path.basename(checkpoint_path) == 'checkpoint.pth'
            ),
        )
        self.model.to(self.device)
        return identity

    def load_lprc(self, setting):
        if not self._lprc_enabled():
            raise RuntimeError('Cannot load LPRC when --use_lprc is disabled.')
        checkpoint_path = self._lprc_checkpoint_path(setting)
        artifact_path = self._resolve_lprc_artifact_path(setting)
        self.lprc_artifact = load_lprc_artifact(
            artifact_path, self.args, checkpoint_path=checkpoint_path
        )
        self.lprc_artifact_path = artifact_path
        print('[LPRC] loaded {}'.format(artifact_path))
        return self.lprc_artifact

    def fit_lprc(self, setting):
        if not self._lprc_enabled():
            raise RuntimeError('Cannot fit LPRC when --use_lprc is disabled.')
        if bool(getattr(self.args, 'use_lpra', False)):
            raise RuntimeError('LPRA and LPRC cannot be combined in this stage.')
        checkpoint_path = self._lprc_checkpoint_path(setting)
        artifact_path = self._resolve_lprc_artifact_path(setting)
        artifact_json_path = os.path.splitext(artifact_path)[0] + '.json'
        for existing_path in (artifact_path, artifact_json_path):
            if os.path.exists(existing_path):
                raise FileExistsError(
                    'Refusing to overwrite existing LPRC artifact evidence: '
                    '{}'.format(existing_path)
                )
        train_data, _ = self._get_data(flag='train')
        print('[LPRC] fitting from TRAIN only: {}'.format(checkpoint_path))
        self.lprc_artifact = fit_lprc_artifact(
            model=self.model,
            train_data=train_data,
            args=self.args,
            device=self.device,
            batch_forward=self._lprc_forward_batch,
            checkpoint_path=checkpoint_path,
            artifact_path=artifact_path,
        )
        self.lprc_artifact_path = artifact_path
        metadata = self.lprc_artifact['metadata']
        print(
            '[LPRC] saved {} requested_rank={} effective_rank={} '
            'period={} alpha=1.0'.format(
                artifact_path,
                metadata['requested_rank'],
                metadata['effective_rank'],
                metadata['period'],
            )
        )
        return self.lprc_artifact

    def _guard_lprc_evaluation_outputs(self, output_dir):
        if self._lprc_enabled():
            guard_lprc_evaluation_output_dir(output_dir)

    def _lprc_phase_indices(self, batch_y_mark):
        indexer = phase_indexer_from_args(self.args)
        return indexer(batch_y_mark)

    def _apply_lprc(self, outputs, batch_y_mark):
        if not self._lprc_enabled():
            return outputs
        if self.lprc_artifact is None:
            raise RuntimeError('LPRC artifact has not been loaded or fitted.')
        phases = self._lprc_phase_indices(batch_y_mark)
        return apply_lprc(
            outputs,
            phases,
            self.lprc_artifact,
            alpha=self.args.lprc_alpha,
        )

    def _write_lprc_evaluation(
        self,
        output_dir,
        paired_accumulator,
        apply_time_sec,
        saved_base_predictions,
    ):
        record = write_lprc_evaluation(
            output_dir=output_dir,
            artifact=self.lprc_artifact,
            paired_accumulator=paired_accumulator,
            alpha=self.args.lprc_alpha,
            artifact_path=self.lprc_artifact_path,
            apply_time_sec=apply_time_sec,
            saved_base_predictions=saved_base_predictions,
            evaluation_provenance=runtime_provenance(
                self.args, self.device
            ),
        )
        metrics = record['evaluation']
        print(
            '[LPRC] base MSE={:.9f} MAE={:.9f}'.format(
                metrics['base_mse'], metrics['base_mae']
            )
        )
        print(
            '[LPRC] corrected MSE={:.9f} MAE={:.9f}'.format(
                metrics['corrected_mse'], metrics['corrected_mae']
            )
        )
        print(
            '[LPRC] gain MSE={:.3f}% MAE={:.3f}%'.format(
                metrics['mse_gain_pct'], metrics['mae_gain_pct']
            )
        )
        return record

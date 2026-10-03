import os
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.utils.data import Dataset

from data_provider.data_loader import Dataset_Custom, Dataset_PEMS
from run import experiment_setting
from VarDrop import (
    efficient_sampler,
    efficient_sampler_fast_exact,
    k_dominant_frequency_hashing,
    k_dominant_frequency_hashing_fast_exact,
)
from utils.lprc import (
    HourOfWeekPhaseIndexer,
    PairedErrorAccumulator,
    ResidualTableAccumulator,
    apply_lprc,
    build_lprc_artifact,
    CompactResidualAccumulator,
    LPRCExperimentMixin,
    StreamingVariantMetrics,
    fit_final_lprc,
    fit_lprc_from_cache,
    fit_lprc_artifact,
    forecast_target_global_offsets,
    guard_lprc_evaluation_output_dir,
    load_backbone_checkpoint,
    load_lprc_artifact,
    load_final_lprc_artifact,
    lprc_result_setting,
    lprc_setting_suffix,
    make_phase_indexer,
    model_config_from_args,
    phase_indexer_from_args,
    runtime_provenance,
    save_lprc_artifact,
    sha256_file,
    truncated_svd,
    validate_lprc_cache_metadata,
)


def make_args(root_path, data_path='traffic.csv'):
    with open(os.path.join(root_path, data_path), 'w', encoding='utf-8') as handle:
        handle.write('date,load,OT\n')
    return SimpleNamespace(
        model_id='synthetic_lprc',
        model='iTransformer',
        data='custom',
        root_path=root_path,
        data_path=data_path,
        features='M',
        target='OT',
        freq='h',
        embed='timeF',
        seq_len=96,
        label_len=48,
        pred_len=168,
        enc_in=2,
        c_out=2,
        d_model=16,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        factor=1,
        class_strategy='projection',
        dropout=0.1,
        activation='gelu',
        use_norm=1,
        k=None,
        group_size=None,
        exact_fast_vardrop=False,
        batch_size=4,
        lprc_phase_mode='hour_of_week',
        lprc_period=168,
        lprc_rank=1,
        lprc_alpha=1.0,
        lprc_checkpoint=None,
    )


def week_marks():
    hour = torch.arange(168) % 24
    weekday = torch.arange(168) // 24
    marks = torch.zeros(168, 4, dtype=torch.float32)
    marks[:, 0] = hour.float() / 23.0 - 0.5
    marks[:, 1] = weekday.float() / 6.0 - 0.5
    return marks


class SyntheticTrainDataset(Dataset):
    def __init__(self):
        self.marks = week_marks()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        batch_x = torch.zeros(96, 2)
        batch_y = torch.zeros(168, 2)
        batch_y[:, 0] = torch.arange(168, dtype=torch.float32)
        batch_y[:, 1] = 2.0 * torch.arange(168, dtype=torch.float32)
        return batch_x, batch_y, torch.zeros(96, 4), self.marks


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, values):
        return values * self.weight


def periodic_cache(period=12, channels=4, train_length=240, val_length=96):
    phase_pattern = np.stack([
        np.sin(2.0 * np.pi * np.arange(period) / period + channel)
        for channel in range(channels)
    ], axis=1)
    train = phase_pattern[np.arange(train_length) % period]
    val = phase_pattern[(train_length + np.arange(val_length)) % period]
    return {
        'metadata': {
            'train_global_offset': 0,
            'val_global_offset': train_length,
            'test_global_offset': train_length + val_length,
        },
        'train_residual_sum': train,
        'train_count': np.ones(train_length, dtype=np.int64),
        'val_residual_sum': val,
        'val_residual_sumsq': val ** 2,
        'val_count': np.ones(val_length, dtype=np.int64),
    }


class TestLPRC(unittest.TestCase):
    def test_original_and_exact_fast_vardrop_remain_identical(self):
        generator = torch.Generator().manual_seed(2023)
        inputs = torch.randn(4, 96, 17, generator=generator)
        original_hash = k_dominant_frequency_hashing(
            inputs, k=4, freq_list=range(1, 25)
        )
        fast_hash = k_dominant_frequency_hashing_fast_exact(
            inputs, k=4, freq_list=range(1, 25)
        )
        np.testing.assert_array_equal(original_hash, fast_hash)

        np.random.seed(2023)
        original_indices, original_groups = efficient_sampler(
            inputs,
            k=4,
            group_size=10,
            freq_list=range(1, 25),
            return_group=True,
        )
        np.random.seed(2023)
        fast_indices, fast_groups = efficient_sampler_fast_exact(
            inputs,
            k=4,
            group_size=10,
            freq_list=range(1, 25),
            return_group=True,
        )
        self.assertEqual(original_indices, fast_indices)
        self.assertEqual(set(original_groups), set(fast_groups))
        for key in original_groups:
            np.testing.assert_array_equal(
                original_groups[key], fast_groups[key]
            )

    def test_legacy_checkpoint_allowlist_is_narrow(self):
        class BufferedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)
                self.register_buffer('lpra_alpha', torch.tensor(0.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            model = BufferedModel()
            checkpoint_path = os.path.join(temp_dir, 'legacy.pth')
            legacy = {
                key: value
                for key, value in model.state_dict().items()
                if key != 'lpra_alpha'
            }
            torch.save(legacy, checkpoint_path)
            load_backbone_checkpoint(
                model, checkpoint_path, allow_missing_lpra_alpha=True
            )
            with self.assertRaises(RuntimeError):
                load_backbone_checkpoint(
                    model, checkpoint_path, allow_missing_lpra_alpha=False
                )

            wrapped_path = os.path.join(temp_dir, 'checkpoint.pth')
            wrapped = dict(model.state_dict())
            wrapped['lpra.sensor_factor.weight'] = torch.zeros(2, 1)
            wrapped['lpra.phase_factor.weight'] = torch.zeros(168, 1)
            torch.save(wrapped, wrapped_path)
            load_backbone_checkpoint(
                model,
                wrapped_path,
                allow_untrained_lpra_module=True,
            )

            broken_path = os.path.join(temp_dir, 'broken.pth')
            torch.save({}, broken_path)
            with self.assertRaises(RuntimeError):
                load_backbone_checkpoint(
                    model, broken_path, allow_missing_lpra_alpha=True
                )

    def test_hour_of_week_matches_existing_semantics(self):
        marks = week_marks().unsqueeze(0)
        indexer = HourOfWeekPhaseIndexer(168, 'h', 168)
        phases = indexer(marks)
        torch.testing.assert_close(phases, torch.arange(168).unsqueeze(0))

        with self.assertRaises(ValueError):
            HourOfWeekPhaseIndexer(24, 'h', 168)
        with self.assertRaises(ValueError):
            HourOfWeekPhaseIndexer(168, '15min', 168)
        with self.assertRaises(ValueError):
            HourOfWeekPhaseIndexer(168, 'h', 168, embed='fixed')
        with self.assertRaises(ValueError):
            make_phase_indexer('guessed_solar_phase', 168, 'h', 168)
        with tempfile.TemporaryDirectory() as temp_dir:
            unsupported_args = make_args(
                temp_dir, data_path='weather.csv'
            )
            with self.assertRaises(ValueError):
                phase_indexer_from_args(unsupported_args)

    def test_residual_table_uses_target_minus_prediction(self):
        accumulator = ResidualTableAccumulator(2, 2, torch.device('cpu'))
        predictions = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        targets = torch.tensor([[[2.0, 4.0], [5.0, 8.0]]])
        phases = torch.tensor([[0, 1]])
        accumulator.update(predictions, targets, phases)
        table, residual_sum, phase_counts, cell_counts = accumulator.finalize()
        expected = torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float64)
        torch.testing.assert_close(table, expected)
        torch.testing.assert_close(residual_sum, expected)
        torch.testing.assert_close(phase_counts, torch.ones(2, dtype=torch.int64))
        torch.testing.assert_close(cell_counts, torch.ones(2, 2, dtype=torch.int64))

    def test_rank_is_only_structurally_clamped(self):
        left = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        right = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
        table = left @ right
        result = truncated_svd(table, requested_rank=32)
        self.assertEqual(result['requested_rank'], 32)
        self.assertEqual(result['effective_rank'], 2)
        reconstructed = result['channel_factor'] @ result['phase_factor']
        torch.testing.assert_close(reconstructed, table, atol=1e-10, rtol=1e-10)

    def test_apply_rejects_out_of_range_phase_without_modulo(self):
        artifact = {
            'metadata': {'period': 2},
            'channel_factor': torch.tensor([[1.0], [2.0]]),
            'phase_factor': torch.tensor([[10.0, 20.0]]),
        }
        outputs = torch.zeros(1, 2, 2)
        corrected = apply_lprc(outputs, torch.tensor([[0, 1]]), artifact)
        expected = torch.tensor([[[10.0, 20.0], [20.0, 40.0]]])
        torch.testing.assert_close(corrected, expected)
        with self.assertRaises(ValueError):
            apply_lprc(outputs, torch.tensor([[0, 2]]), artifact)

    def test_paired_statistics_have_window_channel_and_horizon_views(self):
        stats = PairedErrorAccumulator()
        true = np.zeros((2, 3, 2), dtype=np.float32)
        base = np.ones_like(true)
        corrected = np.full_like(true, 0.5)
        stats.update(base, corrected, true)
        summary, arrays = stats.finalize()
        self.assertEqual(summary['window_count'], 2)
        self.assertAlmostEqual(summary['base_mse'], 1.0)
        self.assertAlmostEqual(summary['corrected_mse'], 0.25)
        self.assertEqual(arrays['per_channel_base_mse'].shape, (2,))
        self.assertEqual(arrays['per_horizon_base_mse'].shape, (3,))
        self.assertEqual(arrays['per_window_base_mse'].shape, (2,))

    def test_artifact_round_trip_and_configuration_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = make_args(temp_dir)
            args.k = 3
            args.group_size = 10
            checkpoint_path = os.path.join(temp_dir, 'checkpoint.pth')
            with open(checkpoint_path, 'wb') as handle:
                handle.write(b'checkpoint identity only')
            config = model_config_from_args(args, channel_count=2)
            table = torch.arange(336, dtype=torch.float64).reshape(2, 168)
            counts = torch.ones(168, dtype=torch.int64)
            artifact = build_lprc_artifact(
                residual_table=table,
                residual_sum=table,
                phase_counts=counts,
                cell_counts=counts.unsqueeze(0).repeat(2, 1),
                requested_rank=1,
                phase_metadata={
                    'phase_mode': 'hour_of_week',
                    'period': 168,
                    'freq': 'h',
                    'formula': 'weekday * 24 + hour',
                },
                model_config=config,
                source_checkpoint={
                    'path': os.path.abspath(checkpoint_path),
                    'sha256': __import__('hashlib').sha256(
                        b'checkpoint identity only'
                    ).hexdigest(),
                    'size_bytes': len(b'checkpoint identity only'),
                },
                fit_timing={'residual_collection_sec': 0.0},
                window_count=1,
                provenance=runtime_provenance(args, torch.device('cpu')),
            )
            artifact_path = os.path.join(temp_dir, 'lprc.pt')
            artifact['metadata']['runtime_provenance'][
                'python_version'
            ] = 'provenance-only-test-value'
            save_lprc_artifact(artifact_path, artifact)
            loaded = load_lprc_artifact(
                artifact_path, args, checkpoint_path=checkpoint_path
            )
            torch.testing.assert_close(
                loaded['residual_table'], artifact['residual_table']
            )

            args.model_id = 'renamed_regression_run'
            load_lprc_artifact(
                artifact_path, args, checkpoint_path=checkpoint_path
            )

            args.k = 4
            with self.assertRaises(RuntimeError):
                load_lprc_artifact(
                    artifact_path, args, checkpoint_path=checkpoint_path
                )

            args.k = 3
            with open(
                os.path.join(temp_dir, args.data_path), 'w', encoding='utf-8'
            ) as handle:
                handle.write('date,reordered,OT\n')
            with self.assertRaises(RuntimeError):
                load_lprc_artifact(
                    artifact_path, args, checkpoint_path=checkpoint_path
                )

    def test_lprc_evidence_guard_refuses_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_lprc_evaluation_output_dir(temp_dir)
            evidence_path = os.path.join(temp_dir, 'lprc_summary.json')
            with open(evidence_path, 'w', encoding='utf-8') as handle:
                handle.write('{}')
            with self.assertRaises(FileExistsError):
                guard_lprc_evaluation_output_dir(temp_dir)

    def test_fit_uses_checkpoint_and_disables_gradients(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = make_args(temp_dir)
            checkpoint_path = os.path.join(temp_dir, 'checkpoint.pth')
            model = TinyModel()
            checkpoint_model = TinyModel()
            checkpoint_model.weight.data.fill_(2.0)
            torch.save(checkpoint_model.state_dict(), checkpoint_path)
            observations = []

            def batch_forward(batch):
                observations.append({
                    'grad_enabled': torch.is_grad_enabled(),
                    'model_training': model.training,
                })
                _, targets, _, marks = batch
                predictions = model(torch.ones_like(targets))
                return predictions, targets, marks

            model.weight.data.fill_(9.0)
            model.train()
            artifact = fit_lprc_artifact(
                model, SyntheticTrainDataset(), args, torch.device('cpu'),
                batch_forward, checkpoint_path,
                os.path.join(temp_dir, 'lprc_train_state.pt'),
            )
            self.assertEqual(float(model.weight.item()), 2.0)
            self.assertTrue(model.training)
            self.assertIsNone(model.weight.grad)
            self.assertEqual(
                observations,
                [{'grad_enabled': False, 'model_training': False}],
            )
            self.assertEqual(float(artifact['residual_table'][0, 10]), 8.0)
            self.assertEqual(
                artifact['metadata']['source_checkpoint']['sha256'],
                sha256_file(checkpoint_path),
            )
            self.assertEqual(artifact['metadata']['fit_split'], 'train')
            self.assertFalse(artifact['metadata']['fit_timing']['shuffle'])
            self.assertFalse(artifact['metadata']['fit_timing']['drop_last'])
            provenance = artifact['metadata']['runtime_provenance']
            self.assertEqual(
                provenance['cli_arguments']['model_id'], args.model_id
            )
            for key in (
                'git_commit', 'seed', 'python_version', 'pytorch_version',
                'cuda_version', 'device', 'gpu_name', 'numpy_version',
            ):
                self.assertIn(key, provenance)
            self.assertEqual(
                artifact['metadata']['overlapping_window_weighting'],
                'each forecast element from each overlapping TRAIN window contributes once',
            )

            observations.clear()
            model.weight.data.fill_(11.0)
            model.eval()
            fit_lprc_artifact(
                model, SyntheticTrainDataset(), args, torch.device('cpu'),
                batch_forward, checkpoint_path,
                os.path.join(temp_dir, 'lprc_eval_state.pt'),
            )
            self.assertFalse(model.training)
            self.assertIsNone(model.weight.grad)
            self.assertEqual(float(model.weight.item()), 2.0)
            self.assertEqual(
                observations,
                [{'grad_enabled': False, 'model_training': False}],
            )

    def test_final_fit_uses_one_selected_period_svd(self):
        cache = periodic_cache()
        with mock.patch(
            'utils.lprc.np.linalg.svd', wraps=np.linalg.svd
        ) as svd:
            fit = fit_final_lprc(cache)
        self.assertEqual(fit['selected_period'], 12)
        self.assertEqual(svd.call_count, 1)
        self.assertIn('tau=.90', fit['variants'])
        self.assertIn('selected-P full rank', fit['variants'])
        self.assertIn('P=1', fit['variants'])

    def test_cache_metadata_rejects_explicit_use_norm_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = make_args(temp_dir)
            args.exact_fast_vardrop = True
            args.k = 4
            args.group_size = 10
            metadata = {
                'checkpoint_sha256': 'a' * 64,
                'dataset': args.data,
                'data_path': args.data_path,
                'seq_len': args.seq_len,
                'pred_len': args.pred_len,
                'channels': args.enc_in,
                'use_norm': 0,
                'k': args.k,
                'group_size': args.group_size,
                'exact_fast_vardrop': True,
                'cache_version': 1,
                'train_global_offset': 0,
                'val_global_offset': 240,
                'test_global_offset': 336,
            }
            with self.assertRaisesRegex(RuntimeError, 'use_norm'):
                validate_lprc_cache_metadata(metadata, args)

    def test_fit_from_cache_is_numpy_only_and_artifact_is_low_rank(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = make_args(temp_dir)
            args.exact_fast_vardrop = True
            args.k = 4
            args.group_size = 10
            args.checkpoints = temp_dir
            args.seed = 2023
            cache = periodic_cache(channels=2)
            metadata = {
                'checkpoint_sha256': 'b' * 64,
                'dataset': args.data,
                'data_path': args.data_path,
                'seq_len': args.seq_len,
                'pred_len': args.pred_len,
                'channels': args.enc_in,
                'use_norm': args.use_norm,
                'k': args.k,
                'group_size': args.group_size,
                'exact_fast_vardrop': True,
                'cache_version': 1,
                **cache['metadata'],
            }
            cache_path = os.path.join(temp_dir, 'cache.npz')
            artifact_path = os.path.join(temp_dir, 'artifact.npz')
            np.savez_compressed(
                cache_path,
                metadata_json=np.asarray(json.dumps(metadata)),
                train_residual_sum=cache['train_residual_sum'],
                train_residual_sumsq=cache['train_residual_sum'] ** 2,
                train_count=cache['train_count'],
                val_residual_sum=cache['val_residual_sum'],
                val_residual_sumsq=cache['val_residual_sumsq'],
                val_count=cache['val_count'],
            )
            with mock.patch('utils.lprc.torch.load') as torch_load, mock.patch(
                'utils.lprc.torch.cuda.is_available'
            ) as cuda_available:
                fitted = fit_lprc_from_cache(
                    cache_path,
                    artifact_path,
                    args,
                    'synthetic_xf1_lp1_pa_w1_t95',
                )
            torch_load.assert_not_called()
            cuda_available.assert_not_called()
            self.assertIn(fitted['selected_P'], (12, 24))
            artifact = load_final_lprc_artifact(artifact_path)
            self.assertEqual(
                artifact['phase_factor'].shape[1], fitted['selected_rank']
            )
            with np.load(artifact_path, allow_pickle=False) as stored:
                self.assertNotIn('residual_table', stored.files)
                self.assertNotIn('train_residual_sum', stored.files)
                self.assertIn('phase_factor', stored.files)
                self.assertIn('variable_factor', stored.files)

    def test_short_setting_suffixes(self):
        self.assertEqual(
            lprc_setting_suffix(True, True), '_xf1_lp1_pa_w1_t95'
        )
        args = SimpleNamespace(exact_fast_vardrop=True)
        self.assertEqual(
            lprc_result_setting('run_fastdfh', args),
            'run_xf1_lp1_pa_w1_t95',
        )

    def test_historical_exact_fast_backbone_setting_is_unchanged(self):
        args = SimpleNamespace(
            model_id='traffic_96_96', model='OURS', data='custom',
            features='M', seq_len=96, label_len=48, pred_len=96,
            d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
            factor=3, embed='timeF', distil=True, des='Exp', k=4,
            group_size=10, class_strategy='projection',
            exact_fast_vardrop=True, use_lpra=False,
        )
        self.assertEqual(
            experiment_setting(args, 0),
            'traffic_96_96_OURS_custom_M_ft96_sl48_ll96_pl512_dm8_nh2_'
            'el1_dl2048_df3_fctimeF_ebTrue_dtExp_k4_gs10_projection_0_'
            'fastdfh',
        )

    def test_custom_overlapping_split_target_alignment(self):
        train_data = Dataset_Custom.__new__(Dataset_Custom)
        val_data = Dataset_Custom.__new__(Dataset_Custom)
        train_data.data_x = np.empty((70, 1))
        val_data.data_x = np.empty((14, 1))
        self.assertEqual(
            forecast_target_global_offsets(train_data, val_data, 4),
            {'train': 4, 'val': 70, 'test': 80},
        )

    def test_pems_disjoint_split_target_alignment(self):
        train_data = Dataset_PEMS.__new__(Dataset_PEMS)
        val_data = Dataset_PEMS.__new__(Dataset_PEMS)
        train_data.data_x = np.empty((60, 1))
        val_data.data_x = np.empty((20, 1))
        self.assertEqual(
            forecast_target_global_offsets(train_data, val_data, 4),
            {'train': 4, 'val': 64, 'test': 84},
        )

    def test_final_ablation_evaluation_is_one_streaming_pass(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = make_args(temp_dir)
            args.seed = 2023
            args.checkpoints = temp_dir
            args.exact_fast_vardrop = True
            args.k = 4
            args.group_size = 10
            args.inverse = False
            args.lprc_eval_ablations = True

            metadata = {
                'selected_P': 2,
                'selected_rank': 1,
                'channels': 2,
                'checkpoint_sha256': 'c' * 64,
                'cache_version': 1,
                'artifact_version': 3,
                'test_global_offset': 4,
                'variant_periods': {
                    'P=1': 1,
                    'selected-P full rank': 2,
                    'tau=.90': 2,
                    'tau=.95': 2,
                    'tau=.97': 2,
                },
            }
            one_phase = np.ones((2, 1), dtype=np.float64) * 0.25
            variable = np.ones((1, 2), dtype=np.float64)

            class FinalHarness(LPRCExperimentMixin):
                def __init__(self):
                    self.args = args
                    self.model = TinyModel()
                    self.device = torch.device('cpu')
                    self.calls = 0
                    self.loader_calls = 0
                    self.lprc_artifact_path = os.path.join(
                        temp_dir, 'artifact.npz'
                    )
                    self.lprc_artifact = {
                        'metadata': metadata,
                        'phase_factor': one_phase,
                        'variable_factor': variable,
                        'variants': {
                            label: (one_phase[:period], variable)
                            for label, period in (
                                ('P=1', 1),
                                ('selected-P full rank', 2),
                                ('tau=.90', 2),
                                ('tau=.95', 2),
                                ('tau=.97', 2),
                            )
                        },
                    }

                def _get_data(self, flag):
                    self.loader_calls += 1
                    self.assert_flag = flag
                    return SimpleNamespace(scale=False), [0, 1, 2]

                def _lprc_forward_batch(self, batch):
                    self.calls += 1
                    return (
                        torch.zeros(2, 3, 2),
                        torch.ones(2, 3, 2),
                        None,
                    )

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                harness = FinalHarness()
                record = harness._test_final_lprc(
                    'backbone_xf1_lp0', 0, 'run_xf1_lp1_pa_w1_t95'
                )
            finally:
                os.chdir(old_cwd)
            self.assertEqual(harness.assert_flag, 'test')
            self.assertEqual(harness.loader_calls, 1)
            self.assertEqual(harness.calls, 3)
            self.assertEqual(record['test_model_passes'], 1)
            saved = []
            for root, _, files in os.walk(temp_dir):
                saved.extend(os.path.join(root, name) for name in files)
            self.assertFalse(any(
                os.path.basename(path) in ('pred.npy', 'true.npy', 'base_pred.npy')
                for path in saved
            ))


if __name__ == '__main__':
    unittest.main()

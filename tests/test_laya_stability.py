#!/usr/bin/env python3
"""CPU checks of campaign commands and canonical comparison integration."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import laya_stability as stability


class LayaStabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='puffer-laya-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prepare(self, precision):
        return stability.prepare(self.root, self.root / 'trainer', self.root / 'evaluator',
            self.root / 'bundle', self.root / 'initial.bin', self.root / precision, precision)

    def evaluation(self, command, food):
        argv = dict(arg[2:].split('=', 1) for arg in command['argv'][1:])
        protocol = {'type': 'protocol', 'version': 1, 'environment': 'decision_laya',
            'bundle': argv['policy.bundle'], 'checkpoint': argv['weights'], 'params': 4,
            'grid_size': 10, 'action_count': 4, 'primary_metric': 'food_score',
            'mask_rule': 'reverse_only', 'observation_format': 1, 'zero_init_critic': False,
            'episodes': 512, 'episode_offset': 0, 'batch': 16, 'model_init_seed': 73,
            'environment_seed_base': stability.ENV_SEED, 'action_seed_base': stability.ACTION_SEED,
            'sampling': 'puffer_philox_categorical', 'max_steps': 500, 'temperature': 1.8,
            'gemm_precision': 'fp32', 'padded_tokens': 384, 'bundle_max_tokens': 512}
        episodes = []
        for i in range(512):
            board = [0] * 100
            board[55], board[54], board[53], board[0] = 1, 2, 3, -1
            length = food + 5
            episodes.append({'type': 'episode', 'episode': i, 'food_score': food,
                'length': length, 'return': food - 1, 'environment_seed': stability.ENV_SEED + i,
                'action_seed': stability.ACTION_SEED + i, 'initial_board': board,
                'initial_action_mask': [1, 1, 0, 1], 'terminated': True, 'truncated': False,
                'time_limit_reached': False, 'pure_timeout': False, 'outcome': -1,
                'collision': True, 'full_board': False, 'action_counts': [length, 0, 0, 0],
                'mean_action_probabilities': [.25] * 4})
        summary = {'type': 'summary', 'episodes': 512, 'mean_food_score': food,
            'food_score_standard_error': 0., 'mean_return': food - 1,
            'mean_length': food + 5, 'collisions': 512, 'full_boards': 0,
            'pure_timeouts': 0, 'evaluation_seconds': 1.}
        Path(command['stdout']).write_text(
            '\n'.join(json.dumps(row) for row in [protocol, *episodes, summary]) + '\n')

    def completed(self, precision):
        manifest = self.prepare(precision)
        self.evaluation(manifest['initial_evaluation'], 6)
        for trial in manifest['trials']:
            food = 8 if trial['learning_rate'] == stability.RATES[1] else 5
            if precision == 'tf32':
                food += 1
                if (trial['learning_rate'], trial['training_seed']) == (stability.RATES[1], 200073):
                    food = 4
            self.evaluation(trial['evaluation'], food)
            config = {}
            for key, value in trial['settings'].items():
                section, setting = key.split('.', 1)
                config.setdefault(section, {})[setting] = value
            records = [{'schema': 'puffer.training_metrics.v1', 'event': 'run',
                'world_size': 4, 'config': config},
                {'schema': 'puffer.training_metrics.v1', 'event': 'metrics',
                    'metrics': {'agent_steps': 256, 'epoch': 1, 'uptime': 2.}},
                {'schema': 'puffer.training_metrics.v1', 'event': 'metrics',
                    'metrics': {'agent_steps': 8192, 'epoch': 32, 'uptime': 60.}}]
            Path(trial['settings']['base.metrics_path']).write_text(
                '\n'.join(json.dumps(row) for row in records) + '\n')
        return manifest

    def test_prepare_uses_one_fp32_evaluation_per_policy_and_selected_training_precision(self):
        for precision in ('fp32', 'tf32'):
            with self.subTest(precision=precision):
                manifest = self.prepare(precision)
                self.assertEqual(len(manifest['trials']), 6)
                for trial in manifest['trials']:
                    settings = trial['settings']
                    self.assertIn('--policy.gemm_precision=' + precision, trial['training']['argv'])
                    self.assertEqual(
                        settings['base.load_model_path'], str(self.root / 'initial.bin'))
                    self.assertEqual(settings['policy.zero_init_critic'], 0)
                    self.assertEqual(settings['base.resume_path'], 'None')
                    self.assertEqual(settings['train.total_timesteps'], 8192)
                    self.assertFalse(Path(trial['training']['stdout']).exists())
                commands = [manifest['initial_evaluation'],
                    *[trial['evaluation'] for trial in manifest['trials']]]
                for command in commands:
                    self.assertIn('--episodes=512', command['argv'])
                    self.assertIn('--policy.gemm_precision=fp32', command['argv'])
                    self.assertIn('--base.seed=73', command['argv'])

    def test_all_contrasts_use_canonical_comparator_and_all_seeds_decide_success(self):
        manifests = {mode: self.completed(mode) for mode in ('fp32', 'tf32')}
        canonical = stability.compare
        with mock.patch.object(stability, 'compare',
            side_effect=lambda before, after: canonical(before, after, resamples=1000)) as compare:
            result = stability.analyze([self.root / mode / 'protocol.json' for mode in manifests])
        self.assertEqual(compare.call_count, 24)
        fp32, tf32 = (result['precisions'][mode] for mode in ('fp32', 'tf32'))
        self.assertTrue(fp32['lower_rate_preserved_all_three'])
        self.assertTrue(fp32['lower_rate_advantage_all_three'])
        self.assertEqual(fp32['improved_all_three_by_rate'], {'0.0001': False, '0.00001': True})
        self.assertFalse(tf32['lower_rate_preserved_all_three'])
        self.assertFalse(tf32['lower_rate_advantage_all_three'])
        self.assertFalse(tf32['improved_all_three_by_rate']['0.00001'])
        for row in fp32['trials']:
            self.assertEqual(row['training_metrics']['logged_events'], 2)
        contrast = result['precision_comparisons']
        self.assertTrue(contrast['0.0001']['improved_all_three'])
        self.assertFalse(contrast['0.00001']['preserved_all_three'])
        self.assertEqual(contrast['0.00001']['tf32_minus_fp32']['200073']
            ['intervals_95']['mean_food_score_change'], [-4, -4])
        for mode, manifest in manifests.items():
            for trial in manifest['trials']:
                self.assertIn(mock.call(manifest['initial_evaluation']['stdout'],
                    trial['evaluation']['stdout']), compare.call_args_list)

    def test_missing_metrics_or_wrong_final_counters_prevent_analysis(self):
        manifest = self.completed('fp32')
        path = Path(manifest['trials'][0]['settings']['base.metrics_path'])
        original = path.read_text()
        for mutate in (
            lambda rows: rows.__delitem__(slice(1, None)),
            lambda rows: rows[-1]['metrics'].update(agent_steps=2048),
            lambda rows: rows[-1]['metrics'].update(epoch=31),
        ):
            rows = [json.loads(line) for line in original.splitlines()]
            mutate(rows)
            path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
            with self.assertRaises(AssertionError):
                stability.analyze([self.root / 'fp32/protocol.json'])


if __name__ == '__main__':
    unittest.main()

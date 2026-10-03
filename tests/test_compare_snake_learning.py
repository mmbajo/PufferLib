#!/usr/bin/env python3
"""Admission and paired primary-metric tests for native Snake evaluation."""
import importlib.util
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('snake_comparison', ROOT / 'tools/compare_snake_learning.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class SnakeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, name, foods=(0, 1, 3, 8), lengths=None, outcomes=None,
                checkpoint='', changed=None, offset=0):
        lengths = lengths or [score + 8 for score in foods]
        outcomes = outcomes or [-1] * len(foods)
        protocol = dict(type='protocol', version=1, environment='decision_laya', bundle='bundles/laya',
            checkpoint=checkpoint, episodes=len(foods), episode_offset=offset, batch=16,
            model_init_seed=73, environment_seed_base=670001, action_seed_base=770001,
            sampling='puffer_philox_categorical', max_steps=500, temperature=1.9,
            padded_tokens=512, params=421294860, grid_size=10, action_count=4,
            primary_metric='food_score', mask_rule='reverse_only')
        episodes = []
        for i, (food, length, outcome) in enumerate(zip(foods, lengths, outcomes)):
            board = [0] * 100
            board[55], board[54], board[53], board[10 + i] = 1, 2, 3, -1
            counts = [length // 4] * 4
            counts[0] += length % 4
            episodes.append(dict(type='episode', episode=offset+i, food_score=food, length=length,
                **{'return': food - int(outcome == -1)},
                environment_seed=670001+offset+i, action_seed=770001+offset+i,
                initial_board=board, initial_action_mask=[1, 1, 0, 1],
                terminated=outcome != 0, truncated=outcome == 0,
                time_limit_reached=length == 500, pure_timeout=outcome == 0,
                outcome=outcome, collision=outcome == -1, full_board=outcome == 1,
                action_counts=counts, mean_action_probabilities=[.25] * 4))
        summary = dict(type='summary', episodes=len(foods), mean_food_score=statistics.mean(foods),
            food_score_standard_error=statistics.stdev(foods)/math.sqrt(len(foods)) if len(foods) > 1 else 0,
            mean_return=statistics.mean(r['return'] for r in episodes),
            mean_length=statistics.mean(lengths), collisions=outcomes.count(-1),
            full_boards=outcomes.count(1), pure_timeouts=outcomes.count(0), evaluation_seconds=1.2)
        records = [protocol, *reversed(episodes), summary]
        if changed:
            changed(records)
        path = self.root/name
        path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
        return path

    def test_self_comparison_has_exact_zero_primary_and_secondary_changes(self):
        path = self.fixture('same', foods=(0, 2, 97), lengths=(12, 500, 250), outcomes=(-1, 0, 1))
        result = comparison.compare(path, path, resamples=1000)
        for key in ('mean_food_score_change', 'mean_return_change', 'mean_length_change',
                    'collision_rate_change', 'pure_timeout_rate_change', 'full_board_rate_change'):
            self.assertEqual(result[key], 0)
            self.assertEqual(result['intervals_95'][key], [0, 0])
        self.assertEqual(result['food_tied_episodes'], 3)
        self.assertFalse(result['positive_mean_food_score_change_interval'])

    def test_constant_food_gain_has_exact_paired_interval(self):
        before = self.fixture('before')
        after = self.fixture('after', foods=(2, 3, 5, 10), checkpoint='trained.bin')
        result = comparison.compare(before, after, resamples=1000)
        self.assertEqual(result['mean_food_score_change'], 2)
        self.assertEqual(result['intervals_95']['mean_food_score_change'], [2, 2])
        self.assertEqual(result['mean_return_change'], 2)
        self.assertTrue(result['positive_mean_food_score_change_interval'])
        self.assertEqual(result['food_improved_episodes'], 4)

    def test_survival_and_return_improvement_do_not_replace_primary_food_metric(self):
        before = self.fixture('before', foods=(0, 1), lengths=(12, 20))
        after = self.fixture('after', foods=(0, 1), lengths=(500, 500), outcomes=(0, 0), checkpoint='after.bin')
        result = comparison.compare(before, after, resamples=1000)
        self.assertEqual(result['mean_food_score_change'], 0)
        self.assertEqual(result['mean_return_change'], 1)
        self.assertEqual(result['pure_timeout_rate_change'], 1)
        self.assertEqual(result['collision_rate_change'], -1)
        self.assertGreater(result['mean_length_change'], 0)
        self.assertFalse(result['positive_mean_food_score_change_interval'])

    def test_collision_and_full_board_at_cap_are_not_timeouts(self):
        path = self.fixture('cap', foods=(2, 97), lengths=(500, 500), outcomes=(-1, 1))
        _, summary, rows = comparison.load(path)
        self.assertEqual(summary['pure_timeouts'], 0)
        self.assertTrue(all(row['time_limit_reached'] for row in rows.values()))
        self.assertTrue(all(not row['truncated'] for row in rows.values()))

    def test_disjoint_episode_offsets_and_record_order_are_supported(self):
        path = self.fixture('offset', offset=1024)
        _, _, episodes = comparison.load(path)
        self.assertEqual(sorted(episodes), list(range(1024, 1028)))
        self.assertEqual(episodes[1024]['environment_seed'], 671025)

    def test_food_seed_or_initial_board_mismatch_rejected(self):
        before = self.fixture('before')
        def change_food(rows):
            board = rows[1]['initial_board']
            board[board.index(-1)] = 0
            board[0] = -1
        for name, change, message in (
            ('seed', lambda r: r[1].update(action_seed=12), 'action_seed'),
            ('board', change_food, 'unpaired initial_board'),
            ('mask', lambda r: r[1].update(initial_action_mask=[1, 1, 1, 0]), 'action mask'),
        ):
            with self.subTest(name=name):
                after = self.fixture(name, changed=change)
                with self.assertRaisesRegex(ValueError, message):
                    comparison.compare(before, after, resamples=1000)

    def test_changed_protocol_rejected(self):
        before = self.fixture('before')
        for key, value in (('batch', 8), ('model_init_seed', 74), ('temperature', 2.0),
                           ('bundle', 'different/tokenizer'), ('max_steps', 400)):
            with self.subTest(key=key):
                after = self.fixture(key, changed=lambda r: r[0].update({key: value}))
                with self.assertRaisesRegex(ValueError, 'protocols differ'):
                    comparison.compare(before, after, resamples=1000)

    def test_missing_duplicate_and_unknown_records_rejected(self):
        for name, mutation, message in (
            ('missing', lambda r: r.pop(1), 'episode count'),
            ('duplicate', lambda r: r[1].update(episode=r[2]['episode']), 'episode identifier'),
            ('unknown', lambda r: r.insert(1, {'type': 'debug'}), 'record type'),
            ('summary-first', lambda r: r.insert(0, r.pop()), 'protocol must be first'),
            ('missing-field', lambda r: r[1].pop('food_score'), 'missing fields'),
        ):
            with self.subTest(name=name):
                path = self.fixture(name, changed=mutation)
                with self.assertRaisesRegex(ValueError, message):
                    comparison.load(path)
        path = self.fixture('duplicate-json-key')
        path.write_text(path.read_text().replace('"version": 1', '"version": 1, "version": 1', 1))
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            comparison.load(path)

    def test_native_board_greedy_and_random_protocols_are_admitted(self):
        for sampling, mode in comparison.SAMPLING_MODES.items():
            path = self.fixture(mode, changed=lambda rows: rows[0].update(
                environment='decision_snake', bundle='', sampling=sampling,
                sampling_mode=mode, padded_tokens=101, bundle_max_tokens=101,
                model='board_transformer', zero_init_critic=False, observation_format=0))
            comparison.load(path)
        bad = self.fixture('mixed-mode', changed=lambda rows: rows[0].update(sampling_mode='greedy'))
        with self.assertRaisesRegex(ValueError, 'sampling mode'):
            comparison.load(bad)

    def test_treatments_require_explicit_declaration_and_are_recorded(self):
        before = self.fixture('control', changed=lambda rows: rows[0].update(observation_format=0))
        after = self.fixture('coordinate-treatment', changed=lambda rows: rows[0].update(observation_format=1))
        with self.assertRaisesRegex(ValueError, 'protocols differ'):
            comparison.compare(before, after, resamples=1000)
        result = comparison.compare(before, after, resamples=1000,
                                    allow_protocol_differences=['observation_format'])
        self.assertEqual(result['protocol_differences'], {'observation_format': {'before': 0, 'after': 1}})
        self.assertEqual(result['declared_treatment_fields'], ['observation_format'])
        self.assertEqual(result['after_protocol']['observation_format'], 1)
        self.assertEqual(result['mean_food_score_change'], 0)

    def test_treatment_cannot_waive_pairing_or_game_rules(self):
        path = self.fixture('valid')
        for field in ('environment_seed_base', 'action_seed_base', 'model_init_seed',
                      'max_steps', 'mask_rule', 'episode_offset', 'episodes', 'batch'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'permitted treatment'):
                comparison.compare(path, path, resamples=1000, allow_protocol_differences=[field])
        other = self.fixture('unpaired', changed=lambda rows: rows[1].update(action_seed=1))
        with self.assertRaisesRegex(ValueError, 'action_seed'):
            comparison.compare(path, other, resamples=1000,
                               allow_protocol_differences=['observation_format'])

    def test_food_reward_and_terminal_accounting_rejected(self):
        for key, value, message in (
            ('food_score', True, 'length/food'), ('food_score', -1, 'length/food'),
            ('food_score', 98, 'length/food'), ('food_score', 97, 'food/full-board'),
            ('return', 100, 'food/reward'), ('return', float('nan'), 'food/reward'),
            ('collision', False, 'termination/timeout'), ('full_board', True, 'termination/timeout'),
            ('terminated', 2, 'boolean flag'), ('outcome', 0, 'outcome'),
            ('time_limit_reached', True, 'termination/timeout'),
            ('pure_timeout', True, 'termination/timeout'), ('truncated', True, 'termination/timeout'),
        ):
            with self.subTest(key=key, value=value):
                path = self.fixture('bad', changed=lambda r: r[1].update({key: value}))
                with self.assertRaisesRegex(ValueError, message):
                    comparison.load(path)

    def test_initial_board_structure_rejected(self):
        for state in ([0]*99, [0]*100, [False]*100, [float('nan')]*100, None):
            with self.subTest(state=state):
                path = self.fixture('bad-board', changed=lambda r: r[1].update(initial_board=state))
                with self.assertRaisesRegex(ValueError, 'initial_board'):
                    comparison.load(path)

    def test_action_counts_and_probabilities_rejected(self):
        for key, value in (
            ('action_counts', [1, 2, 3]), ('action_counts', [0]*4),
            ('action_counts', [True]*4), ('action_counts', [-1, 0, 0, 1]),
            ('mean_action_probabilities', [.5]*4), ('mean_action_probabilities', [.5, .5]),
            ('mean_action_probabilities', [float('nan'), 0, 0, 1]),
            ('mean_action_probabilities', [-.01, .51, .25, .25]),
            ('mean_action_probabilities', [True, 0, 0, 0]),
        ):
            with self.subTest(key=key, value=value):
                path = self.fixture('bad-action', changed=lambda r: r[1].update({key: value}))
                with self.assertRaisesRegex(ValueError, 'action'):
                    comparison.load(path)

    def test_summary_means_counts_and_standard_error_rejected(self):
        for key, value in (
            ('mean_food_score', 0), ('mean_return', 123), ('mean_length', float('inf')),
            ('food_score_standard_error', 0), ('collisions', 0), ('full_boards', True),
            ('pure_timeouts', 1), ('evaluation_seconds', -1),
        ):
            with self.subTest(key=key, value=value):
                path = self.fixture('bad-summary', changed=lambda r: r[-1].update({key: value}))
                with self.assertRaisesRegex(ValueError, key):
                    comparison.load(path)

    def test_seed_overflow_and_invalid_bootstrap_controls_rejected(self):
        for key, value in (('environment_seed_base', (1 << 32)-1),
                           ('action_seed_base', (1 << 64)-1), ('episode_offset', -1)):
            path = self.fixture('overflow', changed=lambda r: r[0].update({key: value}))
            with self.assertRaises(ValueError):
                comparison.load(path)
        valid = self.fixture('valid')
        for resamples, seed in ((999, 1), (1000, -1), (True, 1)):
            with self.assertRaisesRegex(ValueError, 'bootstrap'):
                comparison.compare(valid, valid, resamples=resamples, bootstrap_seed=seed)

    def test_statistics_are_recomputed_from_rows_and_are_deterministic(self):
        before = self.fixture('before', changed=lambda r: r[-1].update(mean_food_score=3.0000005))
        after = self.fixture('after', foods=(0, 2, 4, 6), checkpoint='after.bin')
        a = comparison.compare(before, after, resamples=1000)
        b = comparison.compare(before, after, resamples=1000)
        self.assertEqual(a, b)
        self.assertEqual(a['before_mean_food_score'], 3)
        self.assertEqual(a['food_improved_episodes'], 2)
        self.assertEqual(a['food_worsened_episodes'], 1)
        self.assertEqual(a['food_tied_episodes'], 1)


if __name__ == '__main__':
    unittest.main()

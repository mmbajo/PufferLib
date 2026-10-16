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

    def test_precision_treatment_requires_explicit_declaration(self):
        before = self.fixture('fp32-before',
            changed=lambda rows: rows[0].update(gemm_precision='fp32'))
        explicit = self.fixture('explicit-fp32', changed=lambda rows: rows[0].update(gemm_precision='fp32'))
        self.assertEqual(comparison.compare(before, explicit, resamples=1000)['protocol_differences'], {})
        after = self.fixture('tf32', changed=lambda rows: rows[0].update(gemm_precision='tf32'))
        with self.assertRaisesRegex(ValueError, 'protocols differ'):
            comparison.compare(before, after, resamples=1000)
        result = comparison.compare(before, after, resamples=1000,
                                    allow_protocol_differences=['gemm_precision'])
        self.assertEqual(result['protocol_differences'], {'gemm_precision': {'before': 'fp32', 'after': 'tf32'}})


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
            ('mask', lambda r: r[1].update(initial_action_mask=[1, 1, 1, 0]), 'initial_action_mask'),
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


    def test_mixed_sign_pairs_have_nonzero_uncertainty(self):
        deltas = [-2] * 9 + [3] * 9 + [1, 1]
        before = self.fixture('mixed-before', foods=[10] * 20)
        after = self.fixture('mixed-after', foods=[10 + d for d in deltas])
        result = comparison.compare(before, after, resamples=1000)
        self.assertEqual(result['mean_food_score_change'], .55)
        lower, upper = result['intervals_95']['mean_food_score_change']
        self.assertLess(lower, 0)
        self.assertGreater(upper, 0)
        self.assertFalse(result['positive_mean_food_score_change_interval'])

    def test_incomplete_or_duplicate_records_are_rejected(self):
        for name, mutate, message in (
            ('missing-summary', lambda rows: rows.pop(), 'protocol and summary'),
            ('duplicate-protocol', lambda rows: rows.append(rows[0]), 'protocol and summary'),
            ('missing-episode', lambda rows: rows.pop(1), 'episode identifier'),
            ('duplicate-episode', lambda rows: rows.append(rows[1]), 'episode identifier'),
        ):
            with self.subTest(name=name):
                path = self.fixture(name, foods=(1, 2), changed=mutate)
                with self.assertRaisesRegex(ValueError, message):
                    comparison.load(path)


if __name__ == '__main__':
    unittest.main()

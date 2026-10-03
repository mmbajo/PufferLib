#!/usr/bin/env python3
"""Validate paired-inference data admission and statistical change accounting."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparison', ROOT / 'tools/compare_decision_learning.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, name, lengths, checkpoint='', changed=None):
        protocol = dict(type='protocol', episodes=len(lengths), episode_offset=0,
                        max_steps=200, environment_seed_base=170001,
                        action_seed_base=270001, model_init_seed=73, checkpoint=checkpoint, batch=2)
        episodes = []
        for i, length in enumerate(lengths):
            episodes.append(dict(type='episode', episode=i, length=length,
                **{'return': length - 1}, environment_seed=170001+i, action_seed=270001+i,
                initial_state=[i*.001, 0, 0, 0], max_abs_theta_observed=.02,
                mean_probability_right=.5, left_actions=length//2,
                right_actions=length-length//2, pure_timeout=length == 200))
        summary = dict(type='summary', episodes=len(lengths),
                       mean_length=sum(lengths)/len(lengths),
                       mean_return=sum(lengths)/len(lengths)-1)
        records = [protocol, *reversed(episodes), summary]
        if changed: changed(records)
        path = self.root/name
        path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
        return path

    def test_identical_checkpoint_has_exact_zero_paired_change(self):
        before = self.fixture('before', [4, 12, 27, 200])
        result = comparison.compare(before, before, resamples=1000)
        self.assertEqual(result['mean_length_change'], 0)
        self.assertEqual(result['intervals_95']['mean_length_change'], [0, 0])
        self.assertEqual(result['tied_episodes'], 4)
        self.assertFalse(result['positive_mean_change_interval'])

    def test_constant_paired_gain_has_exact_interval(self):
        before = self.fixture('before', [4, 12, 27, 190])
        after = self.fixture('after', [14, 22, 37, 200], checkpoint='trained.bin')
        result = comparison.compare(before, after, resamples=1000)
        self.assertEqual(result['mean_length_change'], 10)
        self.assertEqual(result['intervals_95']['mean_length_change'], [10, 10])
        self.assertEqual(result['pure_timeout_rate_change'], .25)
        self.assertTrue(result['positive_mean_change_interval'])

    def test_changed_initial_state_rejects_pairing(self):
        before = self.fixture('before', [4, 12])
        after = self.fixture('after', [5, 13], changed=lambda rows: rows[1].update(initial_state=[0,0,0,0]))
        with self.assertRaisesRegex(ValueError, 'unpaired initial_state'):
            comparison.compare(before, after, resamples=1000)

    def test_missing_episode_rejected(self):
        path = self.fixture('incomplete', [4, 12], changed=lambda rows: rows.pop(1))
        with self.assertRaisesRegex(ValueError, 'episode count'):
            comparison.load(path)

    def test_seed_or_reward_or_batch_changes_rejected(self):
        before = self.fixture('before', [4, 12])
        for name, change, message in [
            ('seed', lambda rows: rows[1].update(action_seed=1), 'action seed'),
            ('reward', lambda rows: rows[1].update({'return': 12}), 'accounting'),
            ('batch', lambda rows: rows[0].update(batch=8), 'protocols differ'),
            ('initialization-seed', lambda rows: rows[0].update(model_init_seed=74), 'protocols differ'),
        ]:
            with self.subTest(name=name):
                after = self.fixture(name, [4, 12], changed=change)
                with self.assertRaisesRegex(ValueError, message):
                    comparison.compare(before, after, resamples=1000)

    def test_corrupt_summary_means_rejected(self):
        for field in ('mean_length', 'mean_return'):
            for value in (42, float('nan'), float('inf'), '12', True):
                with self.subTest(field=field, value=value):
                    path = self.fixture('bad-summary', [4, 12],
                        changed=lambda rows: rows[-1].update({field: value}))
                    with self.assertRaisesRegex(ValueError, f'summary {field}'):
                        comparison.load(path)

    def test_initial_state_must_have_four_finite_numbers(self):
        for state in ([0, 0, 0], [0, 0, 0, 0, 0], [0, 0, float('nan'), 0],
                      [0, float('inf'), 0, 0], [0, '0', 0, 0], [0, True, 0, 0], None):
            with self.subTest(state=state):
                path = self.fixture('bad-state', [4, 12],
                    changed=lambda rows: rows[1].update(initial_state=state))
                with self.assertRaisesRegex(ValueError, 'four finite numbers'):
                    comparison.load(path)

    def test_probabilities_must_be_finite_and_in_unit_interval(self):
        for probability in (-.01, 1.01, float('nan'), float('inf'), '0.5', True):
            with self.subTest(probability=probability):
                path = self.fixture('bad-probability', [4, 12],
                    changed=lambda rows: rows[1].update(mean_probability_right=probability))
                with self.assertRaisesRegex(ValueError, 'mean_probability_right'):
                    comparison.load(path)

    def test_return_means_come_from_raw_rows(self):
        # Permit harmless summary formatting error within the admission tolerance,
        # but do not propagate it into the independently recomputed statistics.
        before = self.fixture('before', [4, 12],
            changed=lambda rows: rows[-1].update(mean_return=7.0000005))
        after = self.fixture('after', [5, 13], checkpoint='after.bin',
            changed=lambda rows: rows[-1].update(mean_return=8.0000005))
        result = comparison.compare(before, after, resamples=1000)
        self.assertEqual(result['before_mean_return'], 7)
        self.assertEqual(result['after_mean_return'], 8)


if __name__ == '__main__':
    unittest.main()

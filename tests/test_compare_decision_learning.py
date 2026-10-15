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

    def test_precision_is_recorded_and_mixed_modes_rejected(self):
        before = self.fixture('fp32-before', [4, 12],
            changed=lambda rows: rows[0].update(gemm_precision='fp32'))
        explicit_fp32 = self.fixture('fp32', [4, 12],
            changed=lambda rows: rows[0].update(gemm_precision='fp32'))
        tf32 = self.fixture('tf32', [4, 12],
            changed=lambda rows: rows[0].update(gemm_precision='tf32'))
        result = comparison.compare(before, explicit_fp32, resamples=1000)
        self.assertEqual(result['protocol']['gemm_precision'], 'fp32')
        self.assertEqual(result['mean_length_change'], 0)
        self.assertEqual(comparison.compare(tf32, tf32, resamples=1000)['mean_length_change'], 0)
        for before, after in ((before, tf32), (tf32, explicit_fp32)):
            with self.assertRaisesRegex(ValueError, 'protocols differ'):
                comparison.compare(before, after, resamples=1000)


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
        with self.assertRaisesRegex(ValueError, 'episode identifier'):
            comparison.load(path)

    def test_seed_or_batch_changes_rejected(self):
        before = self.fixture('before', [4, 12])
        for name, change, message in [
            ('seed', lambda rows: rows[1].update(action_seed=1), 'action_seed'),
            ('batch', lambda rows: rows[0].update(batch=8), 'protocols differ'),
            ('initialization-seed', lambda rows: rows[0].update(model_init_seed=74), 'protocols differ'),
        ]:
            with self.subTest(name=name):
                after = self.fixture(name, [4, 12], changed=change)
                with self.assertRaisesRegex(ValueError, message):
                    comparison.compare(before, after, resamples=1000)


    def test_effective_policy_controls_are_validated_and_compared(self):
        controls = dict(padded_tokens=512, bundle_max_tokens=512, zero_init_critic=False)
        before = self.fixture('controlled', [4, 12], changed=lambda rows: rows[0].update(controls))
        self.assertEqual(comparison.compare(before, before, resamples=1000)['mean_length_change'], 0)
        for field, value in (('padded_tokens', 384), ('bundle_max_tokens', 1024),
                             ('zero_init_critic', True)):
            with self.subTest(field=field):
                after = self.fixture('different', [4, 12],
                    changed=lambda rows: rows[0].update({**controls, field: value}))
                with self.assertRaisesRegex(ValueError, 'protocols differ'):
                    comparison.compare(before, after, resamples=1000)

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


    def test_mixed_sign_pairs_have_nonzero_uncertainty(self):
        deltas = [-2] * 9 + [3] * 9 + [1, 1]
        before = self.fixture('mixed-before', [10] * 20)
        after = self.fixture('mixed-after', [10 + d for d in deltas])
        result = comparison.compare(before, after, resamples=1000)
        self.assertEqual(result['mean_length_change'], .55)
        lower, upper = result['intervals_95']['mean_length_change']
        self.assertLess(lower, 0)
        self.assertGreater(upper, 0)
        self.assertFalse(result['positive_mean_change_interval'])

    def test_incomplete_or_duplicate_records_are_rejected(self):
        for name, mutate, message in (
            ('missing-summary', lambda rows: rows.pop(), 'protocol and summary'),
            ('duplicate-protocol', lambda rows: rows.append(rows[0]), 'protocol and summary'),
            ('missing-episode', lambda rows: rows.pop(1), 'episode identifier'),
            ('duplicate-episode', lambda rows: rows.append(rows[1]), 'episode identifier'),
        ):
            with self.subTest(name=name):
                path = self.fixture(name, [4, 12], changed=mutate)
                with self.assertRaisesRegex(ValueError, message):
                    comparison.load(path)


if __name__ == '__main__':
    unittest.main()

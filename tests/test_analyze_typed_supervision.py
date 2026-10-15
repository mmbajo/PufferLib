"""Hand examples for the paired native typed-supervision analysis."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'analyzer', ROOT / 'tools/analyze_typed_supervision.py')
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


class TypedAnalysisTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def fixture(self, boards):
        rows, records = [], []
        for state, metrics in boards:
            for kind in analyzer.TYPES:
                index = len(rows)
                rows.append({'state_id': state, 'questions': {'q': {'type': kind}}})
                for epoch, (correct, loss) in zip((0, 5), metrics[kind]):
                    records.append({'event': 'validation_row', 'epoch': epoch,
                        'row': index, 'correct': correct, 'log_loss': loss})
        records = [*reversed(records),
            {'event': 'evaluation', 'split': 'train', 'epoch': 0},
            {'event': 'evaluation', 'split': 'validation', 'epoch': 0},
            {'event': 'train', 'epoch': 5},
            {'event': 'evaluation', 'split': 'validation', 'epoch': 5}]
        dataset = self.directory / 'data.jsonl'
        metrics = self.directory / 'metrics.jsonl'
        self.write(dataset, rows)
        self.write(metrics, records)
        return dataset, metrics, records

    @staticmethod
    def write(path, records):
        path.write_text(''.join(json.dumps(record) + '\n' for record in records))

    def test_hand_losses_row_binding_and_positive_change(self):
        counts = {'choice': (2, 3), 'noul': (2, 2), 'score': (4, 4)}
        boards = [(state, {kind: [(0, math.log(counts[kind][index])),
            (1, math.log((counts[kind][index] + 2) / 3))] for kind in analyzer.TYPES})
            for index, state in enumerate((9, 7))]
        dataset, metrics, _ = self.fixture(boards)
        result = analyzer.analyze(dataset, metrics)
        self.assertEqual(result['metrics_file'], str(metrics))
        self.assertEqual((result['initial_epoch'], result['final_epoch']), (0, 5))
        self.assertEqual((result['states'], result['questions']), (2, 6))
        for kind in (*analyzer.TYPES, 'overall'):
            actual = result['overall'] if kind == 'overall' else result['types'][kind]
            accuracy = actual['accuracy']
            self.assertEqual((accuracy['initial'], accuracy['final']), (0, 1))
            self.assertEqual(accuracy['final_minus_initial'], 1)
            self.assertEqual(accuracy['change_interval_95'], [1, 1])
            self.assertTrue(accuracy['improved_interval'])
            kinds = analyzer.TYPES if kind == 'overall' else (kind,)
            before = sum(math.log(n) for q in kinds for n in counts[q]) / (2 * len(kinds))
            after = sum(math.log((n + 2) / 3) for q in kinds for n in counts[q]) / (2 * len(kinds))
            self.assertAlmostEqual(actual['log_loss']['initial'], before)
            self.assertAlmostEqual(actual['log_loss']['final'], after)
            self.assertAlmostEqual(actual['log_loss']['final_minus_initial'], after - before)
            self.assertTrue(actual['log_loss']['improved_interval'])

    def test_three_boards_use_paired_95_percent_interval(self):
        boards = [(state, {kind: [(0, 1), (int(state == 0), int(state != 0))]
            for kind in analyzer.TYPES}) for state in range(3)]
        dataset, metrics, _ = self.fixture(boards)
        result = analyzer.analyze(dataset, metrics)
        for kind in (*analyzer.TYPES, 'overall'):
            actual = result['overall'] if kind == 'overall' else result['types'][kind]
            self.assertAlmostEqual(actual['accuracy']['final_minus_initial'], 1 / 3)
            # One board improves. A 90% interval incorrectly ends at 2/3;
            # resampling nine independent questions also narrows the interval.
            self.assertEqual(actual['accuracy']['change_interval_95'], [0, 1])
            self.assertAlmostEqual(actual['log_loss']['final_minus_initial'], -1 / 3)
            self.assertEqual(actual['log_loss']['change_interval_95'], [-1, 0])
            self.assertFalse(actual['accuracy']['improved_interval'])
            self.assertFalse(actual['log_loss']['improved_interval'])

    def test_regression_keeps_the_opposite_sign(self):
        boards = [(state, {kind: [(1, .25), (0, 1.25)] for kind in analyzer.TYPES})
            for state in (7, 9)]
        dataset, metrics, _ = self.fixture(boards)
        result = analyzer.analyze(dataset, metrics)
        for kind in (*analyzer.TYPES, 'overall'):
            actual = result['overall'] if kind == 'overall' else result['types'][kind]
            self.assertEqual(actual['accuracy']['final_minus_initial'], -1)
            self.assertEqual(actual['accuracy']['change_interval_95'], [-1, -1])
            self.assertEqual(actual['log_loss']['final_minus_initial'], 1)
            self.assertEqual(actual['log_loss']['change_interval_95'], [1, 1])
            self.assertFalse(actual['accuracy']['improved_interval'])
            self.assertFalse(actual['log_loss']['improved_interval'])

    def test_missing_duplicate_and_out_of_range_pairs_are_rejected(self):
        boards = [(state, {kind: [(0, 1), (1, 0)] for kind in analyzer.TYPES})
            for state in (7, 9)]
        dataset, metrics, records = self.fixture(boards)
        for epoch in (0, 5):
            index = next(i for i, row in enumerate(records)
                if row['event'] == 'validation_row' and row['epoch'] == epoch)
            for mutation in ('missing', 'duplicate', 'wrong-index'):
                changed = [row.copy() for row in records]
                if mutation == 'missing':
                    changed.pop(index)
                elif mutation == 'duplicate':
                    changed.append(changed[index])
                else:
                    changed[index]['row'] = 6
                self.write(metrics, changed)
                with self.subTest(epoch=epoch, mutation=mutation):
                    with self.assertRaisesRegex(AssertionError, 'incomplete validation pairs'):
                        analyzer.analyze(dataset, metrics)

    def test_one_arm_and_unfinished_final_evaluation_are_rejected(self):
        boards = [(state, {kind: [(0, 1), (1, 0)] for kind in analyzer.TYPES})
            for state in (7, 9)]
        dataset, metrics, records = self.fixture(boards)
        for changed in ([row for row in records if row['epoch'] == 0], records[:-1]):
            self.write(metrics, changed)
            with self.assertRaisesRegex(AssertionError, 'initial and final validation'):
                analyzer.analyze(dataset, metrics)

    def test_one_board_has_no_paired_uncertainty(self):
        dataset, metrics, _ = self.fixture([
            (7, {kind: [(0, 1), (1, 0)] for kind in analyzer.TYPES})])
        with self.assertRaisesRegex(AssertionError, 'at least two states'):
            analyzer.analyze(dataset, metrics)


if __name__ == '__main__':
    unittest.main(verbosity=2)

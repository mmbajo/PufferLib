#!/usr/bin/env python3
"""Paired board bootstrap from native typed-supervision validation records."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import statistics

TYPES = ('choice', 'noul', 'score')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def baseline_features(row, rule):
    question = next(iter(row['questions'].values()))
    kind, target = question['type'], row['target']
    options = tuple(question['criteria']) if kind == 'choice' else tuple(range(
        2 if kind == 'noul' else len(question['criteria'])))
    if rule == 'majority':
        return (), tuple(range(len(options))), target
    if kind == 'choice':
        if rule == 'ordered_options':
            return options, tuple(range(len(options))), target
        return tuple(sorted(options)), tuple(sorted(options)), options[target]
    pattern = r'attempting (\w+) now' if kind == 'noul' else r'first moving (\w+),'
    return (re.search(pattern, question['instructions'])[1],), options, target


def board_free_baselines(training, rows):
    result = {}
    for kind in TYPES:
        result[kind] = {}
        rules = ('majority', 'ordered_options', 'directions') if kind == 'choice' else (
            'majority', 'directions')
        train, heldout = [[row for row in split
            if next(iter(row['questions'].values()))['type'] == kind] for split in (training, rows)]
        for rule in rules:
            groups, totals = defaultdict(Counter), Counter()
            for row in train:
                key, _, label = baseline_features(row, rule)
                groups[key][label] += 1
                totals[label] += 1
            correct = {}
            for row in heldout:
                key, allowed, label = baseline_features(row, rule)
                counts = groups.get(key, totals)
                prediction = max((value for value in counts if value in allowed),
                    key=counts.get, default=allowed[0])
                correct[row['state_id']] = int(prediction == label)
            result[kind][rule] = correct
    return result


def confidence_interval(samples):
    percentiles = statistics.quantiles(samples, n=40, method='inclusive')
    return [percentiles[0], percentiles[-1]]


def analyze(train, dataset, metrics):
    rows, records = map(read_rows, (dataset, metrics))
    epochs = [record['epoch'] for record in records
        if record['event'] == 'evaluation' and record['split'] == 'validation']
    assert len(epochs) == 2 and epochs[0] == 0 and epochs[1] > 0, \
        'require initial and final validation evaluations'
    arms = [sorted((record for record in records
        if record['event'] == 'validation_row' and record['epoch'] == epoch),
        key=lambda record: record['row']) for epoch in epochs]
    assert all([record['row'] for record in arm] == list(range(len(rows)))
        for arm in arms), 'incomplete validation pairs: row counts or indices differ'
    values = {kind: {} for kind in TYPES}
    for row, before, after in zip(rows, *arms):
        kind = next(iter(row['questions'].values()))['type']
        values[kind][row['state_id']] = [
            [record['correct'], record['log_loss']] for record in (before, after)]
    ids = sorted({row['state_id'] for row in rows})
    n = len(ids)
    assert n >= 2, 'paired uncertainty requires at least two states'
    baselines = board_free_baselines(read_rows(train), rows)
    kinds = (*TYPES, 'overall')
    values['overall'] = {state: [[statistics.fmean(values[kind][state][arm][metric]
        for kind in TYPES) for metric in range(2)] for arm in range(2)] for state in ids}
    changes = {}
    for kind in kinds:
        changes[kind] = {name: [values[kind][state][1][metric] - values[kind][state][0][metric]
            for state in ids] for metric, name in enumerate(('accuracy', 'log_loss'))}
        for rule, correct in baselines.get(kind, {}).items():
            changes[kind][rule] = [values[kind][state][1][0] - correct[state] for state in ids]
    samples = {kind: {name: [] for name in changes[kind]} for kind in kinds}
    rng = random.Random(73)
    for _ in range(10000):
        selected = rng.choices(range(n), k=n)
        for kind in kinds:
            for name, differences in changes[kind].items():
                samples[kind][name].append(sum(differences[i] for i in selected) / n)
    result = {}
    for kind in kinds:
        measures = {}
        for metric, name in enumerate(('accuracy', 'log_loss')):
            interval = confidence_interval(samples[kind][name])
            measures[name] = {
                'initial': statistics.fmean(values[kind][state][0][metric] for state in ids),
                'final': statistics.fmean(values[kind][state][1][metric] for state in ids),
                'final_minus_initial': statistics.fmean(changes[kind][name]),
                'change_interval_95': interval,
                'improved_interval': interval[0] > 0 if metric == 0 else interval[1] < 0,
            }
        result[kind] = {'states': n, 'questions': n if kind in TYPES else len(rows), **measures}
        if kind in baselines:
            result[kind]['baselines'] = {rule: {
                'accuracy': statistics.fmean(correct.values()),
                'final_minus_baseline': statistics.fmean(changes[kind][rule]),
                'change_interval_95': confidence_interval(samples[kind][rule]),
            } for rule, correct in baselines[kind].items()}
    return {
        'train': str(train), 'dataset': str(dataset), 'metrics_file': str(metrics),
        'initial_epoch': epochs[0], 'final_epoch': epochs[1],
        'states': n, 'questions': len(rows),
        'types': {kind: result[kind] for kind in TYPES}, 'overall': result['overall'],
        'bootstrap_resamples': 10000, 'bootstrap_seed': 73,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=Path)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('metrics', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.train, args.dataset, args.metrics)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as output:
            output.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()

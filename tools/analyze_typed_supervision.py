#!/usr/bin/env python3
"""Paired board bootstrap from native typed-supervision validation records."""
import argparse
import json
from pathlib import Path
import random
import statistics

TYPES = ('choice', 'noul', 'score')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def analyze(dataset, metrics):
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
    kinds = (*TYPES, 'overall')
    values['overall'] = {state: [[statistics.fmean(values[kind][state][arm][metric]
        for kind in TYPES) for metric in range(2)] for arm in range(2)] for state in ids}
    changes = {}
    for kind in kinds:
        changes[kind] = [[values[kind][state][1][metric] - values[kind][state][0][metric]
            for state in ids] for metric in range(2)]
    samples = {kind: [[], []] for kind in kinds}
    rng = random.Random(73)
    for _ in range(10000):
        selected = rng.choices(range(n), k=n)
        for kind in kinds:
            for metric in range(2):
                samples[kind][metric].append(sum(changes[kind][metric][i] for i in selected) / n)
    result = {}
    for kind in kinds:
        measures = {}
        for metric, name in enumerate(('accuracy', 'log_loss')):
            percentiles = statistics.quantiles(samples[kind][metric], n=40, method='inclusive')
            interval = [percentiles[0], percentiles[-1]]
            measures[name] = {
                'initial': statistics.fmean(values[kind][state][0][metric] for state in ids),
                'final': statistics.fmean(values[kind][state][1][metric] for state in ids),
                'final_minus_initial': statistics.fmean(changes[kind][metric]),
                'change_interval_95': interval,
                'improved_interval': interval[0] > 0 if metric == 0 else interval[1] < 0,
            }
        result[kind] = {'states': n, 'questions': n if kind in TYPES else len(rows), **measures}
    return {
        'dataset': str(dataset), 'metrics_file': str(metrics),
        'initial_epoch': epochs[0], 'final_epoch': epochs[1],
        'states': n, 'questions': len(rows),
        'types': {kind: result[kind] for kind in TYPES}, 'overall': result['overall'],
        'bootstrap_resamples': 10000, 'bootstrap_seed': 73,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('metrics', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.dataset, args.metrics)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as output:
            output.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Paired episode bootstrap for the native decision CartPole evaluation runner."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def load(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    protocols = [r for r in records if r.get('type') == 'protocol']
    summaries = [r for r in records if r.get('type') == 'summary']
    episodes = [r for r in records if r.get('type') == 'episode']
    if len(protocols) != 1 or len(summaries) != 1:
        raise ValueError(f'{path}: require one complete protocol and summary')
    protocol, summary = protocols[0], summaries[0]
    if len(episodes) != protocol['episodes'] or len(episodes) != summary['episodes']:
        raise ValueError(f'{path}: episode count differs from requested/completed count')
    keyed = {r['episode']: r for r in episodes}
    expected = set(range(protocol['episode_offset'], protocol['episode_offset'] + protocol['episodes']))
    if len(keyed) != len(episodes) or set(keyed) != expected:
        raise ValueError(f'{path}: duplicated or missing episode identifier')
    for episode in episodes:
        length = episode['length']
        if type(length) is not int or not 1 <= length <= protocol['max_steps'] or episode['return'] != length - 1:
            raise ValueError(f'{path}: stock CartPole length/reward accounting mismatch')
        initial = episode['initial_state']
        if not isinstance(initial, list) or len(initial) != 4 or not all(finite_number(v) for v in initial):
            raise ValueError(f'{path}: initial_state must contain four finite numbers')
        if any(type(episode[key]) is not int or episode[key] < 0 for key in ('left_actions', 'right_actions')):
            raise ValueError(f'{path}: action counts must be nonnegative integers')
        if episode['left_actions'] + episode['right_actions'] != length:
            raise ValueError(f'{path}: action count mismatch')
        if episode['environment_seed'] != protocol['environment_seed_base'] + episode['episode']:
            raise ValueError(f'{path}: environment seed does not follow frozen protocol')
        if episode['action_seed'] != protocol['action_seed_base'] + episode['episode']:
            raise ValueError(f'{path}: action seed does not follow frozen protocol')
        for field in ('return', 'max_abs_theta_observed', 'mean_probability_right'):
            if not finite_number(episode[field]):
                raise ValueError(f'{path}: nonfinite episode field {field}')
        if not 0 <= episode['mean_probability_right'] <= 1:
            raise ValueError(f'{path}: mean_probability_right must be between zero and one')
    for field, summary_field in (('length', 'mean_length'), ('return', 'mean_return')):
        actual = statistics.mean(r[field] for r in episodes)
        reported = summary[summary_field]
        if not finite_number(reported) or abs(actual - reported) > 1e-6:
            raise ValueError(f'{path}: summary {summary_field} does not match episode records')
    return protocol, summary, keyed


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def compare(before_path, after_path, resamples=10000, bootstrap_seed=606001):
    bp, _, before = load(before_path)
    ap, _, after = load(after_path)
    # Different checkpoints are intended. Every other experimental setting,
    # including inference batch, stays fixed in this primary comparison.
    if {k: v for k, v in bp.items() if k != 'checkpoint'} != {k: v for k, v in ap.items() if k != 'checkpoint'}:
        raise ValueError('evaluation protocols differ beyond checkpoint path')
    if len(before) < 2:
        raise ValueError('paired uncertainty requires at least two episodes')
    ids = sorted(before)
    for episode in ids:
        for field in ('environment_seed', 'action_seed', 'initial_state'):
            if before[episode][field] != after[episode][field]:
                raise ValueError(f'episode {episode}: unpaired {field}')
    b = [before[i]['length'] for i in ids]
    a = [after[i]['length'] for i in ids]
    delta = [new - old for old, new in zip(b, a)]
    timeout_delta = [int(after[i]['pure_timeout']) - int(before[i]['pure_timeout']) for i in ids]
    rng = random.Random(bootstrap_seed)
    intervals = {name: [] for name in ('before_mean_length', 'after_mean_length', 'mean_length_change', 'pure_timeout_rate_change')}
    n = len(ids)
    for _ in range(resamples):
        sample = rng.choices(range(n), k=n)
        intervals['before_mean_length'].append(sum(b[i] for i in sample) / n)
        intervals['after_mean_length'].append(sum(a[i] for i in sample) / n)
        intervals['mean_length_change'].append(sum(delta[i] for i in sample) / n)
        intervals['pure_timeout_rate_change'].append(sum(timeout_delta[i] for i in sample) / n)
    result = {
        'protocol': bp,
        'before_file': str(before_path), 'after_file': str(after_path),
        'before_checkpoint': bp['checkpoint'], 'after_checkpoint': ap['checkpoint'],
        'episodes': n, 'bootstrap_resamples': resamples, 'bootstrap_seed': bootstrap_seed,
        'confidence_method': 'paired episode percentile bootstrap; two-sided 95%; seeds fixed before comparison',
        'before_mean_length': statistics.mean(b), 'after_mean_length': statistics.mean(a),
        'before_mean_return': statistics.mean(before[i]['return'] for i in ids),
        'after_mean_return': statistics.mean(after[i]['return'] for i in ids),
        'mean_length_change': statistics.mean(delta),
        'paired_change_standard_error': statistics.stdev(delta) / math.sqrt(n),
        'before_pure_timeout_rate': sum(before[i]['pure_timeout'] for i in ids) / n,
        'after_pure_timeout_rate': sum(after[i]['pure_timeout'] for i in ids) / n,
        'pure_timeout_rate_change': statistics.mean(timeout_delta),
        'improved_episodes': sum(v > 0 for v in delta), 'worsened_episodes': sum(v < 0 for v in delta),
        'tied_episodes': sum(v == 0 for v in delta),
        'intervals_95': {name: [quantile(values, .025), quantile(values, .975)] for name, values in intervals.items()},
        'limitation': 'Evaluation uncertainty for this trained checkpoint and CartPole protocol; does not measure variation across training seeds or establish transfer to other tasks.',
    }
    result['positive_mean_change_interval'] = result['intervals_95']['mean_length_change'][0] > 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before')
    parser.add_argument('after')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=606001)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.resamples < 1000:
        parser.error('--resamples must be at least 1000')
    result = compare(args.before, args.after, args.resamples, args.bootstrap_seed)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()

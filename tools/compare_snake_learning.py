#!/usr/bin/env python3
"""Validate paired native Snake episodes and bootstrap the change in food eaten."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path


# Only model/execution treatments may differ in an explicitly declared ablation.
# Seeds, episode counts, game rules and masks must always remain paired.
TREATMENT_FIELDS = frozenset(('environment', 'bundle', 'model', 'representation',
    'observation_format', 'zero_init_critic', 'padded_tokens', 'bundle_max_tokens',
    'pooling', 'coordinates', 'model_width', 'model_layers', 'params',
    'temperature', 'sampling', 'sampling_mode'))
def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    protocols = [r for r in records if r.get('type') == 'protocol']
    summaries = [r for r in records if r.get('type') == 'summary']
    episodes = [r for r in records if r.get('type') == 'episode']
    require(len(protocols) == 1 and len(summaries) == 1,
            f'{path}: require one complete protocol and summary')
    protocol, summary = protocols[0], summaries[0]
    keyed = {r['episode']: r for r in episodes}
    expected = set(range(protocol['episode_offset'],
        protocol['episode_offset'] + protocol['episodes']))
    require(len(keyed) == len(episodes) and set(keyed) == expected,
            f'{path}: duplicated or missing episode identifier')
    return protocol, summary, keyed


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def compare(before_path, after_path, resamples=10000, bootstrap_seed=606001,
            allow_protocol_differences=()):
    bp, _, before = load(before_path)
    ap, _, after = load(after_path)
    allowed = set(allow_protocol_differences)
    differences = {key: {'before': bp.get(key), 'after': ap.get(key)}
        for key in sorted(bp.keys() | ap.keys()) if key != 'checkpoint' and
        (key not in bp or key not in ap or type(bp[key]) is not type(ap[key]) or bp[key] != ap[key])}
    require(set(differences) <= allowed,
            f'evaluation protocols differ beyond checkpoint path: {sorted(set(differences) - allowed)}')
    require(len(before) >= 2, 'paired uncertainty requires at least two episodes')
    ids = sorted(before)
    for episode in ids:
        for field in ('environment_seed', 'action_seed', 'initial_board', 'initial_action_mask'):
            require(before[episode][field] == after[episode][field], f'episode {episode}: unpaired {field}')
    metrics = {'food_score': 'food_score', 'return': 'return', 'length': 'length',
               'collision_rate': 'collision', 'pure_timeout_rate': 'pure_timeout',
               'full_board_rate': 'full_board'}
    samples = {}
    for name, field in metrics.items():
        prefix = 'mean_' if field in ('food_score', 'return', 'length') else ''
        b, a = [before[i][field] for i in ids], [after[i][field] for i in ids]
        samples['before_' + prefix + name] = b
        samples['after_' + prefix + name] = a
        samples[prefix + name + '_change'] = [new - old for old, new in zip(b, a)]
    n = len(ids)
    rng = random.Random(bootstrap_seed)
    intervals = {name: [] for name in samples}
    # Every metric resamples the same episode pairs; secondary metrics are
    # descriptive, not additional routes to declare primary food improvement.
    for _ in range(resamples):
        chosen = rng.choices(range(n), k=n)
        for name, values in samples.items():
            intervals[name].append(sum(values[i] for i in chosen) / n)
    primary = samples['mean_food_score_change']
    result = {'protocol': bp, 'primary_metric': 'food_score',
        'after_protocol': ap, 'declared_treatment_fields': sorted(allowed),
        'protocol_differences': differences,
        'before_file': str(before_path), 'after_file': str(after_path),
        'before_checkpoint': bp['checkpoint'], 'after_checkpoint': ap['checkpoint'],
        'episodes': n, 'bootstrap_resamples': resamples, 'bootstrap_seed': bootstrap_seed,
        'confidence_method': 'paired episode percentile bootstrap; two-sided 95%; seeds fixed before comparison',
        **{name: statistics.mean(values) for name, values in samples.items()},
        'paired_food_change_standard_error': statistics.stdev(primary) / math.sqrt(n),
        'food_improved_episodes': sum(v > 0 for v in primary),
        'food_worsened_episodes': sum(v < 0 for v in primary),
        'food_tied_episodes': sum(v == 0 for v in primary),
        'intervals_95': {name: [quantile(values, .025), quantile(values, .975)] for name, values in intervals.items()},
        'limitation': 'Evaluation uncertainty for this trained checkpoint and Snake protocol; does not measure training-seed variation, establish task transfer or correct checkpoint selection. Food eaten is the primary metric; survival alone does not establish improvement.',
    }
    result['positive_mean_food_score_change_interval'] = result['intervals_95']['mean_food_score_change'][0] > 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before')
    parser.add_argument('after')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=606001)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-protocol-difference', action='append', default=[],
                        choices=sorted(TREATMENT_FIELDS),
                        help='Explicit ablation field; repeat for each allowed difference. Seeds and rules cannot differ.')
    args = parser.parse_args()
    if args.resamples < 1000 or args.bootstrap_seed < 0:
        parser.error('require at least 1000 resamples and a nonnegative bootstrap seed')
    result = compare(args.before, args.after, args.resamples, args.bootstrap_seed,
                     args.allow_protocol_difference)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()

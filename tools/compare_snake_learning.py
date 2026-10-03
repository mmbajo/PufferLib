#!/usr/bin/env python3
"""Validate paired native Snake episodes and bootstrap the change in food eaten."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def integer(value, minimum=0, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def flag(value):
    return type(value) is bool or type(value) is int and value in (0, 1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(record, names, label):
    missing = set(names) - record.keys()
    require(not missing, f'{label}: missing fields {sorted(missing)}')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key {key}')
        result[key] = value
    return result


def load(path):
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line, object_pairs_hook=unique_object)
        require(isinstance(record, dict), f'{path}: every record must be an object')
        require(record.get('type') in ('protocol', 'episode', 'summary'),
                f'{path}: unknown or missing record type')
        records.append(record)
    protocols = [r for r in records if r['type'] == 'protocol']
    summaries = [r for r in records if r['type'] == 'summary']
    episodes = [r for r in records if r['type'] == 'episode']
    require(len(protocols) == 1 and len(summaries) == 1,
            f'{path}: require one complete protocol and summary')
    protocol, summary = protocols[0], summaries[0]
    require(records[0] is protocol and records[-1] is summary,
            f'{path}: protocol must be first and summary last')
    fields(protocol, ('version', 'environment', 'bundle', 'checkpoint', 'episodes',
        'episode_offset', 'batch', 'model_init_seed', 'environment_seed_base',
        'action_seed_base', 'sampling', 'max_steps', 'temperature', 'padded_tokens',
        'params', 'grid_size', 'action_count', 'primary_metric', 'mask_rule'), 'protocol')
    for key, expected in (('version', 1), ('environment', 'decision_laya'),
                         ('grid_size', 10), ('action_count', 4),
                         ('primary_metric', 'food_score'), ('mask_rule', 'reverse_only'),
                         ('sampling', 'puffer_philox_categorical')):
        require(protocol[key] == expected and type(protocol[key]) is type(expected),
                f'{path}: unsupported protocol {key}')
    for key in ('episodes', 'batch', 'padded_tokens', 'params'):
        require(integer(protocol[key], 1), f'{path}: protocol {key} must be a positive integer')
    require(integer(protocol['max_steps'], 1, (1 << 31) - 1),
            f'{path}: invalid max_steps')
    require(integer(protocol['episode_offset'], 0, UINT32_MAX),
            f'{path}: invalid episode offset')
    for key, maximum in (('environment_seed_base', UINT32_MAX),
                         ('action_seed_base', UINT64_MAX), ('model_init_seed', UINT64_MAX)):
        require(integer(protocol[key], 0, maximum), f'{path}: invalid protocol {key}')
    last_id = protocol['episode_offset'] + protocol['episodes'] - 1
    require(last_id <= UINT32_MAX and
            protocol['environment_seed_base'] + last_id <= UINT32_MAX and
            protocol['action_seed_base'] + last_id <= UINT64_MAX,
            f'{path}: episode or seed range overflows')
    require(isinstance(protocol['bundle'], str) and bool(protocol['bundle']) and
            isinstance(protocol['checkpoint'], str), f'{path}: invalid bundle/checkpoint path')
    require(finite_number(protocol['temperature']) and protocol['temperature'] > 0,
            f'{path}: invalid temperature')
    fields(summary, ('episodes', 'mean_food_score', 'food_score_standard_error',
        'mean_return', 'mean_length', 'collisions', 'full_boards', 'pure_timeouts',
        'evaluation_seconds'), 'summary')
    require(integer(summary['episodes'], 1) and len(episodes) == protocol['episodes'] == summary['episodes'],
            f'{path}: episode count differs from requested/completed count')
    for episode in episodes:
        fields(episode, ('episode', 'environment_seed', 'action_seed', 'initial_board',
            'initial_action_mask', 'length', 'food_score', 'return', 'terminated',
            'truncated', 'time_limit_reached', 'pure_timeout', 'outcome', 'collision',
            'full_board', 'action_counts', 'mean_action_probabilities'), 'episode')
        require(integer(episode['episode'], 0, UINT32_MAX), f'{path}: invalid episode identifier')
    keyed = {r['episode']: r for r in episodes}
    expected_ids = set(range(protocol['episode_offset'], last_id + 1))
    require(len(keyed) == len(episodes) and set(keyed) == expected_ids,
            f'{path}: duplicated or missing episode identifier')
    for episode in episodes:
        length, food = episode['length'], episode['food_score']
        require(integer(length, 1, protocol['max_steps']) and integer(food, 0, 97),
                f'{path}: invalid Snake length/food score')
        for key in ('environment_seed', 'action_seed'):
            maximum = UINT32_MAX if key == 'environment_seed' else UINT64_MAX
            require(integer(episode[key], 0, maximum) and
                    episode[key] == protocol[key + '_base'] + episode['episode'],
                    f'{path}: {key} does not follow frozen protocol')
        board = episode['initial_board']
        require(isinstance(board, list) and len(board) == 100 and
                all(type(v) is int and -1 <= v <= 3 for v in board),
                f'{path}: initial_board must contain 100 integer cells')
        require(board[55] == 1 and board[54] == 2 and board[53] == 3 and
                [board.count(v) for v in (-1, 0, 1, 2, 3)] == [1, 96, 1, 1, 1],
                f'{path}: initial_board differs from fixed Snake reset layout')
        mask = episode['initial_action_mask']
        require(isinstance(mask, list) and len(mask) == 4 and all(flag(v) for v in mask)
                and mask == [1, 1, 0, 1], f'{path}: invalid initial reverse-only action mask')
        for key in ('terminated', 'truncated', 'time_limit_reached', 'pure_timeout',
                    'collision', 'full_board'):
            require(flag(episode[key]), f'{path}: {key} must be a boolean flag')
        collision, full, timeout = (bool(episode[k]) for k in ('collision', 'full_board', 'pure_timeout'))
        require(sum((collision, full, timeout)) == 1 and
                bool(episode['terminated']) == (collision or full) and
                bool(episode['truncated']) == timeout and
                bool(episode['time_limit_reached']) == (length == protocol['max_steps']) and
                (not timeout or length == protocol['max_steps']),
                f'{path}: inconsistent termination/timeout accounting')
        require(type(episode['outcome']) is int and
                episode['outcome'] == (-1 if collision else 1 if full else 0),
                f'{path}: outcome differs from termination flags')
        require(full == (food == 97) and food <= length - int(collision),
                f'{path}: food/full-board accounting mismatch')
        require(finite_number(episode['return']) and episode['return'] == food - int(collision),
                f'{path}: Snake food/reward accounting mismatch')
        counts = episode['action_counts']
        require(isinstance(counts, list) and len(counts) == 4 and
                all(integer(v) for v in counts) and sum(counts) == length,
                f'{path}: invalid action counts')
        probabilities = episode['mean_action_probabilities']
        require(isinstance(probabilities, list) and len(probabilities) == 4 and
                all(finite_number(v) and 0 <= v <= 1 for v in probabilities) and
                abs(sum(probabilities) - 1) <= 1e-6,
                f'{path}: invalid mean_action_probabilities')
    for field, summary_field in (('food_score', 'mean_food_score'), ('return', 'mean_return'),
                                 ('length', 'mean_length')):
        actual, reported = statistics.mean(r[field] for r in episodes), summary[summary_field]
        require(finite_number(reported) and abs(actual - reported) <= 1e-6,
                f'{path}: summary {summary_field} does not match episode records')
    expected_se = statistics.stdev(r['food_score'] for r in episodes) / math.sqrt(len(episodes)) if len(episodes) > 1 else 0
    require(finite_number(summary['food_score_standard_error']) and
            abs(expected_se - summary['food_score_standard_error']) <= 1e-6,
            f'{path}: summary food_score_standard_error does not match episode records')
    for flag_name, summary_name in (('collision', 'collisions'), ('full_board', 'full_boards'),
                                   ('pure_timeout', 'pure_timeouts')):
        require(integer(summary[summary_name]) and
                summary[summary_name] == sum(r[flag_name] for r in episodes),
                f'{path}: summary {summary_name} does not match episode records')
    require(finite_number(summary['evaluation_seconds']) and summary['evaluation_seconds'] >= 0,
            f'{path}: invalid evaluation_seconds')
    return protocol, summary, keyed


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def compare(before_path, after_path, resamples=10000, bootstrap_seed=606001):
    require(integer(resamples, 1000), 'bootstrap resamples must be at least 1000')
    require(integer(bootstrap_seed), 'bootstrap seed must be a nonnegative integer')
    bp, _, before = load(before_path)
    ap, _, after = load(after_path)
    require({k: v for k, v in bp.items() if k != 'checkpoint'} ==
            {k: v for k, v in ap.items() if k != 'checkpoint'},
            'evaluation protocols differ beyond checkpoint path')
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
    args = parser.parse_args()
    if args.resamples < 1000 or args.bootstrap_seed < 0:
        parser.error('require at least 1000 resamples and a nonnegative bootstrap seed')
    result = compare(args.before, args.after, args.resamples, args.bootstrap_seed)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()

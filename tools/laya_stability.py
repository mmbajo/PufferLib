#!/usr/bin/env python3
"""Prepare Laya warm starts and compare completed Snake evaluations."""
import argparse
import json
from pathlib import Path

from compare_snake_learning import compare

SEEDS = (73, 100073, 200073)
RATES = ('0.0001', '0.00001')
EPISODES = 512
STEPS = 8192
EVAL_SEED = 73
ENV_SEED = 6140001
ACTION_SEED = 6240001


def options(values):
    return [f'--{key}={value}' for key, value in values.items()]


def evaluation_command(evaluator, cwd, bundle, weights, root, name):
    settings = {
        'weights': weights, 'policy.bundle': bundle, 'policy.zero_init_critic': 0,
        'policy.sequence_length': 384, 'policy.gemm_precision': 'fp32',
        'env.observation_format': 1, 'base.seed': EVAL_SEED, 'base.gpu_offset': 0,
        'env.max_steps': 500, 'episodes': EPISODES, 'batch': 16,
        'env-seed': ENV_SEED, 'action-seed': ACTION_SEED, 'sampling': 'sampled',
    }
    return {'argv': [str(evaluator), *options(settings)], 'cwd': str(cwd),
        'stdout': str(root / 'evaluation' / f'{name}.jsonl'),
        'stderr': str(root / 'evaluation' / f'{name}.stderr')}


def prepare(cwd, trainer, evaluator, bundle, initial_weights, output_dir,
    training_precision='fp32'):
    cwd, trainer, evaluator, bundle, initial_weights, root = (
        Path(p).resolve() for p in (cwd, trainer, evaluator, bundle, initial_weights, output_dir))
    root.mkdir(parents=True)
    for name in ('evaluation', 'training', 'logs', 'checkpoints'):
        (root / name).mkdir()
    trials = []
    for rate in RATES:
        for seed in SEEDS:
            name = f'lr{rate}-seed{seed}'
            final = root / 'checkpoints/decision_laya' / name / f'{STEPS:016d}.bin'
            (root / 'training' / name).mkdir()
            settings = {
                'policy.bundle': str(bundle), 'policy.zero_init_critic': 0,
                'policy.sequence_length': 384, 'policy.gemm_precision': training_precision,
                'env.observation_format': 1,
                'base.seed': seed, 'base.async': 0, 'base.cudagraphs': -1,
                'base.gpu_offset': 0, 'base.load_model_path': str(initial_weights),
                'base.resume_path': 'None', 'base.save_training_state': 0,
                'base.stop_after_steps': 0, 'base.profile': 0,
                'base.metrics_path': str(root / 'training' / name / 'metrics.jsonl'),
                'train.gpus': 4, 'train.optimizer': 'muon', 'train.distributed_optimizer': 1,
                'vec.total_agents': 4, 'vec.num_buffers': 1, 'vec.num_threads': 4,
                'vec.num_policies': 1, 'selfplay.enabled': 0, 'env.max_steps': 500,
                'train.horizon': 16, 'train.minibatch_size': 16,
                'train.total_timesteps': STEPS, 'train.learning_rate': float(rate),
                'train.anneal_lr': 0, 'train.replay_ratio': 1, 'train.gamma': .99,
                'train.gae_lambda': .95, 'train.ent_coef': .01, 'train.anneal_ent_coef': 0,
                'train.clip_coef': .2, 'train.vf_coef': 2, 'train.vf_clip_coef': .2,
                'train.max_grad_norm': 1.5, 'train.momentum': .95, 'train.vtrace': 0,
                'base.eval_episodes': 0, 'base.checkpoint_interval': 32,
                'base.log_dir': str(root / 'logs'),
                'base.checkpoint_dir': str(root / 'checkpoints'), 'base.run_id': name,
            }
            trials.append({'id': name, 'learning_rate': rate, 'training_seed': seed,
                'settings': settings, 'final_weights': str(final),
                'training': {'argv': [str(trainer), 'train', '--headless', *options(settings)],
                    'cwd': str(cwd), 'stdout': str(root / 'training' / f'{name}.stdout'),
                    'stderr': str(root / 'training' / f'{name}.stderr')},
                'evaluation': evaluation_command(evaluator, cwd, bundle, final, root, name)})
    manifest = {'training_precision': training_precision,
        'initial_evaluation': evaluation_command(
            evaluator, cwd, bundle, initial_weights, root, 'initial'),
        'trials': trials}
    (root / 'protocol.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def analyze(paths):
    protocols, results = {}, {}
    for path in paths:
        manifest = json.loads(Path(path).read_text())
        precision = manifest['training_precision']
        assert precision not in protocols, 'provide only one protocol per training precision'
        trials = {(t['learning_rate'], t['training_seed']): t for t in manifest['trials']}
        protocols[precision] = trials
        comparisons = {}
        for key, trial in trials.items():
            metrics_path = trial['settings']['base.metrics_path']
            records = [json.loads(line) for line in Path(metrics_path).read_text().splitlines()]
            assert len(records) >= 2 and records[0]['event'] == 'run', metrics_path
            assert all(row['event'] == 'metrics' for row in records[1:]), metrics_path
            final = records[-1]['metrics']
            assert final['epoch'] == 32 and final['agent_steps'] == STEPS, metrics_path
            paired = compare(
                manifest['initial_evaluation']['stdout'], trial['evaluation']['stdout'])
            assert paired['episodes'] == EPISODES, 'require all 512 evaluation episodes'
            comparisons[key] = paired
            trial['training_metrics'] = {'logged_events': len(records) - 1, 'final': final}
        rate_changes = {str(seed): compare(trials[RATES[0], seed]['evaluation']['stdout'],
            trials[RATES[1], seed]['evaluation']['stdout']) for seed in SEEDS}
        results[precision] = {
            'trials': [{'trial': trial['id'], 'initial_to_final': comparisons[key],
                'training_metrics': trial['training_metrics']} for key, trial in trials.items()],
            'lower_minus_higher_rate': rate_changes,
            'lower_rate_preserved_all_three': all(
                comparisons[RATES[1], seed]['intervals_95']['mean_food_score_change'][0] >= -.5
                for seed in SEEDS),
            'improved_all_three_by_rate': {rate: all(
                comparisons[rate, seed]['positive_mean_food_score_change_interval']
                for seed in SEEDS)
                for rate in RATES},
            'lower_rate_advantage_all_three': all(
                row['positive_mean_food_score_change_interval'] for row in rate_changes.values()),
        }
    output = {'precisions': results}
    if 'fp32' in protocols and 'tf32' in protocols:
        precision_changes = {}
        for rate in RATES:
            contrasts = {str(seed): compare(
                protocols['fp32'][rate, seed]['evaluation']['stdout'],
                protocols['tf32'][rate, seed]['evaluation']['stdout']) for seed in SEEDS}
            precision_changes[rate] = {
                'tf32_minus_fp32': contrasts,
                'preserved_all_three': all(
                    row['intervals_95']['mean_food_score_change'][0] >= -.5
                    for row in contrasts.values()),
                'improved_all_three': all(
                    row['positive_mean_food_score_change_interval'] for row in contrasts.values()),
            }
        output['precision_comparisons'] = precision_changes
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='write commands without executing them')
    for name in ('working-directory', 'trainer', 'evaluator', 'bundle',
        'initial-weights', 'output-dir'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--training-precision', choices=('fp32', 'tf32'), default='fp32')
    analysis = sub.add_parser('analyze', help='compare one or both completed precision protocols')
    analysis.add_argument('protocol', type=Path, nargs='+')
    analysis.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare(args.working_directory, args.trainer, args.evaluator, args.bundle,
                args.initial_weights, args.output_dir, args.training_precision)
            print(f'Prepared {len(result["trials"])} trials: {args.output_dir / "protocol.json"}')
        else:
            result = analyze(args.protocol)
            with args.output.open('x') as output:
                json.dump(result, output, indent=2, allow_nan=False)
                output.write('\n')
            print(f'Analysis written: {args.output}')
    except (AssertionError, ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()

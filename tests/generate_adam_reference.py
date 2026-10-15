#!/usr/bin/env python3
"""Write real CPU torch.optim.Adam updates for the native CUDA optimizer test."""
import argparse
import json
from pathlib import Path

import torch


def generate(destination):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(173)
    cases = [
        ('default', .9, .999, 1e-8, 0., 1.5),
        ('coupled_decay', .9, .999, 1e-8, .08, .2),
        ('nondefault', .7, .93, .003, .05, .1),
        ('zero_betas', 0., 0., .03, .1, .4),
        ('unclipped', .9, .999, 1e-8, 0., 1e20),
    ]
    lines = ['ADAMREF1', str(len(cases))]
    steps = 12
    for name, beta1, beta2, eps, decay, norm in cases:
        # Native registry: an 8-element matrix and a 3-element bias padded to 4.
        # Torch has only the 11 live values; padding must stay exactly zero.
        initial = torch.linspace(-.6, .7, 11, dtype=torch.float32)
        params = [torch.nn.Parameter(initial[:8].clone().reshape(2, 4)),
                  torch.nn.Parameter(initial[8:].clone())]
        optimizer = torch.optim.Adam(params, lr=.001, betas=(beta1, beta2),
                                     eps=eps, weight_decay=decay, foreach=False)
        lines.append(f'{name} 12 {steps} {beta1:.17g} {beta2:.17g} {eps:.17g} {decay:.17g} {norm:.17g}')

        def write(values):
            lines.append(' '.join(f'{float(x):.17g}' for x in values))

        write([*initial.tolist(), 0.])
        for step in range(steps):
            gradient = torch.randn(11, generator=generator) * .7
            if step in (0, 2):
                gradient.zero_()
            elif step == 3:
                gradient = torch.linspace(1., -1., 11)
            elif step == 4:
                gradient *= 1e8
            elif step == 5:
                gradient *= 1e-12
            gradient[2] = 0.
            # Include scheduled LR changes; the native optimizer reads its
            # device scalar at every execution rather than caching the value.
            lr = .001 * (1. if step < 6 else .4)
            # Native stores LR in FP32 before launch. Match that exact input.
            lr = float(torch.tensor(lr, dtype=torch.float32))
            optimizer.param_groups[0]['lr'] = lr
            params[0].grad = gradient[:8].clone().reshape(2, 4)
            params[1].grad = gradient[8:].clone()
            lines.append(f'{lr:.17g}')
            write([*gradient.tolist(), 0.])
            torch.nn.utils.clip_grad_norm_(params, norm, foreach=False)
            optimizer.step()
            for key in (None, 'exp_avg', 'exp_avg_sq'):
                values = torch.cat([(p.detach() if key is None else optimizer.state[p][key]).flatten()
                                    for p in params])
                write([*values.tolist(), 0.])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines) + '\n')
    metadata = {'reference': 'CPU torch.optim.Adam + torch.nn.utils.clip_grad_norm_',
                'torch_version': torch.__version__, 'cases': [case[0] for case in cases],
                'steps_per_case': steps, 'live_values': 11, 'flat_slots': 12,
                'weight_decay': 'coupled L2, applied after clipping',
                'covers': ['bias correction', 'zero gradients before/after momentum',
                           'large/tiny gradients', 'clipping', 'zero betas',
                           'nondefault epsilon and betas', 'LR changes', 'zero padding']}
    destination.with_suffix(destination.suffix + '.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    generate(parser.parse_args().output)

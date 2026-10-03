[![Discord](https://dcbadge.limes.pink/api/server/puffer?style=plastic)](https://discord.gg/puffer)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez)](https://twitter.com/jsuarez)

PufferLib is a fast and sane reinforcement learning library that can train tiny, super-human models in seconds. The included learning algorithm, hyperparameter tuning, and simulation methods are the product of our own research. All our tools are free and open source. Need a high performance environment for your application? We build them professionally and offer training + extended support. Contact jsuarez🐡puffer🐡ai.

All of our documentation is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support. Post there before opening issues. We're always looking for new contributors!

For native Transformer policies, see [benchmark Snake training](ocean/decision_snake/README.md)
and [importing Laya/BERT/ModernBERT models and tokenizers](ocean/decision_laya/README.md).
[CartPole and the reusable decision-policy adapter](ocean/decision_cartpole/README.md)
show how to train the same pretrained models in another native environment.
Additional text decision environments are [Connect Four](ocean/decision_connect4/README.md),
[Lights Out](ocean/decision_lightsout/README.md) and [2048](ocean/decision_2048/README.md).
Use the [native training benchmark](tools/benchmark_native.md) to measure rollout
and learner throughput separately from startup, evaluation and checkpoint I/O.
The [distributed training guide](tools/distributed_training.md) covers multi-GPU
data parallelism, optional Muon state sharding, and current larger-model limits.
Use [paired CartPole evaluation](tools/evaluate_decision_learning.md) to compare
policy quality before and after training on explicitly matched episode seeds.

## Star to puff up the project!

<a href="https://star-history.com/#pufferai/pufferlib&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
 </picture>
</a>

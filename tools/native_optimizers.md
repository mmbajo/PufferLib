# Native optimizer selection

Native training defaults to the existing Muon optimizer. Select Adam explicitly
and set its learning rate; selecting Adam does not silently change any PPO
setting or the inherited learning rate.

```sh
./build.sh decision_snake build/puffer_snake
./build/puffer_snake train --headless \
  --train.optimizer=adam --train.learning_rate=0.001 --train.anneal_lr=0 \
  --train.adam_beta1=0.9 --train.adam_beta2=0.999 \
  --train.adam_eps=1e-8 --train.adam_weight_decay=0 \
  --train.distributed_optimizer=0
```

For example, the same training command without `--train.optimizer` uses Muon;
adding `--train.optimizer=adam` uses Adam at the command's existing learning rate.
The default Muon rate of `0.015` is preserved, so an Adam comparison should always
state its rate explicitly. Comparisons using different rates compare complete
optimizer recipes, not just the optimizer algorithm.

| Setting | Default | Accepted values |
| --- | --- | --- |
| `train.optimizer` | `muon` | `muon`, `adam` |
| `train.adam_beta1` | `0.9` | finite number in `[0, 1)` |
| `train.adam_beta2` | `0.999` | finite number in `[0, 1)` |
| `train.adam_eps` | `1e-8` | positive finite FP32 value |
| `train.adam_weight_decay` | `0` | nonnegative finite FP32 value |

Adam follows `torch.optim.Adam` with ordinary bias correction and **coupled L2
weight decay**, not AdamW. The native trainer first averages gradients across
data-parallel ranks, then clips their global L2 norm with the existing
`train.max_grad_norm` rule. Adam adds the weight-decay contribution after clipping,
updates its two FP32 moments, and updates the FP32 weights. All registered flat
parameter slots participate; zero padding remains zero when its weights and
gradients are zero. A zero gradient still advances the moments and update counter.

Adam currently requires FP32 and rejects `train.distributed_optimizer=1`.
Data-parallel Adam keeps weights, gradients and both moments replicated on every
rank. Muon's optional whole-matrix optimizer sharding remains available only for
Muon. No model, tensor or context parallelism is introduced here.

The benchmark reports the selected `optimizer`. Its existing
`optimizer_momentum_bytes` fields count both Adam moment tensors, or Muon's one
momentum tensor. Adam's persistent optimizer state for continuation consists of
`first_moment`, `second_moment`, the 64-bit device `step`, and the device learning
rate, with the beta/epsilon/decay settings as metadata. A flat model-weight file
alone does not contain that optimizer state. Scratch reductions and derived
update scalars are recomputed and need not be restored.

Both Snake adapters support [full training checkpoints](native_training_checkpoints.md)
with Adam as well as Muon. Set `base.save_training_state=1` and later use
`base.resume_path` with the same optimizer and settings. Adam restoration validates
that its step agrees with the completed epochs and that second moments are
nonnegative. Model-only warm starts intentionally begin with fresh moments.

## Validation

Generate reference updates with an installed CPU PyTorch, then build and run the
native test on a CUDA GPU:

```sh
python tests/generate_adam_reference.py build/adam-reference.txt
tests/build_native_adam_test.sh
build/test_native_adam build/adam-reference.txt
# Optional: two visible GPUs, one process per GPU, distinct averaged gradients.
build/test_native_adam build/adam-reference.txt --two-gpus
```

The fixture compares weights and both moments at every update, including bias
correction, clipping, coupled decay, nondefault betas/epsilon, zero and large
gradients, learning-rate changes, and padded slots. Single-GPU tests also replay a
CUDA graph to verify the device step advances on each execution. Malformed
optimizer controls and the unchanged default are checked without a GPU via
`build/test_native_adam --host-only`. Floating-point moment comparisons allow
small reduction and arithmetic rounding differences from CPU PyTorch.

`tests/build_native_optimizer_dispatch_test.sh` builds a separate board-policy
check: default Muon, explicit Muon and Adam initialization must be byte-identical;
Adam dispatch must advance its own counter; and four fixed-gradient updates must
reproduce direct Muon weights and momentum exactly.
This isolates optimizer dispatch from the existing model backward reductions,
whose atomic additions can make complete training trajectories non-bitwise
deterministic even when repeating an unchanged binary and seed.

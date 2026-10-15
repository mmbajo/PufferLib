# Native training metrics

Set `base.metrics_path` to save the existing native training logging events as
full-precision JSONL. The default is `None` (disabled).

```sh
mkdir -p build/experiments/snake
./build/puffer_snake train --headless \
  --base.metrics_path=build/experiments/snake/run-73.jsonl
```

Before: the dashboard prints losses to three decimal places, and the final INI
omits `loss/*` metrics. An illustrative KL of `0.0000000012345678901234567`
appears as `0.000`. With JSONL enabled, that KL is retained as
`1.2345678901234566e-09` (the round-trippable double representation), alongside
entropy, critic loss, policy loss, clip fraction, environment metrics, and timing.

Only the artifact-owning rank writes. The path is created exclusively before
model allocation; an existing file or symlink is rejected rather than overwritten
or appended to. The parent directory must exist. Each fresh training invocation,
including a checkpoint resume, needs a new metrics path. This path is not part of
the full-checkpoint configuration identity.

The first record has `event: "run"`, schema `puffer.training_metrics.v1`, world
size, and every effective INI section, including `selfplay`. Configured text
values remain strings; `policy.gemm_precision` records the requested GEMM mode.
Retain the code branch and commit, executable path, and model/checkpoint paths
alongside it.

Each subsequent `event: "metrics"` record contains the full numeric dictionary
from that logging event. Its `epoch`, `agent_steps` (global across ranks), and
`uptime` fields are the dashboard's counters. Floating-point values use 17
significant digits. Records include events with zero completed episodes;
unavailable episode statistics are omitted rather than copied from an older
event. Loss values remain averages over updates since the previous logging
event, reduced across ranks, not individual minibatch measurements. The cadence
is the trainer's existing logging cadence.

Nonfinite numbers are JSON `null`; downstream analysis must not interpret these
as healthy zero-valued metrics. Recording a nonfinite value does not itself stop
training. File creation, write, flush, or close failures fail the invocation.

`sweep` rejects a configured `base.metrics_path` before starting any children.
Use separate training invocations with a distinct metrics path for each run.

The writer flushes after every record. This adds synchronous host I/O when
enabled, so compare throughput with the same logging setting on every arm.
There are no additional GPU reductions or device reads. A killed or failed run
may leave a final incomplete line; it has no completion marker. Retain its
complete records and check the process exit status before treating a run as
successful. JSONL files are not full training checkpoints and are not fsynced.

Run the CPU-only serialization and file-safety checks with:

```sh
python3 -m unittest tests.test_training_metrics -v
```

The tests build inside a temporary directory under `build/` and remove that
directory and all its contents when finished.

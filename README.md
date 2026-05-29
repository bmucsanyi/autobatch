# autobatch

Picking a batch size by hand is tedious. You guess a number, hit a CUDA out-of-memory error, lower it, try again, and eventually settle on something that probably leaves performance on the table. `autobatch` does that search for you in virtually any CUDA setup you could think of.

`autobatch` tunes a single integer hyperparameter for a PyTorch CUDA workload. That integer does not have to be a batch size. It can be a chunk size, tile size, beam width, candidate count, token block count, or anything else, as long as larger values never become feasible again once they stop fitting in memory.

There is one function to learn: `autobatch.find`. It returns the tuned value as an integer.

## Single process

Give `find` a probe that runs your workload at a given value, the range of values to consider, and a goal:

```python
import autobatch

def probe(batch_size: int) -> None:
    train_step(batch_size)

value = autobatch.find(
    probe,
    values=range(1, 257),
    goal=autobatch.Goal.largest_safe(),
    cache_key=("train-step", model_name, dataset_name),
    warmup_steps=1,
    measure_steps=3,
    devices=[0],
)
```

`autobatch` runs the probe in your current process, catches and classifies CUDA OOM errors, records peak allocated and reserved memory, and caches memory-only
goal results so later calls in the same process skip the search.

A few things to keep in mind about the cache key. It must be a hashable JSON scalar, or a tuple of such scalars. Put everything that changes how the probe behaves into it, because anything you leave out risks getting a stale answer. Memory-only goals revalidate a cache hit before trusting it; timing goals ignore cache entries and scan the whole declared range.

## Candidate tables

The integer can also be an index into an explicit table of workload settings. This is useful when one candidate changes several knobs at once:

```python
import autobatch


candidates = {
    1: {"token_block": 64, "attention_block": 1_000_000},
    2: {"token_block": 64, "attention_block": 2_000_000},
    3: {"token_block": 128, "attention_block": 1_000_000},
}


def probe(value: int) -> None:
    run_step(**candidates[value])


value = autobatch.find(
    probe,
    values=tuple(candidates),
    goal=autobatch.Goal.fastest_step(),
    cache_key=("candidate-table", model_name, dataset_name),
    warmup_steps=1,
    measure_steps=3,
    devices=[0],
)
chosen = candidates[value]
```

For `Goal.fastest_step()`, `autobatch` measures every declared value and skips values that OOM, so the table can mix settings whose memory use is not ordered by index. For binary-search goals such as `Goal.largest_safe()`, order the table so feasibility is monotone.

## Distributed

For data-parallel runs, every rank calls `find` with the same domain, goal, cache key, step counts, and number of monitored devices. `autobatch` checks that agreement before it starts probing, and checks the selected value again before returning it, so a misconfigured rank fails loudly instead of quietly diverging.

Distributed runs need a `StagedProbe`:

```python
import torch
import torch.distributed as dist
import autobatch

def local(batch_size: int) -> None:
    forward_backward_without_collectives(batch_size)

def sync(batch_size: int, active: bool) -> None:
    payload = current_gradient_chunk() if active else placeholder_chunk
    dist.all_reduce(payload)

value = autobatch.find(
    autobatch.StagedProbe([autobatch.ProbeStage(local, sync)]),
    values=range(1, 257),
    goal=autobatch.Goal.largest_safe(),
    cache_key=("train-step", model_name, dataset_name),
    warmup_steps=1,
    measure_steps=3,
    devices=[local_cuda_device],
)
```

Each stage splits into a local phase and a sync phase. The split exists because collectives have to stay in lockstep across ranks even when one rank just ran out of memory. `autobatch` reduces the local verdict before the sync phase, so every rank reaches every sync phase for every candidate. When some rank OOMs, `active` arrives as `False` on all ranks, and the sync phase still has to issue the same collectives, just over placeholder tensors instead of real ones.

The rule for splitting your workload is: anything that allocates memory or runs compute that can OOM goes in `local`; `sync` only runs the planned collectives over buffers or placeholders that already exist. Control collectives go through a CPU Gloo group, even when your data-parallel backend is NCCL. Every rank must declare the same number of stages.

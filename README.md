# autobatch

`autobatch` tunes an integer hyperparameter for a PyTorch CUDA workload. The setting can be a
batch size, chunk size, tile size, beam width, candidate count, token block count, or any
other positive integer whose CUDA memory feasibility is monotone.

`autobatch.find` is the single entrypoint that returns the tuned hyperparameter as an integer. Example use:

```python
import autobatch

value = autobatch.find(
    workload="my_project.probes:step",
    values=range(1, 257),
    goal=autobatch.Goal.largest_safe(),
    kwargs={},
    reserve_fraction=0.05,
    reserve_bytes=0,
    warmup_steps=1,
    measure_steps=3,
    timeout_s=180.0,
    devices=[0],
    cache_dir=".autobatch-cache",
)
```

The package runs each probe in a separate worker process, classifies CUDA OOM,
records PyTorch peak allocated and reserved memory, validates memory headroom, and caches tuned values.

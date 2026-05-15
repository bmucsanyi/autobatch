# autobatch

`autobatch` finds one integer CUDA setting for a PyTorch workload. The setting can be a
batch size, chunk size, tile size, beam width, candidate count, token block count, or any
other positive integer whose CUDA memory feasibility is monotone over the declared domain.

The public success path returns one integer:

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

The package runs each probe in a worker process, classifies CUDA OOM inside the package,
records PyTorch peak allocated and reserved memory, validates memory headroom, and
revalidates cache hits before returning them.

Timed goals evaluate the declared finite domain exactly.

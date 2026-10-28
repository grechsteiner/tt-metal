# Op support matrix

Finds which memory configurations an op supports at the shapes a model actually uses, and why the others fail. The first op covered is `ttnn.add` for the MoonViT-V2 residual add, `[1, 1, 16384, 1024]` bf16 per chip.

The test is `tests/ttnn/nightly/unit_tests/operations/eltwise/test_add_support_matrix.py`. Its vectors and `run()` live in the sweep module `tests/sweep_framework/sweeps/eltwise/binary/add/add_support_matrix.py`. Every combination of input A, input B and output memory config is run in TILE and ROW_MAJOR layout, and each output is compared element by element with `torch.add`; a pass is every element within 1 ULP.

Sharded configs follow two rules, in order: every core or DRAM bank used holds the same amount of data, and as many cores as possible are used, on a fixed 12x10 grid.

## Running on the Blackhole simulator

```bash
export TT_METAL_SIMULATOR=/path/to/libttsim.so
tests/sweep_framework/support_matrix/run_all.sh 32
```

This builds a runtime root that gives ttsim a 12x10 grid (`make_runtime_root.sh`), runs the vectors as 32 parallel pytest processes, and writes `runs/add_explorer.html`, a page for browsing the results.

| File | What it does |
|---|---|
| `run_matrix.py` | Runs one shard of the vectors. Records each result as it finishes, kills a pytest process that produces nothing for 60 s (a device hang), marks that vector as hung and carries on. Safe to restart. |
| `results_plugin.py` | pytest plugin that writes one JSON line per finished test. |
| `render_page.py`, `explorer_template.html` | Turn the results into the explorer page. |
| `make_runtime_root.sh` | Mirror of the repo whose Blackhole core descriptor exposes a 12x10 compute grid to the simulator. |
| `results/add_explorer_full_12x10.html` | Results from the run on 27 Oct 2026, ttsim v1.14.0. |

# Op support matrix

Finds which memory configurations each op of a model supports at the shapes the model actually uses, and why the others fail. It covers the MoonViT-V2 encoder block (Kimi-K3), per chip with a sequence of 16384 rows, in bf16:

| Group | Op | Shapes | Sweep module |
|---|---|---|---|
| `rms_norm` | `ttnn.rms_norm` (norm0, norm1) | `[M,1024]`, weight `[1,1,32,32]` row-major | `sweeps/normalization/rms_norm/rms_norm_support_matrix.py` |
| `linear_wqkv` | `ttnn.linear` | `[M,1024] × [1024,4608]` | `sweeps/matmul/support_matrix/linear_support_matrix.py` |
| `concat_heads` | `ttnn.experimental.nlp_concat_heads` | `[1,12,M,128] → [M,1536]` | `sweeps/transformer/nlp_concat_heads/nlp_concat_heads_support_matrix.py` |
| `linear_wo` | `ttnn.linear` | `[M,1536] × [1536,1024]` | `sweeps/matmul/support_matrix/linear_support_matrix.py` |
| `add` | `ttnn.add` (residuals) | `[M,1024] + [M,1024]` | `sweeps/eltwise/binary/add/add_support_matrix.py` |
| `linear_fc0` | `ttnn.linear` | `[M,1024] × [1024,4096]` | `sweeps/matmul/support_matrix/linear_support_matrix.py` |
| `gelu` | `ttnn.gelu` (tanh) | `[M,4096]` | `sweeps/eltwise/unary/gelu/gelu_support_matrix.py` |
| `linear_fc1` | `ttnn.linear` | `[M,4096] × [4096,1024]` | `sweeps/matmul/support_matrix/linear_support_matrix.py` |

Each group runs every combination of memory config for each input and the output (DRAM and L1 interleaved; L1 height, width, block and ND sharded; DRAM width and ND sharded; plus the op's own default output), in TILE and, where the op has one, ROW_MAJOR layout. `harness.py` builds the configs, runs a vector and judges it; `groups.py` lists the groups. The pytest entry point is `tests/ttnn/nightly/unit_tests/operations/test_op_support_matrix.py`.

Sharded configs follow three rules, in order: every core or DRAM bank used holds the same amount of data; as many cores as possible are used, on a fixed 12x10 grid; and the cores form a rectangle when one of that size fits.

A vector passes when its output matches its reference: torch within 1 ULP for add, exactly for concat_heads, and the op's own DRAM-interleaved TILE output (what the model runs today) for gelu (bit for bit), rms_norm and linear (within a small tolerance, since sharded kernels may reduce in a different order).

## Running on the Blackhole simulator

```bash
export TT_METAL_SIMULATOR=/path/to/libttsim.so
tests/sweep_framework/support_matrix/start_tmux.sh support-matrix 40 /path/to/out   # all groups
tests/sweep_framework/support_matrix/start_tmux.sh support-matrix 40 /path/to/out add,gelu
tmux attach -t support-matrix    # watch; detach with Ctrl-b d
```

`start_tmux.sh` runs `run_all.sh` in a detached tmux session, so it survives logging out. `run_all.sh` builds a runtime root that gives ttsim a 12x10 grid (`make_runtime_root.sh`), runs the vectors as N parallel pytest processes, and writes `support_explorer.html`, a page for browsing the results. Re-running into the same directory only runs vectors that have no result yet.

| File | What it does |
|---|---|
| `harness.py` | Memory configs, the run loop with tagged failures, comparisons and per-tensor L1 footprints. |
| `groups.py` | Every group, in encoder-block order. |
| `run_matrix.py` | Writes the vector ids, and runs one shard of them. Records each result as it finishes. Kills a pytest process that produces nothing for 120 s (40 min for linear groups, whose vectors take minutes on the simulator), marks that vector as hung and carries on. Safe to restart. |
| `results_plugin.py` | pytest plugin that writes one JSON line per finished test. |
| `run_all.sh`, `start_tmux.sh` | Run everything, in or outside tmux. |
| `render_page.py`, `explorer_template.html` | Turn the results into the explorer page. |
| `make_runtime_root.sh` | Mirror of the repo whose Blackhole core descriptor exposes a 12x10 compute grid to the simulator. |
| `results/add_explorer_full_12x10.html` | add-only results from the run on 27 Oct 2026, ttsim v1.14.0. |

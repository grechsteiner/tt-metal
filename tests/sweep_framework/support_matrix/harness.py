# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared pieces of the op support matrix.

An OpGroup is one op at one model signature (for example ttnn.add on the MoonViT-V2 residual
shape). Its vectors are every combination of tensor layout and memory config for each input and
the output. run_vector() runs one vector and compares the output element by element with torch.

Shards follow three rules, in order: every core (or DRAM bank) used holds the same amount of data;
as many cores as possible are used, on a fixed 12x10 (120-core Blackhole) grid; and the cores form a
rectangle when one of that size fits, since sharded program configs are described by a grid.
Interleaved configs are left to the allocator, which spreads pages over every bank.

Failure messages are tagged so results can be split by cause:
  [SETUP]         an input could not be built, did not land in the requested config, or did not read
                  back equal to its torch source -- the op never ran (or never got the right data)
  [WRONG_RESULT]  the op ran but is more than the group's ULP threshold from torch
  [WRONG_CONFIG]  the op ran and was correct, but the output is not in the requested config or layout
  (untagged)      the op itself raised -- usually a TT_FATAL from validation, or L1 out of memory
"""

import fcntl
import itertools
import math
import os
import random
from dataclasses import dataclass, field
from typing import Callable

import torch
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc, comp_ulp, start_measuring_time, stop_measuring_time

SHARD_GRID = ttnn.CoreCoord(12, 10)

MEMORY_CONFIGS = [
    "dram_interleaved",
    "l1_interleaved",
    "l1_height_sharded",
    "l1_width_sharded",
    "l1_block_sharded",
    "l1_nd_sharded",
    "dram_width_sharded",
    "dram_nd_sharded",
]
# "default" passes memory_config=None and lets the op pick the output config.
OUTPUT_CONFIGS = ["default"] + MEMORY_CONFIGS

# ND shards are 256 x 256 blocks (smaller where the tensor is) dealt round-robin to the cores, so each core holds several blocks
# from different parts of the tensor instead of one contiguous piece.
ND_SHARD_HW = (256, 256)

LAYOUTS = {"tile": ttnn.TILE_LAYOUT, "row_major": ttnn.ROW_MAJOR_LAYOUT}
BYTES_PER_ELEMENT = 2  # every tensor here is bfloat16


@dataclass
class Operand:
    """One input of the op. `layout` is "follow" to use the vector's layout, or a fixed layout name."""

    role: str
    label: str
    shape: list
    configs: list = field(default_factory=lambda: list(MEMORY_CONFIGS))
    layout: str = "follow"
    low: float = -100.0
    high: float = 100.0


@dataclass
class OpGroup:
    name: str
    title: str
    operands: list
    output_shape: list
    # (inputs: dict of role -> ttnn.Tensor, memory_config, ctx) -> ttnn.Tensor, where ctx holds
    # "configs" (role -> the input's MemoryConfig) and "layout", for ops whose program config must
    # match how their inputs are sharded.
    call: Callable
    golden: Callable  # (inputs: dict of role -> torch.Tensor) -> torch.Tensor
    layouts: list = field(default_factory=lambda: list(LAYOUTS))
    out_configs: list = field(default_factory=lambda: list(OUTPUT_CONFIGS))
    # "torch": compare with golden(). "baseline": compare with the op's own DRAM-interleaved TILE
    # output (what the model runs today), for ops whose ttnn numerics differ from torch by design, so
    # that only a config-induced change counts as wrong. "as_accurate": compare with golden() and
    # pass when no element is further from it than accuracy_factor times the DRAM-interleaved TILE
    # output's worst error (or atol), for ops whose program configs legitimately round differently.
    reference: str = "torch"
    accuracy_factor: float = 2.0
    # Each element must be within ulp_threshold ULP, or, when rtol is set, within atol + rtol * |expected|
    # (for ops whose sharded kernels reduce in a different order, so last bits may legitimately differ).
    ulp_threshold: float = 1
    rtol: float = None
    atol: float = 0.0

    def roles(self):
        return [operand.role for operand in self.operands] + ["out"]

    def vectors(self):
        """Every vector of the group, as {"layout": ..., role: config name, ..., "out": ...}."""
        axes = [self.layouts] + [operand.configs for operand in self.operands] + [self.out_configs]
        for values in itertools.product(*axes):
            yield dict(zip(["layout"] + self.roles(), values))

    def vector_id(self, vector):
        return "-".join([self.name, vector["layout"]] + [f"{role}={vector[role]}" for role in self.roles()])


def _even_parts(count, max_parts):
    """The most parts, at most `max_parts`, that split `count` units into equal whole parts."""
    return max(p for p in range(1, min(count, max_parts) + 1) if count % p == 0)


def _sharded(memory_layout, buffer_type, cores, shard_shape):
    shard_spec = ttnn.ShardSpec(cores, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(memory_layout, buffer_type, shard_spec)


def _nd_sharded(buffer_type, cores, shard_shape):
    nd_spec = ttnn.NdShardSpec(
        ttnn.Shape(shard_shape), cores, ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
    )
    return ttnn.MemoryConfig(buffer_type, nd_spec)


def _dram_banks(device):
    dram_grid = device.dram_grid_size()
    return dram_grid.x * dram_grid.y


def _core_grid(num_cores):
    """`num_cores` cores as the widest rectangle that fits SHARD_GRID, or filled row by row if none does."""
    for cols in range(SHARD_GRID.x, 0, -1):
        if num_cores % cols == 0 and num_cores // cols <= SHARD_GRID.y:
            end = ttnn.CoreCoord(cols - 1, num_cores // cols - 1)
            return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), end)})
    return ttnn.num_cores_to_corerangeset(num_cores, SHARD_GRID, row_wise=True)


def grid_size(core_range_set):
    """(x, y) size of the rectangle that bounds a core range set."""
    box = core_range_set.bounding_box()
    return box.end.x - box.start.x + 1, box.end.y - box.start.y + 1


def build_memory_config(name, shape, device, layout=ttnn.TILE_LAYOUT):
    if name == "default":
        return None
    if name == "dram_interleaved":
        return ttnn.DRAM_MEMORY_CONFIG
    if name == "l1_interleaved":
        return ttnn.L1_MEMORY_CONFIG

    # Sharding sees the tensor as [prod(leading dims), width] and splits it in whole tiles. A width
    # shard spans every row, and only TILE pads rows up to a multiple of 32; ROW_MAJOR requires the
    # exact row count.
    height, width = math.prod(shape[:-1]), shape[-1]
    h_tiles, w_tiles = math.ceil(height / ttnn.TILE_SIZE), math.ceil(width / ttnn.TILE_SIZE)
    padded_height = height if layout == ttnn.ROW_MAJOR_LAYOUT else h_tiles * ttnn.TILE_SIZE
    padded_width = w_tiles * ttnn.TILE_SIZE
    grid = SHARD_GRID
    max_cores = grid.x * grid.y

    if name == "l1_height_sharded":
        num_cores = _even_parts(h_tiles, max_cores)
        cores = _core_grid(num_cores)
        shard = [h_tiles // num_cores * ttnn.TILE_SIZE, padded_width]
        return _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, cores, shard)
    if name == "l1_width_sharded":
        num_cores = _even_parts(w_tiles, max_cores)
        cores = _core_grid(num_cores)
        shard = [padded_height, w_tiles // num_cores * ttnn.TILE_SIZE]
        return _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, cores, shard)
    if name == "l1_block_sharded":
        # The rows x cols core rectangle with the most cores whose sides split the tiles evenly.
        rows, cols = max(
            (
                (r, c)
                for r in range(1, grid.y + 1)
                for c in range(1, grid.x + 1)
                if h_tiles % r == 0 and w_tiles % c == 0
            ),
            key=lambda rc: (rc[0] * rc[1], -abs(rc[0] - rc[1])),
        )
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))})
        shard = [h_tiles // rows * ttnn.TILE_SIZE, w_tiles // cols * ttnn.TILE_SIZE]
        return _sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, cores, shard)
    if name == "dram_width_sharded":
        num_banks = _even_parts(w_tiles, _dram_banks(device))
        banks = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
        shard = [padded_height, w_tiles // num_banks * ttnn.TILE_SIZE]
        return _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, banks, shard)
    if name in ("l1_nd_sharded", "dram_nd_sharded"):
        shard_h, shard_w = min(ND_SHARD_HW[0], shape[-2]), min(ND_SHARD_HW[1], width)
        shard = [1] * (len(shape) - 2) + [shard_h, shard_w]
        num_shards = math.prod(shape[:-2]) * math.ceil(shape[-2] / shard_h) * math.ceil(width / shard_w)
        if name == "l1_nd_sharded":
            num_cores = _even_parts(num_shards, max_cores)
            cores = _core_grid(num_cores)
            return _nd_sharded(ttnn.BufferType.L1, cores, shard)
        num_banks = _even_parts(num_shards, _dram_banks(device))
        banks = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
        return _nd_sharded(ttnn.BufferType.DRAM, banks, shard)
    raise ValueError(f"Unknown memory config name: {name}")


_CONFIG_NAMES = {
    ("BufferType.DRAM", "TensorMemoryLayout.INTERLEAVED"): "dram_interleaved",
    ("BufferType.L1", "TensorMemoryLayout.INTERLEAVED"): "l1_interleaved",
    ("BufferType.L1", "TensorMemoryLayout.HEIGHT_SHARDED"): "l1_height_sharded",
    ("BufferType.L1", "TensorMemoryLayout.WIDTH_SHARDED"): "l1_width_sharded",
    ("BufferType.L1", "TensorMemoryLayout.BLOCK_SHARDED"): "l1_block_sharded",
    ("BufferType.L1", "TensorMemoryLayout.ND_SHARDED"): "l1_nd_sharded",
    ("BufferType.DRAM", "TensorMemoryLayout.WIDTH_SHARDED"): "dram_width_sharded",
    ("BufferType.DRAM", "TensorMemoryLayout.ND_SHARDED"): "dram_nd_sharded",
}


def config_name(memory_config):
    """The MEMORY_CONFIGS name of a memory config, or its type if it is none of them."""
    key = (str(memory_config.buffer_type), str(memory_config.memory_layout))
    return _CONFIG_NAMES.get(key, f"{key[0]}/{key[1]}")


def describe(memory_config):
    text = f"{memory_config.buffer_type}/{memory_config.memory_layout}"
    if memory_config.shard_spec is not None:
        text += f" shard={list(memory_config.shard_spec.shape)}"
    elif memory_config.nd_shard_spec is not None:
        text += f" nd_shard={list(memory_config.nd_shard_spec.shard_shape)}"
    return text


def footprint(memory_config, shape, layout):
    """(cores or banks used, KB of L1 per core) for a tensor of `shape` in `memory_config`."""
    if memory_config.buffer_type == ttnn.BufferType.DRAM:
        return None, 0
    if memory_config.shard_spec is not None:
        spec = memory_config.shard_spec
        return spec.num_cores(), math.prod(spec.shape) * BYTES_PER_ELEMENT / 1024
    if memory_config.nd_shard_spec is not None:
        spec = memory_config.nd_shard_spec
        shard = list(spec.shard_shape)
        num_shards = math.prod(math.ceil(dim / s) for dim, s in zip(shape, shard))
        cores = spec.grid.num_cores()
        return cores, math.ceil(num_shards / cores) * math.prod(shard) * BYTES_PER_ELEMENT / 1024
    # Interleaved: pages (tiles, or rows for ROW_MAJOR) dealt round-robin over every core's bank.
    banks = SHARD_GRID.x * SHARD_GRID.y
    height, width = math.prod(shape[:-1]), shape[-1]
    if layout == ttnn.TILE_LAYOUT:
        pages = math.ceil(height / ttnn.TILE_SIZE) * math.ceil(width / ttnn.TILE_SIZE)
        page_bytes = ttnn.TILE_SIZE * ttnn.TILE_SIZE * BYTES_PER_ELEMENT
    else:
        pages, page_bytes = height, width * BYTES_PER_ELEMENT
    return banks, math.ceil(pages / banks) * page_bytes / 1024


def compare_with_torch(expected, actual, ulp_threshold=1):
    """Element-wise comparison against torch: passes within ulp_threshold, and says which rows differ."""
    if actual.shape != expected.shape:
        return False, f"shape {list(actual.shape)} != torch {list(expected.shape)}"
    if torch.equal(expected, actual):  # fast path; the full ULP pass below is slow on 67M elements
        return True, f"Max ULP Delta: 0.0; exact {expected.numel()}/{expected.numel()}"
    passed, message = comp_ulp(expected, actual, ulp_threshold)
    differs = actual != expected
    message += f"; exact {expected.numel() - int(differs.sum())}/{expected.numel()}"
    if not passed:  # PCC is slow on 16M elements and only helps describe a failure
        _, pcc = comp_pcc(expected, actual)
        message += f"; PCC {pcc:.6f}"
    rows = differs.reshape(-1, expected.shape[-1]).any(dim=-1).nonzero().flatten().tolist()
    if rows:
        total_rows = expected.numel() // expected.shape[-1]
        message += f"; differing rows {rows[0]}..{rows[-1]} ({len(rows)} of {total_rows})"
    return bool(passed), message


def compare_close(expected, actual, rtol, atol):
    """Element-wise |actual - expected| <= atol + rtol * |expected|, saying how many elements miss and where."""
    if actual.shape != expected.shape:
        return False, f"shape {list(actual.shape)} != expected {list(expected.shape)}"
    if torch.equal(expected, actual):
        return True, f"max |diff| 0; exact {expected.numel()}/{expected.numel()}"
    diff = (actual.float() - expected.float()).abs()
    outside = diff > atol + rtol * expected.float().abs()
    num_outside = int(outside.sum())
    exact = expected.numel() - int((actual != expected).sum())
    message = (
        f"max |diff| {diff.max().item():.4g}; {num_outside}/{expected.numel()} outside rtol {rtol} atol {atol}; "
        f"exact {exact}/{expected.numel()}"
    )
    if num_outside:
        _, pcc = comp_pcc(expected, actual)
        rows = outside.reshape(-1, expected.shape[-1]).any(dim=-1).nonzero().flatten().tolist()
        message += f"; PCC {pcc:.6f}; rows outside {rows[0]}..{rows[-1]} ({len(rows)} of {outside.numel() // expected.shape[-1]})"
    return num_outside == 0, message


# Per-process caches. The inputs are the same seeded tensors for every vector of a group, and
# reading an input back proves its config transfers data intact, which does not change from one
# vector to the next.
_TORCH_DATA = {}
_VERIFIED_INPUTS = set()


def _torch_inputs(group):
    if group.name not in _TORCH_DATA:
        torch.manual_seed(0)
        inputs = {
            operand.role: torch.rand(operand.shape, dtype=torch.bfloat16) * (operand.high - operand.low) + operand.low
            for operand in group.operands
        }
        _TORCH_DATA[group.name] = (inputs, group.golden(inputs))
    return _TORCH_DATA[group.name]


_BASELINES = {}


def _baseline_output(group, device):
    """The op's output with every tensor DRAM interleaved in TILE layout.

    Computed once per process, or once per run when SUPPORT_MATRIX_CACHE names a directory that many
    test processes share: the first to take the lock computes it, the others load it.
    """
    if group.name in _BASELINES:
        return _BASELINES[group.name]
    cache_dir = os.environ.get("SUPPORT_MATRIX_CACHE")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"baseline_{group.name}.pt")
        with open(path + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if os.path.exists(path):
                _BASELINES[group.name] = torch.load(path)
            else:
                torch.save(_compute_baseline(group, device), path + ".tmp")
                os.replace(path + ".tmp", path)
                _BASELINES[group.name] = torch.load(path)
        return _BASELINES[group.name]
    _BASELINES[group.name] = _compute_baseline(group, device)
    return _BASELINES[group.name]


def _compute_baseline(group, device):
    torch_inputs, _ = _torch_inputs(group)
    inputs = {
        operand.role: ttnn.from_torch(
            torch_inputs[operand.role],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT if operand.layout == "follow" else LAYOUTS[operand.layout],
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for operand in group.operands
    }
    ctx = {"configs": {role: ttnn.DRAM_MEMORY_CONFIG for role in inputs}, "layout": ttnn.TILE_LAYOUT}
    result = group.call(inputs, ttnn.DRAM_MEMORY_CONFIG, ctx)
    output = ttnn.to_torch(result)
    for tensor in [result, *inputs.values()]:
        ttnn.deallocate(tensor)
    return output


def run_vector(group, vector, *, device, info=None):
    """Run one vector. Returns (passed, message, info); info describes every tensor for the report.

    Pass a dict as `info` to keep that description when the op raises, which is when it matters most.
    """
    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < SHARD_GRID.x or device_grid.y < SHARD_GRID.y:
        raise RuntimeError(
            f"Device compute grid {device_grid.x}x{device_grid.y} is smaller than {SHARD_GRID.x}x{SHARD_GRID.y}"
        )
    random.seed(0)
    layout = LAYOUTS[vector["layout"]]
    torch_inputs, torch_output = _torch_inputs(group)
    info = {} if info is None else info
    info.update({"group": group.name, "title": group.title, "layout": vector["layout"], "tensors": {}, "chosen": None})

    def note(role, label, shape, memory_config, tensor_layout):
        cores, kb = footprint(memory_config, shape, tensor_layout) if memory_config is not None else (None, None)
        info["tensors"][role] = {"label": label, "shape": list(shape), "config": vector[role], "cores": cores, "kb": kb}

    # Deallocated explicitly so a failing vector cannot leave L1 fragmented for the next one.
    device_tensors = {}
    try:
        configs, layouts = {}, {}
        try:
            for operand in group.operands:
                layouts[operand.role] = layout if operand.layout == "follow" else LAYOUTS[operand.layout]
                configs[operand.role] = build_memory_config(
                    vector[operand.role], operand.shape, device, layouts[operand.role]
                )
                note(operand.role, operand.label, operand.shape, configs[operand.role], layouts[operand.role])
            out_config = build_memory_config(vector["out"], group.output_shape, device, layout)
            note("out", "Output", group.output_shape, out_config, layout)
            for operand in group.operands:
                device_tensors[operand.role] = ttnn.from_torch(
                    torch_inputs[operand.role],
                    dtype=ttnn.bfloat16,
                    layout=layouts[operand.role],
                    device=device,
                    memory_config=configs[operand.role],
                )
        except Exception as e:
            return False, f"[SETUP] {e}", info

        # Only blame the op for a wrong output if every input reached it intact and as requested.
        for operand in group.operands:
            tensor, config = device_tensors[operand.role], configs[operand.role]
            if tensor.memory_config() != config:
                actual = describe(tensor.memory_config())
                return False, f"[SETUP] input {operand.label} requested {describe(config)}, got {actual}", info
            key = (group.name, vector["layout"], operand.role, vector[operand.role])
            if key not in _VERIFIED_INPUTS:
                if not torch.equal(ttnn.to_torch(tensor), torch_inputs[operand.role]):
                    return False, f"[SETUP] input {operand.label} does not read back equal in {describe(config)}", info
                _VERIFIED_INPUTS.add(key)

        start_time = start_measuring_time()
        result = group.call(device_tensors, out_config, {"configs": configs, "layout": layout})
        stop_measuring_time(start_time)
        device_tensors["out"] = result

        baseline = group.reference == "baseline"
        expected = _baseline_output(group, device) if baseline else torch_output
        actual = ttnn.to_torch(result)
        if group.reference == "as_accurate":
            baseline_error = (_baseline_output(group, device).float() - torch_output.float()).abs().max().item()
            bound = max(group.accuracy_factor * baseline_error, group.atol)
            passed, message = compare_close(torch_output, actual, 0.0, bound)
            message += f"; vs torch, the DRAM-interleaved TILE output is within {baseline_error:.4g}"
        elif group.rtol is not None:
            passed, message = compare_close(expected, actual, group.rtol, group.atol)
        else:
            passed, message = compare_with_torch(expected, actual, 0 if baseline else group.ulp_threshold)
        if baseline:
            message += "; vs the DRAM-interleaved TILE output"
        actual_config = result.memory_config()
        if vector["out"] == "default":
            info["chosen"] = config_name(actual_config)
            info["tensors"]["out"]["cores"], info["tensors"]["out"]["kb"] = footprint(
                actual_config, group.output_shape, layout
            )
        if not passed:
            message = f"[WRONG_RESULT] {message}"
        elif (out_config is not None and actual_config != out_config) or result.layout != layout:
            passed = False
            requested = "default" if out_config is None else describe(out_config)
            message = f"[WRONG_CONFIG] requested {requested} {layout}, got {describe(actual_config)} {result.layout}"
        else:
            message = f"{message}; out={describe(actual_config)}"
        return passed, message, info
    finally:
        for tensor in device_tensors.values():
            ttnn.deallocate(tensor)

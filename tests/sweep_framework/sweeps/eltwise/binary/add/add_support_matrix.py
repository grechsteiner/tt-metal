# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Support-matrix sweep for ttnn.add, using the residual adds of the MoonViT-V2 encoder block (Kimi-K3).
# Both residuals add two [1, 1, 65536, 1024] bf16 tensors that are sequence-parallel over a 4x1 mesh,
# so each chip runs a plain single-device add on [1, 1, 16384, 1024] with no CCL. The model runs it
# DRAM interleaved / TILE; this sweep tries every combination of buffer type (DRAM/L1), memory layout
# (interleaved / height / width / block sharded) and tensor layout (TILE / ROW_MAJOR) for A, B and
# the output.
#
# Nothing is invalidated up front, since finding out what fails is the point. Failure messages are
# tagged so the results can be split by cause:
#   [SETUP]         an input tensor could not be built, did not land in the requested config, or did not
#                   read back equal to its torch source -- the op never ran (or never got the right data)
#   [WRONG_RESULT]  the op ran but is more than ULP_THRESHOLD from torch.add -- accepted but incorrect
#   [WRONG_CONFIG]  the op ran and was correct, but the output is not in the requested memory config
#                   or layout -- the request was silently ignored
#   (untagged)      the op itself raised -- usually a TT_FATAL from validation, or L1 out of memory
#
# Shards are laid out by two rules, in order: every core (or DRAM bank) used gets the same amount of
# data, and as many cores as possible are used. They are placed on a fixed 12x10 (120-core
# Blackhole) grid, so a vector means the same core count on every machine; run() refuses to start
# on a device whose compute grid is smaller. Interleaved configs are left to the allocator, which
# always spreads pages over every bank.

import math
import random

import torch
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc, comp_ulp, start_measuring_time, stop_measuring_time

TIMEOUT = 30

# bf16 add rounds once, so anything beyond 1 ULP of torch.add is a real error. PCC is not enough here:
# at the full shape a single zeroed tile or row still scores above 0.9999, which hides shard and
# padding bugs.
ULP_THRESHOLD = 1

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

# ND shards are 256 x 256 blocks dealt round-robin to the cores, so each core holds several blocks
# from different parts of the tensor instead of one contiguous piece.
ND_SHARD_HW = (256, 256)

# The real per-chip shape of both residual adds.
SHAPES = {
    "full": [1, 1, 16384, 1024],
}

parameters = {
    "moonvit_add": {
        "shape_kind": list(SHAPES),
        "layout": ["tile", "row_major"],
        "input_a_memory_config": MEMORY_CONFIGS,
        "input_b_memory_config": MEMORY_CONFIGS,
        # "default" passes memory_config=None and lets the op pick the output config.
        "out_memory_config": ["default"] + MEMORY_CONFIGS,
    },
}


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
        cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
        shard = [h_tiles // num_cores * ttnn.TILE_SIZE, padded_width]
        return _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, cores, shard)
    if name == "l1_width_sharded":
        num_cores = _even_parts(w_tiles, max_cores)
        cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
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
        shard_h, shard_w = ND_SHARD_HW
        shard = [1] * (len(shape) - 2) + [shard_h, shard_w]
        num_shards = math.ceil(height / shard_h) * math.ceil(width / shard_w)
        if name == "l1_nd_sharded":
            num_cores = _even_parts(num_shards, max_cores)
            cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
            return _nd_sharded(ttnn.BufferType.L1, cores, shard)
        num_banks = _even_parts(num_shards, _dram_banks(device))
        banks = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
        return _nd_sharded(ttnn.BufferType.DRAM, banks, shard)
    raise ValueError(f"Unknown memory config name: {name}")


def _describe(memory_config):
    text = f"{memory_config.buffer_type}/{memory_config.memory_layout}"
    if memory_config.shard_spec is not None:
        text += f" shard={list(memory_config.shard_spec.shape)}"
    elif memory_config.nd_shard_spec is not None:
        text += f" nd_shard={list(memory_config.nd_shard_spec.shard_shape)}"
    return text


def compare_with_torch(expected, actual):
    """Element-wise comparison against torch: passes within ULP_THRESHOLD, and says which rows differ."""
    if actual.shape != expected.shape:
        return False, f"shape {list(actual.shape)} != torch {list(expected.shape)}"
    passed, message = comp_ulp(expected, actual, ULP_THRESHOLD)
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


# Per-process caches. The inputs are the same seeded tensors for every vector, and reading an input
# back proves its config transfers data intact, which does not change from one vector to the next.
_TORCH_DATA = {}
_VERIFIED_INPUTS = set()


def _torch_data(shape_kind):
    if shape_kind not in _TORCH_DATA:
        torch.manual_seed(0)
        shape = SHAPES[shape_kind]
        torch_a = torch.rand(shape, dtype=torch.bfloat16) * 200 - 100
        torch_b = torch.rand(shape, dtype=torch.bfloat16) * 200 - 100
        _TORCH_DATA[shape_kind] = (torch_a, torch_b, torch.add(torch_a, torch_b))
    return _TORCH_DATA[shape_kind]


def run(
    shape_kind,
    layout,
    input_a_memory_config,
    input_b_memory_config,
    out_memory_config,
    *,
    device,
) -> list:
    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < SHARD_GRID.x or device_grid.y < SHARD_GRID.y:
        raise RuntimeError(
            f"Device compute grid {device_grid.x}x{device_grid.y} is smaller than {SHARD_GRID.x}x{SHARD_GRID.y}"
        )

    random.seed(0)
    shape = SHAPES[shape_kind]
    tt_layout = ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT
    torch_a, torch_b, torch_output = _torch_data(shape_kind)

    # Deallocated explicitly so a failing vector cannot leave L1 fragmented for the next one.
    device_tensors = []
    try:
        try:
            input_configs = [
                build_memory_config(name, shape, device, tt_layout)
                for name in (input_a_memory_config, input_b_memory_config)
            ]
            for torch_tensor, config in zip((torch_a, torch_b), input_configs):
                device_tensors.append(
                    ttnn.from_torch(
                        torch_tensor, dtype=ttnn.bfloat16, layout=tt_layout, device=device, memory_config=config
                    )
                )
            out_config = build_memory_config(out_memory_config, shape, device, tt_layout)
        except Exception as e:
            return [(False, f"[SETUP] {e}"), None]
        a, b = device_tensors

        # Only blame the op for a wrong output if both inputs reached it intact and as requested.
        for name, tensor, torch_tensor, config_name, config in zip(
            "AB", (a, b), (torch_a, torch_b), (input_a_memory_config, input_b_memory_config), input_configs
        ):
            if tensor.memory_config() != config:
                actual = _describe(tensor.memory_config())
                return [(False, f"[SETUP] input {name} requested {_describe(config)}, got {actual}"), None]
            key = (shape_kind, layout, config_name, name)
            if key not in _VERIFIED_INPUTS:
                if not torch.equal(ttnn.to_torch(tensor), torch_tensor):
                    return [(False, f"[SETUP] input {name} does not read back equal in {_describe(config)}"), None]
                _VERIFIED_INPUTS.add(key)

        start_time = start_measuring_time()
        result = ttnn.add(a, b, memory_config=out_config)
        e2e_perf = stop_measuring_time(start_time)
        device_tensors.append(result)

        passed, message = compare_with_torch(torch_output, ttnn.to_torch(result))
        actual_config = result.memory_config()
        if not passed:
            message = f"[WRONG_RESULT] {message}"
        elif (out_config is not None and actual_config != out_config) or result.layout != tt_layout:
            passed = False
            requested = "default" if out_config is None else _describe(out_config)
            message = (
                f"[WRONG_CONFIG] requested {requested} {tt_layout}, " f"got {_describe(actual_config)} {result.layout}"
            )
        else:
            message = f"{message}; out={_describe(actual_config)}"
        return [(passed, message), e2e_perf]
    finally:
        for tensor in device_tensors:
            ttnn.deallocate(tensor)

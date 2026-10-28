# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Support-matrix group for ttnn.rms_norm in the MoonViT-V2 encoder block (Kimi-K3): norm0 and norm1,
# on the per-chip [1, 1, 16384, 1024] bf16 activation with a [1, 1, 32, 32] row-major weight. HF uses
# nn.RMSNorm(eps=None), i.e. torch.finfo(bfloat16).eps = 2^-7. A sharded input is given a
# LayerNormShardedMultiCoreProgramConfig that matches its shard grid, since the op cannot run a
# sharded input without one. See tests/sweep_framework/support_matrix/harness.py.

import torch
import ttnn

from tests.sweep_framework.support_matrix.harness import OpGroup, Operand, grid_size, run_vector

TIMEOUT = 30

SHAPE = [1, 1, 16384, 1024]
WEIGHT_SHAPE = [1, 1, 32, 32]
EPS = 2**-7


def _program_config(input_config):
    """A sharded-norm program config for a legacy-sharded input, else None to let the op choose."""
    spec = input_config.shard_spec if input_config is not None else None
    if spec is None:
        return None
    grid_x, grid_y = grid_size(spec.grid)
    block_h, block_w = spec.shape[0] // ttnn.TILE_SIZE, spec.shape[1] // ttnn.TILE_SIZE
    subblock_w = max(d for d in range(1, min(block_w, 4) + 1) if block_w % d == 0)
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[grid_x, grid_y],
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )


def _golden(inputs):
    x = inputs["input"].float()
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (x / rms * inputs["weight"].float().reshape(-1)).bfloat16()


GROUP = OpGroup(
    name="rms_norm",
    title="ttnn.rms_norm · norm0 / norm1",
    operands=[
        Operand("input", "Input", SHAPE, low=-4.0, high=4.0),
        Operand(
            "weight",
            "Weight",
            WEIGHT_SHAPE,
            configs=["dram_interleaved", "l1_interleaved"],
            layout="row_major",
            low=0.5,
            high=1.5,
        ),
    ],
    output_shape=SHAPE,
    call=lambda inputs, memory_config, ctx: ttnn.rms_norm(
        inputs["input"],
        epsilon=EPS,
        weight=inputs["weight"],
        memory_config=memory_config,
        program_config=_program_config(ctx["configs"]["input"]),
    ),
    golden=_golden,
    # Sharded norms reduce each row across cores in a different order than the interleaved kernel, so
    # last bits may differ: judge against the op's own DRAM-interleaved output with a tolerance that a
    # zeroed tile or garbage (|diff| ~ the value itself) still fails.
    reference="baseline",
    rtol=2**-6,
    atol=2**-6,
)

VECTORS = {GROUP.vector_id(vector): vector for vector in GROUP.vectors()}

parameters = {GROUP.name: {"vector": list(VECTORS)}}


def run(vector, *, device) -> list:
    passed, message, _ = run_vector(GROUP, VECTORS[vector], device=device)
    return [(passed, message), None]

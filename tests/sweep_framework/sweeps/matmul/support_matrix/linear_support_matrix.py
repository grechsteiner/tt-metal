# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Support-matrix groups for ttnn.linear in the MoonViT-V2 encoder block (Kimi-K3), one per projection,
# since each size shards differently: wqkv, wo, fc0 and fc1 on the per-chip sequence of 16384 rows.
# Weights are replicated on every chip and stay in DRAM or L1 interleaved, or DRAM width-sharded.
# The program config is left to ttnn (program_config=None), which picks one from how the
# activation and output are sharded. TILE only: matmul has no row-major path.
# See tests/sweep_framework/support_matrix/harness.py.

import torch
import ttnn

from tests.sweep_framework.support_matrix.harness import MEMORY_CONFIGS, OpGroup, Operand, run_vector

TIMEOUT = 30

SEQ = 16384
WEIGHT_CONFIGS = ["dram_interleaved", "l1_interleaved", "dram_width_sharded"]

# name, title, in features, out features
PROJECTIONS = [
    ("linear_wqkv", "ttnn.linear · wqkv 1024 → 4608", 1024, 4608),
    ("linear_wo", "ttnn.linear · wo 1536 → 1024", 1536, 1024),
    ("linear_fc0", "ttnn.linear · fc0 1024 → 4096", 1024, 4096),
    ("linear_fc1", "ttnn.linear · fc1 4096 → 1024", 4096, 1024),
]


def _group(name, title, k, n):
    return OpGroup(
        name=name,
        title=title,
        operands=[
            Operand("act", "Activation", [1, 1, SEQ, k], configs=list(MEMORY_CONFIGS), low=-1.0, high=1.0),
            # Scaled so outputs stay near unit size whatever the inner dimension.
            Operand(
                "weight", "Weight", [1, 1, k, n], configs=WEIGHT_CONFIGS, low=-((3 / k) ** 0.5), high=(3 / k) ** 0.5
            ),
        ],
        output_shape=[1, 1, SEQ, n],
        call=lambda inputs, memory_config, ctx: ttnn.linear(
            inputs["act"], inputs["weight"], memory_config=memory_config, dtype=ttnn.bfloat16
        ),
        golden=lambda inputs: torch.matmul(inputs["act"].float(), inputs["weight"].float()).bfloat16(),
        layouts=["tile"],
        # Program configs differ in how they block the inner dimension, so accumulation order and the
        # last bits of the output may differ: judge against the op's own DRAM-interleaved output with
        # a tolerance that a zeroed tile or garbage (|diff| ~ the value itself, about 1) still fails.
        reference="baseline",
        rtol=2**-5,
        atol=2**-5,
    )


GROUPS = [_group(*projection) for projection in PROJECTIONS]

VECTORS = {group.vector_id(vector): (group, vector) for group in GROUPS for vector in group.vectors()}

parameters = {group.name: {"vector": [v for v, (g, _) in VECTORS.items() if g is group]} for group in GROUPS}


def run(vector, *, device) -> list:
    group, values = VECTORS[vector]
    passed, message, _ = run_vector(group, values, device=device)
    return [(passed, message), None]

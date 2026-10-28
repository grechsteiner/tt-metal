# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Support-matrix group for ttnn.add, using the residual adds of the MoonViT-V2 encoder block (Kimi-K3).
# Both residuals add two [1, 1, 65536, 1024] bf16 tensors that are sequence-parallel over a 4x1 mesh,
# so each chip runs a plain single-device add on [1, 1, 16384, 1024] with no CCL. The model runs it
# DRAM interleaved / TILE. See tests/sweep_framework/support_matrix/harness.py for how vectors are
# built, run and judged.

import torch
import ttnn

from tests.sweep_framework.support_matrix.harness import OpGroup, Operand, run_vector

TIMEOUT = 30

SHAPE = [1, 1, 16384, 1024]

GROUP = OpGroup(
    name="add",
    title="ttnn.add · residual add",
    operands=[Operand("a", "Input A", SHAPE), Operand("b", "Input B", SHAPE)],
    output_shape=SHAPE,
    call=lambda inputs, memory_config, ctx: ttnn.add(inputs["a"], inputs["b"], memory_config=memory_config),
    golden=lambda inputs: torch.add(inputs["a"], inputs["b"]),
    # bf16 add rounds once, so anything beyond 1 ULP of torch.add is a real error.
    ulp_threshold=1,
)

VECTORS = {GROUP.vector_id(vector): vector for vector in GROUP.vectors()}

parameters = {GROUP.name: {"vector": list(VECTORS)}}


def run(vector, *, device) -> list:
    passed, message, _ = run_vector(GROUP, VECTORS[vector], device=device)
    return [(passed, message), None]

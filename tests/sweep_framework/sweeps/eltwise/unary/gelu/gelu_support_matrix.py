# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Support-matrix group for ttnn.gelu in the MoonViT-V2 encoder block (Kimi-K3): the GELU between fc0
# and fc1, on the per-chip [1, 1, 16384, 4096] bf16 fc0 output. HF uses
# PytorchGELUTanh, so this runs ttnn.GeluVariant.Tanh against
# torch.nn.functional.gelu(approximate="tanh"). See tests/sweep_framework/support_matrix/harness.py.

import torch
import ttnn

from tests.sweep_framework.support_matrix.harness import OpGroup, Operand, run_vector

TIMEOUT = 30

SHAPE = [1, 1, 16384, 4096]

GROUP = OpGroup(
    name="gelu",
    title="ttnn.gelu (tanh) · between fc0 and fc1",
    operands=[Operand("input", "Input", SHAPE, low=-6.0, high=6.0)],
    output_shape=SHAPE,
    call=lambda inputs, memory_config, ctx: ttnn.gelu(
        inputs["input"], variant=ttnn.GeluVariant.Tanh, memory_config=memory_config
    ),
    golden=lambda inputs: torch.nn.functional.gelu(inputs["input"], approximate="tanh"),
    # ttnn's SFPU gelu is within 14 ULP of torch for 99% of elements, and differs from torch's exact
    # -0.0 near x = -6 by ~1e-7, so a memory config is judged against ttnn's own DRAM-interleaved
    # output instead: it must not change a single bit.
    reference="baseline",
)

VECTORS = {GROUP.vector_id(vector): vector for vector in GROUP.vectors()}

parameters = {GROUP.name: {"vector": list(VECTORS)}}


def run(vector, *, device) -> list:
    passed, message, _ = run_vector(GROUP, VECTORS[vector], device=device)
    return [(passed, message), None]

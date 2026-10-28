# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Support-matrix group for ttnn.experimental.nlp_concat_heads in the MoonViT-V2 encoder block
# (Kimi-K3): ring attention's per-chip output [1, 12, 16384, 128] becomes [1, 1, 16384, 1536] for wo.
# It only moves data, so every element must match torch exactly. See
# tests/sweep_framework/support_matrix/harness.py.

import torch
import ttnn

from tests.sweep_framework.support_matrix.harness import OpGroup, Operand, run_vector

TIMEOUT = 30

NUM_HEADS, SEQ, HEAD_DIM = 12, 16384, 128

GROUP = OpGroup(
    name="concat_heads",
    title="nlp_concat_heads · before wo",
    operands=[Operand("input", "Input", [1, NUM_HEADS, SEQ, HEAD_DIM])],
    output_shape=[1, 1, SEQ, NUM_HEADS * HEAD_DIM],
    call=lambda inputs, memory_config, ctx: ttnn.experimental.nlp_concat_heads(
        inputs["input"], memory_config=memory_config
    ),
    golden=lambda inputs: inputs["input"].permute(0, 2, 1, 3).reshape(1, 1, SEQ, NUM_HEADS * HEAD_DIM),
    ulp_threshold=0,
)

VECTORS = {GROUP.vector_id(vector): vector for vector in GROUP.vectors()}

parameters = {GROUP.name: {"vector": list(VECTORS)}}


def run(vector, *, device) -> list:
    passed, message, _ = run_vector(GROUP, VECTORS[vector], device=device)
    return [(passed, message), None]

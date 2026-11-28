# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""MLP: `MLP2` in the reference. fc0 -> tanh GELU -> fc1, no biases.

The GELU is the tanh approximation (`PytorchGELUTanh` in the reference), applied after fc0's
matmul as `activation_fn="gelu_tanh"` (`UnaryOpType.GELU_TANH`). Sequence parallel: each device
works on its own rows and no collectives are needed.
"""

from __future__ import annotations

import ttnn

from .linear import Linear
from .model_config import MLP_FC0_BLOCKS, MLP_FC1_BLOCKS, MoonViTConfig
from .module import Module


class TT_Vision_MLP(Module):
    def __init__(self, config: MoonViTConfig, *, mesh_device: ttnn.MeshDevice, compute_kernel_config) -> None:
        super().__init__()
        kwargs = dict(bias=config.linear_bias, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config)
        self.fc0 = Linear(
            config.hidden_size, config.intermediate_size, activation_fn="gelu_tanh", blocks=MLP_FC0_BLOCKS, **kwargs
        )
        # fp32 out: added to the fp32 residual stream (vision_block.py).
        self.fc1 = Linear(
            config.intermediate_size, config.hidden_size, output_dtype=ttnn.float32, blocks=MLP_FC1_BLOCKS, **kwargs
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, rows, hidden_size] bf16 -> the same shape, fp32; rows are this device's shard,
        padding included."""
        h = self.fc0(x)
        out = self.fc1(h)
        ttnn.deallocate(h)
        return out

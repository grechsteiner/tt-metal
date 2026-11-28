# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Replicated linear layer. Bias is supported, though MoonViT has none."""

from __future__ import annotations

import torch

import ttnn

from .model_config import Matmul2DBlocks, MatmulBlocks
from .module import Module, Parameter

# `activation_fn` names -> the unary op `ttnn.linear` applies for the same name.
_ACTIVATIONS = {
    "gelu": lambda: ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0),  # exact erf GELU
    "gelu_tanh": lambda: ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH),
}


class Linear(Module):
    """`y = x @ W.T`, weight stored transposed as [in, out]; loads a torch `nn.Linear` state.

    Input, weight and output are DRAM interleaved. `blocks` picks the op: `model_config.MatmulBlocks`
    runs `ttnn.experimental.minimal_matmul` with those block sizes on the device's whole worker grid;
    `model_config.Matmul2DBlocks` runs `ttnn.linear` with a 2D multicast program config; without it,
    `minimal_matmul` picks its defaults.

    `activation_fn` is `"gelu"` (exact) or `"gelu_tanh"`. It runs as a separate eltwise op on the
    matmul's output, as `ttnn.linear` applies it without a `core_grid`, not fused into the matmul.

    `output_dtype` sets the output's dtype (default: the input's). fp32 keeps the fp32 accumulator
    instead of rounding it to bf16; the layers that write the fp32 residual stream use it.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        mesh_device: ttnn.MeshDevice,
        compute_kernel_config,
        activation_fn: str | None = None,
        output_dtype: ttnn.DataType | None = None,
        blocks: MatmulBlocks | Matmul2DBlocks | None = None,
    ) -> None:
        super().__init__()
        if activation_fn is not None and activation_fn not in _ACTIVATIONS:
            raise ValueError(f"activation_fn {activation_fn!r} is not one of {sorted(_ACTIVATIONS)}")
        self.in_features = in_features
        self.out_features = out_features
        self.mesh_device = mesh_device
        self.compute_kernel_config = compute_kernel_config
        self.activation_fn = activation_fn
        self.output_dtype = output_dtype
        self.blocks = blocks
        self.weight = Parameter(shape=(in_features, out_features), device=mesh_device)
        self.bias = Parameter(shape=(1, out_features), device=mesh_device) if bias else None
        # Built on the first forward: it reads the device grid, and construction stays device-free
        # (test_module.py builds the tree on a stand-in device).
        self._cached_minimal_matmul_config: ttnn.MinimalMatmulConfig | None = None
        # `ttnn.linear` 2D program configs by input row count, in tiles.
        self._linear_program_configs: dict[int, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = state["weight"].transpose(0, 1)
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def _minimal_matmul_config(self) -> ttnn.MinimalMatmulConfig | None:
        if isinstance(self.blocks, MatmulBlocks) and self._cached_minimal_matmul_config is None:
            grid = self.mesh_device.compute_with_storage_grid_size()
            self._cached_minimal_matmul_config = ttnn.MinimalMatmulConfig(
                M_block_size=self.blocks.m,
                K_block_size=self.blocks.k,
                N_block_size=self.blocks.n,
                subblock_h=self.blocks.subblock_h,
                subblock_w=self.blocks.subblock_w,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            )
        return self._cached_minimal_matmul_config

    def _linear_program_config(self, x: ttnn.Tensor) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
        blocks = self.blocks
        row_tiles = x.padded_shape[-2] // ttnn.TILE_SIZE
        if row_tiles not in self._linear_program_configs:
            per_core_rows = -(-row_tiles // blocks.grid_y)
            per_core_columns = -(-self.out_features // ttnn.TILE_SIZE // blocks.grid_x)
            # The output block must divide the per-core rows; row counts other than production's
            # get the largest divisor no larger than the tuned one.
            out_block_h = max(h for h in range(1, blocks.out_block_h + 1) if per_core_rows % h == 0)
            self._linear_program_configs[row_tiles] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(blocks.grid_x, blocks.grid_y),
                in0_block_w=blocks.in0_block_w,
                out_subblock_h=blocks.subblock_h,
                out_subblock_w=blocks.subblock_w,
                out_block_h=out_block_h,
                out_block_w=blocks.out_block_w,
                per_core_M=per_core_rows,
                per_core_N=per_core_columns,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
        return self._linear_program_configs[row_tiles]

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if isinstance(self.blocks, Matmul2DBlocks):
            y = ttnn.linear(
                x,
                self.weight.data,
                bias=self.bias.data if self.bias is not None else None,
                program_config=self._linear_program_config(x),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.output_dtype,
                compute_kernel_config=self.compute_kernel_config,
            )
        else:
            y = ttnn.experimental.minimal_matmul(
                x,
                self.weight.data,
                bias_tensor=self.bias.data if self.bias is not None else None,
                config=self._minimal_matmul_config(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.output_dtype,
                compute_kernel_config=self.compute_kernel_config,
            )
        if self.activation_fn is None:
            return y
        out = ttnn.unary_chain(y, [_ACTIVATIONS[self.activation_fn]()], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(y)
        return out

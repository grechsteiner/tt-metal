# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: each `Linear` of the model, built by its production module (block sizes, op, kernel
config, output dtype and activation as in the model), against the reference `nn.Linear` at the row
count it sees in the model."""

from __future__ import annotations

import functools

import pytest
import torch
import torch.nn.functional as F

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    SYNTHETIC,
    SAMPLED_LAYERS,
    random_tensor,
    assert_quality,
    device_config,
    padding_layout_grid,
    PADDING_LAYOUT_GRIDS,
    MeshRows,
    P150,
    SIMULATOR_GRID,
    SIMULATOR,
    random_patches,
    simulator_config,
)
from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.tt.linear import Linear
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import compute_kernel_config, resolve_parallel
from models.experimental.kimi_k3_moonvit_v2.tt.patch_embed import TT_Patch_Embed
from models.experimental.kimi_k3_moonvit_v2.tt.projector import TT_Projector
from models.experimental.kimi_k3_moonvit_v2.tt.vision_block import TT_Vision_Block

# Per-layer linears: reference path under `vision_tower.encoder.blocks.<layer>` -> path in
# `TT_Vision_Block`. They see every patch.
ENCODER_LINEARS = {
    "wqkv": ("wqkv", "attention.wqkv"),
    "wo": ("wo", "attention.wo"),
    "mlp_fc0": ("mlp.fc0", "mlp.fc0"),
    "mlp_fc1": ("mlp.fc1", "mlp.fc1"),
}
ENCODER_LINEAR_PATHS = pytest.mark.parametrize("paths", ENCODER_LINEARS.values(), ids=ENCODER_LINEARS.keys())
# Projector linears: reference path -> path in `TT_Projector`. They see merged tokens, a quarter of
# the rows.
PROJECTOR_LINEARS = {"projector_fc0": ("mm_projector.proj.0", "fc0"), "projector_fc1": ("mm_projector.proj.2", "fc1")}
PROJECTOR_LINEAR_PATHS = pytest.mark.parametrize("paths", PROJECTOR_LINEARS.values(), ids=PROJECTOR_LINEARS.keys())

# Reference GELU for each `Linear.activation_fn`.
_GELU_APPROXIMATION = {"gelu": "none", "gelu_tanh": "tanh"}


def submodule(module, path: str) -> Linear:
    """The submodule at dotted `path`, e.g. `attention.wqkv`."""
    return functools.reduce(getattr, path.split("."), module)


def check_linear(mesh_device, tt: Linear, torch_linear, rows: MeshRows) -> None:
    """`tt` is the production module's linear; its activation, if any, is applied to the golden too."""
    tt.load_torch_state_dict(torch_linear.state_dict())

    x = random_tensor(rows.n, torch_linear.in_features)
    with torch.no_grad():
        golden = torch_linear(rows.real_rows(x))
        if tt.activation_fn is not None:
            golden = F.gelu(golden, approximate=_GELU_APPROXIMATION[tt.activation_fn])

    out = tt(rows.shard(x, mesh_device))
    assert_quality(golden, rows.gather(out, mesh_device), relative_rmse=0.003)


def check_encoder_linear(mesh_device, sp_axis, config, reference, layer, paths, grid) -> None:
    """`grid` padded to `config.max_patches`, the ring's length; one chip runs its quarter (`MeshRows`)."""
    reference_path, tt_path = paths
    tt_config = device_config(config, sp_axis)
    block = TT_Vision_Block(
        tt_config,
        mesh_device=mesh_device,
        parallel=resolve_parallel(mesh_device, None, None, tt_config),
        compute_kernel_config=compute_kernel_config(mesh_device),
    )
    torch_linear = reference.vision_tower.encoder.blocks[layer].get_submodule(reference_path)
    ((_, h, w),) = grid
    check_linear(mesh_device, submodule(block, tt_path), torch_linear, MeshRows(h * w, config.max_patches, sp_axis))


def check_projector_linear(mesh_device, sp_axis, config, reference, paths, grid) -> None:
    """The merged tokens of `grid`, padded to `config.max_patches / 4` (the ring's length after merging);
    one chip runs its quarter (`MeshRows`)."""
    reference_path, tt_path = paths
    projector = TT_Projector(
        device_config(config, sp_axis),
        mesh_device=mesh_device,
        compute_kernel_config=compute_kernel_config(mesh_device),
    )
    ((_, h, w),) = grid
    rows = MeshRows(h * w // config.merge_area, config.max_patches // config.merge_area, sp_axis)
    check_linear(mesh_device, submodule(projector, tt_path), reference.get_submodule(reference_path), rows)


def check_patch_embed_linear(mesh_device, sp_axis, config, reference, grid) -> None:
    """`patch_embed.proj`, loaded through `TT_Patch_Embed` (conv weight zero-padded to
    `padded_patch_dim`), on zero-padded patch rows, against the reference Conv2d. `grid` is padded to
    `config.max_patches`; one chip runs its quarter (`MeshRows`)."""
    tt = TT_Patch_Embed(
        device_config(config, sp_axis),
        mesh_device=mesh_device,
        compute_kernel_config=compute_kernel_config(mesh_device),
    )
    tt.load_torch_state_dict(reference.vision_tower.patch_embed.state_dict())

    pixels, _ = random_patches(config, grid)
    pixels = pixels.to(torch.bfloat16)
    rows = MeshRows(pixels.shape[0], config.max_patches, sp_axis)
    with torch.no_grad():
        golden = reference.vision_tower.patch_embed.proj(rows.real_rows(pixels)).flatten(1)

    out = tt.proj(rows.shard(host_prep.patch_rows(config, pixels), mesh_device))
    assert_quality(golden, rows.gather(out, mesh_device), relative_rmse=0.004)


@P150
@SYNTHETIC
@SAMPLED_LAYERS
@PADDING_LAYOUT_GRIDS
@ENCODER_LINEAR_PATHS
def test_encoder_linear(mesh_device, sp_axis, num_links, config, reference, size, layer, paths):
    check_encoder_linear(mesh_device, sp_axis, config, reference, layer, paths, padding_layout_grid(size))


@SIMULATOR
@SYNTHETIC
@ENCODER_LINEAR_PATHS
def test_encoder_linear_small(mesh_device, sp_axis, num_links, config, reference, paths):
    """Sized for ttsim: one chip's quarter of a ring padded to SIMULATOR_ROWS."""
    check_encoder_linear(mesh_device, sp_axis, simulator_config(config), reference, 0, paths, SIMULATOR_GRID)


@P150
@SYNTHETIC
@PADDING_LAYOUT_GRIDS
@PROJECTOR_LINEAR_PATHS
def test_projector_linear(mesh_device, sp_axis, num_links, config, reference, size, paths):
    check_projector_linear(mesh_device, sp_axis, config, reference, paths, padding_layout_grid(size))


@SIMULATOR
@SYNTHETIC
@PROJECTOR_LINEAR_PATHS
def test_projector_linear_small(mesh_device, sp_axis, num_links, config, reference, paths):
    """Sized for ttsim: one chip's quarter of a ring padded to SIMULATOR_ROWS patches."""
    check_projector_linear(mesh_device, sp_axis, simulator_config(config), reference, paths, SIMULATOR_GRID)


@P150
@SYNTHETIC
@PADDING_LAYOUT_GRIDS
def test_patch_embed_linear(mesh_device, sp_axis, num_links, config, reference, size):
    check_patch_embed_linear(mesh_device, sp_axis, config, reference, padding_layout_grid(size))


@SIMULATOR
@SYNTHETIC
def test_patch_embed_linear_small(mesh_device, sp_axis, num_links, config, reference):
    """Sized for ttsim: one chip's quarter of a ring padded to SIMULATOR_ROWS."""
    check_patch_embed_linear(mesh_device, sp_axis, simulator_config(config), reference, SIMULATOR_GRID)

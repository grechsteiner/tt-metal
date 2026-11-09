# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Configuration for the MoonViT-V2 encoder, patch merger and projector.

`MoonViTConfig` holds the architecture; its defaults are the Kimi-K3 `vision_config`, and
`from_hf_config` reads the same fields from a checkpoint's `config.json`. `VisionParallel` is the
mesh layout the TT modules receive, resolved from a sequence-parallel `ParallelFactor` and the
`TT_CCL` that owns the ring resources (`ccl.py`). This module imports no ttnn at import time, so host
prep and the CPU tests can use it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import torch

    import ttnn
    from .ccl import ParallelFactor, TT_CCL


# The processor's `in_patch_limit` (`preprocessor_config.json`).
IN_PATCH_LIMIT = 65536
# The layout the model targets: the padded sequence, RING_MAX_PATCHES rows, split over a ring of
# RING_SIZE chips. The padded length is the processor's patch limit. One device without the split
# runs at most one ring chip's share of it, ONE_DEVICE_MAX_PATCHES rows (`resolve_parallel`).
RING_SIZE = 4
RING_MAX_PATCHES = IN_PATCH_LIMIT
ONE_DEVICE_MAX_PATCHES = RING_MAX_PATCHES // RING_SIZE


@dataclass(frozen=True)
class MoonViTConfig:
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 27
    num_attention_heads: int = 12
    qkv_hidden_size: int = 1536
    in_channels: int = 3
    patch_size: int = 14
    pos_emb_interpolation_mode: str = "bilinear"
    attn_bias: bool = False
    linear_bias: bool = False
    patch_embed_proj_bias: bool = False
    rope_theta: float = 10000.0
    # Upstream norms use nn.RMSNorm(eps=None), i.e. finfo(bf16).eps for bf16 activations.
    norm_eps: float = 2.0**-7
    # Processor (`preprocessor_config.json`): `patch_limit_on_one_side` (also the RoPE table size) and
    # `in_patch_limit`, which only sets the resize (`image_prep.resize_plan`).
    max_side_patches: int = 512
    in_patch_limit: int = IN_PATCH_LIMIT
    # Device row count: every image is padded to `max_patches` rows, and ring attention masks the
    # padding. Images whose grid exceeds it are rejected, never shrunk to fit.
    max_patches: int = RING_MAX_PATCHES
    # Patch merger (`sd2_tpool`): each 2x2 window of patches becomes one token.
    merge_kernel_size: tuple[int, int] = (2, 2)
    # Projector (`patchmergerv2`): merged token -> language-model width, then RMSNorm.
    text_hidden_size: int = 7168
    projector_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.qkv_hidden_size // self.num_attention_heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size * self.patch_size

    @property
    def padded_patch_dim(self) -> int:
        """`patch_dim` (588) rounded up to a whole tile (608), the width the patch rows and the patch
        embed weight have on device. The extra columns are explicit zeros on both sides, so the matmul
        does not depend on the values ttnn leaves in tile padding.

        This can potentially be removed in the future: today both tensors come from `ttnn.from_torch`,
        which zero-fills tile padding, so 588 works as long as that holds and nothing on device
        produces the patch rows. Drop it once that is guaranteed and tested.
        """
        return -(-self.patch_dim // 32) * 32

    @property
    def merge_area(self) -> int:
        return self.merge_kernel_size[0] * self.merge_kernel_size[1]

    @property
    def merged_dim(self) -> int:
        """Width of a merged token: the window's patches concatenated."""
        return self.hidden_size * self.merge_area

    @classmethod
    def from_hf_config(cls, config: dict | str | Path) -> MoonViTConfig:
        """Read the `vision_config` block of a Kimi-K3 `config.json` (a dict, the file, or its folder).

        A missing key takes upstream's effective default, which is `KimiK3VisionConfig`'s and also
        this class's (the K3 values). Upstream's modeling code has other fallbacks (`bicubic`,
        biases on), but they never apply: the vision config always sets those attributes.
        The processor fields (`in_patch_limit`) are not in `config.json` and always keep their defaults.
        """
        if not isinstance(config, dict):
            path = Path(config)
            if path.is_dir():
                path = path / "config.json"
            config = json.loads(path.read_text())
        vc = config.get("vision_config", config)
        unsupported = {
            k: vc[k]
            for k, want in (
                ("norm_type", "rmsnorm"),
                ("mlp_type", "mlp2"),
                ("pos_emb_type", "divided_fixed"),
                ("activation_func", "gelu_pytorch_tanh"),
                ("merge_type", "sd2_tpool"),
                ("mm_projector_type", "patchmergerv2"),
                ("projector_hidden_act", "gelu"),
            )
            if k in vc and vc[k] != want
        }
        if unsupported:
            raise ValueError(f"vision_config differs from MoonViT-V2 in {unsupported}")
        d = cls()
        hidden_size = vc.get("vt_hidden_size", d.hidden_size)
        if vc.get("mm_hidden_size") not in (None, hidden_size):
            raise ValueError(f"mm_hidden_size {vc['mm_hidden_size']} != vt_hidden_size {hidden_size}")
        if "qkv_hidden_size" in vc and vc["qkv_hidden_size"] is None:
            raise ValueError("qkv_hidden_size null (attention width = vt_hidden_size) is not MoonViT-V2")
        return cls(
            hidden_size=hidden_size,
            intermediate_size=vc.get("vt_intermediate_size", d.intermediate_size),
            num_hidden_layers=vc.get("vt_num_hidden_layers", d.num_hidden_layers),
            num_attention_heads=vc.get("vt_num_attention_heads", d.num_attention_heads),
            qkv_hidden_size=vc.get("qkv_hidden_size", d.qkv_hidden_size),
            patch_size=vc.get("patch_size", d.patch_size),
            pos_emb_interpolation_mode=vc.get("pos_emb_interpolation_mode", d.pos_emb_interpolation_mode),
            attn_bias=vc.get("attn_bias", d.attn_bias),
            linear_bias=vc.get("linear_bias", d.linear_bias),
            patch_embed_proj_bias=vc.get("patch_embed_proj_bias", d.patch_embed_proj_bias),
            merge_kernel_size=tuple(vc.get("merge_kernel_size", d.merge_kernel_size)),
            text_hidden_size=vc.get("text_hidden_size", d.text_hidden_size),
            projector_norm_eps=vc.get("projector_ln_eps", d.projector_norm_eps),
        )


def check_grid(grid_thws: torch.Tensor, config: MoonViTConfig) -> tuple[int, int]:
    """Validate a `grid_thws` of one still image and return its (h, w) in patches.

    One image per call; no packing and no video. The processor pads images to a multiple of
    `merge_kernel_size * patch_size` (28 px), so h and w are multiples of the merge kernel.
    """
    if tuple(grid_thws.shape) != (1, 3):
        raise ValueError(f"one image per call, got grid_thws {grid_thws.tolist()}")
    t, h, w = grid_thws[0].tolist()
    if t != 1:
        raise ValueError(f"video is not supported, got t={t}")
    kh, kw = config.merge_kernel_size
    if h < kh or w < kw or h % kh or w % kw:
        raise ValueError(f"grid {h}x{w} is not a positive multiple of the {kh}x{kw} merge kernel")
    if h > config.max_side_patches or w > config.max_side_patches or h * w > config.max_patches:
        raise ValueError(f"grid {h}x{w} exceeds {config.max_side_patches} per side or {config.max_patches} patches")
    return h, w


class VisionParallel(NamedTuple):
    """Sequence parallelism only: weights are replicated, the sequence is split along `sp_axis`."""

    sp_axis: int | None
    sp_factor: int
    ccl: TT_CCL | None

    @property
    def sp(self) -> bool:
        return self.sp_factor > 1


def resolve_parallel(
    mesh_device: ttnn.MeshDevice,
    sequence_parallel: ParallelFactor | None,
    ccl: TT_CCL | None,
    config: MoonViTConfig,
) -> VisionParallel:
    sp = sequence_parallel
    if sp is None or sp.factor == 1:
        if config.max_patches > ONE_DEVICE_MAX_PATCHES:
            raise ValueError(
                f"max_patches={config.max_patches} on one device: it runs at most one ring chip's share, "
                f"{ONE_DEVICE_MAX_PATCHES} rows"
            )
        return VisionParallel(sp_axis=None, sp_factor=1, ccl=ccl)
    if ccl is None:
        raise ValueError("sequence parallelism needs a TT_CCL")
    if tuple(mesh_device.shape)[sp.mesh_axis] != sp.factor:
        raise ValueError(f"sp factor {sp.factor} != mesh axis {sp.mesh_axis} size {tuple(mesh_device.shape)}")
    if config.max_patches % (sp.factor * config.merge_area * 32):
        raise ValueError(
            f"max_patches={config.max_patches} does not split into tiles over {sp.factor} devices after merging"
        )
    return VisionParallel(sp_axis=sp.mesh_axis, sp_factor=sp.factor, ccl=ccl)


@dataclass(frozen=True)
class Precision:
    """Math fidelity per op group, whether Q/K/V enter ring attention as bfp8, and RoPE's accumulator.

    Names, not ttnn objects, so this module stays ttnn-free; `kernel_configs` builds the configs.
    Every group keeps fp32 accumulation (`fp32_dest_acc_en=True`), and the residual stream stays fp32.

    - `linear_fidelity`: the encoder's wqkv, wo, fc0 and fc1.
    - `projector_fidelity`: the projector's fc0 and fc1.
    - `sdpa_fidelity`: ring attention's compute config. With fp32 accumulation the op takes no separate
      `matmul_math_fidelity`, so this sets both of its matmuls (Q K^T and P V).
    - `qkv_bfp8`: Q, K and V typecast to bfp8 after RoPE (`rotary_embedding_llama` takes bf16 only). The
      op needs one dtype for all three and for its gather buffers, and always outputs bf16.
    - `rope_fp32_dest`: run `rotary_embedding_llama` with fp32 accumulation (`fp32_dest_acc_en=True`,
      HiFi4). The kernel packs `x*cos` and `rot(x)*sin` to bf16 buffers before the add either way, but
      its default 16-bit dest rounds them stochastically; with fp32 dest they round to nearest. On ttsim
      that cuts RoPE's error against exact math from 3.4e-3 to 2.35e-3.

    The norms and the patch embed always run at HiFi4.
    """

    linear_fidelity: str = "HiFi4"
    projector_fidelity: str = "HiFi4"
    sdpa_fidelity: str = "HiFi4"
    qkv_bfp8: bool = False
    rope_fp32_dest: bool = False


# Highest fidelity, no quantization. The default.
PRECISION_BASELINE = Precision()
# HiFi3 linears and ring attention, HiFi2 projector, RoPE with fp32 accumulation; bf16 Q/K/V. Estimated on
# CPU at the baseline's accuracy and ~1.11-1.16x its speed per image. Not measured on silicon.
PRECISION_HIFI3 = Precision(
    linear_fidelity="HiFi3", projector_fidelity="HiFi2", sdpa_fidelity="HiFi3", rope_fp32_dest=True
)
# Ring attention at HiFi2 on bfp8 Q/K/V (typecast after RoPE), HiFi3 linears, HiFi2 projector. Estimated on
# CPU at ~2x the baseline's error and ~1.2-1.35x its speed per image, before counting the halved K/V
# bytes. Not measured on silicon.
PRECISION_HIFI2_BFP8 = Precision(
    linear_fidelity="HiFi3", projector_fidelity="HiFi2", sdpa_fidelity="HiFi2", qkv_bfp8=True
)
PRECISIONS = {"baseline": PRECISION_BASELINE, "hifi3": PRECISION_HIFI3, "hifi2_bfp8": PRECISION_HIFI2_BFP8}
PRECISION_ENV = "MOONVIT_PRECISION"


def resolve_precision(precision: Precision | str | None = None) -> Precision:
    """`precision` itself, the preset of that name, or with None the preset named by $MOONVIT_PRECISION
    (`baseline` when unset), so evals and the device tests can compare presets without code changes.

    The variable is read on each call, not at import, so a bad value fails only the device path and
    not the host-only image prep that also imports this module."""
    if isinstance(precision, Precision):
        return precision
    source = "precision"
    if precision is None:
        source, precision = PRECISION_ENV, os.environ.get(PRECISION_ENV, "baseline")
    if precision not in PRECISIONS:
        raise ValueError(f"{source}={precision!r}; expected one of {sorted(PRECISIONS)}")
    return PRECISIONS[precision]


class KernelConfigs(NamedTuple):
    """Compute kernel configs per op group (see `Precision`) and the dtype of Q/K/V into ring attention."""

    default: object  # norms, patch embed
    linear: object  # encoder wqkv, wo, fc0, fc1
    sdpa: object
    projector: object
    qkv_dtype: object = None  # None: bf16, as RoPE and `nlp_create_qkv_heads` produce
    rope: object = None  # None: the op's default (HiFi4, bf16 dest)

    @classmethod
    def uniform(cls, compute_kernel_config) -> KernelConfigs:
        """`compute_kernel_config` for every op group, bf16 Q/K/V, and RoPE at the op's default."""
        return cls(compute_kernel_config, compute_kernel_config, compute_kernel_config, compute_kernel_config)


def compute_kernel_config(mesh_device: ttnn.MeshDevice, math_fidelity: str = "HiFi4"):
    """`math_fidelity` (HiFi4 by default) with fp32 accumulation."""
    import ttnn

    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, math_fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def kernel_configs(mesh_device: ttnn.MeshDevice, precision: Precision | str | None = None) -> KernelConfigs:
    """The compute configs and Q/K/V dtype of `precision` (see `resolve_precision`)."""
    import ttnn

    precision = resolve_precision(precision)
    return KernelConfigs(
        default=compute_kernel_config(mesh_device),
        linear=compute_kernel_config(mesh_device, precision.linear_fidelity),
        sdpa=compute_kernel_config(mesh_device, precision.sdpa_fidelity),
        projector=compute_kernel_config(mesh_device, precision.projector_fidelity),
        qkv_dtype=ttnn.bfloat8_b if precision.qkv_bfp8 else None,
        rope=compute_kernel_config(mesh_device) if precision.rope_fp32_dest else None,
    )

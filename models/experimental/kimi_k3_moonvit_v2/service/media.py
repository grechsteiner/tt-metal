# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3 media module for the encoder service.

The service loads the class by import path (`models.experimental.kimi_k3_moonvit_v2.service.media:KimiK3Media`),
builds it with the checkpoint's config.json and calls only the members below. The media key, the store, request
limits, image decoding and batching belong to the service; this module enforces only the model's own rules, in
`prepare`.

The encoder runs on a (1, 4) ring of Blackhole chips, since attention is ring attention. Every image is padded to
`max_patches` (65,536) patch rows and `logical_n` is a device tensor, so the one trace captured in `load` serves every
image size.

ttnn is imported only where the chips are used, in `device_params` and `load`: `prepare`, `expansion` and
`stored_nbytes` must also run in a process without ttnn. With the option `stub_embeddings`, `load` and `encode` write
seeded stand-in embeddings and use no chip, so the service and the prefill side can be tested without a ring.

The service must not call `load`, `encode` or `close` from several threads at once: they share the device inputs and
the one captured trace.
"""

from __future__ import annotations

import gc
import hashlib
import os
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from models.experimental.kimi_k3_moonvit_v2.service import format as media_format
from models.experimental.kimi_k3_moonvit_v2.tt.image_prep import image_to_patches
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import RING_SIZE, MoonViTConfig

if TYPE_CHECKING:
    from models.experimental.kimi_k3_moonvit_v2.tt.trace import Trace
    from models.experimental.kimi_k3_moonvit_v2.tt.vision_model import TT_MoonViT_Vision_Model

# The ring is along mesh axis 1.
SP_AXIS = 1


@dataclass(frozen=True)
class K3Image:
    pixel_values: torch.Tensor
    """(h*w, 3, 14, 14) float32, as HF's processor returns it."""
    grid_thws: torch.Tensor
    """[[1, h, w]], the grid in patches."""
    n_tokens: int
    """Embeddings: one per 2 x 2 window of patches, h*w/4."""


def _clear_frames(error: BaseException) -> None:
    """Clear the local variables of the frames in error's traceback, and in those of the errors it chains.

    The frames of a failed load or encode hold device tensors, which the caller holding the error would keep alive
    after close. The tracebacks still show their lines."""
    pending, seen = [error], set()
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        pending += [error.__cause__, error.__context__]


def _stub_rows(item: K3Image, hidden: int) -> np.ndarray:
    """Stand-in embeddings for item, seeded by its patches and grid: bf16 bits as uint16, [n_tokens, hidden]."""
    digest = hashlib.sha256(item.grid_thws.numpy().tobytes() + item.pixel_values.numpy().tobytes()).digest()
    seed = int.from_bytes(digest[:8], "little")
    rows = torch.randn(item.n_tokens, hidden, generator=torch.Generator().manual_seed(seed)) * 0.02
    return rows.to(torch.bfloat16).view(torch.uint16).numpy()


def _check_options(options: dict) -> None:
    if not isinstance(options, dict):
        raise ValueError(f"options must be a dict, got {type(options).__name__}")
    cache_dir = options.get("cache_dir")
    if cache_dir is not None and not isinstance(cache_dir, (str, os.PathLike)):
        raise ValueError(f"cache_dir must be a folder path, got {cache_dir!r}")
    if not isinstance(options.get("stub_embeddings", False), bool):
        raise ValueError(f"stub_embeddings must be a bool, got {options['stub_embeddings']!r}")


class KimiK3Media:
    """The Kimi-K3 media module that the encoder service calls.

    Not thread safe: the service must not call load, encode or close from several threads at once."""

    @property
    def device_params(self) -> dict:
        """How the service opens the mesh: the ring's fabric, and DRAM for the one captured trace.

        Reading it imports ttnn (the fabric config is a ttnn enum); prepare, expansion and stored_nbytes do not."""
        from models.experimental.kimi_k3_moonvit_v2.tt.ccl import RING_FABRIC_CONFIG
        from models.experimental.kimi_k3_moonvit_v2.tt.trace import TRACE_REGION_SIZE

        return {"fabric_config": RING_FABRIC_CONFIG, "trace_region_size": TRACE_REGION_SIZE}

    def __init__(self, hf_config: dict):
        # Raises for a vision_config that is not MoonViT-V2.
        self.config = MoonViTConfig.from_hf_config(hf_config)
        self.hidden = self.config.text_hidden_size
        # <|media_pad|>: the chat template emits one per image, and expansion repeats it once per embedding.
        self.placeholder_token_ids = {"image": hf_config["media_placeholder_token_id"]}
        self._model: TT_MoonViT_Vision_Model | None = None
        self._trace: Trace | None = None
        self._stub = False

    def prepare(self, kind: str, media: np.ndarray) -> K3Image:
        """Resize, pad and patchify one image on the CPU as HF's processor does, bit for bit (`tt/image_prep.py`).

        media: non-empty uint8 RGB [height, width, 3]. An image whose grid is over 65,536 patches or 512 patches a
        side raises ValueError: it is rejected, never shrunk.
        """
        if kind not in self.placeholder_token_ids:
            raise ValueError(f"Kimi-K3 does not accept {kind}")
        if media.dtype != np.uint8 or media.ndim != 3 or media.shape[2] != 3 or 0 in media.shape:
            raise ValueError(
                f"expected a non-empty uint8 RGB [height, width, 3] array, got {media.dtype} {media.shape}"
            )
        pixel_values, grid_thws = image_to_patches(Image.fromarray(media), self.config)
        _, h, w = grid_thws[0].tolist()
        return K3Image(pixel_values, grid_thws, h * w // self.config.merge_area)

    def expansion(self, item: K3Image) -> list[tuple[int, int]]:
        """The run of placeholders that replaces the template's one <|media_pad|>. The tokens around it
        (<|media_begin|>, the "image WxH" text, <|media_content|>, <|media_end|>) come from the chat template."""
        return [(self.placeholder_token_ids["image"], item.n_tokens)]

    def stored_nbytes(self, item: K3Image) -> int:
        return media_format.stored_nbytes(item.n_tokens, self.hidden)

    def load(self, mesh_device, weights_path: str, kind: str, options: dict) -> None:
        """Put the vision weights on `mesh_device`, run the encoder once and capture its trace.

        mesh_device: the (1, 4) ring, opened by the service with `device_params`; it is not opened, closed or reset
        here, and any other width along axis 1 raises ValueError. weights_path: the Kimi-K3 checkpoint folder; only
        the shards that hold the vision tower and projector are read. options:
        - `cache_dir` (optional): a folder that keeps the converted weights for the next load
          (`tt/module.py:load_cached`); valid only for this mesh shape.
        - `stub_embeddings` (optional, bool): encode writes seeded stand-in embeddings; mesh_device and weights_path
          are not used.

        Raises ValueError before ttnn is imported for another kind, bad options, a weights_path that is not a folder or
        a mesh_device of None; RuntimeError for a second load before close. A failed load frees what it put on the
        chips before it raises.
        """
        if kind not in self.placeholder_token_ids:
            raise ValueError(f"Kimi-K3 does not accept {kind}")
        if self._stub or self._model is not None or self._trace is not None:
            raise RuntimeError("load after load: close first")
        _check_options(options)
        if options.get("stub_embeddings"):
            self._stub = True
            return
        if not weights_path or not os.path.isdir(weights_path):
            raise ValueError(f"weights_path must be a Kimi-K3 checkpoint folder, got {weights_path!r}")
        if mesh_device is None:
            raise ValueError("mesh_device is None: load runs the encoder on the ring that the service opened")
        import ttnn
        from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import load_vision_state_dict
        from models.experimental.kimi_k3_moonvit_v2.tt.ccl import RING_NUM_LINKS, TT_CCL, ParallelFactor
        from models.experimental.kimi_k3_moonvit_v2.tt.module import load_cached
        from models.experimental.kimi_k3_moonvit_v2.tt.trace import Trace
        from models.experimental.kimi_k3_moonvit_v2.tt.vision_model import TT_MoonViT_Vision_Model

        try:
            ccl = TT_CCL(mesh_device, num_links=RING_NUM_LINKS, topology=ttnn.Topology.Ring)
            model = self._model = TT_MoonViT_Vision_Model(
                self.config, mesh_device=mesh_device, sequence_parallel=ParallelFactor(RING_SIZE, SP_AXIS), ccl=ccl
            )
            load_cached(model, options.get("cache_dir"), lambda: load_vision_state_dict(weights_path))
            # The inputs are written in place on every call, so a trace captured on this image replays any other.
            inputs = model.prepare_inputs(*image_to_patches(Image.new("RGB", (28, 28)), self.config))
            # Untraced first: that compiles the programs and allocates the ring-attention buffers, neither of which
            # can happen inside a capture. forward_rows untilizes on the chips, inside the trace, so the replay's
            # output is read back as plain rows.
            ttnn.deallocate(model.forward_rows(**inputs))
            self._trace = Trace(mesh_device, lambda: model.forward_rows(**inputs))
        except BaseException as error:
            self.close()
            _clear_frames(error)  # the failed call's frames hold device tensors
            gc.collect()
            raise

    def encode(self, items: list[K3Image], out: list[memoryview]) -> None:
        """Write each item's stored value into its slot in out: format.Header("image"), then its bf16 embeddings.

        out holds one slot per item: a writable, C-contiguous memoryview of `stored_nbytes(item)` bytes, in any format.
        Every slot's size is checked before anything is written. Each chip's rows are copied straight into the slot
        (`TT_MoonViT_Vision_Model.read_output`); the waits on the chips release the GIL. If encode raises, the caller
        must not use the slots of that call.

        Raises ValueError for a slot of the wrong size, and RuntimeError before load or after close.
        """
        if len(items) != len(out):
            raise ValueError(f"{len(items)} items for {len(out)} slots")
        for item, buf in zip(items, out):
            size = self.stored_nbytes(item)
            if buf.nbytes != size:
                raise ValueError(
                    f"a slot of {buf.nbytes} B for an image of {item.n_tokens} tokens, which needs {size} B"
                )
        if not self._stub and self._trace is None:
            raise RuntimeError("encode before load, or after close")
        for item, buf in zip(items, out):
            buf = buf.cast("B")  # the header and the rows are at byte offsets, whatever the slot's format
            media_format.Header("image").write(buf)
            rows = media_format.embeddings(buf, self.hidden)
            if self._stub:
                rows[:] = _stub_rows(item, self.hidden)
                continue
            try:
                self._model.prepare_inputs(item.pixel_values, item.grid_thws)
                self._model.read_output(self._trace.replay(), rows)
            except BaseException as error:
                _clear_frames(error)  # so that close, while the caller holds the error, frees the device tensors
                raise

    def close(self) -> None:
        """Release the trace and drop the weights and the device inputs. Safe after a failed load or encode, and
        when called twice. The mesh is the service's: it stays open."""
        trace, self._trace = self._trace, None
        loaded = trace is not None or self._model is not None
        self._model = None
        self._stub = False
        if trace is not None:
            trace.release()
        if loaded:
            gc.collect()  # device tensors that only reference cycles still hold

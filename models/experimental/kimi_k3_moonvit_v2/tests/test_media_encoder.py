# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: the Kimi-K3 media module's load, encode and close on the ring, as the encoder service calls them, against
the fp32 reference. Two images of different sizes go through the one captured trace."""

from __future__ import annotations

import copy
import json

import numpy as np
import torch
from safetensors.torch import save_file

from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import checkpoint_path
from models.experimental.kimi_k3_moonvit_v2.service import format as media_format
from models.experimental.kimi_k3_moonvit_v2.service.media import KimiK3Media
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    MEDIA_CONFIG,
    REAL,
    REAL_MARK,
    RING_TRACE,
    SIMULATOR,
    SYNTHETIC,
    assert_quality,
    random_array,
)
from models.experimental.kimi_k3_moonvit_v2.tests.goldens import to_fp32

SHARD = "model-00001-of-00001.safetensors"
# (height, width) in pixels: 4 x 6 and 10 x 4 patches after the padding, so the second image has a grid the trace
# was not captured with.
IMAGES = [(56, 84), (130, 50)]


def write_checkpoint(reference, directory) -> str:
    """`reference`'s weights as a one-shard checkpoint folder, as `load_vision_state_dict` reads it."""
    state = {key: value.contiguous().clone() for key, value in reference.state_dict().items()}
    save_file(state, directory / SHARD)
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: SHARD for k in state}}))
    return str(directory)


def encode(media: KimiK3Media, array: np.ndarray):
    item = media.prepare("image", array)
    buf = memoryview(bytearray(media.stored_nbytes(item)))
    media.encode([item], [buf])
    rows = torch.from_numpy(media_format.embeddings(buf, media.hidden).copy()).view(torch.bfloat16)
    return item, rows


@RING_TRACE
@SYNTHETIC
@REAL
def test_load_encode_close(mesh_device, sp_axis, num_links, weights, reference, tmp_path, expect_error):
    weights_path = str(checkpoint_path()) if weights == REAL_MARK else write_checkpoint(reference, tmp_path)
    golden_model = to_fp32(copy.deepcopy(reference))
    media = KimiK3Media(MEDIA_CONFIG)

    media.load(mesh_device, weights_path, "image", {})
    try:
        for seed, hw in enumerate(IMAGES):
            item, rows = encode(media, random_array(*hw, seed=seed))
            assert rows.shape == (item.n_tokens, media.hidden)
            with torch.no_grad():
                (golden,) = golden_model(item.pixel_values, item.grid_thws)
            assert_quality(golden, rows, relative_rmse=0.07)
    finally:
        media.close()
    media.close()  # a second close is a no-op

    item = media.prepare("image", random_array(*IMAGES[0]))
    with expect_error(RuntimeError, "encode before load"):
        media.encode([item], [memoryview(bytearray(media.stored_nbytes(item)))])


@SIMULATOR
def test_load_rejects_one_chip(mesh_device, sp_axis, num_links, expect_error):
    with expect_error(ValueError, r"sp factor 4 != mesh axis 1 size \(1, 1\)"):
        KimiK3Media(MEDIA_CONFIG).load(mesh_device, "", "image", {})

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU: the Kimi-K3 media module (`service/media.py`) and its stored-value format, against the HF processor
copied into `reference/`. The device side, load and encode, is `test_media_encoder.py`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision_processing import KimiK3VisionProcessor
from models.experimental.kimi_k3_moonvit_v2.service import format as media_format
from models.experimental.kimi_k3_moonvit_v2.service.media import KimiK3Media
from models.experimental.kimi_k3_moonvit_v2.tests.common import MEDIA_CONFIG, MEDIA_PAD, random_array

REFERENCE_DIR = Path(__file__).parents[1] / "reference"
TEXT_HIDDEN = 7168


@pytest.fixture(scope="module")
def processor() -> KimiK3VisionProcessor:
    cfg = json.loads((REFERENCE_DIR / "preprocessor_config.json").read_text())["media_proc_cfg"]
    return KimiK3VisionProcessor(media_proc_cfg=cfg)


# (height, width): unresized, odd sizes the padding rounds up, and resized by the side limit (7168 px).
@pytest.mark.parametrize("hw", [(1024, 1024), (20, 33), (48, 49), (1080, 1920), (300, 8000), (1, 1)])
def test_prepare_matches_hf(processor, hw):
    media = KimiK3Media(MEDIA_CONFIG)
    array = random_array(*hw)
    item = media.prepare("image", array)

    want = processor.preprocess([{"type": "image", "image": Image.fromarray(array)}])
    assert torch.equal(item.pixel_values, want["pixel_values"])
    assert item.grid_thws.tolist() == want["grid_thws"].tolist()
    _, h, w = item.grid_thws[0].tolist()
    assert item.n_tokens == h * w // 4
    assert media.placeholder_token_ids == {"image": MEDIA_PAD}
    assert media.expansion(item) == [(MEDIA_PAD, item.n_tokens)]
    assert media.stored_nbytes(item) == item.n_tokens * TEXT_HIDDEN * 2


@pytest.mark.parametrize(
    "kind,array,message",
    [
        ("video", random_array(28, 28), "does not accept video"),
        ("image", np.zeros((28, 28, 4), np.uint8), "RGB"),
        ("image", np.zeros((28, 28, 3), np.float32), "uint8"),
        ("image", np.zeros((28, 28), np.uint8), "uint8"),
        ("image", random_array(0, 28), "non-empty"),
        # 258 x 258 patches after the 28 px padding: over 65,536, rejected rather than shrunk.
        ("image", np.zeros((3590, 3590, 3), np.uint8), "image 3590x3590: grid 258x258 exceeds"),
    ],
)
def test_prepare_rejects(expect_error, kind, array, message):
    with expect_error(ValueError, message):
        KimiK3Media(MEDIA_CONFIG).prepare(kind, array)


def test_prepare_does_not_import_ttnn():
    """prepare, expansion and stored_nbytes run where the service prepares images, which may have no ttnn. A fresh
    interpreter, since this one has ttnn imported already."""
    code = (
        "import sys, numpy as np\n"
        "from models.experimental.kimi_k3_moonvit_v2.service.media import KimiK3Media\n"
        f"media = KimiK3Media({MEDIA_CONFIG!r})\n"
        "item = media.prepare('image', np.zeros((30, 40, 3), np.uint8))\n"
        "media.expansion(item), media.stored_nbytes(item)\n"
        "assert 'ttnn' not in sys.modules, 'ttnn imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[4])


def test_device_params():
    import ttnn

    params = KimiK3Media(MEDIA_CONFIG).device_params
    assert params["fabric_config"] == ttnn.FabricConfig.FABRIC_2D_TORUS_X
    assert params["trace_region_size"] > 0


def test_encode_before_load(expect_error):
    media = KimiK3Media(MEDIA_CONFIG)
    item = media.prepare("image", random_array(28, 28))
    with expect_error(RuntimeError, "encode before load"):
        media.encode([item], [memoryview(bytearray(media.stored_nbytes(item)))])
    media.close()  # without a load: a no-op


def test_format_rejects_bad_values(expect_error):
    buf = memoryview(bytearray(media_format.stored_nbytes(6, 64)))
    assert media_format.embeddings(buf, 64).shape == (6, 64)
    with expect_error(ValueError, "whole bf16 rows"):
        media_format.embeddings(buf[:-2], 64)
    with expect_error(ValueError, "whole bf16 rows"):
        media_format.embeddings(buf[:0], 64)

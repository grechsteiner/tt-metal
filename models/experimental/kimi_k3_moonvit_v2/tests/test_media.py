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
    assert media.stored_nbytes(item) == 16 + item.n_tokens * TEXT_HIDDEN * 2


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
    assert media_format.stored_nbytes(6, 64) == 16 + 6 * 64 * 2
    buf = memoryview(bytearray(media_format.stored_nbytes(6, 64)))
    media_format.Header("image").write(buf)
    assert media_format.Header.read(buf) == media_format.Header("image")
    rows = media_format.embeddings(buf, 64)
    assert rows.shape == (6, 64)
    rows[:] = 0xFFFF
    assert bytes(buf[:16]) == bytes(16)  # the rows start after the header
    with expect_error(ValueError, "whole bf16 rows"):
        media_format.embeddings(buf[:-2], 64)
    with expect_error(ValueError, "whole bf16 rows"):
        media_format.embeddings(buf[:16], 64)
    with expect_error(ValueError, "shorter than its 16 B header"):
        media_format.Header.read(buf[:15])
    buf[0] = len(media_format.KINDS)
    with expect_error(ValueError, "unknown media kind"):
        media_format.Header.read(buf)


def stub_media() -> KimiK3Media:
    media = KimiK3Media(MEDIA_CONFIG)
    media.load(None, "", "image", {"stub_embeddings": True})
    return media


def slot(media: KimiK3Media, item, extra: int = 0) -> memoryview:
    return memoryview(bytearray(media.stored_nbytes(item) + extra))


def test_stub_encode_is_deterministic():
    """With stub_embeddings, load and encode use no chip: the stored value is a header and seeded rows."""
    items = [KimiK3Media(MEDIA_CONFIG).prepare("image", random_array(56, 84, seed=s)) for s in (0, 1)]
    values = []
    for _ in range(2):
        media = stub_media()
        bufs = [slot(media, item) for item in items]
        media.encode(items, bufs)
        media.close()
        values.append([bytes(buf) for buf in bufs])
    assert values[0] == values[1]
    first, second = values[0]
    assert first != second  # seeded by the image
    assert media_format.Header.read(memoryview(first)).kind == "image"
    rows = torch.from_numpy(media_format.embeddings(memoryview(first), TEXT_HIDDEN).copy()).view(torch.bfloat16)
    assert rows.shape == (items[0].n_tokens, TEXT_HIDDEN)
    assert torch.isfinite(rows).all() and rows.abs().max() > 0


def test_encode_checks_every_slot_first(expect_error):
    """A wrong-sized slot raises before any slot is written."""
    media = stub_media()
    items = [media.prepare("image", random_array(56, 84, seed=s)) for s in (0, 1)]
    bufs = [slot(media, items[0]), slot(media, items[1], extra=2)]
    with expect_error(ValueError, "a slot of"):
        media.encode(items, bufs)
    assert bytes(bufs[0]) == bytes(len(bufs[0]))
    with expect_error(ValueError, "2 items for 1 slots"):
        media.encode(items, bufs[:1])
    media.close()


def test_encode_takes_slots_of_any_format():
    """A slot is cast to bytes: a uint16 memoryview gets the same value as a byte one."""
    media = stub_media()
    item = media.prepare("image", random_array(56, 84))
    as_bytes, as_uint16 = slot(media, item), slot(media, item).cast("H")
    media.encode([item, item], [as_bytes, as_uint16])
    assert bytes(as_bytes) == as_uint16.tobytes()
    media.close()


def test_load_twice(expect_error):
    media = stub_media()
    with expect_error(RuntimeError, "load after load: close first"):
        media.load(None, "", "image", {"stub_embeddings": True})
    media.close()
    media.load(None, "", "image", {"stub_embeddings": True})  # after close, a load is fine
    media.close()
    item = media.prepare("image", random_array(28, 28))
    with expect_error(RuntimeError, "after close"):
        media.encode([item], [slot(media, item)])


@pytest.mark.parametrize(
    "weights_path,mesh,options,message",
    [
        ("", object(), {}, "weights_path must be a Kimi-K3 checkpoint folder"),
        ("/nonexistent/k3", object(), {}, "weights_path must be a Kimi-K3 checkpoint folder"),
        (".", None, {}, "mesh_device is None"),
        (".", object(), {"cache_dir": 3}, "cache_dir must be a folder path"),
        (".", object(), {"stub_embeddings": "yes"}, "stub_embeddings must be a bool"),
    ],
)
def test_load_rejects_before_ttnn(expect_error, weights_path, mesh, options, message):
    """Bad inputs raise before load touches ttnn or the chips."""
    media = KimiK3Media(MEDIA_CONFIG)
    with expect_error(ValueError, message):
        media.load(mesh, weights_path, "image", options)
    with expect_error(ValueError, "does not accept video"):
        media.load(mesh, weights_path, "video", options)


def test_stub_does_not_import_ttnn():
    """A stub load and encode, as the service runs them without a ring, import no ttnn."""
    code = (
        "import sys, numpy as np\n"
        "from models.experimental.kimi_k3_moonvit_v2.service.media import KimiK3Media\n"
        f"media = KimiK3Media({MEDIA_CONFIG!r})\n"
        "media.load(None, '', 'image', {'stub_embeddings': True})\n"
        "item = media.prepare('image', np.zeros((30, 40, 3), np.uint8))\n"
        "media.encode([item], [memoryview(bytearray(media.stored_nbytes(item)))])\n"
        "media.close()\n"
        "assert 'ttnn' not in sys.modules, 'ttnn imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[4])

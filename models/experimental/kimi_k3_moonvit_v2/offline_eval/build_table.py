"""Concatenate the saved bf16 image rows into the extra embedding-table rows and write the manifest.

  python -m models.experimental.kimi_k3_moonvit_v2.offline_eval.build_table \
      --images DIR [DIR ...] --embeddings DIR --hf-files DIR --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision_processing import KimiK3VisionProcessor

FIRST_TEMP_ID = 163840
HIDDEN = 7168
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_key(image: Image.Image) -> str:
    h = hashlib.sha256(f"{image.mode}:{image.size[0]}x{image.size[1]}".encode())
    if image.mode == "P":
        h.update(bytes(image.getpalette() or []))
    if "transparency" in image.info:
        h.update(repr(image.info["transparency"]).encode())
    h.update(image.tobytes())
    return h.hexdigest()


def rgba_key(image: Image.Image) -> str:
    rgba = image.convert("RGBA")
    h = hashlib.sha256(f"{rgba.size[0]}x{rgba.size[1]}".encode())
    h.update(rgba.tobytes())
    return h.hexdigest()


def keys_of(data: bytes) -> dict:
    import io

    image = Image.open(io.BytesIO(data))
    image.load()
    return {"file_sha256": file_sha256(data), "strict_key": strict_key(image), "rgba_key": rgba_key(image)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--hf-files", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_files), trust_remote_code=True)

    images = sorted(p for d in args.images for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    rows, entries, skipped = [], [], []
    next_id = FIRST_TEMP_ID
    for path in images:
        emb_path = args.embeddings / f"{path.stem}.pt"
        if not emb_path.exists():
            skipped.append(path.name)
            continue
        emb = torch.load(emb_path, map_location="cpu", weights_only=True)
        assert emb.dtype == torch.bfloat16 and emb.dim() == 2 and emb.shape[1] == HIDDEN, emb.shape
        data = path.read_bytes()
        image = Image.open(path)
        w, h = image.size
        entries.append(
            {
                "image": path.name,
                "mode": image.mode,
                "size": [w, h],
                **keys_of(data),
                "first_id": next_id,
                "rows": emb.shape[0],
                "image_prompt": KimiK3VisionProcessor.make_image_prompt(w, h),
                "dims_token_ids": tokenizer.encode(f"image {w}x{h}", allow_special_tokens=False),
            }
        )
        rows.append(emb)
        next_id += emb.shape[0]

    table = torch.cat(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(table, args.out / "extra_rows.pt")
    manifest = {
        "first_temp_id": FIRST_TEMP_ID,
        "total_rows": table.shape[0],
        "vocab_size_with_extra": FIRST_TEMP_ID + table.shape[0],
        "extra_bytes": table.numel() * table.element_size(),
        "extra_rows_sha256": hashlib.sha256(table.view(torch.int16).numpy().tobytes()).hexdigest(),
        "skipped_no_embedding": skipped,
        "images": entries,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        f"{len(entries)} images, {table.shape[0]} rows, {manifest['extra_bytes'] / 2**30:.2f} GiB, "
        f"ids {FIRST_TEMP_ID}..{next_id - 1}, skipped {skipped}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

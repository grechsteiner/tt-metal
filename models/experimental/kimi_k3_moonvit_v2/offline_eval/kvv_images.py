"""Write the exact image bytes the Kimi Vendor Verifier sends for OCRBench and MMMU-Pro (10 options).

Runs KVV's own dataset loaders (ocr_bench.record_to_sample, mmmu_pro_vision.load_mmmu_pro_dataset), takes the
data URL of every image in every sample, and writes each distinct payload once. index.json maps each sample to its
images in prompt order. Run it with KVV's own environment, the one the eval runs with:

  HF_HOME=... <kvv>/.venv/bin/python kvv_images.py --kvv <kvv checkout> --out DIR
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def data_url_bytes(url: str) -> tuple[str, bytes]:
    head, sep, payload = url.partition(",")
    if not url.startswith("data:") or not sep or not head.endswith(";base64"):
        raise ValueError(f"not a base64 data URL: {url[:40]}")
    return head[5:].split(";")[0], base64.b64decode(payload, validate=True)


def environment(kvv: Path) -> dict:
    import datasets
    import inspect_ai
    import PIL
    from PIL import features

    commit = subprocess.run(["git", "-C", str(kvv), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "kvv_commit": commit,
        "pillow": PIL.__version__,
        "libjpeg_turbo": features.version("libjpeg_turbo"),
        "inspect_ai": inspect_ai.__version__,
        "datasets": datasets.__version__,
        "python": sys.version.split()[0],
    }


def dataset_revision(path: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(path).sha
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def write_set(name: str, samples, out: Path) -> dict:
    from inspect_ai.model import ContentImage

    images_dir = out / name / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    mimes: dict[str, int] = {}
    index = []
    for position, sample in enumerate(samples):
        files = []
        for message in sample.input:
            for part in message.content if isinstance(message.content, list) else []:
                if not isinstance(part, ContentImage):
                    continue
                mime, data = data_url_bytes(part.image)
                mimes[mime] = mimes.get(mime, 0) + 1
                sha = hashlib.sha256(data).hexdigest()
                if sha not in seen:
                    seen[sha] = f"{len(seen):05d}_{sha[:12]}.jpg"
                    (images_dir / seen[sha]).write_bytes(data)
                files.append(seen[sha])
        index.append({"position": position, "id": sample.id, "images": files})
    (out / name / "index.json").write_text(json.dumps(index, indent=1))
    return {
        "samples": len(index),
        "image_refs": sum(len(s["images"]) for s in index),
        "unique_images": len(seen),
        "data_url_mime": mimes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.kvv))
    import mmmu_pro_vision
    import ocr_bench
    from inspect_ai.dataset import hf_dataset

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"environment": environment(args.kvv), "sets": {}}
    ocr = hf_dataset(path=ocr_bench.DATASET_PATH, split="test", sample_fields=ocr_bench.record_to_sample)
    meta["sets"]["ocrbench"] = write_set("ocrbench", ocr, args.out) | {
        "dataset": ocr_bench.DATASET_PATH,
        "revision": dataset_revision(ocr_bench.DATASET_PATH),
    }
    mmmu = mmmu_pro_vision.load_mmmu_pro_dataset()
    meta["sets"]["mmmu_pro_10c"] = write_set("mmmu_pro_10c", mmmu, args.out) | {
        "dataset": f"{mmmu_pro_vision.MMMU_PRO_10c_DATASET} / {mmmu_pro_vision.MMMU_PRO_10c_SUBSET}",
        "revision": dataset_revision(mmmu_pro_vision.MMMU_PRO_10c_DATASET),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

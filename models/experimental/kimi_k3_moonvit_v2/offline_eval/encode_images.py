"""Offline image embeddings for the K3 temporary-token eval hack (verification only, not for merge).

For each image: compare the host prep options (tt image_prep, HF processor on the image as opened, HF on
.convert("RGB") first), run the TT encoder on the LoudBox ring, compare against an fp32 CPU reference,
and save the bf16 rows. Run from the tt-metal root with KIMI_K3_HF_MODEL set.

  python -m models.experimental.kimi_k3_moonvit_v2.offline_eval.encode_images \
      --images DIR [DIR ...] --golden-dir DIR --out DIR [--cpu-only] [--no-reference] [--only PREFIX,...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from PIL import Image

from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import checkpoint_path
from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision_processing import KimiK3VisionProcessor
from models.experimental.kimi_k3_moonvit_v2.tests.common import REAL_MARK, build_reference, quality
from models.experimental.kimi_k3_moonvit_v2.tests.goldens import to_fp32, use_chunked_attention
from models.experimental.kimi_k3_moonvit_v2.tt.image_prep import image_to_patches
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import MoonViTConfig

PREPROCESSOR_CONFIG = Path(__file__).resolve().parents[1] / "reference" / "preprocessor_config.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CPU_REFERENCE_MAX_PATCHES = 20000


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image.load()
    return image


def hf_prep(processor, image):
    out = processor.preprocess([{"type": "image", "image": image}])
    return out["pixel_values"].float(), out["grid_thws"]


def prep_variants(image, config, processor) -> dict:
    variants = {}
    for name, fn in (
        ("tt", lambda: image_to_patches(image.copy(), config)),
        ("hf", lambda: hf_prep(processor, image.copy())),
        ("hf_rgb_first", lambda: hf_prep(processor, image.copy().convert("RGB"))),
    ):
        try:
            variants[name] = fn()
        except ValueError as e:
            variants[name] = str(e)
    return variants


def compare_prep(a, b) -> dict:
    if isinstance(a, str) or isinstance(b, str):
        return {"comparable": False}
    grid_match = torch.equal(a[1], b[1])
    r = {"comparable": True, "grid_match": grid_match, "bit_exact": grid_match and torch.equal(a[0], b[0])}
    if grid_match and not r["bit_exact"]:
        diff = (a[0] - b[0]).abs()
        r["max_abs"] = diff.max().item()
        r["frac_diff"] = (diff > 0).float().mean().item()
    return r


def token_cosine(golden: torch.Tensor, actual: torch.Tensor) -> dict:
    cos = torch.nn.functional.cosine_similarity(golden.double(), actual.double(), dim=-1)
    return {
        "cos_mean": cos.mean().item(),
        "cos_min": cos.min().item(),
        "cos_p01": cos.quantile(0.01).item() if cos.numel() > 1 else cos.item(),
    }


def compare_out(golden: torch.Tensor, actual: torch.Tensor) -> dict:
    if golden.shape != actual.shape:
        return {"shape_mismatch": [list(golden.shape), list(actual.shape)]}
    q = quality(golden.float(), actual.float())
    return {k: q[k] for k in ("pcc", "relative_rmse", "max_abs", "nonfinite")} | token_cosine(golden, actual)


def open_ring():
    import ttnn
    from tests.scripts.common import get_updated_device_params

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_X, ttnn.FabricReliabilityMode.STRICT_INIT)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4), **get_updated_device_params({}))
    ring = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    return parent, ring


def close_ring(parent, ring):
    import ttnn

    ttnn.close_mesh_device(ring)
    ttnn.close_mesh_device(parent)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def build_tt_model(mesh, config, reference):
    import ttnn
    from models.experimental.kimi_k3_moonvit_v2.tt.ccl import ParallelFactor, TT_CCL
    from models.experimental.kimi_k3_moonvit_v2.tt.vision_model import TT_MoonViT_Vision_Model

    ccl = TT_CCL(mesh, num_links=2, topology=ttnn.Topology.Ring)
    tt = TT_MoonViT_Vision_Model(config, mesh_device=mesh, sequence_parallel=ParallelFactor(4, 1), ccl=ccl)
    tt.load_torch_state_dict(reference.state_dict())
    return tt


def cpu_reference(fp32_reference, pixels, grid_thws, cache: Path) -> torch.Tensor:
    if cache.exists():
        return torch.load(cache, weights_only=True)
    with torch.no_grad(), use_chunked_attention():
        out = fp32_reference(pixels.float(), grid_thws)[0].float()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--golden-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    path = checkpoint_path()
    if path is None:
        raise SystemExit("set KIMI_K3_HF_MODEL")
    config = MoonViTConfig.from_hf_config(path)
    processor = KimiK3VisionProcessor(media_proc_cfg=json.loads(PREPROCESSOR_CONFIG.read_text())["media_proc_cfg"])

    images = sorted(p for d in args.images for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.only:
        images = [p for p in images if p.name.startswith(tuple(args.only.split(",")))]
    (args.out / "embeddings").mkdir(parents=True, exist_ok=True)

    reference = build_reference(REAL_MARK)
    tt = parent = ring = None
    if not args.cpu_only:
        parent, ring = open_ring()
        t0 = time.perf_counter()
        tt = build_tt_model(ring, config, reference)
        logger.info(f"TT model built in {time.perf_counter() - t0:.0f} s")
    fp32_reference = to_fp32(reference)

    results = []
    results_path = args.out / ("results_cpu.json" if args.cpu_only else "results.json")
    try:
        for img_path in images:
            image = load_image(img_path)
            r = {"image": img_path.name, "mode": image.mode, "size": list(image.size)}
            results.append(r)
            variants = prep_variants(image, config, processor)
            hf = variants["hf"]
            r["prep"] = {
                "tt_vs_hf": compare_prep(variants["tt"], hf),
                "hf_rgb_first_vs_hf": compare_prep(variants["hf_rgb_first"], hf),
            }
            if isinstance(variants["tt"], str):
                r["tt_prep_error"] = variants["tt"]
            if isinstance(hf, str):
                r["status"] = f"HF prep error: {hf}"
                continue
            grid = hf[1][0].tolist()
            r["grid_thw"] = grid
            r["patches"] = grid[1] * grid[2]
            r["tokens"] = r["patches"] // config.merge_area
            if r["patches"] > config.max_patches:
                r["status"] = f"over max_patches ({r['patches']} > {config.max_patches})"
                logger.warning(f"{img_path.name}: {r['status']}")
                continue

            golden = None
            golden_file = args.golden_dir / f"{img_path.stem}.pt" if args.golden_dir else None
            if golden_file is not None and golden_file.exists():
                golden = torch.load(golden_file, map_location="cpu", weights_only=True).float()
                r["golden"] = "eval_embeddings fp32"
            elif not args.no_reference and r["patches"] <= CPU_REFERENCE_MAX_PATCHES:
                t0 = time.perf_counter()
                golden = cpu_reference(fp32_reference, *hf, args.out / "cpu_ref" / f"{img_path.stem}.pt")
                r["golden"] = "cpu fp32 chunked, hf prep"
                r["cpu_ref_seconds"] = time.perf_counter() - t0
            else:
                r["golden"] = None

            if (
                golden is not None
                and not args.no_reference
                and variants["hf_rgb_first"] is not None
                and not r["prep"]["hf_rgb_first_vs_hf"].get("bit_exact", True)
            ):
                if r["patches"] <= CPU_REFERENCE_MAX_PATCHES:
                    wrong = cpu_reference(
                        fp32_reference, *variants["hf_rgb_first"], args.out / "cpu_ref" / f"{img_path.stem}.rgbfirst.pt"
                    )
                    r["cpu_rgb_first_vs_golden"] = compare_out(golden, wrong)

            if tt is None:
                r["status"] = "cpu only"
                logger.info(f"{img_path.name}: {json.dumps(r['prep'])}")
                continue

            outs = {}
            runs = {"hf": hf}
            if not r["prep"]["tt_vs_hf"].get("bit_exact", False) and not isinstance(variants["tt"], str):
                runs["tt"] = variants["tt"]
            if not r["prep"]["hf_rgb_first_vs_hf"].get("bit_exact", True):
                runs["hf_rgb_first"] = variants["hf_rgb_first"]
            for name, (pixels, grid_thws) in runs.items():
                t0 = time.perf_counter()
                outs[name] = tt.run(pixels, grid_thws).float()
                r[f"seconds_{name}"] = time.perf_counter() - t0
            if golden is not None:
                r["device"] = {name: compare_out(golden, out) for name, out in outs.items()}
            r["device_vs_canonical"] = {
                name: compare_out(outs["hf"], out) for name, out in outs.items() if name != "hf"
            }
            torch.save(outs["hf"].to(torch.bfloat16), args.out / "embeddings" / f"{img_path.stem}.pt")
            r["status"] = "ok"
            logger.info(f"{img_path.name}: {r['tokens']} tokens, {json.dumps(r.get('device'))}")
            results_path.write_text(json.dumps(results, indent=1))
    finally:
        results_path.write_text(json.dumps(results, indent=1))
        if parent is not None:
            close_ring(parent, ring)
    return 0


if __name__ == "__main__":
    sys.exit(main())

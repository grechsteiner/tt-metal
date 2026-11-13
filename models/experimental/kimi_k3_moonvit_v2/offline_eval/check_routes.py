"""CPU checks of the two ways to send temporary image token IDs, from build_table's manifest.

Route A: the chat request carries the image; Dynamo renders one bare <|media_pad|> per image and the tt-d-gen
worker splices in the encoder's runs (tt-d-gen's real EncoderService and splice, driven by ManifestModel).
Route B: the client renders the HF template with the image prompt and replaces <|media_pad|> itself.

Also checks which manifest key survives each client's way of sending the image, and that every temporary
ID maps to the right saved row.

  python -m models.experimental.kimi_k3_moonvit_v2.offline_eval.check_routes \
      --images DIR [DIR ...] --table DIR --embeddings DIR --hf-files DIR --tt-d-gen DIR [--questions FILE ...]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import types
from pathlib import Path

import torch
from PIL import Image

from models.experimental.kimi_k3_moonvit_v2.offline_eval.build_table import keys_of
from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision_processing import KimiK3VisionProcessor

MEDIA_BEGIN, MEDIA_CONTENT, MEDIA_END, MEDIA_PAD = 163602, 163603, 163604, 163605
KEYS = ("file_sha256", "strict_key", "rgba_key")
PREPROCESSOR_CONFIG = Path(__file__).resolve().parents[1] / "reference" / "preprocessor_config.json"


def import_tt_d_gen(root: Path):
    import importlib.util

    sys.path.insert(0, str(root / "multimodal"))
    exceptions = types.ModuleType("dynamo.llm.exceptions")
    exceptions.HttpError = type("HttpError", (Exception,), {})
    for name in ("dynamo", "dynamo.llm"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["dynamo.llm.exceptions"] = exceptions
    spec = importlib.util.spec_from_file_location("tt_dynamo_encoder", root / "adapters/dynamo/tt_dynamo/encoder.py")
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    from tt_encoder.model import ManifestModel as TtDGenManifestModel
    from tt_encoder.service import EncoderService, Limits

    return encoder.splice, EncoderService, Limits, TtDGenManifestModel


class ManifestModel:
    placeholder = MEDIA_PAD

    def __init__(self, manifest: dict, tokenizer, key: str) -> None:
        self._key = key
        self._index = {}
        for entry in manifest["images"]:
            self._index.setdefault(entry[key], entry)
        self._tokenizer = tokenizer

    def prepare(self, kind: str, data: bytes) -> dict:
        if kind != "image":
            raise ValueError(f"unsupported media kind {kind!r}")
        key = keys_of(data)[self._key]
        if key not in self._index:
            raise ValueError(f"image not in the manifest ({self._key} {key[:12]})")
        return self._index[key]

    def expansion(self, entry: dict) -> list[tuple[int, int]]:
        w, h = entry["size"]
        dims = self._tokenizer.encode(f"image {w}x{h}", allow_special_tokens=False)
        return (
            [(MEDIA_BEGIN, 1)]
            + [(t, 1) for t in dims]
            + [(MEDIA_CONTENT, 1)]
            + [(entry["first_id"] + i, 1) for i in range(entry["rows"])]
            + [(MEDIA_END, 1)]
        )


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def client_variants(path: Path) -> dict[str, bytes]:
    original = Image.open(path)
    original.load()
    return {
        "file_bytes": path.read_bytes(),
        "lmms_pil_png": png_bytes(original.copy()),
        "lmms_path_rgb_png": png_bytes(Image.open(path).convert("RGB")),
    }


def data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def hf_pixels(processor, data: bytes):
    image = Image.open(io.BytesIO(data))
    image.load()
    out = processor.preprocess([{"type": "image", "image": image}])
    return out["pixel_values"], out["grid_thws"]


def messages_for(images: list[str], question: str) -> list[dict]:
    content = [{"type": "image_url", "image_url": {"url": f"file://{name}"}} for name in images]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def route_b(tokenizer, messages, entries) -> tuple[list[int], list[int]]:
    prompts = [e["image_prompt"] for e in entries]
    ids = list(tokenizer.apply_chat_template(messages, image_prompts=prompts, tokenize=True))
    with_pad = list(ids)
    pads = [i for i, t in enumerate(ids) if t == MEDIA_PAD]
    assert len(pads) == len(entries), (len(pads), len(entries))
    out, start = [], 0
    for pos, e in zip(pads, entries):
        out += ids[start:pos] + list(range(e["first_id"], e["first_id"] + e["rows"]))
        start = pos + 1
    return with_pad, out + ids[start:]


def dynamo_render(tokenizer, messages, n_images) -> list[int]:
    return list(tokenizer.apply_chat_template(messages, image_prompts=["<|media_pad|>"] * n_images, tokenize=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--hf-files", type=Path, required=True)
    ap.add_argument("--tt-d-gen", type=Path, required=True)
    ap.add_argument("--questions", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_files), trust_remote_code=True)
    processor = KimiK3VisionProcessor(media_proc_cfg=json.loads(PREPROCESSOR_CONFIG.read_text())["media_proc_cfg"])
    splice, EncoderService, Limits, TtDGenManifestModel = import_tt_d_gen(args.tt_d_gen)
    manifest = json.loads((args.table / "manifest.json").read_text())
    table = torch.load(args.table / "extra_rows.pt", weights_only=True)
    by_name = {e["image"]: e for e in manifest["images"]}
    paths = {p.name: p for d in args.images for p in d.iterdir()}
    report = {"keys": [], "routes": [], "rows": [], "summary": {}}

    for entry in manifest["images"]:
        rows = table[entry["first_id"] - manifest["first_temp_id"] :][: entry["rows"]]
        saved = torch.load(args.embeddings / f"{Path(entry['image']).stem}.pt", weights_only=True)
        report["rows"].append({"image": entry["image"], "rows_match_saved": bool(torch.equal(rows, saved))})

    services = {key: EncoderService(ManifestModel(manifest, tokenizer, key), Limits()) for key in KEYS}
    for entry in manifest["images"]:
        path = paths[entry["image"]]
        ref_pixels, ref_grid = hf_pixels(processor, path.read_bytes())
        for client, data in client_variants(path).items():
            pixels, grid = hf_pixels(processor, data)
            row = {
                "image": entry["image"],
                "client": client,
                "bytes": len(data),
                "model_sees_same_image": bool(torch.equal(grid, ref_grid) and torch.equal(pixels, ref_pixels)),
            }
            for key, service in services.items():
                reply = service.expand({"images": [data_url(data)], "embeddings": False})
                if "error" in reply:
                    row[key] = "miss: " + reply["error"][:60]
                else:
                    first = reply["items"][0]["runs"]
                    temp = [t for t, _ in first if t >= manifest["first_temp_id"]]
                    row[key] = "hit" if temp[0] == entry["first_id"] else f"WRONG ENTRY first_id {temp[0]}"
            report["keys"].append(row)

    questions = []
    for qfile in args.questions:
        for q in json.loads(qfile.read_text()):
            text = q["question"] + ("\nOptions: " + q["options"] if q.get("options") not in (None, "[]") else "")
            questions.append((q["id"], q["images"], text))
    covered = {name for _, imgs, _ in questions for name in imgs}
    for entry in manifest["images"]:
        if entry["image"] not in covered:
            questions.append((entry["image"], [entry["image"]], "Describe this image in detail."))

    route_service = EncoderService(TtDGenManifestModel(str(args.table / "manifest.json")), Limits())
    for qid, names, text in questions:
        if not all(n in by_name for n in names):
            report["routes"].append({"id": qid, "status": "image(s) missing from manifest"})
            continue
        entries = [by_name[n] for n in names]
        messages = messages_for(names, text)
        hf_with_pad, ids_b = route_b(tokenizer, messages, entries)
        rendered = dynamo_render(tokenizer, messages, len(names))
        reply = route_service.expand({"images": [data_url(paths[n].read_bytes()) for n in names], "embeddings": False})
        r = {"id": qid, "images": len(names)}
        if "error" in reply:
            r["status"] = "encoder error: " + reply["error"]
            report["routes"].append(r)
            continue
        ids_a = splice(rendered, reply["items"])
        r["route_a_equals_route_b"] = ids_a == ids_b
        r["prompt_tokens"] = len(ids_b)
        r["max_id"] = max(ids_b)
        r["temp_ids_contiguous_per_image"] = all(
            ids_b[ids_b.index(e["first_id"]) : ids_b.index(e["first_id"]) + e["rows"]]
            == list(range(e["first_id"], e["first_id"] + e["rows"]))
            for e in entries
        )
        stripped = [t for t in ids_b if t < manifest["first_temp_id"]]
        hf_without_pad = [t for t in hf_with_pad if t != MEDIA_PAD]
        r["text_tokens_match_hf"] = stripped == hf_without_pad
        image_block = ids_b[ids_b.index(MEDIA_BEGIN) : ids_b.index(MEDIA_CONTENT) + 1]
        r["image_block_head"] = tokenizer.decode(image_block)
        if not r["route_a_equals_route_b"]:
            n = min(len(ids_a), len(ids_b))
            first_diff = next((i for i in range(n) if ids_a[i] != ids_b[i]), n)
            r["first_diff"] = [first_diff, ids_a[first_diff : first_diff + 8], ids_b[first_diff : first_diff + 8]]
        report["routes"].append(r)

    keys = report["keys"]
    report["summary"] = {
        "rows_all_match_saved": all(r["rows_match_saved"] for r in report["rows"]),
        "routes_all_equal": all(r.get("route_a_equals_route_b") for r in report["routes"]),
        "routes_checked": len(report["routes"]),
        "key_hits": {
            client: {key: sum(r[key] == "hit" for r in keys if r["client"] == client) for key in KEYS}
            | {
                "model_sees_same_image": sum(r["model_sees_same_image"] for r in keys if r["client"] == client),
                "total": sum(r["client"] == client for r in keys),
            }
            for client in ("file_bytes", "lmms_pil_png", "lmms_path_rgb_png")
        },
        "unsafe_hits": [
            {k: r[k] for k in ("image", "client", *KEYS)}
            for r in keys
            if not r["model_sees_same_image"] and any(r[k] == "hit" for k in KEYS)
        ],
        "max_id": manifest["vocab_size_with_extra"] - 1,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "routes_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

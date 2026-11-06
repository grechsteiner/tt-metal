# Kimi-K3 MoonViT-V2 vision encoder

The vision path of [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3): patch embedding,
27 encoder blocks, the final norm, the 2x2 patch merger and the projector into the language
model's width (7168).

Scope: one still image per call. No packing, no video. The encoder takes 14x14 px patches; the
merger turns each 2x2 window into one 28 px token, and the projector maps it to a language-model
embedding.

Target layout: 4 Blackhole chips in a ring along mesh axis 1 (`(1, 4)` mesh, 2 links). On a LoudBox
(8 chips, 2x4) that is one row; the tests open all 8 chips and run on the first row
(`tests/conftest.py`), with `FABRIC_2D_TORUS_X`.
Each image's patches are put in window order (each 2x2 merge window's four patches consecutive),
padded to 65,536 rows and split along the ring, so the merger is a reshape on each chip. Attention is
ring attention with the padding masked; norms, MLP, merger and projector are sequence parallel.
Weights are bf16 and replicated, activations DRAM interleaved, matmuls HiFi4 with fp32 accumulation.
Activations are bf16 except the encoder's residual stream, which is fp32 (`tt/vision_block.py`).

## Structure

The TT code depends only on `ttnn` and `torch` (plus numpy and Pillow in `tt/image_prep.py`), and
imports nothing from `reference/`. The reference needs `transformers`, as upstream does; tests also
use `models.common.utility_functions`.

| Path | Contents |
|---|---|
| `reference/modeling_kimi_k3.py`, `reference/configuration_kimi_k3.py` | Copied from [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3); the language model is removed |
| `reference/kimi_k3_vision.py` | `KimiK3VisionModel`: the vision path of `KimiK3ForConditionalGeneration` (tower, then `mm_projector`), and `load_vision_config`, which selects upstream's `eager` attention |
| `reference/checkpoint.py` | Reads `vision_tower.*` and `mm_projector.*` from a Kimi-K3 HF checkpoint |
| `reference/config.json`, `LICENSE-Kimi-K3` | The K3 `vision_config` and the upstream license |
| `reference/media_utils.py`, `reference/kimi_k3_vision_processing.py`, `reference/preprocessor_config.json` | Upstream's image processor and its config, copied: the golden for `tt/image_prep.py` |
| `tt/image_prep.py` | A PIL image to `(pixel_values, grid_thws)` as the HF processor makes them: resize, black padding to a multiple of 28 px, normalize, patchify |
| `tt/host_prep.py` | Host prep for the device, torch only: patch rows, position embeddings, RoPE cos/sin, window order, padding to `max_patches` rows |
| `tt/model_config.py` | `MoonViTConfig`, the grid check, `VisionParallel`, the compute kernel config |
| `tt/module.py` | `Module`, `ModuleList`, `Parameter`, state-dict loading, `save`/`load`, `load_cached` |
| `tt/linear.py`, `tt/rms_norm.py` | The leaf layers |
| `tt/ccl.py` | `ParallelFactor` and `TT_CCL` (links, topology, ring-attention semaphores and buffers) |
| `tt/device_input.py` | `DeviceInput`: a per-image input allocated once and rewritten in place, so a trace can replay it |
| `tt/patch_embed.py`, `rope.py`, `vision_attention.py`, `vision_mlp.py`, `vision_block.py` | One TT module per encoder part |
| `tt/patch_merger.py`, `tt/projector.py` | `TT_Patch_Merger` (no weights) and `TT_Projector` |
| `tt/vision_model.py` | `TT_MoonViT_Vision_Tower` (patch embed, encoder, merger) and `TT_MoonViT_Vision_Model` (tower, projector) |
| `tests/` | `common.py` and `conftest.py`, CPU tests for the reference and host prep, one device test per part, the full-size tests, and `goldens.py`, which saves their reference outputs |

## Torch and ttnn

The TT modules are `tt/module.py` `Module`s whose children carry the reference's names, so each
loads the matching reference `state_dict()` with `load_torch_state_dict`. `load_cached` caches the
loaded weights as `.tensorbin` files. Tests build both, run the same input, and compare with
`assert_quality` (`tests/common.py`): PCC (`comp_pcc`) and relative RMSE (RMS error over the reference's
standard deviation) are asserted; CCC and the largest element differences are logged.

```python
ccl = TT_CCL(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
tt_model = TT_MoonViT_Vision_Model(config, mesh_device=mesh_device, sequence_parallel=ParallelFactor(4, 1), ccl=ccl)
tt_model.load_torch_state_dict(reference.state_dict())
pixel_values, grid_thws = image_to_patches(Image.open(path), config)  # tt/image_prep.py
out = tt_model.run(pixel_values, grid_thws)  # (h*w, 3, 14, 14), [[1, h, w]] -> (h*w/4, 7168)
```

Image limit: an image is accepted when the HF processor's grid fits `max_patches` (65,536 patches).
That rejects nearly every image over about 12.8 MP, and some smaller ones that the 28 px padding
pushes over (3590x3590 -> 258x258). Over-limit images are rejected, never shrunk, because shrinking
would diverge from HF. Only RGB images are accepted; other modes (including anything with
transparency, which HF composites onto a chessboard) raise `ValueError`.

TODO (image pre-processing, `tt/image_prep.py`):

- Handle non-RGB images as HF does, instead of rejecting them: convert other modes to RGB *after*
  the resize (Pillow resizes P and 1 with nearest neighbour and RGBA with premultiplied alpha, so
  converting first does not match HF). Test against `reference/kimi_k3_vision_processing.py` for
  RGBA, LA, PA, L, P, 1, CMYK and I;16.
- Handle `transparent_bg_config` and `transparent_bg_fill_stage` (`reference/preprocessor_config.json`):
  composite images with alpha (RGBA, LA, PA, and P or L with `tRNS`) onto an 8 px chessboard
  (white 255, gray 180, white square top left), after the resize and before the black pad. Read the
  values from the config rather than hard-coding them.

## Weights

The `reference` fixture in `tests/conftest.py` builds with the weights a test's marks ask for. A test
with `@SYNTHETIC` runs with random weights, one with `@REAL` with the checkpoint, and one with both runs
once with each (ids `synthetic` and `real`; `-m synthetic` or `-m real` selects by weights). CPU tests
use random weights only (`test_load_checkpoint.py` reads the checkpoint itself). Device tests use the
checkpoint only for attention, the MLP, the projector and what contains them (block, full model,
full size); the other device tests use random weights only:

- `synthetic`: the reference initialized under `torch.manual_seed(0)`, with norm weights drawn around
  one so a norm that fails to load shows up.
- `real`: the vision tower and projector of the checkpoint at `KIMI_K3_HF_MODEL`. The loader opens
  only the shards holding `vision_tower.*` and `mm_projector.*` keys, so a folder with
  `model.safetensors.index.json`, `config.json`, `model-00095-of-000096.safetensors` and
  `model-00096-of-000096.safetensors` is enough. These cases skip when the variable is unset.

Per-layer tests take a `layer` parameter, set apart from the weights: `@SAMPLED_LAYERS` (layers 0,
13 and 26; attention, MLP, block, encoder linears and norms) or `@ALL_LAYERS` (the full-size layer
test). `MOONVIT_TEST_LAYER=5` runs only layer 5 in the `@SAMPLED_LAYERS` tests; `@ALL_LAYERS`
ignores it.

## Tests

```bash
# CPU: reference, image and host prep, checkpoint loader, TT module names (device upload stubbed),
# the golden generator's helpers, and `assert_quality`
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_reference.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_image_prep.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_host_prep.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_load_checkpoint.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_module.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_goldens.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_quality.py

# Device, one P150: parts that work on each row alone
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_linear.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_rms_norm.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_rope.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_patch_embed.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_vision_mlp.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_patch_merger.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_projector.py

# Device, a LoudBox: attention needs the ring
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_vision_attention.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_vision_block.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_vision_model.py
```

Only tests with attention in them (attention, block, full model, full size) run on the ring
(`ring1x4`); pytest skips them when the machine has fewer chips. The row-wise tests run on one P150
(`p150`), which stands in for one chip of the ring: it gets that chip's quarter of the padded
sequence (16,384 rows), cut out by `MeshRows` (`tests/common.py`).

Image sizes (`tests/common.py`):

- Row-wise tests (`PADDING_LAYOUT_GRIDS`): `full` (256x256) and `max_side` (512x128) fill the padded
  length exactly, `partial` (250x250) ends inside a tile, and `small` (40x100) is almost all padding.
  The P150 runs the quarter of the last ring chip that holds real rows (chip 0 for `small`, chip 3
  for the others), and the CPU golden covers that quarter's real rows. Nothing there scales with the
  square of the length, so the golden takes seconds.
- Attention, the block and the full model (`ATTENTION_GRIDS`, 64x64 and 40x100) stay small: their CPU
  golden builds L x L scores per head. All their real rows are on chip 0; chips 1-3 hold only padding.
- Attention and the block with real rows on every chip (`REAL_ROWS_FROM_EACH_CHIP_ASSERTED_GRIDS`:
  `full`, `max_side`, `partial`; the `*_real_rows_from_each_chip` tests): the fp32 golden covers
  `SAMPLED_ROWS_PER_CHIP` query rows of each chip (`sampled_rows`), each against every key, and each
  chip's rows are checked on their own.

Simulator (ttsim, `TT_METAL_SIMULATOR` set): each row-wise test has a `_small` variant marked
`simulator` that runs `SIMULATOR_GRID` (6x20) in a config padded to `SIMULATOR_ROWS` (512), so the
chip runs a 128-row quarter. On ttsim, every other device test is skipped. The `_small` tests also run
on hardware:

```bash
pytest models/experimental/kimi_k3_moonvit_v2/tests -m simulator
```

Full size (`test_full_size.py`, LoudBox, checkpoint only): a 256x256 image fills all 65,536 rows,
and each encoder layer on its own, the full 27-layer encoder and the full model (merger and
projector included) are compared against reference outputs saved once by `tests/goldens.py`. They
are fp32, with attention computed in chunks of query rows: the bf16 reference rounds more than the
device in deeper layers, and upstream's eager attention in fp32 needs more memory than the host has.
`--dtype bf16` writes the bf16 eager reference instead. The outputs go in `$MOONVIT_GOLDEN_DIR`; the
tests skip while it is unset or the outputs do not exist:

```bash
export KIMI_K3_HF_MODEL=/path/to/Kimi-K3
export MOONVIT_GOLDEN_DIR=/path/to/goldens
python -m models.experimental.kimi_k3_moonvit_v2.tests.goldens --grid 256x256   # resumable
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_full_size.py
```

Traced perf tests, full size only (`test_full_size_layer_trace` for one layer,
`test_full_size_encoder_trace` for patch embed and encoder, `test_full_size_model_trace` for the full
model): the forward runs once untraced, then is captured in one trace (`Trace` in
`tests/common.py`). The first replay must equal the untraced run exactly and pass the untraced
test's checks. Then `TRACE_WARMUP` untimed and `TRACE_ITERATIONS` timed replays run back to back with
one wait at the end, and the time per replay is logged. The model test also logs one host-in,
host-out run (host prep, input writes, replay, gather):

```bash
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_full_size.py -k trace
```

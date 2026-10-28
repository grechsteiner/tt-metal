# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Pytest entry point for the ttnn.add support-matrix sweep (MoonViT-V2 residual add). The vectors and
# run() live in the sweep module so the sweep runner and support_matrix_report.py can use the same
# definitions. A config whose input tensors cannot be built is skipped rather than failed, since the
# op never ran; see the sweep module for the meaning of the other failure tags.
#
# The test_compare_* tests run on the host only. They check that the torch comparison used by run()
# accepts a correctly rounded result and rejects the faults a broken shard or padding path produces,
# so a pass from test_add_support_matrix means the output really matches torch.add.

import pytest
import torch
import ttnn

from tests.sweep_framework.sweeps.eltwise.binary.add.add_support_matrix import (
    SHAPES,
    compare_with_torch,
    parameters,
    run,
)

pytestmark = pytest.mark.use_module_device

suite = parameters["moonvit_add"]


def _lose_mantissa_bits(tensor, bits):
    tensor.view(torch.int16).bitwise_and_(~((1 << bits) - 1))


# Each fault edits the output in place, given input a.
FAULTS = {
    "zeroed_tile": lambda a, out: out[..., :32, :32].zero_(),
    "zeroed_row": lambda a, out: out[..., 5, :].zero_(),
    "zeroed_last_rows": lambda a, out: out[..., -16:, :].zero_(),
    "b_dropped_in_one_tile": lambda a, out: out[..., :32, :32].copy_(a[..., :32, :32]),
    "swapped_tiles": lambda a, out: out[..., :64, :32].copy_(out[..., :64, :32].roll(32, dims=-2)),
    "two_mantissa_bits_lost": lambda a, out: _lose_mantissa_bits(out, 2),
}


def _torch_inputs(shape_kind):
    torch.manual_seed(0)
    a = torch.rand(SHAPES[shape_kind], dtype=torch.bfloat16) * 200 - 100
    b = torch.rand(SHAPES[shape_kind], dtype=torch.bfloat16) * 200 - 100
    return a, b


@pytest.mark.parametrize("shape_kind", SHAPES)
def test_compare_golden_function_is_torch_add(shape_kind):
    a, b = _torch_inputs(shape_kind)
    assert torch.equal(ttnn.get_golden_function(ttnn.add)(a, b), torch.add(a, b))


@pytest.mark.parametrize("shape_kind", SHAPES)
def test_compare_accepts_round_toward_zero(shape_kind):
    # torch rounds a + b to nearest; hardware may truncate instead. That is at most 1 ULP and must pass.
    a, b = _torch_inputs(shape_kind)
    exact = a.float() + b.float()
    truncated = (exact.view(torch.int32) & ~0xFFFF).view(torch.float32).bfloat16()
    assert not torch.equal(truncated, torch.add(a, b))
    passed, message = compare_with_torch(torch.add(a, b), truncated)
    assert passed, message


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("shape_kind", SHAPES)
def test_compare_rejects_fault(shape_kind, fault):
    a, b = _torch_inputs(shape_kind)
    expected = torch.add(a, b)
    actual = expected.clone()
    FAULTS[fault](a, actual)
    passed, message = compare_with_torch(expected, actual)
    assert not passed, message


@pytest.mark.parametrize("out_memory_config", suite["out_memory_config"])
@pytest.mark.parametrize("input_b_memory_config", suite["input_b_memory_config"])
@pytest.mark.parametrize("input_a_memory_config", suite["input_a_memory_config"])
@pytest.mark.parametrize("layout", suite["layout"])
@pytest.mark.parametrize("shape_kind", suite["shape_kind"])
def test_add_support_matrix(
    device, record_property, shape_kind, layout, input_a_memory_config, input_b_memory_config, out_memory_config
):
    (passed, message), _ = run(
        shape_kind,
        layout,
        input_a_memory_config,
        input_b_memory_config,
        out_memory_config,
        device=device,
    )
    # Kept for passing vectors too: the ULP figures and the output config the op chose.
    record_property("message", message)
    if message.startswith("[SETUP]"):
        pytest.skip(message)
    assert passed, message

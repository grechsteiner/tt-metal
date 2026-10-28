# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Pytest entry point for the op support matrix (tests/sweep_framework/support_matrix). One test per
# vector of every group, with ids like "add-tile-a=dram_interleaved-b=l1_height_sharded-out=default".
# A vector whose inputs cannot be built is skipped rather than failed, since the op never ran; see
# harness.py for the other failure tags. Each test records its message and a JSON description of
# its tensors, which the support-matrix tools turn into the explorer page.
#
# The test_compare_* tests run on the host only. They check that the comparison used by run_vector
# accepts a correctly rounded result and rejects the faults a broken shard or padding path produces,
# so a pass really means the output matches its reference.

import json

import pytest
import torch
import ttnn

from tests.sweep_framework.support_matrix.groups import GROUPS
from tests.sweep_framework.support_matrix.harness import compare_with_torch, run_vector

pytestmark = pytest.mark.use_module_device

VECTORS = [(group.name, vector) for group in GROUPS.values() for vector in group.vectors()]


@pytest.mark.parametrize(
    "group_name, vector", VECTORS, ids=[GROUPS[name].vector_id(vector) for name, vector in VECTORS]
)
def test_support_matrix(device, record_property, group_name, vector):
    info = {}
    try:
        passed, message, _ = run_vector(GROUPS[group_name], vector, device=device, info=info)
    finally:
        # Kept even when the op raises: every tensor's config, cores and L1 per core.
        record_property("info", json.dumps(info))
    # Kept for passing vectors too: the ULP figures and the output config the op chose.
    record_property("message", message)
    if message.startswith("[SETUP]"):
        pytest.skip(message)
    assert passed, message


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


def _add_inputs():
    torch.manual_seed(0)
    shape = GROUPS["add"].output_shape
    return torch.rand(shape, dtype=torch.bfloat16) * 200 - 100, torch.rand(shape, dtype=torch.bfloat16) * 200 - 100


def test_compare_golden_function_is_torch_add():
    a, b = _add_inputs()
    assert torch.equal(ttnn.get_golden_function(ttnn.add)(a, b), torch.add(a, b))


def test_compare_accepts_round_toward_zero():
    # torch rounds a + b to nearest; hardware may truncate instead. That is at most 1 ULP and must pass.
    a, b = _add_inputs()
    exact = a.float() + b.float()
    truncated = (exact.view(torch.int32) & ~0xFFFF).view(torch.float32).bfloat16()
    assert not torch.equal(truncated, torch.add(a, b))
    passed, message = compare_with_torch(torch.add(a, b), truncated)
    assert passed, message


def test_compare_baseline_rejects_one_ulp():
    # Groups judged against their own DRAM-interleaved output require bit-identical results.
    a, b = _add_inputs()
    exact = a.float() + b.float()
    truncated = (exact.view(torch.int32) & ~0xFFFF).view(torch.float32).bfloat16()
    passed, message = compare_with_torch(torch.add(a, b), truncated, ulp_threshold=0)
    assert not passed, message


@pytest.mark.parametrize("fault", FAULTS)
def test_compare_rejects_fault(fault):
    a, b = _add_inputs()
    expected = torch.add(a, b)
    actual = expected.clone()
    FAULTS[fault](a, actual)
    passed, message = compare_with_torch(expected, actual)
    assert not passed, message

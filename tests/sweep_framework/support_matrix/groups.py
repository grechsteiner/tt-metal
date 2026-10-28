# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Every support-matrix group, by name, in encoder-block order. Each lives in its op's sweep module."""

from tests.sweep_framework.sweeps.eltwise.binary.add.add_support_matrix import GROUP as ADD
from tests.sweep_framework.sweeps.eltwise.unary.gelu.gelu_support_matrix import GROUP as GELU
from tests.sweep_framework.sweeps.matmul.support_matrix.linear_support_matrix import GROUPS as LINEAR
from tests.sweep_framework.sweeps.normalization.rms_norm.rms_norm_support_matrix import GROUP as RMS_NORM
from tests.sweep_framework.sweeps.transformer.nlp_concat_heads.nlp_concat_heads_support_matrix import (
    GROUP as CONCAT_HEADS,
)

WQKV, WO, FC0, FC1 = LINEAR

GROUPS = {group.name: group for group in (RMS_NORM, WQKV, CONCAT_HEADS, WO, ADD, FC0, GELU, FC1)}

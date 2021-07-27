# coding=utf-8
# Copyright 2021 The HuggingFace Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from ..utils.py_utils import map_nested
from .formatting import Formatter


if TYPE_CHECKING:
    import paddle


class PaddleFormatter(Formatter[dict, "paddle.Tensor", dict]):
    def __init__(self, **paddle_tensor_kwargs):
        self.paddle_tensor_kwargs = paddle_tensor_kwargs
        import paddle  # noqa import torch at initialization

    def _tensorize(self, value):
        import paddle

        default_device = {"place": paddle.CPUPlace()}
        default_dtype = {}
        if np.issubdtype(value.dtype, np.integer):
            default_dtype = {"dtype": paddle.int64}
        elif np.issubdtype(value.dtype, np.floating):
            default_dtype = {"dtype": paddle.float32}

        return paddle.to_tensor(value, **{**default_dtype, **default_device, **self.paddle_tensor_kwargs})

    def _recursive_tensorize(self, data_struct: dict):
        # support for nested types like struct of list of struct
        if isinstance(data_struct, (list, np.ndarray)):
            data_struct = np.array(data_struct, copy=False)
            if data_struct.dtype == np.object:  # pytorch tensors cannot be instantied from an array of objects
                return [self.recursive_tensorize(substruct) for substruct in data_struct]
        return self._tensorize(data_struct)

    def recursive_tensorize(self, data_struct: dict):
        return map_nested(self._recursive_tensorize, data_struct, map_list=False)

    def format_row(self, pa_table: pa.Table) -> dict:
        row = self.numpy_arrow_extractor().extract_row(pa_table)
        return self.recursive_tensorize(row)

    def format_column(self, pa_table: pa.Table) -> "paddle.Tensor":
        col = self.numpy_arrow_extractor().extract_column(pa_table)
        return self.recursive_tensorize(col)

    def format_batch(self, pa_table: pa.Table) -> dict:
        batch = self.numpy_arrow_extractor().extract_batch(pa_table)
        return self.recursive_tensorize(batch)

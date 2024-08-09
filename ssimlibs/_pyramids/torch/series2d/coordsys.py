"""
    Copyright 2024 Michael Tsai (win10_Mike@outlook.com)

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from abc import abstractmethod
from typing import Tuple, Union

import torch


class CoordSys:
    def __init__(self, window: Union[int, Tuple[int], Tuple[int, int]], dtype: torch.dtype = torch.float64):
        if isinstance(window, int):
            window = (window,)
        if isinstance(window, Tuple) and len(window) == 1:
            window = (window[0], window[0])
        assert isinstance(window, Tuple) and len(window) == 2, 'Input size should be int, Tuple[int] or Tuple[int, int]'
        self.window = window
        x_indices = torch.arange(-window[1] / 2 + 0.5, +window[1] / 2 + 0.0, +1.0, dtype=dtype) / window[1] * max(window)
        y_indices = torch.arange(-window[0] / 2 + 0.5, +window[0] / 2 + 0.0, +1.0, dtype=dtype) / window[0] * max(window)
        self.y_axis, self.x_axis = torch.meshgrid(y_indices, x_indices, indexing='ij')

    @property
    @abstractmethod
    def groups(self):
        raise NotImplementedError

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

import functools
import operator
from typing import List, Tuple

import torch
from torch.fft import fft2 as _fft2
from torch.fft import fftshift as _fftshift
from torch.fft import ifft2
from torch.fft import ifftshift

from .series2d import Series2D


class SteerablePyramid2D:
    def __init__(self, levels: int, orientations: int, is_complex: bool, transition: float = 1.0, offset: float = 0.0 * torch.pi, dtype: torch.dtype = torch.float64):
        self._levels = levels
        self._orientations = orientations
        self._is_complex = is_complex
        self._transition = transition
        self._offset, self._dtype = offset, dtype

    def forward(self, data: torch.Tensor) -> List[Tuple[torch.Tensor, ...]]:
        result = []
        window = Series2D((data.shape[-2], data.shape[-1]), self._levels, self._orientations, self._is_complex, self._transition, self._offset, self._dtype)
        series2d, slices2d = window.groups, window.slices
        freq = _fftshift(_fft2(data), dim=(-1, -2))
        for group, index in zip(series2d, slices2d):
            tido = [ifft2(ifftshift((freq * item)[(Ellipsis,) + index], dim=(-1, -2))) for item in group]
            tido = [item if self._is_complex else torch.real(item) for item in tido]
            result.append(tuple(tido))
        return result

    def reverse(self, data: List[Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        result = torch.zeros_like(data[0][0])
        window = Series2D((data[0][0].shape[-2], data[0][0].shape[-1]), self._levels, self._orientations, self._is_complex, self._transition, self._offset, self._dtype)
        series2d, slices2d = window.groups, window.slices
        index: Tuple[slice, slice]
        for times, group, index in zip(data, series2d, slices2d):
            freq = [torch.nn.functional.pad(_fftshift(_fft2(time), dim=(-1, -2)), tuple(value for bound in index for value in [abs(bound.start if bound.start else 0), abs(-bound.stop if bound.stop else 0)])) * item for time, item in zip(times, group)]
            result = result + functools.reduce(operator.add, freq)
        result = ifft2(ifftshift(result, dim=(-1, -2)))
        return result if self._is_complex else torch.real(result)

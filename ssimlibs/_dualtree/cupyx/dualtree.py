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

from typing import List, Tuple

import cupy

from .reformat import LevelCommonReformatOperator
from .operator.series import LevelAlpha1SeriesOperator
from .operator.series import LevelOthersSeriesOperator
from .wavelets import BiorthogonalWavelet, OrthogonalWavelet


class DualTreeComplexWaveletTransform:
    def __init__(self, levels: int, dimension: int, level_alpha1: BiorthogonalWavelet, level_others: OrthogonalWavelet):
        assert levels > 0 and dimension > 0
        self.levels, self.level_common = levels, LevelCommonReformatOperator(dimension)
        self.level_alpha1 = LevelAlpha1SeriesOperator(dimension, level_alpha1)
        self.level_others = LevelOthersSeriesOperator(dimension, level_others)

    def forward(self, data: cupy.ndarray) -> Tuple[cupy.ndarray, List[Tuple[Tuple[cupy.ndarray, ...], ...]]]:
        res = []
        low, raw = self.raw_forward(data)
        for tup in raw:
            res.append(tuple(self.level_common.forward(val) for val in tup))
        return low, res

    def reverse(self, low: cupy.ndarray, arrays: List[Tuple[Tuple[cupy.ndarray, ...], ...]]) -> cupy.ndarray:
        raw = []
        for tup in arrays:
            raw.append(tuple(self.level_common.reverse(val) for val in tup))
        res = self.raw_reverse(low, raw)
        return res

    def raw_forward(self, tensor: cupy.ndarray) -> Tuple[cupy.ndarray, List[Tuple[cupy.ndarray, ...]]]:
        stack, array = [], [tensor]
        if self.levels > 0:
            array = self.level_alpha1.forward(array[0])
            stack.append(array[1:])
        for level in range(+1, +self.levels, +1):
            array = self.level_others.forward(array[0])
            stack.append(array[1:])
        return array[0], stack

    def raw_reverse(self, final: cupy.ndarray, arrays: List[Tuple[cupy.ndarray, ...]]) -> cupy.ndarray:
        # Reverse
        for level in range(-1, -self.levels, -1):
            if abs(level) - 1 < len(arrays):
                final = self.level_others.reverse(tuple([final] + list(arrays[level])))
        if self.levels == len(arrays):
            final = self.level_alpha1.reverse(
                tuple([final] + list(arrays[0])))
        return final

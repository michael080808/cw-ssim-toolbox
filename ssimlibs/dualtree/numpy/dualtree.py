from typing import List, Tuple

import numpy

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

    def forward(self, data: numpy.ndarray) -> Tuple[numpy.ndarray, List[Tuple[Tuple[numpy.ndarray, ...], ...]]]:
        res = []
        low, raw = self.raw_forward(data)
        for tup in raw:
            res.append(tuple(self.level_common.forward(val) for val in tup))
        return low, res

    def reverse(self, low: numpy.ndarray, arrays: List[Tuple[Tuple[numpy.ndarray, ...], ...]]) -> numpy.ndarray:
        raw = []
        for tup in arrays:
            raw.append(tuple(self.level_common.reverse(val) for val in tup))
        res = self.raw_reverse(low, raw)
        return res

    def raw_forward(self, tensor: numpy.ndarray) -> Tuple[numpy.ndarray, List[Tuple[numpy.ndarray, ...]]]:
        stack, array = [], [tensor]
        if self.levels > 0:
            array = self.level_alpha1.forward(array[0])
            stack.append(array[1:])
        for level in range(+1, +self.levels, +1):
            array = self.level_others.forward(array[0])
            stack.append(array[1:])
        return array[0], stack

    def raw_reverse(self, final: numpy.ndarray, arrays: List[Tuple[numpy.ndarray, ...]]) -> numpy.ndarray:
        # Reverse
        for level in range(-1, -self.levels, -1):
            if abs(level) - 1 < len(arrays):
                final = self.level_others.reverse(tuple([final] + list(arrays[level])))
        if self.levels == len(arrays):
            final = self.level_alpha1.reverse(
                tuple([final] + list(arrays[0])))
        return final

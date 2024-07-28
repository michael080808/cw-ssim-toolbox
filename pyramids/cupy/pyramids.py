import functools
import operator
from typing import List, Tuple, Union

import cupy
from cupy.fft import fft2 as _fft2
from cupy.fft import fftshift as _fftshift
from cupy.fft import ifft2
from cupy.fft import ifftshift

from .series2d import Series2D


class SteerablePyramid2D:
    def __init__(self, window: Union[int, Tuple[int], Tuple[int, int]], levels: int, orientations: int, is_complex: bool, transition: float = 1.0, offset: float = 0.0 * cupy.pi, dtype: cupy.dtype = cupy.float64):
        fb = Series2D(window, levels, orientations, is_complex, transition, offset, dtype)
        self.is_complex, self.series2d, self.slices2d = is_complex, fb.groups, fb.slices

    def forward(self, data: cupy.ndarray) -> List[Tuple[cupy.ndarray, ...]]:
        result = []
        freq = _fftshift(_fft2(data), axes=(-1, -2))
        for group, index in zip(self.series2d, self.slices2d):
            tido = [ifft2(ifftshift((freq * item)[(Ellipsis,) + index], axes=(-1, -2))) for item in group]
            tido = [item if self.is_complex else cupy.real(item) for item in tido]
            result.append(tuple(tido))
        return result

    def reverse(self, data: List[Tuple[cupy.ndarray, ...]]) -> cupy.ndarray:
        result = cupy.zeros_like(data[0][0])
        index: Tuple[slice, slice]
        for times, group, index in zip(data, self.series2d, self.slices2d):
            freq = [cupy.pad(_fftshift(_fft2(time), axes=(-1, -2)), [[abs(bound.start if bound.start else 0), abs(-bound.stop if bound.stop else 0)] for bound in index]) * item for time, item in zip(times, group)]
            result = result + functools.reduce(operator.add, freq)
        result = ifft2(ifftshift(result, axes=(-1, -2)))
        return result if self.is_complex else cupy.real(result)

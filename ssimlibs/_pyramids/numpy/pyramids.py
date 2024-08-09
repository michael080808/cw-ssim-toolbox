import functools
import operator
from typing import List, Tuple

import numpy
from numpy.fft import fft2 as _fft2
from numpy.fft import fftshift as _fftshift
from numpy.fft import ifft2
from numpy.fft import ifftshift

from .series2d import Series2D


class SteerablePyramid2D:
    def __init__(self, levels: int, orientations: int, is_complex: bool, transition: float = 1.0, offset: float = 0.0 * numpy.pi, dtype: numpy.dtype = numpy.float64):
        self._levels = levels
        self._orientations = orientations
        self._is_complex = is_complex
        self._transition = transition
        self._offset, self._dtype = offset, dtype

    def forward(self, data: numpy.ndarray) -> List[Tuple[numpy.ndarray, ...]]:
        result = []
        window = Series2D((data.shape[-2], data.shape[-1]), self._levels, self._orientations, self._is_complex, self._transition, self._offset, self._dtype)
        series2d, slices2d = window.groups, window.slices
        freq = _fftshift(_fft2(data), axes=(-1, -2))
        for group, index in zip(series2d, slices2d):
            tido = [ifft2(ifftshift((freq * item)[(Ellipsis,) + index], axes=(-1, -2))) for item in group]
            tido = [item if self._is_complex else numpy.real(item) for item in tido]
            result.append(tuple(tido))
        return result

    def reverse(self, data: List[Tuple[numpy.ndarray, ...]]) -> numpy.ndarray:
        result = numpy.zeros_like(data[0][0])
        window = Series2D((data[0][0].shape[-2], data[0][0].shape[-1]), self._levels, self._orientations, self._is_complex, self._transition, self._offset, self._dtype)
        series2d, slices2d = window.groups, window.slices
        index: Tuple[slice, slice]
        for times, group, index in zip(data, series2d, slices2d):
            freq = [numpy.pad(_fftshift(_fft2(time), axes=(-1, -2)), numpy.array([[abs(bound.start if bound.start else 0), abs(-bound.stop if bound.stop else 0)] for bound in index])) * item for time, item in zip(times, group)]
            result = result + functools.reduce(operator.add, freq)
        result = ifft2(ifftshift(result, axes=(-1, -2)))
        return result if self._is_complex else numpy.real(result)

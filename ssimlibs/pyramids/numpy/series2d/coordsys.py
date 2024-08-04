from abc import abstractmethod
from typing import Tuple, Union

import numpy


class CoordSys:
    def __init__(self, window: Union[int, Tuple[int], Tuple[int, int]], dtype: numpy.dtype = numpy.float64):
        if isinstance(window, int):
            window = (window,)
        if isinstance(window, Tuple) and len(window) == 1:
            window = (window[0], window[0])
        assert isinstance(window, Tuple) and len(window) == 2, 'Input size should be int, Tuple[int] or Tuple[int, int]'
        self.window = window
        x_indices = numpy.arange(-window[1] / 2 + 0.5, +window[1] / 2 + 0.0, +1.0, dtype=dtype) / window[1] * max(window)
        y_indices = numpy.arange(-window[0] / 2 + 0.5, +window[0] / 2 + 0.0, +1.0, dtype=dtype) / window[0] * max(window)
        self.y_axis, self.x_axis = numpy.meshgrid(y_indices, x_indices, indexing='ij')

    @property
    @abstractmethod
    def groups(self):
        raise NotImplementedError

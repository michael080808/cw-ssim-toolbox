from typing import Tuple, Union

import numpy
import scipy

from .coordsys import CoordSys


class Rotation(CoordSys):
    def __init__(self, window: Union[int, Tuple[int], Tuple[int, int]], orientations: int, is_complex: bool, offset: float = 0.0 * numpy.pi, dtype: numpy.dtype = numpy.float64):
        assert orientations > 0
        CoordSys.__init__(self, window, dtype)
        self.orientations, self.is_complex = orientations, is_complex
        self.offset, self.angles = offset, numpy.arctan2(self.y_axis, self.x_axis)

    @property
    def _param(self):
        order = self.orientations - 1
        p_numerator = 2 ** order * scipy.special.factorial(order)
        denominator = numpy.sqrt(self.orientations * scipy.special.factorial(2 * order))
        return p_numerator / denominator

    @property
    def groups(self):
        angles = [numpy.remainder(self.angles - self.offset + numpy.pi - b * numpy.pi / self.orientations, 2 * numpy.pi) - numpy.pi for b in range(self.orientations)]
        masked = [numpy.abs(self._param * numpy.cos(angle) ** (self.orientations - 1)) for angle in angles]
        masked = [duals * (numpy.abs(angle) < numpy.pi / 2) for angle, duals in zip(angles, masked)] + [duals * (numpy.abs(angle) > numpy.pi / 2) for angle, duals in zip(angles, masked)] if self.is_complex else masked
        return masked

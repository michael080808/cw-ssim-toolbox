from typing import Tuple, Union

import cupy
import scipy

from .coordsys import CoordSys


class Rotation(CoordSys):
    def __init__(self, window: Union[int, Tuple[int], Tuple[int, int]], orientations: int, is_complex: bool, offset: float = 0.0 * cupy.pi, dtype: cupy.dtype = cupy.float64):
        assert orientations > 0
        CoordSys.__init__(self, window, dtype)
        self.orientations, self.is_complex = orientations, is_complex
        self.offset, self.angles = offset, cupy.arctan2(self.y_axis, self.x_axis)

    @property
    def _param(self):
        order = self.orientations - 1
        p_numerator = 2 ** order * scipy.special.factorial(order)
        denominator = cupy.sqrt(self.orientations * scipy.special.factorial(2 * order))
        return p_numerator / denominator

    @property
    def groups(self):
        angles = [cupy.remainder(self.angles - self.offset + cupy.pi - b * cupy.pi / self.orientations, 2 * cupy.pi) - cupy.pi for b in range(self.orientations)]
        masked = [cupy.abs(self._param * cupy.cos(angle) ** (self.orientations - 1)) for angle in angles]
        masked = [duals * (cupy.abs(angle) < cupy.pi / 2) for angle, duals in zip(angles, masked)] + [duals * (cupy.abs(angle) > cupy.pi / 2) for angle, duals in zip(angles, masked)] if self.is_complex else masked
        return masked

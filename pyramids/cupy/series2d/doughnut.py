import itertools
from typing import Tuple, Union

import cupy

from .coordsys import CoordSys


class Doughnut(CoordSys):
    def __init__(self, window: Union[int, Tuple[int], Tuple[int, int]], levels: int, transition: float = 1.0, dtype: cupy.dtype = cupy.float64):
        assert levels >= 2
        CoordSys.__init__(self, window, dtype)
        self.levels, self.transition = levels, transition
        self.radius = cupy.sqrt(self.x_axis ** 2 + self.y_axis ** 2)

    def _logarithm_indices(self, _lower: float, _upper: float):
        log_upper = cupy.log(_upper)
        return cupy.clip(-(cupy.log(cupy.where(self.radius <= 0, cupy.finfo(self.radius.dtype).eps, self.radius)) - log_upper) / (cupy.log(_lower) - log_upper), a_min=-1, a_max=+0)

    def _logarithm_indices_with_transition(self, _lower: float, _upper: float):
        new_lower, new_upper = (_upper + _lower) / 2 - (_upper - _lower) * self.transition / 2, (_upper + _lower) / 2 + (_upper - _lower) * self.transition / 2
        return self._logarithm_indices(new_lower, new_upper)

    @property
    def groups(self):
        result = [1.]
        for _upper, _lower in itertools.pairwise([max(self.window) // 2 ** level for level in range(1, self.levels + 1)]):
            log_id = self._logarithm_indices_with_transition(_lower, _upper) * cupy.pi / 2
            result.extend([result.pop() * cupy.cos(+log_id), cupy.sin(-log_id)])
        return result

    @property
    def slices(self):
        none_s = [(slice(None, None),) * 2]
        bounds = [tuple(int(axis_l / 2 ** (i - 0) * (2 ** (i - 1) - 1)) for axis_l in self.window) for i in range(2, self.levels)]
        slices = [tuple(slice(+b_item, -b_item) for b_item in b_pair) for b_pair in bounds]
        slices = none_s * 2 + slices
        return slices

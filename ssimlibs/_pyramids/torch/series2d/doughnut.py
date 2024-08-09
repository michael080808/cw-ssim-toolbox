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

import itertools
from typing import Tuple, Union

import numpy
import torch

from .coordsys import CoordSys


class Doughnut(CoordSys):
    def __init__(self, window: Union[int, Tuple[int], Tuple[int, int]], levels: int, transition: float = 1.0, dtype: torch.dtype = torch.float64):
        assert levels >= 2
        CoordSys.__init__(self, window, dtype)
        self.levels, self.transition = levels, transition
        self.radius = torch.sqrt(self.x_axis ** 2 + self.y_axis ** 2)

    def _logarithm_indices(self, _lower: float, _upper: float):
        log_upper = numpy.log(_upper)
        return torch.clip(-(torch.log(torch.where(self.radius <= 0, torch.finfo(self.radius.dtype).eps, self.radius)) - log_upper) / (numpy.log(_lower) - log_upper), min=-1, max=+0)

    def _logarithm_indices_with_transition(self, _lower: float, _upper: float):
        new_lower, new_upper = (_upper + _lower) / 2 - (_upper - _lower) * self.transition / 2, (_upper + _lower) / 2 + (_upper - _lower) * self.transition / 2
        return self._logarithm_indices(new_lower, new_upper)

    @property
    def groups(self):
        result = [1.]
        for _upper, _lower in itertools.pairwise([max(self.window) // 2 ** level for level in range(1, self.levels + 1)]):
            log_id = self._logarithm_indices_with_transition(_lower, _upper) * torch.pi / 2
            result.extend([result.pop() * torch.cos(+log_id), torch.sin(-log_id)])
        return result

    @property
    def slices(self):
        none_s = [(slice(None, None),) * 2]
        bounds = [tuple(int(axis_l / 2 ** (i - 0) * (2 ** (i - 1) - 1)) for axis_l in self.window) for i in range(2, self.levels)]
        slices = [tuple(slice(+b_item, -b_item) for b_item in b_pair) for b_pair in bounds]
        slices = none_s * 2 + slices
        return slices

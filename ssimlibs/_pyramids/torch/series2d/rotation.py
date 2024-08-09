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

from typing import Tuple, Union

import numpy
import scipy
import torch

from .coordsys import CoordSys


class Rotation(CoordSys):
    def __init__(self, window: Union[int, Tuple[int], Tuple[int, int]], orientations: int, is_complex: bool, offset: float = 0.0 * torch.pi, dtype: torch.dtype = torch.float64):
        assert orientations > 0
        CoordSys.__init__(self, window, dtype)
        self.orientations, self.is_complex = orientations, is_complex
        self.offset, self.angles = offset, torch.arctan2(self.y_axis, self.x_axis)

    @property
    def _param(self):
        order = self.orientations - 1
        p_numerator = 2 ** order * scipy.special.factorial(order)
        denominator = numpy.sqrt(self.orientations * scipy.special.factorial(2 * order))
        return p_numerator / denominator

    @property
    def groups(self):
        angles = [torch.remainder(self.angles - self.offset + torch.pi - b * torch.pi / self.orientations, 2 * torch.pi) - torch.pi for b in range(self.orientations)]
        masked = [torch.abs(self._param * torch.cos(angle) ** (self.orientations - 1)) for angle in angles]
        masked = [duals * (torch.abs(angle) < torch.pi / 2) for angle, duals in zip(angles, masked)] + [duals * (torch.abs(angle) > torch.pi / 2) for angle, duals in zip(angles, masked)] if self.is_complex else masked
        return masked

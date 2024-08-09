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

import cupy

from .doughnut import Doughnut
from .rotation import Rotation


class Series2D(Rotation, Doughnut):
    def __init__(self, window: Union[int, Tuple[int], Tuple[int, int]], levels: int, orientations: int, is_complex: bool, transition: float = 1.0, offset: float = 0.0 * cupy.pi, dtype: cupy.dtype = cupy.float64):
        Doughnut.__init__(self, levels=levels, window=window, transition=transition, dtype=dtype)
        Rotation.__init__(self, orientations=orientations, window=window, is_complex=is_complex, offset=offset, dtype=dtype)

    @property
    def groups(self):
        rotation = Rotation.groups.fget(self)
        doughnut = [[item] for item in Doughnut.groups.fget(self)]
        for index, group in enumerate(doughnut[+1: -1]):
            doughnut[index + 1] = [window * masked / 2 for window, masked in itertools.product(group, rotation)]
        return doughnut

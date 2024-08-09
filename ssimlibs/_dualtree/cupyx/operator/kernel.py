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

from abc import ABC, ABCMeta
from abc import abstractmethod

import cupy
import cupyx.scipy.ndimage
import numpy


class _LevelBasicKernelOperator(ABC, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, tensor: cupy.ndarray, axis: int) -> cupy.ndarray:
        return cupy.asarray(tensor) if not isinstance(tensor, cupy.ndarray) else tensor

    @abstractmethod
    def reverse(self, tensor: cupy.ndarray, axis: int) -> cupy.ndarray:
        return cupy.asarray(tensor) if not isinstance(tensor, cupy.ndarray) else tensor


class LevelAlpha1KernelOperator(_LevelBasicKernelOperator):
    def __init__(self, kernel: numpy.ndarray):
        self.kernel = cupy.asarray(kernel).flatten()

    def forward(self, tensor: cupy.ndarray, axis: int) -> cupy.ndarray:
        tensor = super().forward(tensor, axis=axis)
        return cupyx.scipy.ndimage.convolve1d(tensor, self.kernel, axis=axis, mode='reflect')

    def reverse(self, tensor: cupy.ndarray, axis: int) -> cupy.ndarray:
        tensor = super().reverse(tensor, axis=axis)
        return cupyx.scipy.ndimage.convolve1d(tensor, self.kernel, axis=axis, mode='reflect')


class LevelOthersKernelOperator(_LevelBasicKernelOperator):
    def __init__(self, coef_a: numpy.ndarray, coef_b: numpy.ndarray):
        assert len(coef_a) == len(coef_b)
        assert len(coef_a) % 0x02 == 0x00
        self.coef_a, self.coef_b = cupy.asarray(coef_a), cupy.asarray(coef_b)
        self.calc_c = cupy.sum(self.coef_a * self.coef_b)

    def forward(self, tensor: cupy.ndarray, axis: int) -> cupy.ndarray:
        tensor = super().forward(tensor, axis=axis)
        assert tensor.shape[axis] % 4 == 0, f'Rows on Axis {axis} should be completely divisible by 4'
        conv_a = cupy.take(cupyx.scipy.ndimage.convolve1d(tensor, cupy.kron(self.coef_a, cupy.asarray([1, 0])), axis=axis, mode='reflect'), list(range(0, tensor.shape[axis], 4)), axis=axis)
        conv_b = cupy.take(cupyx.scipy.ndimage.convolve1d(tensor, cupy.kron(self.coef_b, cupy.asarray([0, 1])), axis=axis, mode='reflect'), list(range(2, tensor.shape[axis], 4)), axis=axis)
        concat = cupy.concatenate([conv_a, conv_b] if self.calc_c > 0 else [conv_b, conv_a], axis=axis)
        _l = conv_a.shape[axis]
        return cupy.take(concat, [_l * i + j for j in range(_l) for i in range(2)], axis=axis)

    def reverse(self, tensor: cupy.ndarray, axis: int) -> cupy.ndarray:
        tensor = super().reverse(tensor, axis=axis)
        assert tensor.shape[axis] % 2 == 0, f'Rows on Axis {axis} should be completely divisible by 2'
        f_bank = [cupy.kron(self.coef_a[0::2], cupy.asarray([0, 1] if self.calc_c > 0 else [1, 0])), cupy.kron(self.coef_b[0::2], cupy.asarray([1, 0] if self.calc_c > 0 else [0, 1])),
                  cupy.kron(self.coef_a[1::2], cupy.asarray([0, 1] if self.calc_c > 0 else [1, 0])), cupy.kron(self.coef_b[1::2], cupy.asarray([1, 0] if self.calc_c > 0 else [0, 1]))]
        concat = cupy.concatenate([cupy.take(cupyx.scipy.ndimage.convolve1d(tensor, f, axis=axis, mode='reflect'), list(range(0, tensor.shape[axis], 2)), axis=axis) for f in f_bank], axis=axis)
        _l = tensor.shape[axis]
        return cupy.take(concat, [_l // 2 * i + j for j in range(_l // 2) for i in range(4)], axis=axis)

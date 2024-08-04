from abc import ABC, ABCMeta
from abc import abstractmethod

import numpy
import scipy


class _LevelBasicKernelOperator(ABC, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
        return numpy.asarray(tensor) if not isinstance(tensor, numpy.ndarray) else tensor

    @abstractmethod
    def reverse(self, tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
        return numpy.asarray(tensor) if not isinstance(tensor, numpy.ndarray) else tensor


class LevelAlpha1KernelOperator(_LevelBasicKernelOperator):
    def __init__(self, kernel: numpy.ndarray):
        self.kernel = kernel.flatten()

    def forward(self, tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
        tensor = super().forward(tensor, axis=axis)
        return scipy.ndimage.convolve1d(tensor, self.kernel, axis=axis, mode='reflect')

    def reverse(self, tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
        tensor = super().reverse(tensor, axis=axis)
        return scipy.ndimage.convolve1d(tensor, self.kernel, axis=axis, mode='reflect')


class LevelOthersKernelOperator(_LevelBasicKernelOperator):
    def __init__(self, coef_a: numpy.ndarray, coef_b: numpy.ndarray):
        assert len(coef_a) == len(coef_b)
        assert len(coef_a) % 0x02 == 0x00
        self.coef_a, self.coef_b = coef_a, coef_b
        self.calc_c = numpy.sum(self.coef_a * self.coef_b)

    def forward(self, tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
        tensor = super().forward(tensor, axis=axis)
        assert tensor.shape[axis] % 4 == 0, f'Rows on Axis {axis} should be completely divisible by 4'
        conv_a = numpy.take(scipy.ndimage.convolve1d(tensor, numpy.kron(self.coef_a, numpy.asarray([1, 0])), axis=axis, mode='reflect'), list(range(0, tensor.shape[axis], 4)), axis=axis)
        conv_b = numpy.take(scipy.ndimage.convolve1d(tensor, numpy.kron(self.coef_b, numpy.asarray([0, 1])), axis=axis, mode='reflect'), list(range(2, tensor.shape[axis], 4)), axis=axis)
        concat = numpy.concatenate([conv_a, conv_b] if self.calc_c > 0 else [conv_b, conv_a], axis=axis)
        _l = conv_a.shape[axis]
        return numpy.take(concat, [_l * i + j for j in range(_l) for i in range(2)], axis=axis)

    def reverse(self, tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
        tensor = super().reverse(tensor, axis=axis)
        assert tensor.shape[axis] % 2 == 0, f'Rows on Axis {axis} should be completely divisible by 2'
        f_bank = [numpy.kron(self.coef_a[0::2], numpy.asarray([0, 1] if self.calc_c > 0 else [1, 0])), numpy.kron(self.coef_b[0::2], numpy.asarray([1, 0] if self.calc_c > 0 else [0, 1])),
                  numpy.kron(self.coef_a[1::2], numpy.asarray([0, 1] if self.calc_c > 0 else [1, 0])), numpy.kron(self.coef_b[1::2], numpy.asarray([1, 0] if self.calc_c > 0 else [0, 1]))]
        concat = numpy.concatenate([numpy.take(scipy.ndimage.convolve1d(tensor, f, axis=axis, mode='reflect'), list(range(0, tensor.shape[axis], 2)), axis=axis) for f in f_bank], axis=axis)
        _l = tensor.shape[axis]
        return numpy.take(concat, [_l // 2 * i + j for j in range(_l // 2) for i in range(4)], axis=axis)

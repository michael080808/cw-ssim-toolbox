import functools
import operator
from typing import Tuple

import cupy
import cupyx.scipy.linalg


class LevelCommonReformatOperator:
    def __init__(self, n: int):
        assert n > 0
        self.dimension = n

    @property
    def poly_forward(self):
        return cupyx.scipy.linalg.block_diag(*self.dual_partial)

    @property
    def poly_reverse(self):
        return cupy.transpose(self.poly_forward)

    @property
    def cplx_forward(self):
        return cupy.kron(cupy.asarray([[1.0, +1j], [1.0, -1j]]), cupy.eye(2 ** (self.dimension - 1)))

    @property
    def cplx_reverse(self):
        return cupy.kron(cupy.asarray([[1.0, 1.0], [-1j, +1j]]), cupy.eye(2 ** (self.dimension - 1)))

    @property
    def prod_forward(self):
        return 1 / 2 ** (self.dimension - 1) * (self.cplx_forward @ self.poly_forward)

    @property
    def prod_reverse(self):
        return 1 / 2 ** (self.dimension - 1) * (self.poly_reverse @ self.cplx_reverse)

    def forward(self, tensor: cupy.ndarray) -> Tuple[cupy.ndarray, ...]:
        assert tensor.ndim >= self.dimension
        return self._tup_forward(self._forward_cut(tensor))

    def reverse(self, arrays: Tuple[cupy.ndarray, ...]) -> cupy.ndarray:
        assert len(arrays) == 2 ** self.dimension
        return self._reverse_rec(self._tup_reverse(arrays))

    def _tup_forward(self, arrays: Tuple[cupy.ndarray, ...]) -> Tuple[cupy.ndarray, ...]:
        assert len(arrays) == 2 ** self.dimension
        return tuple(functools.reduce(operator.add, [ratio * array for array, ratio in zip(arrays, ratios)]) for ratios in self.prod_forward)

    def _tup_reverse(self, arrays: Tuple[cupy.ndarray, ...]) -> Tuple[cupy.ndarray, ...]:
        assert len(arrays) == 2 ** self.dimension
        return tuple(functools.reduce(operator.add, [ratio * array for array, ratio in zip(arrays, ratios)]) for ratios in self.prod_reverse)

    def _forward_cut(self, tensor: cupy.ndarray) -> Tuple[cupy.ndarray, ...]:
        assert tensor.ndim >= self.dimension
        result = []
        for slicing in self.ndim_slicing:
            result.append(tensor[(Ellipsis,) + slicing])
        return tuple(result)

    def _reverse_rec(self, arrays: Tuple[cupy.ndarray, ...]) -> cupy.ndarray:
        assert len(arrays) == 2 ** self.dimension
        result = cupy.zeros([2 * s if i >= arrays[0].ndim - self.dimension else s for i, s in enumerate(arrays[0].shape)], dtype=cupy.complex_)
        for slicing, ndim_arr in zip(self.ndim_slicing, arrays):
            result[(Ellipsis,) + slicing] = ndim_arr
        return result

    @property
    def dual_partial(self):
        partial_real, partial_imag = cupy.ones([1, 1]), cupy.ones([1, 1])
        for _ in range(1, self.dimension):
            partial_real, partial_imag = (cupy.kron(cupy.asarray([[+1, +0], [+1, +0]]), partial_real) + cupy.kron(cupy.asarray([[-0, -1], [+0, +1]]), partial_imag),
                                          cupy.kron(cupy.asarray([[+1, +0], [+1, +0]]), partial_real) + cupy.kron(cupy.asarray([[+0, +1], [-0, -1]]), partial_imag))
        return partial_real, partial_imag

    @property
    def ndim_slicing(self):
        slice_tuples = [(slice(0, None, 2),), (slice(1, None, 2),)]
        for i in range(self.dimension - 1):
            slice_append = [slice(0, None, 2)] * (2 ** i) + [slice(1, None, 2)] * (2 ** i)
            slice_tuples = [group for group in zip(slice_append[::+1] + slice_append[::-1], *zip(*(slice_tuples * 2)))]
        return [tuple(reversed(group)) for group in slice_tuples]

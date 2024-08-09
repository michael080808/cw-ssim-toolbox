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

import functools
import operator

import numpy
import scipy


class GlobalStatistics:
    def __init__(self, _dim: int):
        assert _dim > 0
        self._dim = _dim

    def __call__(self, _a: numpy.ndarray, _b: numpy.ndarray):
        assert _a.ndim == _b.ndim >= self._dim
        assert _a.shape == _b.shape

        axis = tuple(range(-self._dim, 0))
        _avg_a_ = numpy.mean(_a, axis=axis)
        _avg_b_ = numpy.mean(_b, axis=axis)

        _avg_ab = (_a * _b).mean(axis=axis)
        _avg_a2 = (_a * _a).mean(axis=axis)
        _avg_b2 = (_b * _b).mean(axis=axis)

        _cov_ab = _avg_ab - _avg_a_ * _avg_b_
        _var_a2 = numpy.clip(_avg_a2 - _avg_a_ * _avg_a_, a_min=0x00, a_max=None)
        _var_b2 = numpy.clip(_avg_b2 - _avg_b_ * _avg_b_, a_min=0x00, a_max=None)

        return {'μ(A^1)': _avg_a_, 'μ(B^1)': _avg_b_, 'σ(A*B)': _cov_ab, 'σ(A^2)': _var_a2, 'σ(B^2)': _var_b2}


class WindowStatistics:
    _kernel: numpy.ndarray

    def __init__(self, _dim: int):
        assert _dim > 0
        self._dim = _dim

    def __call__(self, _a: numpy.ndarray, _b: numpy.ndarray):
        assert _a.ndim == _b.ndim >= self._dim
        assert _a.shape == _b.shape

        _kernel = numpy.expand_dims(self._kernel, axis=tuple(range(_a.ndim - self._dim)))
        _avg_a_ = scipy.signal.convolve(_a, _kernel, mode='valid', method='direct')
        _avg_b_ = scipy.signal.convolve(_b, _kernel, mode='valid', method='direct')

        _avg_ab = scipy.signal.convolve(_a * _b, _kernel, mode='valid', method='direct')
        _avg_a2 = scipy.signal.convolve(_a * _a, _kernel, mode='valid', method='direct')
        _avg_b2 = scipy.signal.convolve(_b * _b, _kernel, mode='valid', method='direct')

        _cov_ab = _avg_ab - _avg_a_ * _avg_b_
        _var_a2 = numpy.clip(_avg_a2 - _avg_a_ * _avg_a_, a_min=0.00, a_max=None)
        _var_b2 = numpy.clip(_avg_b2 - _avg_b_ * _avg_b_, a_min=0.00, a_max=None)

        return {'μ(A^1)': _avg_a_, 'μ(B^1)': _avg_b_, 'σ(A*B)': _cov_ab, 'σ(A^2)': _var_a2, 'σ(B^2)': _var_b2}


class UnifyWindowStatistics(WindowStatistics):
    @classmethod
    def unify(cls, _win: int) -> numpy.ndarray:
        return numpy.ones(_win, dtype=numpy.float64)

    def __init__(self, _dim: int, _win: int = 11):
        super().__init__(_dim)
        self._kernel = self.unify(_win)
        self._kernel = functools.reduce(operator.mul, [numpy.expand_dims(self._kernel, axis=tuple(j for j in range(_dim) if j != i)) for i in range(_dim)])
        self._kernel = self._kernel / numpy.sum(self._kernel)


class GaussWindowStatistics(WindowStatistics):
    @classmethod
    def gauss(cls, _win: int, _sigma: float) -> numpy.ndarray:
        return numpy.exp(-0.5 * ((numpy.arange(_win, dtype=numpy.float64) - _win / 2 + 0.5) / _sigma) ** 2)

    def __init__(self, _dim: int, _win: int = 11):
        super().__init__(_dim)
        self._kernel = self.gauss(_win, _sigma=1.5)
        self._kernel = functools.reduce(operator.mul, [numpy.expand_dims(self._kernel, axis=tuple(j for j in range(_dim) if j != i)) for i in range(_dim)])
        self._kernel = self._kernel / numpy.sum(self._kernel)


class SlideWindowStatistics:
    def __init__(self, _dim: int, _win: int = 7):
        assert _dim > 0
        self._dim = _dim
        # Kernel
        self._kernel = numpy.ones(shape=(_win,) * self._dim, dtype=numpy.float64)

    def __call__(self, _a: numpy.ndarray, _b: numpy.ndarray):
        assert _a.ndim == _b.ndim >= self._dim
        assert _a.shape == _b.shape

        _kernel = numpy.expand_dims(self._kernel, axis=tuple(range(_a.ndim - self._dim)))
        _sum_a2 = scipy.signal.convolve(numpy.abs(_a) ** 2, _kernel, mode='valid', method='direct')
        _sum_b2 = scipy.signal.convolve(numpy.abs(_b) ** 2, _kernel, mode='valid', method='direct')
        _sum_ab = scipy.signal.convolve(numpy.abs(_a) * numpy.abs(_b), _kernel, mode='valid', method='direct')

        _sum_sa = numpy.abs(scipy.signal.convolve(_a * numpy.conj(_b), _kernel, mode='valid', method='direct'))
        _sum_as = scipy.signal.convolve(numpy.abs(_a * numpy.conj(_b)), _kernel, mode='valid', method='direct')

        return {'Σ(|A||B|)': _sum_ab, 'Σ(|A|^2)': _sum_a2, 'Σ(|B|^2)': _sum_b2, '|Σ(AB*)|': _sum_sa, 'Σ(|AB*|)': _sum_as}

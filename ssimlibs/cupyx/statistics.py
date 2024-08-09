import functools
import operator

import cupy
import cupyx.scipy.signal


class GlobalStatistics:
    def __init__(self, _dim: int):
        assert _dim > 0
        self._dim = _dim

    def __call__(self, _a: cupy.ndarray, _b: cupy.ndarray):
        assert _a.ndim == _b.ndim >= self._dim
        assert _a.shape == _b.shape

        axis = tuple(range(-self._dim, 0))
        _avg_a_ = cupy.mean(_a, axis=axis)
        _avg_b_ = cupy.mean(_b, axis=axis)

        _avg_ab = (_a * _b).mean(axis=axis)
        _avg_a2 = (_a * _a).mean(axis=axis)
        _avg_b2 = (_b * _b).mean(axis=axis)

        _cov_ab = _avg_ab - _avg_a_ * _avg_b_
        _var_a2 = cupy.clip(_avg_a2 - _avg_a_ * _avg_a_, a_min=0x00, a_max=None)
        _var_b2 = cupy.clip(_avg_b2 - _avg_b_ * _avg_b_, a_min=0x00, a_max=None)

        return {'μ(A^1)': _avg_a_, 'μ(B^1)': _avg_b_, 'σ(A*B)': _cov_ab, 'σ(A^2)': _var_a2, 'σ(B^2)': _var_b2}


class WindowStatistics:
    _kernel: cupy.ndarray

    def __init__(self, _dim: int):
        assert _dim > 0
        self._dim = _dim

    def __call__(self, _a: cupy.ndarray, _b: cupy.ndarray):
        assert _a.ndim == _b.ndim >= self._dim
        assert _a.shape == _b.shape

        _kernel = cupy.expand_dims(self._kernel, axis=tuple(range(_a.ndim - self._dim)))
        _avg_a_ = cupyx.scipy.signal.convolve(_a, _kernel, mode='valid', method='direct')
        _avg_b_ = cupyx.scipy.signal.convolve(_b, _kernel, mode='valid', method='direct')

        _avg_ab = cupyx.scipy.signal.convolve(_a * _b, _kernel, mode='valid', method='direct')
        _avg_a2 = cupyx.scipy.signal.convolve(_a * _a, _kernel, mode='valid', method='direct')
        _avg_b2 = cupyx.scipy.signal.convolve(_b * _b, _kernel, mode='valid', method='direct')

        _cov_ab = _avg_ab - _avg_a_ * _avg_b_
        _var_a2 = cupy.clip(_avg_a2 - _avg_a_ * _avg_a_, a_min=0.00, a_max=None)
        _var_b2 = cupy.clip(_avg_b2 - _avg_b_ * _avg_b_, a_min=0.00, a_max=None)

        return {'μ(A^1)': _avg_a_, 'μ(B^1)': _avg_b_, 'σ(A*B)': _cov_ab, 'σ(A^2)': _var_a2, 'σ(B^2)': _var_b2}


class UnifyWindowStatistics(WindowStatistics):
    @classmethod
    def unify(cls, _win: int) -> cupy.ndarray:
        return cupy.ones(_win, dtype=cupy.float64)

    def __init__(self, _dim: int, _win: int = 11):
        super().__init__(_dim)
        self._kernel = self.unify(_win)
        self._kernel = functools.reduce(operator.mul, [cupy.expand_dims(self._kernel, axis=tuple(j for j in range(_dim) if j != i)) for i in range(_dim)])
        self._kernel = self._kernel / cupy.sum(self._kernel)


class GaussWindowStatistics(WindowStatistics):
    @classmethod
    def gauss(cls, _win: int, _sigma: float) -> cupy.ndarray:
        return cupy.exp(-0.5 * ((cupy.arange(_win, dtype=cupy.float64) - _win / 2 + 0.5) / _sigma) ** 2)

    def __init__(self, _dim: int, _win: int = 11):
        super().__init__(_dim)
        self._kernel = self.gauss(_win, _sigma=1.5)
        self._kernel = functools.reduce(operator.mul, [cupy.expand_dims(self._kernel, axis=tuple(j for j in range(_dim) if j != i)) for i in range(_dim)])
        self._kernel = self._kernel / cupy.sum(self._kernel)


class SlideWindowStatistics:
    def __init__(self, _dim: int, _win: int = 7):
        assert _dim > 0
        self._dim = _dim
        # Kernel
        self._kernel = cupy.ones(shape=(_win,) * self._dim, dtype=cupy.float64)

    def __call__(self, _a: cupy.ndarray, _b: cupy.ndarray):
        assert _a.ndim == _b.ndim >= self._dim
        assert _a.shape == _b.shape

        _kernel = cupy.expand_dims(self._kernel, axis=tuple(range(_a.ndim - self._dim)))
        _sum_a2 = cupyx.scipy.signal.convolve(cupy.abs(_a) ** 2, _kernel, mode='valid', method='direct')
        _sum_b2 = cupyx.scipy.signal.convolve(cupy.abs(_b) ** 2, _kernel, mode='valid', method='direct')
        _sum_ab = cupyx.scipy.signal.convolve(cupy.abs(_a) * cupy.abs(_b), _kernel, mode='valid', method='direct')

        _sum_sa = cupy.abs(cupyx.scipy.signal.convolve(_a * cupy.conj(_b), _kernel, mode='valid', method='direct'))
        _sum_as = cupyx.scipy.signal.convolve(cupy.abs(_a * cupy.conj(_b)), _kernel, mode='valid', method='direct')

        return {'Σ(|A||B|)': _sum_ab, 'Σ(|A|^2)': _sum_a2, 'Σ(|B|^2)': _sum_b2, '|Σ(AB*)|': _sum_sa, 'Σ(|AB*|)': _sum_as}
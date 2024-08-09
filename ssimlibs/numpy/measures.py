import functools
import operator
from abc import ABC, ABCMeta
from fractions import Fraction
from typing import Union, Sequence, Literal, Any

import numpy

from .components import RealComponents, CplxComponents
from .statistics import GlobalStatistics
from .statistics import SlideWindowStatistics
from .statistics import UnifyWindowStatistics, GaussWindowStatistics
from .._dualtree.numpy import DualTreeComplexWaveletTransform
from .._pyramids.numpy import SteerablePyramid2D


class _MeasureWithRealComponents(ABC, metaclass=ABCMeta):
    components: RealComponents
    statistics: Union[GlobalStatistics, GaussWindowStatistics]

    def luminance(self, _a: numpy.ndarray, _b: numpy.ndarray):
        statistics = self.statistics(_a, _b)
        return self.components.luminance(statistics['μ(A^1)'], statistics['μ(B^1)'])

    def contrasts(self, _a: numpy.ndarray, _b: numpy.ndarray):
        statistics = self.statistics(_a, _b)
        return self.components.contrasts(statistics['σ(A^2)'], statistics['σ(B^2)'])

    def structure(self, _a: numpy.ndarray, _b: numpy.ndarray):
        statistics = self.statistics(_a, _b)
        return self.components.structure(statistics['σ(A*B)'], statistics['σ(A^2)'], statistics['σ(B^2)'])


class _MeasureWithMeanComponents(_MeasureWithRealComponents):
    _dim: int

    def luminance(self, _a: numpy.ndarray, _b: numpy.ndarray):
        axis = tuple(range(-self._dim, 0))
        return numpy.mean(super().luminance(_a, _b), axis=axis)

    def contrasts(self, _a: numpy.ndarray, _b: numpy.ndarray):
        axis = tuple(range(-self._dim, 0))
        return numpy.mean(super().contrasts(_a, _b), axis=axis)

    def structure(self, _a: numpy.ndarray, _b: numpy.ndarray):
        axis = tuple(range(-self._dim, 0))
        return numpy.mean(super().structure(_a, _b), axis=axis)


class _UIQI(ABC, metaclass=ABCMeta):
    statistics: Union[GlobalStatistics, GaussWindowStatistics]

    def __call__(self, _a: numpy.ndarray, _b: numpy.ndarray):
        assert _a.shape == _b.shape
        statistics = self.statistics(_a, _b)
        p_numerator = 4 * statistics['σ(A*B)'] * statistics['μ(A^1)'] * statistics['μ(B^1)']
        denominator = (statistics['σ(A^2)'] + statistics['σ(B^2)']) * (statistics['μ(A^1)'] ** 2 + statistics['μ(B^1)'] ** 2)
        if denominator.size - numpy.count_nonzero(denominator) > 0:
            eps = numpy.finfo(denominator.dtype).eps
            location = numpy.where(denominator == 0)
            p_numerator[location] = p_numerator[location] + eps
            denominator[location] = denominator[location] + eps
        return p_numerator / denominator


class _SSIM(ABC, metaclass=ABCMeta):
    _c1: float
    _c2: float
    statistics: Union[GlobalStatistics, GaussWindowStatistics]

    def __call__(self, _a: numpy.ndarray, _b: numpy.ndarray):
        assert _a.shape == _b.shape
        statistics = self.statistics(_a, _b)
        p_numerator = (2 * statistics['μ(A^1)'] * statistics['μ(B^1)'] + self._c1) * (2 * statistics['σ(A*B)'] + self._c2)
        denominator = (statistics['μ(A^1)'] ** 2 + statistics['μ(B^1)'] ** 2 + self._c1) * (statistics['σ(A^2)'] + statistics['σ(B^2)'] + self._c2)
        return p_numerator / denominator


class gUIQI(_MeasureWithRealComponents, _UIQI):
    def __init__(self, k1: float = 0.01, k2: float = 0.03, _dim: int = 2):
        self._c1, self._c2 = k1 ** 2, k2 ** 2
        self.statistics = GlobalStatistics(_dim)
        self.components = RealComponents(c1=self._c1, c2=self._c2, c3=0.5 * self._c2)


class gSSIM(_MeasureWithRealComponents, _SSIM):
    def __init__(self, k1: float = 0.01, k2: float = 0.03, _dim: int = 2):
        self._c1, self._c2 = k1 ** 2, k2 ** 2
        self.statistics = GlobalStatistics(_dim)
        self.components = RealComponents(c1=self._c1, c2=self._c2, c3=0.5 * self._c2)


class mUIQI(_MeasureWithMeanComponents, _UIQI):
    def __init__(self, k1: float = 0.01, k2: float = 0.03, _dim: int = 2, _win: int = 11, _dis: Literal['unify', 'gauss'] = 'gauss'):
        self._dim = _dim
        self._c1, self._c2 = k1 ** 2, k2 ** 2
        self.statistics = GaussWindowStatistics(_dim, _win) if _dis == 'gauss' else UnifyWindowStatistics(_dim, _win)
        self.components = RealComponents(c1=self._c1, c2=self._c2, c3=0.5 * self._c2)

    def __call__(self, _a: numpy.ndarray, _b: numpy.ndarray):
        return numpy.mean(super().__call__(_a, _b), axis=tuple(range(-self._dim, 0)))


class mSSIM(_MeasureWithMeanComponents, _SSIM):
    def __init__(self, k1: float = 0.01, k2: float = 0.03, _dim: int = 2, _win: int = 11, _dis: Literal['unify', 'gauss'] = 'gauss'):
        self._dim = _dim
        self._c1, self._c2 = k1 ** 2, k2 ** 2
        self.statistics = GaussWindowStatistics(_dim, _win) if _dis == 'gauss' else UnifyWindowStatistics(_dim, _win)
        self.components = RealComponents(c1=self._c1, c2=self._c2, c3=0.5 * self._c2)

    def __call__(self, _a: numpy.ndarray, _b: numpy.ndarray):
        return numpy.mean(super().__call__(_a, _b), axis=tuple(range(-self._dim, 0)))


class MsSSIM(_MeasureWithRealComponents):
    # noinspection NonAsciiCharacters   
    def __init__(self, k1: float = 0.01, k2: float = 0.03, α: float = 0.1333, β: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333), γ: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333), _dim: int = 2, _win: int = 11):
        assert len(β) == len(γ) > 0
        self._dim, self._levels = _dim, len(β)
        self._c1, self._c2 = k1 ** 2, k2 ** 2
        self.statistics = GaussWindowStatistics(_dim, _win)
        self.components = RealComponents(c1=self._c1, c2=self._c2, c3=0.5 * self._c2)
        self._α, self._β, self._γ = α, β, γ

    def calc_slices(self, _shape: Sequence[int], _tier: int):
        _removed = (1 - Fraction(1, 2) ** _tier) / 2
        return tuple(
            slice(None) if index < self._dim else
            slice(+int(_size * _removed),
                  -int(_size * _removed))
            for index, _size in enumerate(reversed(_shape)))

    def down_sample(self, _t: numpy.ndarray, level: int):
        axis = tuple(range(-self._dim, 0))
        _mid = numpy.fft.fftn(_t + 0.0j, axes=axis)
        _mid = numpy.fft.fftshift(_mid, axes=axis)
        _mid = _mid[self.calc_slices(_mid.shape, level)]
        _mid = numpy.fft.ifftshift(_mid, axes=axis)
        return numpy.fft.ifftn(_mid, axes=axis).real

    def luminance(self, _a: numpy.ndarray, _b: numpy.ndarray, level: int = 0):
        return super().luminance(self.down_sample(_a, level), self.down_sample(_b, level))

    def contrasts(self, _a: numpy.ndarray, _b: numpy.ndarray, level: int = 0):
        return super().contrasts(self.down_sample(_a, level), self.down_sample(_b, level))

    def structure(self, _a: numpy.ndarray, _b: numpy.ndarray, level: int = 0):
        return super().structure(self.down_sample(_a, level), self.down_sample(_b, level))

    def __call__(self, _a: numpy.ndarray, _b: numpy.ndarray):
        axis = tuple(range(-self._dim, 0))
        _l = [numpy.mean(self.luminance(_a, _b, level=self._levels - 1) ** self._α, axis=axis)]
        _c = [numpy.mean(self.contrasts(_a, _b, level=level), axis=axis) ** value for level, value in enumerate(self._β)]
        _s = [numpy.mean(self.structure(_a, _b, level=level), axis=axis) ** value for level, value in enumerate(self._γ)]
        # Products
        return functools.reduce(operator.mul, _l + _c + _s)


class CwSSIM:
    def __init__(self, k: float = 0, _dim: int = 2, _win: int = 7, backend: Literal['DualTree', 'Pyramids'] = 'DualTree', *args, **kwargs):
        assert backend.lower() in ['DualTree'.lower(), 'Pyramids'.lower()]
        kwargs = kwargs if backend.lower() == 'Pyramids'.lower() and _dim == 2 else kwargs | {'dimension': _dim}
        self._transform = SteerablePyramid2D(*args, **kwargs) if backend.lower() == 'Pyramids'.lower() and _dim == 2 else DualTreeComplexWaveletTransform(*args, **kwargs)
        self._k = k
        self._dim = _dim
        self.statistics = SlideWindowStatistics(_dim, _win)
        self.components = CplxComponents(k)

    @staticmethod
    def seq(_t: Sequence[Any]):
        status = True
        while status:
            status = False
            _m = []
            for item in _t:
                if isinstance(item, numpy.ndarray):
                    _m.append(item)
                elif isinstance(item, Sequence):
                    _m.extend(item)
                    if len(item[0]) > 0 and not isinstance(item[0], numpy.ndarray):
                        status = True
            _t = _m
        return _t

    def amp(self, _a: numpy.ndarray, _b: numpy.ndarray):
        counter, accumulator = 0, 0
        axis = tuple(range(-self._dim, 0))
        for _a_item, _b_item in zip(self._transform.forward(_a), self._transform.forward(_b)):
            statistics = self.statistics(_a_item, _b_item)
            result = self.components.amp(statistics['Σ(|A||B|)'], statistics['Σ(|A|^2)'], statistics['Σ(|B|^2)'])
            counter, accumulator = counter + 1, accumulator + numpy.mean(result, axis=axis)
        return accumulator / counter

    def phi(self, _a: numpy.ndarray, _b: numpy.ndarray):
        counter, accumulator = 0, 0
        axis = tuple(range(-self._dim, 0))
        for _a_item, _b_item in zip(self._transform.forward(_a), self._transform.forward(_b)):
            statistics = self.statistics(_a_item, _b_item)
            result = self.components.phi(statistics['|Σ(AB*)|'], statistics['Σ(|AB*|)'])
            counter, accumulator = counter + 1, accumulator + numpy.mean(result, axis=axis)
        return accumulator / counter

    def __call__(self, _a: numpy.ndarray, _b: numpy.ndarray):
        counter, accumulator = 0, 0
        axis = tuple(range(-self._dim, 0))
        for _a_item, _b_item in zip(self.seq(self._transform.forward(_a)), self.seq(self._transform.forward(_b))):
            statistics = self.statistics(_a_item, _b_item)
            p_numerator = 2 * statistics['|Σ(AB*)|'] + self._k
            denominator = statistics['Σ(|A|^2)'] + statistics['Σ(|B|^2)'] + self._k
            if denominator.size - numpy.count_nonzero(denominator) > 0:
                eps = numpy.finfo(denominator.dtype).eps
                location = numpy.where(denominator == 0)
                p_numerator[location] = p_numerator[location] + eps
                denominator[location] = denominator[location] + eps
            counter, accumulator = counter + 1, accumulator + numpy.mean(p_numerator / denominator, axis=axis)
        return accumulator / counter

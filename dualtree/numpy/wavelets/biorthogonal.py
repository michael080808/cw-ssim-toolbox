from functools import cached_property

import numpy

from ._baseclass import BiorthogonalWavelet
from ._decorator import validator, temporary, persisted


class _NearSymmetricWavelet(BiorthogonalWavelet):
    @classmethod
    def xfm(cls, t: numpy.ndarray, m: numpy.ndarray):
        pads = (len(m) - 1) // 2
        h, z = numpy.array([t[-1]]), numpy.ones(1)
        for it in reversed(t[:-1]):
            z = numpy.convolve(z, m)
            h = numpy.pad(h, (pads, pads)) + it * z
        return h

    def __init__(self, _px: float = 3.5, _pa: float = 0.0, _pb: float = 0.0, dtype: numpy.floating = numpy.float64):
        super().__init__(dtype)
        # Ht & Ft
        c = -_px
        a, b = 2 * c + 2 / (2 + c), -(2 + c)
        self.Ht = numpy.convolve([1, 1], -numpy.array([1, b, a], dtype=dtype))
        self.Ft = numpy.convolve([1, 1], -numpy.array([1, c], dtype=dtype))

        if _pa > 0 and _pb > 0:
            self.M = numpy.array([+_pb, 0, -_pa, 0, 1 + _pa - _pb, 0, 1 + _pa - _pb, 0, -_pa, 0, +_pb]) / 2
        elif _pa > 0:
            self.M = numpy.array([-_pa, 0, 1 + _pa, 0, 1 + _pa, 0, -_pa]) / 2
        else:
            self.M = numpy.array([1, 0, 1]) / 2

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_channelizer(self) -> numpy.ndarray:
        return self.normalize(self.xfm(self.Ft, self.M))

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_synthesiser(self) -> numpy.ndarray:
        return self.normalize(self.xfm(self.Ht, self.M))


class HaarWavelet(BiorthogonalWavelet):
    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_channelizer(self) -> numpy.ndarray:
        return self.normalize(numpy.array([+1, +1], dtype=numpy.int64))

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_synthesiser(self) -> numpy.ndarray:
        return self.normalize(numpy.array([+1, +1], dtype=numpy.int64))


class LeGallWavelet(BiorthogonalWavelet):
    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_channelizer(self) -> numpy.ndarray:
        return self.normalize(numpy.array([-1, +2, +6, +2, -1], dtype=numpy.int64))

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_synthesiser(self) -> numpy.ndarray:
        return self.normalize(numpy.array([+1, +2, +1], dtype=numpy.int64))


class BiorthogonalT06T10Wavelet(BiorthogonalWavelet):
    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_channelizer(self) -> numpy.ndarray:
        return self.normalize(numpy.convolve(numpy.array([+1, +3, +3, +1], dtype=numpy.int64), numpy.array([-1, +4, -1], dtype=numpy.int64)))

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_synthesiser(self) -> numpy.ndarray:
        return self.normalize(numpy.convolve(numpy.array([+1, +3, +3, +1], dtype=numpy.int64), numpy.array([+1, -2, -5, 28, -5, -2, +1], dtype=numpy.int64)))


class BiorthogonalT06T10ComplexWavelet(BiorthogonalWavelet):
    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_channelizer(self) -> numpy.ndarray:
        return self.normalize(numpy.array([+1, -1, +8, +8, -1, +1], dtype=numpy.int64))

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_synthesiser(self) -> numpy.ndarray:
        return self.normalize(numpy.array([+1, +1, +8, -8, 62, 62, -8, +8, +1, +1], dtype=numpy.int64))


class AntoniniWavelet(BiorthogonalWavelet):
    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__(dtype)
        self.roots = numpy.roots(numpy.array([+5, -40, +131, -208, 131, -40, +5], dtype=self.dtype))

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_channelizer(self) -> numpy.ndarray:
        return self.normalize(numpy.real(numpy.poly([-1] * 4 + self.roots[1: 5: 1].tolist())))

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_synthesiser(self) -> numpy.ndarray:
        return self.normalize(numpy.real(numpy.poly([-1] * 4 + self.roots[0: 6: 5].tolist())))


class DaubechiesT04T04Wavelet(BiorthogonalWavelet):
    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__(dtype)
        self.roots = numpy.roots(numpy.array([-1, +4, -1], dtype=self.dtype))

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_channelizer(self) -> numpy.ndarray:
        return self.normalize(numpy.real(numpy.poly([-1] * 2 + self.roots[[0]].tolist())))

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_synthesiser(self) -> numpy.ndarray:
        return self.normalize(numpy.real(numpy.poly([-1] * 2 + self.roots[[1]].tolist())))


class DaubechiesT06T06Wavelet(BiorthogonalWavelet):
    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__(dtype)
        self.roots = numpy.roots(numpy.array([+3, -18, +38, -18, +3], dtype=self.dtype))

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_channelizer(self) -> numpy.ndarray:
        return self.normalize(numpy.real(numpy.poly([-1] * 3 + self.roots[[0, 1]].tolist())))

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_synthesiser(self) -> numpy.ndarray:
        return self.normalize(numpy.real(numpy.poly([-1] * 3 + self.roots[[2, 3]].tolist())))


class DaubechiesT08T08Wavelet(BiorthogonalWavelet):
    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__(dtype)
        self.roots = numpy.roots(numpy.array([+5, -40, +131, -208, 131, -40, +5], dtype=self.dtype))

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_channelizer(self) -> numpy.ndarray:
        return self.normalize(numpy.real(numpy.poly([-1] * 4 + self.roots[[0, 3, 4]].tolist())))

    @cached_property
    @temporary
    @persisted()
    @validator()
    def lo_pass_synthesiser(self) -> numpy.ndarray:
        return self.normalize(numpy.real(numpy.poly([-1] * 4 + self.roots[[1, 2, 5]].tolist())))


class NearSymmetricT05T07Wavelet(_NearSymmetricWavelet):
    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__(dtype=dtype)


class NearSymmetricT13T19Wavelet(_NearSymmetricWavelet):
    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__(_pa=3 / 16, dtype=dtype)

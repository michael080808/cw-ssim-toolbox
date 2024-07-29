import itertools
from functools import cached_property
from math import pi

import numpy
import scipy
import sympy

from ._baseclass import OrthogonalWavelet
from ._decorator import validator, temporary, persisted


class _QShiftWavelet(OrthogonalWavelet):
    def __init__(self, tap_n: int = 14, f_min: float | sympy.Number = 0.36, dtype: numpy.floating = numpy.float64):
        if tap_n < 10 and tap_n != 6:
            raise ValueError('Q-shift filter order must be even and greater than 8 except 6.')
        if tap_n % 2:
            raise ValueError('Q-shift filter order must be even!')
        super().__init__(dtype)
        self.n, self.f = tap_n, f_min if isinstance(f_min, sympy.Number) else sympy.Number(f_min)

    @classmethod
    def pr_force(cls, h0):
        h1 = h0[::-1] * numpy.cumprod(-numpy.ones_like(h0))
        hh = numpy.concatenate([numpy.transpose([h0]), numpy.transpose([h1])], axis=-1)
        _a, _d = [], []

        while hh.shape[0] > 2:
            _a.append(numpy.sum(numpy.prod(hh[:2], axis=-1)) / numpy.sum(hh[:2, 1] ** 2))
            if abs(_a[-1]) < 1:
                _a[-1] = _a[-1] / 1
                ht = hh @ numpy.array([[1, +_a[-1]], [-_a[-1], 1]])
                hh = numpy.concatenate([ht[+2:, :1], ht[:-2, 1:]], axis=-1)
                _d.append(True)
            else:
                _a[-1] = 1 / _a[-1]
                ht = hh @ numpy.array([[1, -_a[-1]], [+_a[-1], 1]])
                hh = numpy.concatenate([ht[:-2, :1], ht[+2:, 1:]], axis=-1)
                _d.append(False)

        for d, a in zip(reversed(_d), reversed(_a)):
            if d:
                ht = numpy.concatenate([
                    numpy.pad(hh[:, :1], ((2, 0), (0, 0))),
                    numpy.pad(hh[:, 1:], ((0, 2), (0, 0)))], axis=-1)
                hh = ht @ numpy.array([[1, -a], [+a, 1]])
            else:
                ht = numpy.concatenate([
                    numpy.pad(hh[:, :1], ((0, 2), (0, 0))),
                    numpy.pad(hh[:, 1:], ((2, 0), (0, 0)))], axis=-1)
                hh = ht @ numpy.array([[1, +a], [-a, 1]])

        return hh[:, 0] / numpy.prod(1 + numpy.array(_a) ** 2)

    @classmethod
    def lad_sect(cls, mx, h4):
        return numpy.stack([numpy.convolve(mx[0], h4[0]) + numpy.convolve(mx[1], h4[1]), numpy.convolve(mx[0], h4[2]) + numpy.convolve(mx[1], h4[3])])

    def initials(self, np: int = 16, mp: int = 4):
        indices = [sympy.Rational(i, np) for i in range(np + 1)]
        cosines = (sympy.Matrix([i ** mp for i in indices[:np:+1]] + [2 - i ** mp for i in indices[:00:-1]]) * (sympy.pi / 4)).applyfunc(sympy.cos)
        cosines = sympy.Matrix(list(cosines[::+1]) + [0] * (4 * np) + list(cosines[::-1]))
        rotates = (sympy.I * sympy.pi * sympy.Matrix([i for i in itertools.chain(range(0x00 * np, 0x04 * np), range(0x0C * np, 0x10 * np))]) / (8 * np)).applyfunc(sympy.exp)
        results = numpy.fft.ifft(numpy.array(sympy.matrix_multiply_elementwise(cosines, rotates), dtype=numpy.complex128).squeeze()).real[:self.n]
        return 2 * numpy.concatenate([numpy.flip(results), results])

    def iterates(self, it: int = 20, c_mag: int = 20, f_mag: int = 20):
        hc = self.initials()
        f0 = sympy.Matrix([index for index in range(2 * self.n + 1)])
        f0 = sympy.pi * (self.f * sympy.matrices.ones(f0.rows, f0.cols) + (1 - self.f) * f0 / 2 / self.n)
        fz = sympy.I * f0 * sympy.Matrix([list(range(len(hc)))])
        fz = fz.applyfunc(sympy.exp)
        fz = f_mag * numpy.array(numpy.concatenate([
            numpy.array(fz.applyfunc(sympy.re), dtype=self.dtype),
            numpy.array(fz.applyfunc(sympy.im), dtype=self.dtype)]))

        for _ in range(it):
            hp = hc
            fc = numpy.transpose(fz @ numpy.transpose([hp])).squeeze()
            cm = scipy.linalg.toeplitz(
                numpy.pad(hc[0::1], (0, 2 * self.n - 1)),
                numpy.pad(hc[0: 1], (0, 2 * self.n - 1)))
            c2 = numpy.concatenate([cm[3: 2 * self.n + 1: 4] * c_mag, fz])
            c2 = scipy.signal.convolve2d(c2, numpy.array([[1, 0, 2, 0, 1]]), mode='valid')
            _l, _r = numpy.split(c2, 2, axis=-1)
            c2 = numpy.add(_l, numpy.flip(_r, axis=-1))
            h2 = scipy.linalg.lstsq(c2, numpy.pad(numpy.concatenate([[c_mag], -fc]), (c2.shape[0] - 1 - fc.shape[0], 0)))[0]
            hc = (numpy.convolve(numpy.concatenate([h2[::+1], h2[::-1]]), [1, 0, 2, 0, 1]) + hp) / 2
            c_mag = c_mag * 2

        return self.pr_force(hc[1::2])

    def special6(self):
        thetas = numpy.array([+1.00, +1.00, -0.81, -1.62], dtype=self.dtype) * pi / 4
        thetas = -numpy.diff(numpy.pad(thetas, (0, 1)))
        middle = numpy.eye(2)
        cosine, sine_v = numpy.cos(thetas), numpy.sin(thetas)
        # Iterate
        for i, (c, s) in enumerate(zip(cosine, sine_v)):
            if i == 0:
                middle = self.lad_sect(middle, numpy.array([[+c], [+s], [-s], [+c]]))
            else:
                middle = self.lad_sect(middle, numpy.array([[+c, 0, 0], [0, 0, +s], [-s, 0, 0], [0, 0, +c]]))
        return numpy.pad(middle[0], (0, 2))

    @cached_property
    @temporary()
    @persisted()
    @validator()
    def lo_pass_channelizer(self) -> numpy.ndarray:
        return self.normalize(self.special6() if self.n == 6 else self.iterates())


class QShiftT06Wavelet(_QShiftWavelet):
    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__(tap_n=6, dtype=dtype)


class QShiftT10Wavelet(_QShiftWavelet):
    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__(tap_n=10, dtype=dtype)


class QShiftT12Wavelet(_QShiftWavelet):
    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__(tap_n=12, dtype=dtype)


class QShiftT14Wavelet(_QShiftWavelet):
    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__(tap_n=14, dtype=dtype)


class QShiftT16Wavelet(_QShiftWavelet):
    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__(tap_n=16, dtype=dtype)


class QShiftT18Wavelet(_QShiftWavelet):
    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__(tap_n=18, dtype=dtype)


class QShiftT32Wavelet(_QShiftWavelet):
    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__(tap_n=32, dtype=dtype)

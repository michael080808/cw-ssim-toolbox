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

import torch
from torch import Tensor


def convolve(_tensor: Tensor, _kernel: Tensor) -> Tensor:
    assert _tensor.ndim >= _kernel.ndim
    assert functools.reduce(operator.and_, [_t_shape >= _k_shape for _t_shape, _k_shape in zip(reversed(_tensor.shape), reversed(_kernel.shape))])
    if _kernel.ndim > 3:
        _tensor = _tensor.movedim(source=-_kernel.ndim, destination=0)
        _conv_l = []
        for i in range(_tensor.shape[0] - _kernel.shape[0] + 1):
            _conv_s = 0
            for _t, _k in zip(_tensor[slice(i, i + _kernel.shape[0])], _kernel):
                _conv_s = _conv_s + convolve(_t, _k)
            _conv_l.append(_conv_s)
        return torch.stack(_conv_l, dim=-_kernel.ndim)
    elif _kernel.ndim == 3:
        _shapes = _tensor.shape
        if _tensor.ndim > 3:
            _tensor = _tensor.flatten(0, -4)
        _result = torch.nn.functional.conv3d(
            _tensor[(slice(None),) * (_tensor.ndim - 3) +
                    (None,) * 1 + (slice(None),) * 3],
            _kernel[(None,) * 2 + (slice(None),) * 3].to(device=_tensor.device, dtype=_tensor.dtype)
        ).squeeze(dim=-4)
        return _result.unflatten(0, _shapes[:-3]) if len(_shapes) > 3 else _result
    elif _kernel.ndim == 2:
        _shapes = _tensor.shape
        if _tensor.ndim > 2:
            _tensor = _tensor.flatten(0, -3)
        _result = torch.nn.functional.conv2d(
            _tensor[(slice(None),) * (_tensor.ndim - 2) +
                    (None,) * 1 + (slice(None),) * 2],
            _kernel[(None,) * 2 + (slice(None),) * 2].to(device=_tensor.device, dtype=_tensor.dtype)
        ).squeeze(dim=-3)
        return _result.unflatten(0, _shapes[:-2]) if len(_shapes) > 2 else _result
    elif _kernel.ndim == 1:
        _shapes = _tensor.shape
        if _tensor.ndim > 1:
            _tensor = _tensor.flatten(0, -2)
        _result = torch.nn.functional.conv1d(
            _tensor[(slice(None),) * (_tensor.ndim - 1) +
                    (None,) * 1 + (slice(None),) * 1],
            _kernel[(None,) * 2 + (slice(None),) * 1].to(device=_tensor.device, dtype=_tensor.dtype)
        ).squeeze(dim=-2)
        return _result.unflatten(0, _shapes[:-1]) if len(_shapes) > 1 else _result
    else:
        raise ValueError('Unknown kernel shape.')


class GlobalStatistics:
    def __init__(self, _dim: int):
        assert _dim > 0
        self._dim = _dim

    def __call__(self, _a: Tensor, _b: Tensor):
        assert _a.ndim == _b.ndim >= self._dim
        assert _a.shape == _b.shape

        axis = tuple(range(-self._dim, 0))
        _avg_a_ = torch.mean(_a, dim=axis)
        _avg_b_ = torch.mean(_b, dim=axis)

        _avg_ab = (_a * _b).mean(dim=axis)
        _avg_a2 = (_a * _a).mean(dim=axis)
        _avg_b2 = (_b * _b).mean(dim=axis)

        _cov_ab = _avg_ab - _avg_a_ * _avg_b_
        _var_a2 = torch.clip(_avg_a2 - _avg_a_ * _avg_a_, min=0x00, max=None)
        _var_b2 = torch.clip(_avg_b2 - _avg_b_ * _avg_b_, min=0x00, max=None)

        return {'μ(A^1)': _avg_a_, 'μ(B^1)': _avg_b_, 'σ(A*B)': _cov_ab, 'σ(A^2)': _var_a2, 'σ(B^2)': _var_b2}


class WindowStatistics:
    _kernel: Tensor

    def __init__(self, _dim: int):
        assert _dim > 0
        self._dim = _dim

    def __call__(self, _a: Tensor, _b: Tensor):
        assert _a.ndim == _b.ndim >= self._dim
        assert _a.shape == _b.shape

        _kernel = self._kernel[(None,) * (_a.ndim - self._dim) + (slice(None),) * self._dim]
        _avg_a_ = convolve(_a, _kernel)
        _avg_b_ = convolve(_b, _kernel)

        _avg_ab = convolve(_a * _b, _kernel)
        _avg_a2 = convolve(_a * _a, _kernel)
        _avg_b2 = convolve(_b * _b, _kernel)

        _cov_ab = _avg_ab - _avg_a_ * _avg_b_
        _var_a2 = torch.clip(_avg_a2 - _avg_a_ * _avg_a_, min=0.00, max=None)
        _var_b2 = torch.clip(_avg_b2 - _avg_b_ * _avg_b_, min=0.00, max=None)

        return {'μ(A^1)': _avg_a_, 'μ(B^1)': _avg_b_, 'σ(A*B)': _cov_ab, 'σ(A^2)': _var_a2, 'σ(B^2)': _var_b2}


class UnifyWindowStatistics(WindowStatistics):
    @classmethod
    def unify(cls, _win: int) -> Tensor:
        return torch.ones(_win, dtype=torch.float64)

    def __init__(self, _dim: int, _win: int = 11):
        super().__init__(_dim)
        self._kernel = self.unify(_win)
        self._kernel = functools.reduce(operator.mul, [self._kernel[tuple(None if j != i else slice(None) for j in range(_dim))] for i in range(_dim)])
        self._kernel = self._kernel / torch.sum(self._kernel)


class GaussWindowStatistics(WindowStatistics):
    @classmethod
    def gauss(cls, _win: int, _sigma: float) -> Tensor:
        return torch.exp(-0.5 * ((torch.arange(_win, dtype=torch.float64) - _win / 2 + 0.5) / _sigma) ** 2)

    def __init__(self, _dim: int, _win: int = 11):
        super().__init__(_dim)
        self._kernel = self.gauss(_win, _sigma=1.5)
        self._kernel = functools.reduce(operator.mul, [self._kernel[tuple(None if j != i else slice(None) for j in range(_dim))] for i in range(_dim)])
        self._kernel = self._kernel / torch.sum(self._kernel)


class SlideWindowStatistics:
    def __init__(self, _dim: int, _win: int = 7):
        assert _dim > 0
        self._dim = _dim
        # Kernel
        self._kernel = torch.ones((_win,) * self._dim, dtype=torch.float64)

    def __call__(self, _a: Tensor, _b: Tensor):
        assert _a.ndim == _b.ndim >= self._dim
        assert _a.shape == _b.shape

        _kernel = self._kernel[(None,) * (_a.ndim - self._dim) + (slice(None),) * self._dim]
        _sum_a2 = convolve(torch.abs(_a) ** 2, _kernel)
        _sum_b2 = convolve(torch.abs(_b) ** 2, _kernel)
        _sum_ab = convolve(torch.abs(_a) * torch.abs(_b), _kernel)

        _sum_sa = torch.abs(convolve(_a * torch.conj(_b), _kernel))
        _sum_as = convolve(torch.abs(_a * torch.conj(_b)), _kernel)

        return {'Σ(|A||B|)': _sum_ab, 'Σ(|A|^2)': _sum_a2, 'Σ(|B|^2)': _sum_b2, '|Σ(AB*)|': _sum_sa, 'Σ(|AB*|)': _sum_as}

from abc import ABC, ABCMeta
from abc import abstractmethod
from typing import Tuple, Literal, Sequence

import math
import numpy
import torch
import torchvision.transforms.v2.functional
from torch import Tensor


class _LevelBasicSeparateKernelOperator(ABC, metaclass=ABCMeta):
    @classmethod
    def symmetric_pad(cls, tensor: Tensor, pad: Tuple[int, int], axis: int):
        dim_ln = tensor.shape[axis]
        tensor = torchvision.transforms.v2.functional.pad(tensor.movedim(axis, -1), [max(pad), 0], padding_mode='symmetric').movedim(-1, axis)
        tensor = torch.split(tensor, [max(pad) - pad[0], pad[0] + dim_ln + pad[1], max(pad) - pad[1]], dim=axis)[1]
        return tensor

    @classmethod
    def extension_pad(cls, tensor: Tensor, pad: Tuple[int, int], axis: int, mode: Literal['constant', 'reflect', 'symmetric', 'replicate', 'circular'] = 'symmetric', cval: float = 0.0):
        if mode == 'symmetric':
            return cls.symmetric_pad(tensor, pad, axis)
        else:
            return torch.nn.functional.pad(tensor.movedim(axis, -1), pad, mode=mode, value=cval).movedim(axis, -1)

    @classmethod
    def weight_prepare(cls, weight: Tensor, device: torch.device):
        weight = weight.flatten()
        weight = torch.flip(weight, dims=[-1])
        weight = weight.to(device=device)
        return weight.view(+1, +1, -1)

    @classmethod
    def tensor_prev_op(cls, tensor: Tensor, axis: int):
        shape = tensor.shape
        tensor = tensor.movedim(axis, -1).unsqueeze(-2)
        tensor = tensor.flatten(0, -3) if tensor.ndim > 3 else tensor[(None,) * (3 - tensor.ndim) + (slice(None),)] if tensor.ndim < 3 else tensor
        return tensor, shape

    @classmethod
    def tensor_post_op(cls, tensor: Tensor, axis: int, shape: Sequence[int]) -> Tensor:
        tensor = tensor.squeeze()
        if len(shape) > 3:
            tensor = tensor.unflatten(0, shape[:-2])
        return tensor.movedim(-1, axis)

    @abstractmethod
    def forward(self, tensor: Tensor, axis: int) -> Tensor:
        return torch.tensor(tensor) if not isinstance(tensor, Tensor) else tensor

    @abstractmethod
    def reverse(self, tensor: Tensor, axis: int) -> Tensor:
        return torch.tensor(tensor) if not isinstance(tensor, Tensor) else tensor


class LevelAlpha1SeparateKernelOperator(_LevelBasicSeparateKernelOperator):
    @classmethod
    def convolve1d(cls, tensor: Tensor, weight: Tensor, axis: int, mode: Literal['constant', 'reflect', 'symmetric', 'replicate', 'circular'] = 'symmetric', cval: float = 0.0):
        weight = cls.weight_prepare(weight, device=tensor.device)
        tensor, shape = cls.tensor_prev_op(tensor, axis=axis)
        pad = (weight.numel() - 1) / 2
        pad = (math.floor(pad), math.ceil(pad))
        tensor = cls.extension_pad(tensor, pad, axis=-1, mode=mode, cval=cval)
        result = torch.nn.functional.conv1d(tensor, weight)
        return cls.tensor_post_op(result, axis=axis, shape=shape)

    def __init__(self, kernel: numpy.ndarray):
        self.kernel = torch.tensor(kernel.flatten())

    def forward(self, tensor: Tensor, axis: int) -> Tensor:
        tensor = super().forward(tensor, axis=axis)
        return self.convolve1d(tensor, self.kernel.to(device=tensor.device), axis=axis, mode='symmetric')

    def reverse(self, tensor: Tensor, axis: int) -> Tensor:
        tensor = super().reverse(tensor, axis=axis)
        return self.convolve1d(tensor, self.kernel.to(device=tensor.device), axis=axis, mode='symmetric')


class LevelOthersSeparateKernelOperator(_LevelBasicSeparateKernelOperator):
    @classmethod
    def dilation1d(cls, tensor: Tensor, weight: Tensor, stride: int, bias: int, axis: int, mode: Literal['constant', 'reflect', 'symmetric', 'replicate', 'circular'] = 'symmetric', cval: float = 0.0):
        weight = cls.weight_prepare(weight, device=tensor.device)
        tensor, shape = cls.tensor_prev_op(tensor, axis=axis)
        pad = weight.numel()
        pad = (pad - stride // 2 - 0 - bias, pad - stride // 2 - 1 + bias)
        tensor = cls.extension_pad(tensor, pad, axis=-1, mode=mode, cval=cval)
        result = torch.nn.functional.conv1d(tensor, weight, stride=stride, dilation=2)
        return cls.tensor_post_op(result, axis=axis, shape=shape)

    def __init__(self, coef_a: numpy.ndarray, coef_b: numpy.ndarray):
        assert len(coef_a) == len(coef_b)
        assert len(coef_a) % 0x02 == 0x00
        self.coef_a, self.coef_b = torch.tensor(coef_a.copy()), torch.tensor(coef_b.copy())
        self.calc_c = torch.sum(self.coef_a * self.coef_b)

    def forward(self, tensor: Tensor, axis: int) -> Tensor:
        tensor = super().forward(tensor, axis=axis)
        assert tensor.shape[axis] % 4 == 0, f'Rows on Axis {axis} should be completely divisible by 4'
        conv_a = self.dilation1d(tensor, self.coef_a, stride=4, bias=0x00, axis=axis, mode='symmetric')
        conv_b = self.dilation1d(tensor, self.coef_b, stride=4, bias=0x01, axis=axis, mode='symmetric')
        concat = torch.concatenate([conv_a, conv_b] if self.calc_c > 0 else [conv_b, conv_a], dim=axis)
        _l = conv_a.shape[axis]
        return torch.index_select(concat, dim=axis, index=torch.tensor([_l * i + j for j in range(_l) for i in range(2)], dtype=torch.long, device=tensor.device))

    def reverse(self, tensor: Tensor, axis: int) -> Tensor:
        tensor = super().reverse(tensor, axis=axis)
        assert tensor.shape[axis] % 2 == 0, f'Rows on Axis {axis} should be completely divisible by 2'
        f_bank = [(self.coef_a[0::2], 0 if self.calc_c > 0 else 1), (self.coef_b[0::2], 1 if self.calc_c > 0 else 0),
                  (self.coef_a[1::2], 0 if self.calc_c > 0 else 1), (self.coef_b[1::2], 1 if self.calc_c > 0 else 0)]
        concat = torch.concatenate([self.dilation1d(tensor, k, stride=2, bias=b, axis=axis, mode='symmetric') for k, b in f_bank], dim=axis)
        _l = tensor.shape[axis]
        return torch.index_select(concat, dim=axis, index=torch.tensor([_l // 2 * i + j for j in range(_l // 2) for i in range(4)], device=concat.device))

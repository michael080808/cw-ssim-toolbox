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
import itertools
import operator
from abc import ABC, ABCMeta
from typing import Tuple, Union

from torch import Tensor

from .kernel import LevelAlpha1KernelOperator
from .kernel import LevelOthersKernelOperator
from ..wavelets import BiorthogonalWavelet, OrthogonalWavelet


class _LevelBasicSeriesOperator(ABC, metaclass=ABCMeta):
    lo_pass_channelizer: Union[LevelAlpha1KernelOperator, LevelOthersKernelOperator]
    hi_pass_channelizer: Union[LevelAlpha1KernelOperator, LevelOthersKernelOperator]
    lo_pass_synthesiser: Union[LevelAlpha1KernelOperator, LevelOthersKernelOperator]
    hi_pass_synthesiser: Union[LevelAlpha1KernelOperator, LevelOthersKernelOperator]

    def __init__(self, n: int):
        assert n > 0
        self.dimension = n

    def forward(self, tensor: Tensor) -> Tuple[Tensor, ...]:
        assert self.dimension <= tensor.ndim, \
            f'The input dimension (which is {tensor.ndim}) is less than the settings of dimension (which is {self.dimension}).'
        assert functools.reduce(operator.and_, [shape % 4 == 0 for shape in tensor.shape[-self.dimension:]]), \
            f'Not all the size of last {self.dimension} dimensions are completely divisible by 4.'

        buffer = [tensor]
        for dim in range(-self.dimension, -0, +1):
            buffer = [kernel.forward(record, axis=dim) for kernel, record in itertools.product([self.lo_pass_channelizer, self.hi_pass_channelizer], buffer)]
        return tuple(buffer)

    def reverse(self, arrays: Tuple[Tensor, ...]) -> Tensor:
        assert len(arrays) == 2 ** self.dimension, \
            f'The length of input Tensor list is not equal to the settings with {2 ** self.dimension}.'
        assert functools.reduce(operator.and_, [tensor.ndim >= self.dimension for tensor in arrays]), \
            f'Some dimensions of input Tensor from "arrays" are less than the settings dimension {self.dimension}'

        buffer = arrays
        for dim in range(-self.dimension, -0, +1):
            buffer = [functools.reduce(operator.add, batch) for batch in itertools.batched([kernel.reverse(record, axis=dim) for kernel, record in zip(itertools.chain.from_iterable(itertools.repeat([self.lo_pass_synthesiser, self.hi_pass_synthesiser])), buffer)], 2)]
        return buffer[0]


class LevelAlpha1SeriesOperator(_LevelBasicSeriesOperator):
    def __init__(self, n: int, inst: BiorthogonalWavelet):
        super().__init__(n)
        lo_pass_channelizer, hi_pass_channelizer, lo_pass_synthesiser, hi_pass_synthesiser = inst.wavelets
        self.lo_pass_channelizer = LevelAlpha1KernelOperator(lo_pass_channelizer)
        self.hi_pass_channelizer = LevelAlpha1KernelOperator(hi_pass_channelizer)
        self.lo_pass_synthesiser = LevelAlpha1KernelOperator(lo_pass_synthesiser)
        self.hi_pass_synthesiser = LevelAlpha1KernelOperator(hi_pass_synthesiser)


class LevelOthersSeriesOperator(_LevelBasicSeriesOperator):
    def __init__(self, n: int, inst: OrthogonalWavelet):
        super().__init__(n)
        lo_pass_channelizer_a, lo_pass_channelizer_b, hi_pass_channelizer_a, hi_pass_channelizer_b, lo_pass_synthesiser_a, lo_pass_synthesiser_b, hi_pass_synthesiser_a, hi_pass_synthesiser_b = inst.wavelets
        self.lo_pass_channelizer = LevelOthersKernelOperator(lo_pass_channelizer_b, lo_pass_channelizer_a)
        self.hi_pass_channelizer = LevelOthersKernelOperator(hi_pass_channelizer_b, hi_pass_channelizer_a)
        self.lo_pass_synthesiser = LevelOthersKernelOperator(lo_pass_synthesiser_b, lo_pass_synthesiser_a)
        self.hi_pass_synthesiser = LevelOthersKernelOperator(hi_pass_synthesiser_b, hi_pass_synthesiser_a)

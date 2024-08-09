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

from functools import cached_property

import numpy

from ._metaclass import BiorthogonalWaveletMetaClass, OrthogonalWaveletMetaClass


class BiorthogonalWavelet(object, metaclass=BiorthogonalWaveletMetaClass):
    """
    @DynamicAttrs
    """

    def __init__(self, dtype: numpy.floating = numpy.float64):
        super().__init__()
        self.dtype = dtype

    def normalize(self, x):
        x = x.astype(self.dtype)
        return x / numpy.sum(x)

    @cached_property
    def wavelets(self):
        return self.lo_pass_channelizer, self.hi_pass_channelizer, self.lo_pass_synthesiser, self.hi_pass_synthesiser


class OrthogonalWavelet(BiorthogonalWavelet, metaclass=OrthogonalWaveletMetaClass):
    @cached_property
    def raw_wavelets(self):
        return super().wavelets

    @cached_property
    def wavelets(self):
        scale = 1 / numpy.sqrt(numpy.sum(self.lo_pass_channelizer ** 2))
        lo_pass_channelizer_a = scale * self.lo_pass_channelizer
        hi_pass_channelizer_a = scale * self.hi_pass_channelizer
        lo_pass_synthesiser_a = scale * self.lo_pass_synthesiser
        hi_pass_synthesiser_a = scale * self.hi_pass_synthesiser
        lo_pass_channelizer_b = numpy.flip(lo_pass_channelizer_a)
        hi_pass_channelizer_b = numpy.flip(hi_pass_channelizer_a)
        lo_pass_synthesiser_b = numpy.flip(lo_pass_synthesiser_a)
        hi_pass_synthesiser_b = numpy.flip(hi_pass_synthesiser_a)
        return lo_pass_channelizer_a, lo_pass_channelizer_b, hi_pass_channelizer_a, hi_pass_channelizer_b, lo_pass_synthesiser_a, lo_pass_synthesiser_b, hi_pass_synthesiser_a, hi_pass_synthesiser_b

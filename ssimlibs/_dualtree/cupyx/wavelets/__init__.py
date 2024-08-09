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

from ._baseclass import BiorthogonalWavelet
from ._baseclass import OrthogonalWavelet
from .biorthogonal import AntoniniWavelet
from .biorthogonal import BiorthogonalT06T10Wavelet
from .biorthogonal import BiorthogonalT06T10ComplexWavelet
from .biorthogonal import DaubechiesT04T04Wavelet
from .biorthogonal import DaubechiesT06T06Wavelet
from .biorthogonal import DaubechiesT08T08Wavelet
from .biorthogonal import HaarWavelet
from .biorthogonal import LeGallWavelet
from .biorthogonal import NearSymmetricT05T07Wavelet
from .biorthogonal import NearSymmetricT13T19Wavelet
from .orthogonal import QShiftT06Wavelet
from .orthogonal import QShiftT10Wavelet
from .orthogonal import QShiftT12Wavelet
from .orthogonal import QShiftT14Wavelet
from .orthogonal import QShiftT16Wavelet
from .orthogonal import QShiftT18Wavelet
from .orthogonal import QShiftT32Wavelet

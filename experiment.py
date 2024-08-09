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

import cupy
import skimage.io
import torch

import ssimlibs.cupyx
import ssimlibs.numpy
import ssimlibs.torch

cupyx_metrics = [
    ssimlibs.cupyx.gUIQI(),
    ssimlibs.cupyx.gSSIM(),
    ssimlibs.cupyx.mUIQI(),
    ssimlibs.cupyx.mSSIM(),
    ssimlibs.cupyx.MsSSIM(),
    ssimlibs.cupyx.CwSSIM(
        backend='DualTree',
        levels=4,
        level_alpha1=ssimlibs.cupyx.NearSymmetricT13T19Wavelet(),
        level_others=ssimlibs.cupyx.QShiftT14Wavelet()),
    ssimlibs.cupyx.CwSSIM(backend='Pyramids', levels=5, orientations=6, is_complex=True),
]

numpy_metrics = [
    ssimlibs.numpy.gUIQI(),
    ssimlibs.numpy.gSSIM(),
    ssimlibs.numpy.mUIQI(),
    ssimlibs.numpy.mSSIM(),
    ssimlibs.numpy.MsSSIM(),
    ssimlibs.numpy.CwSSIM(
        backend='DualTree',
        levels=4,
        level_alpha1=ssimlibs.numpy.NearSymmetricT13T19Wavelet(),
        level_others=ssimlibs.numpy.QShiftT14Wavelet()),
    ssimlibs.numpy.CwSSIM(backend='Pyramids', levels=5, orientations=6, is_complex=True),
]

torch_metrics = [
    ssimlibs.torch.gUIQI(),
    ssimlibs.torch.gSSIM(),
    ssimlibs.torch.mUIQI(),
    ssimlibs.torch.mSSIM(),
    ssimlibs.torch.MsSSIM(),
    ssimlibs.torch.CwSSIM(
        backend='DualTree',
        levels=4,
        level_alpha1=ssimlibs.torch.NearSymmetricT13T19Wavelet(),
        level_others=ssimlibs.torch.QShiftT14Wavelet()),
    ssimlibs.torch.CwSSIM(backend='Pyramids', levels=5, orientations=6, is_complex=True),
]

if __name__ == '__main__':
    a = skimage.util.img_as_float64(skimage.io.imread('pictures/elephant.jpg'))
    b = skimage.util.noise.random_noise(a, mode='speckle')
    for module_a, module_b, module_c in zip(cupyx_metrics, numpy_metrics, torch_metrics):
        result_a = module_a(cupy.asarray(a), cupy.asarray(b))
        with torch.device(0):
            result_c = module_c(torch.tensor(a), torch.tensor(b))
            print(result_a, float(result_c))

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


class CplxComponents:
    def __init__(self, k: float):
        self._k = k

    def amp(self, _sum_ab, _sum_a2, _sum_b2):
        p_numerator = 2 * _sum_ab + self._k
        denominator = _sum_a2 + _sum_b2 + self._k
        return p_numerator / denominator

    def phi(self, _sum_sa, _sum_as):
        p_numerator = 2 * _sum_sa + self._k
        denominator = 2 * _sum_as + self._k
        return p_numerator / denominator


class RealComponents:
    def __init__(self, c1: float, c2: float, c3: float):
        self._c1, self._c2, self._c3 = c1, c2, c3

    def luminance(self, _avg_a_, _avg_b_):
        p_numerator = 2 * _avg_a_ * _avg_b_ + self._c1
        denominator = _avg_a_ ** 2 + _avg_b_ ** 2 + self._c1
        return p_numerator / denominator

    def contrasts(self, _var_a2, _var_b2):
        p_numerator = 2 * cupy.sqrt(_var_a2) * cupy.sqrt(_var_b2) + self._c2
        denominator = _var_a2 + _var_b2 + self._c2
        return p_numerator / denominator

    def structure(self, _cov_ab, _var_a2, _var_b2):
        p_numerator = _cov_ab + self._c3
        denominator = cupy.sqrt(_var_a2) * cupy.sqrt(_var_b2) + self._c3
        return p_numerator / denominator

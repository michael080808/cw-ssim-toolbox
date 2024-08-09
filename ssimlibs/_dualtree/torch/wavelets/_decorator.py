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

import inspect
import os
import re
from functools import wraps

import numpy


def __assert__(attr):
    params = inspect.signature(attr).parameters
    if not callable(attr):
        raise TypeError(f'Attribute \'{attr.__name__}\' is not callable' % attr)
    if not len(params) == 1 or 'self' not in params:
        raise TypeError(f'Attribute \'{attr.__name__}\' should be a Class member function with \'self\' as parameter and without other parameters' % attr)


def temporary(prefix: str = '__', suffix: str = '__'):
    assert isinstance(prefix, str)
    assert isinstance(suffix, str)
    assert len(prefix + suffix) > 0

    def decorator(attr):
        __assert__(attr)

        @wraps(attr)
        def wrapper(self):
            if not hasattr(self, f'__{attr.__name__}__'):
                result = attr(self)
            elif numpy.issubdtype((values := getattr(self, f'__{attr.__name__}__')).dtype, (d_type := getattr(self, f'dtype'))):
                result = values
            elif numpy.finfo(values.dtype).bits > numpy.finfo(d_type).bits:
                result = values.astype(d_type)
            else:
                result = attr(self)

            if not hasattr(self, f'__{attr.__name__}__'):
                setattr(self, f'__{attr.__name__}__', result)
            elif numpy.finfo(result.dtype).bits > numpy.finfo(getattr(self, f'__{attr.__name__}__').dtype).bits:
                setattr(self, f'__{attr.__name__}__', result)

            return result

        return wrapper

    return decorator


def persisted(rdir: str = os.path.join(os.path.dirname(__file__), 'buffer')):
    if not os.path.exists(rdir):
        os.makedirs(rdir)
    assert os.path.isdir(rdir)

    def decorator(attr):
        __assert__(attr)

        @wraps(attr)
        def wrapper(self):
            if not os.path.exists(path := os.path.join(rdir, self.__class__.__name__, f'{attr.__name__}')):
                os.makedirs(path)
            d_name = dtype.name if isinstance(dtype := getattr(self, 'dtype'), numpy.dtype) else dtype.__name__
            if os.path.exists(f_name := os.path.join(path, f'{d_name}.{numpy.finfo(dtype).bits:03d}.npy')):
                result = numpy.load(f_name)
            else:
                others = sorted([
                    {'path': os.path.join(root, name), 'bits': int(part.group(1))}
                    for root, dirs, docs in os.walk(path)
                    for name in docs
                    if (part := re.match(r'[a-zA-Z_][a-zA-Z0-9_]*\.([0-9]{3})\.npy', name)) is not None
                ], key=lambda x: x['bits'], reverse=True)
                result = numpy.load(others[0]['path']) if len(others) > 0 and others[0]['bits'] > numpy.finfo(getattr(self, 'dtype')).bits else attr(self)

            if not os.path.exists(f_name):
                numpy.save(f_name, result)

            return result

        return wrapper

    return decorator


def validator(dtype: numpy.floating = numpy.float64):
    if not numpy.issubdtype(dtype, numpy.floating):
        raise TypeError(f'"dtype" should be a floating point data type')
    dtype = dtype

    def decorator(attr):
        __assert__(attr)

        @wraps(attr)
        def wrapper(self):
            if not hasattr(self, 'dtype'):
                raise AttributeError(f'\'{self.__class__.__name__}\' object has no attribute \'dtype\'', obj=self, name='dtype')

            result = attr(self).astype(dtype)

            if not isinstance(result, numpy.ndarray):
                raise TypeError(f'The result of attribute \'{attr.__name__}\' should be \'{numpy.ndarray}\' but got \'{type(result)}\'')
            if not numpy.issubdtype(result.dtype, numpy.floating):
                raise TypeError(f'The result of attribute \'{attr.__name__}\' should be floating-point data type but got \'{result.dtype}\'')
            if not result.ndim == 1:
                raise ValueError(f'The result of attribute \'{attr.__name__}\' should be a 1-dimensional numpy.ndarray')
            if not result.size >= 2:
                raise ValueError(f'The result of attribute \'{attr.__name__}\' should contain 2 or more elements in numpy.ndarray')

            return result

        return wrapper

    return decorator

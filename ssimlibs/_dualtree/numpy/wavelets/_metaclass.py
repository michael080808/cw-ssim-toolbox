import functools
import itertools
import operator
from functools import cached_property, wraps
from typing import Dict, Any, List, Callable

import numpy


class BiorthogonalWaveletMetaClass(type):
    # Lo Pass/Scaling Alias
    _lo_ = ['lo', 'lo_pass', 'bass_pass', 'scaling']
    # Hi Pass/Wavelet Alias
    _hi_ = ['hi', 'hi_pass', 'high_pass', 'wavelet']
    # Channelizer Alias
    _chn = ['d', 'chn', 'channeliser', 'channelizer', 'analysis']
    # Synthesiser Alias
    _syn = ['r', 'syn', 'synthesiser', 'synthesizer', 'synthesis']
    # LoD
    _lo_chn = [prefix + '_' + suffix for prefix, suffix in itertools.product(_lo_, _chn)]
    # HiD
    _hi_chn = [prefix + '_' + suffix for prefix, suffix in itertools.product(_hi_, _chn)]
    # LoR
    _lo_syn = [prefix + '_' + suffix for prefix, suffix in itertools.product(_lo_, _syn)]
    # HiR
    _hi_syn = [prefix + '_' + suffix for prefix, suffix in itertools.product(_hi_, _syn)]
    # LoD & HiR
    _loDhiR = _lo_chn + _hi_syn
    # LoR & HiD
    _loRhiD = _lo_syn + _hi_chn
    # LoD & HiR & LoR & HiD
    _combos = _loDhiR + _loRhiD

    @staticmethod
    def _lo_d_and_hi_r_transform_(attr):
        # Convert Function
        @wraps(attr)
        def wrapper(self) -> numpy.ndarray:
            x = attr(self)
            return -x * numpy.cumprod(-numpy.ones_like(x))

        return wrapper

    @staticmethod
    def _lo_r_and_hi_d_transform_(attr):
        # Convert Function
        @wraps(attr)
        def wrapper(self) -> numpy.ndarray:
            x = attr(self)
            return +x * numpy.cumprod(-numpy.ones_like(x))

        return wrapper

    @classmethod
    def _select_itemizing_(cls, attrs: List[str]) -> str:
        itemizing = f''
        if len(attrs) > 0:
            itemizing += f'"{attrs[0]}"'
        if len(attrs) > 2:
            itemizing += functools.reduce(operator.add, [f', "{item}"' for item in attrs[+1: -1]])
        if len(attrs) > 1:
            itemizing += f' and "{attrs[-1]}"'
        return itemizing if len(itemizing) > 0 else 'nothing'

    @classmethod
    def _notice_conflicts_(cls, attrs: Dict[str, Any], group: List[str]) -> List[str]:
        exist = list(attrs.keys() & set(group))
        if len(exist) > 1:
            # Get Conflict Items
            itemize = cls._select_itemizing_(exist)
            # Raise Conflicts Warning
            raise Warning(f'The attributes {itemize} may cause conflict definition. "{exist[0]}" as the first definition will be reserved.')
        return exist

    @classmethod
    def _remove_conflicts_(cls, attrs: Dict[str, Any]) -> Dict[str, Any]:
        for group in [cls._loDhiR, cls._loRhiD]:
            # Notice Conflicts
            exist = cls._notice_conflicts_(attrs, group)
            # Remove Conflicts
            if len(exist) > 0:
                attrs = {k: v for k, v in attrs.items() if k not in exist[1:]}
        return attrs

    @classmethod
    def _rename_attribute_(cls, attrs: Dict[str, Any]) -> Dict[str, Any]:
        for group in [cls._lo_chn, cls._hi_chn, cls._lo_syn, cls._hi_syn]:
            # Exclude Attributes in Group
            diffs = {k: v for k, v in attrs.items() if k not in group}
            # From back to front update all attributes name with simplified one
            if len(group) > 0:
                attrs = diffs | {group[0]: v for k, v in reversed(attrs.items()) if k in group}
        return attrs

    @classmethod
    def _mutate_attribute_(cls, exist: Any, trans: Callable[[Any], Callable]) -> Any:
        function = exist
        if isinstance(exist, property):
            function = exist.fget
        elif isinstance(exist, cached_property):
            function = exist.func
        function = trans(function) if callable(function) else trans(lambda self: function)()
        return (type(exist))(function) if isinstance(exist, (property, cached_property)) else function

    @classmethod
    def _append_relatives_(cls, attrs: Dict[str, Any], trans: Callable[[Any], Callable], team0: List[str], team1: List[str]) -> Dict[str, Any]:
        e_1st = team0[0] in attrs and team1[0] not in attrs
        e_2nd = team1[0] in attrs and team0[0] not in attrs
        exist = team0[0] if e_1st else team1[0] if e_2nd else None
        empty = team1[0] if e_1st else team0[0] if e_2nd else None
        if exist is not None and empty is not None:
            attrs = attrs | {f'{empty}': cls._mutate_attribute_(attrs[f'{exist}'], trans)}
        return attrs

    @classmethod
    def _insert_relatives_(cls, attrs: Dict[str, Any]) -> Dict[str, Any]:
        combo = [
            (cls._lo_chn, cls._hi_syn, cls._lo_d_and_hi_r_transform_),
            (cls._lo_syn, cls._hi_chn, cls._lo_r_and_hi_d_transform_),
        ]
        for team0, team1, trans in combo:
            attrs = cls._append_relatives_(attrs, trans, team0, team1)
        return attrs

    @classmethod
    def _expand_attribute_(cls, attr, self, name):
        for group in [cls._lo_chn, cls._hi_chn, cls._lo_syn, cls._hi_syn]:
            if name in group:
                if (inst := self.__getattribute__(f'{group[0]}')) is not None:
                    return inst
                else:
                    raise AttributeError(f'One of attributes {cls._select_itemizing_(group)} should be defined')
        return attr['__getattr_user_defined__'](self, name) if '__getattr_user_defined__' in attr else self.__getattribute__(name)

    def __new__(cls, title: str, bases: tuple[type, ...], attrs: Dict[str, Any], **kwds):
        attrs = cls._remove_conflicts_(attrs)
        attrs = cls._rename_attribute_(attrs)
        attrs = cls._insert_relatives_(attrs)
        if '__getattr__' in attrs:
            attrs['__getattr_user_defined__'] = attrs.pop('__getattr__')
        attrs['__getattr__'] = lambda _self, _attr: cls._expand_attribute_(attrs, _self, _attr)
        return super().__new__(cls, title, bases, attrs, **kwds)


class OrthogonalWaveletMetaClass(BiorthogonalWaveletMetaClass):
    @staticmethod
    def _lo_d_and_hi_r_transform_(attr):
        # Convert Function
        @wraps(attr)
        def wrapper(self) -> numpy.ndarray:
            x = attr(self)
            return +x * numpy.cumprod(-numpy.ones_like(x))

        return wrapper

    @staticmethod
    def _lo_r_and_hi_d_transform_(attr):
        # Convert Function
        @wraps(attr)
        def wrapper(self) -> numpy.ndarray:
            x = attr(self)
            return -x * numpy.cumprod(-numpy.ones_like(x))

        return wrapper

    @staticmethod
    def _lo_d_and_lo_r_transform_(attr):
        # Convert Function
        @wraps(attr)
        def wrapper(self) -> numpy.ndarray:
            x = attr(self)
            return numpy.flip(x)

        return wrapper

    @staticmethod
    def _hi_d_and_hi_r_transform_(attr):
        # Convert Function
        @wraps(attr)
        def wrapper(self) -> numpy.ndarray:
            x = attr(self)
            return numpy.flip(x)

        return wrapper

    @classmethod
    def _remove_conflicts_(cls, attrs: Dict[str, Any]) -> Dict[str, Any]:
        for group in [cls._combos]:
            # Notice Conflicts
            exist = cls._notice_conflicts_(attrs, group)
            # Remove Conflicts
            if len(exist) > 0:
                attrs = {k: v for k, v in attrs.items() if k not in exist[1:]}
        return attrs

    @classmethod
    def _insert_relatives_(cls, attrs: Dict[str, Any]) -> Dict[str, Any]:
        combo = [
            (cls._lo_chn, cls._lo_syn, cls._lo_d_and_lo_r_transform_),
            (cls._hi_chn, cls._hi_syn, cls._hi_d_and_hi_r_transform_),
        ]
        for team0, team1, trans in combo:
            attrs = cls._append_relatives_(attrs, trans, team0, team1)
        return super()._insert_relatives_(attrs)

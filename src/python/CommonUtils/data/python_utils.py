from enum import Enum
from typing import Hashable
from functools import reduce
from pyspark.sql import functions as fn


class DictClass(Enum):

    def colname(self):
        return self.value

    @classmethod
    def values(cls):
        """
        set of values
        :return: (set)
        """
        return [v.value for v in cls._member_map_.values()]

    @classmethod
    def keys(cls):
        """
        set of keys
        :return: (list)
        """
        return [k for k, v in cls._member_map_.items()]

    @classmethod
    def to_dict(cls):
        """
        Class as dict
        :return: (dict)
        """
        return {k: v for k, v in zip(cls.keys(), cls.values())}

    @classmethod
    def get(cls, item: Hashable):
        """
        Get item
        :return: value in class
        """
        return cls.to_dict().get(item)


class Formater(DictClass):
    """
    Base class from Format classes
    """

    @classmethod
    def recode(cls, colname, other=None, null=None):
        d_recode = cls.to_dict()
        d_recode = [(k.lstrip('_').replace('x', ''), v) for k, v in d_recode.items()]

        d_recode = reduce(
            lambda x, y, cname=colname: x.when(fn.col(cname) == y[0], y[1]) if isinstance(x, Column) else fn.when(
                fn.col(cname) == x[0], x[1]).when(fn.col(cname) == y[0], y[1]), d_recode)

        if null is not None:
            d_recode = d_recode.when(fn.col(colname).isNull(), null)

        if other is not None:
            d_recode = d_recode.otherwise(other)

        return d_recode

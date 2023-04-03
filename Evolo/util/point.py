# -*- coding: utf-8 -*-<模块功能已了解>
from abc import ABC, abstractmethod

"""
module:: point
synopsis: implementation of points of n-dimensions (e.g, ideal point, nadir point, etc.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Point(ABC):
    @abstractmethod
    def update(self, objective_vector: []) -> None:
        pass


class IdealPoint(Point):
    def __init__(self, dimension: int):
        self.point = dimension * [float("inf")]

    def update(self, objective_vector: []) -> None:
        """
        zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        """
        self.point = [y if x > y else x for x, y in zip(self.point, objective_vector)]

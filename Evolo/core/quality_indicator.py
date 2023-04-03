# -*- coding: utf-8 -*-<模块功能已了解>
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial


class QualityIndicator(ABC):
    def __init__(self, is_minimization: bool):
        self.is_minimization = is_minimization

    @abstractmethod
    def compute(self, solutions: np.array):
        """
        param solutions: [m, n] bi-dimensional numpy array, being m the number of solutions and n the dimension of
        each solution
        return: the value of the quality indicator
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_short_name(self) -> str:
        pass


class FitnessValue(QualityIndicator):
    def __init__(self, is_minimization: bool = True):
        super(FitnessValue, self).__init__(is_minimization=is_minimization)

    def compute(self, solutions: np.array):
        if self.is_minimization:
            mean = np.mean([s.objectives for s in solutions])
        else:
            mean = -np.mean([s.objectives for s in solutions])
        return mean

    def get_short_name(self) -> str:
        return "FIT"

    def get_name(self) -> str:
        return "Fitness"



import copy
from typing import List, TypeVar
from Evolo.core.operator import Selection
from Evolo.core.solution import Solution
from Evolo.util.comparator import Comparator, DominanceComparator

S = TypeVar("S", bound=Solution)
"""
module:: Sort population
synopsis: Module implementing selection operators.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class SortPopulation(Selection[List[S], S]):
    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(SortPopulation, self).__init__()
        self.comparator = comparator

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        sorted_population = copy.deepcopy(front)
        for i in range(0, len(front) - 1):
            for j in range(i + 1, len(front)):
                if self.comparator.compare(sorted_population[i], sorted_population[j]) == 1:
                    solution = sorted_population[i]
                    sorted_population[i] = sorted_population[j]
                    sorted_population[j] = solution
        return sorted_population

    def get_name(self) -> str:
        return "Sort population"

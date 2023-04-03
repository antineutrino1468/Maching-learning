# -*- coding: utf-8 -*-<模块功能已了解>
import copy
from enum import Enum
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar
from Evolo.core.problem import Problem
from Evolo.util.comparator import (Comparator, ObjectiveComparator, IdenticalSolutionsComparator, MultiComparator)

S = TypeVar("S")


class Restart(Generic[S], ABC):
    def __init__(self):
        pass

    @abstractmethod
    def execute(self, solution_list: List[S]) -> List[S]:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__


class SimpleReplaceDuplicatedSolution(Restart[S]):
    def __init__(self):
        super(SimpleReplaceDuplicatedSolution, self).__init__()
        self.identical_solutions_comparator = IdenticalSolutionsComparator()

    def execute(self, solution_list: List[S]) -> List[S]:
        result_list = copy.deepcopy(solution_list)
        for j in range(0, len(result_list) - 1):
            for k in range(j + 1, len(result_list)):
                if self.identical_solutions_comparator.compare(result_list[j], result_list[k]) == 0:
                    for i in range(0, result_list[k].number_of_objectives):
                        result_list[k].objectives[i] = 1.0e30
        return result_list

    def get_name(self) -> str:
        self.__class__.__name__


class ReplaceDuplicatedSolutionWithNewSolution(Restart[S]):
    def __init__(self, problem: Problem):
        super(ReplaceDuplicatedSolutionWithNewSolution, self).__init__()
        self.identical_solutions_comparator = IdenticalSolutionsComparator()
        self.problem = problem

    def execute(self, solution_list: List[S]) -> List[S]:
        result_list = copy.deepcopy(solution_list)
        for j in range(0, len(result_list) - 1):
            for k in range(j + 1, len(result_list)):
                if self.identical_solutions_comparator.compare(result_list[j], result_list[k]) == 0:
                    result_list[k] = self.problem.create_solution()
        return result_list

    def get_name(self) -> str:
        return self.__class__.__name__

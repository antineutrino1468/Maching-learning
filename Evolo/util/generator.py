# -*- coding: utf-8 -*-<模块功能已了解>
import copy
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar
from Evolo.core.problem import Problem
from Evolo.core.solution import Solution

R = TypeVar("R")
"""
module:: generator
synopsis: Population generators implementation.
moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Generator(Generic[R], ABC):
    @abstractmethod
    def create_solution(self, problem: Problem) -> R:
        pass

    def create_solutions(self, problem: Problem, pop_size: int) -> R:
        pass


class RandomGenerator(Generator):
    def create_solution(self, problem: Problem):
        return problem.create_solution()

    def create_solutions(self, problem: Problem, pop_size: int):
        return [problem.create_solution() for _ in range(0, pop_size)]


class InjectorGenerator(Generator):
    def __init__(self, solutions: List[Solution]):
        super(InjectorGenerator, self).__init__()
        self.population = copy.deepcopy(solutions)

    def create_solution(self, problem: Problem):
        if len(self.population) > 0:
            # If we have more solutions to inject, return one from the list
            return self.population.pop()
        else:
            # Otherwise generate a new solution
            solution = problem.create_solution()
        return solution

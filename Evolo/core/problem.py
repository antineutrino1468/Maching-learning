# -*- coding: utf-8 -*-<模块功能已了解>
import random
import copy
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar
from Evolo.core.observer import Observer
from Evolo.core.solution import FloatSolution
from Evolo.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")


class Problem(Generic[S], ABC):
    """
    Class representing problems.
    Define the mathematical form of optimization problem.
    """

    MINIMIZE = -1
    MAXIMIZE = 1

    def __init__(self):
        self.number_of_variables = 1
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE]
        self.objective_weights = None
        self.problem_name = None
        self.solution_type = ""

        self.reference_front: List[S] = []
        self.directions: List[int] = []
        self.labels: List[str] = []

    @abstractmethod
    def create_solution(self) -> S:
        """
        Creates a random_search solution to the problem.
        return: Solution.
        """
        pass

    @abstractmethod
    def evaluate_solution(self, solution: S) -> S:
        """
        Evaluate a solution. For any new problem inheriting from :class:`Problem`, this method should be replaced.
        Note that this framework ASSUMES minimization, thus solutions must be evaluated in consequence.
        return: Evaluated solution.
        """
        pass

    def get_name(self) -> str:
        return self.problem_name


class FloatProblem(Problem[FloatSolution], ABC):
    """Class representing float problems."""

    def __init__(self):
        super(FloatProblem, self).__init__()
        self.variable_lb = []
        self.variable_ub = []
        self.solution_type = "Float"

    def create_variables(self, variable_lb=None, variable_ub=None):
        if variable_lb is None or variable_ub is None:
            variables = [random.uniform(self.variable_lb[i] * 1.0, self.variable_ub[i] * 1.0)
                         for i in range(self.number_of_variables)]
        else:
            variables = [random.uniform(variable_lb[i] * 1.0, variable_ub[i] * 1.0)
                         for i in range(self.number_of_variables)]
        return variables

    def create_solution(self):
        new_solution = FloatSolution(
            self.variable_lb, self.variable_ub, self.number_of_variables, self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.variables = self.create_variables()
        return new_solution

    def remedy_solution(self, solution):
        variables = copy.deepcopy(solution.variables)
        for i in range(0, len(variables)):
            if variables[i] < self.variable_lb[i]:
                variables[i] = self.variable_lb[i]
            if variables[i] > self.variable_ub[i]:
                variables[i] = self.variable_ub[i]
        solution.variables = variables
        return solution




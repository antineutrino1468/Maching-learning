# -*- coding: utf-8 -*-<模块功能已了解>
from abc import ABC
from typing import Generic, List, TypeVar
from Evolo.util.checking import Check

BitSet = List[bool]
S = TypeVar("S")


class Solution(Generic[S], ABC):
    """Class representing solutions"""

    def __init__(self, number_of_variables: int = 1, number_of_objectives: int = 1, number_of_constraints: int = 0):
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints
        self.variables = [[] for _ in range(self.number_of_variables)]
        self.objectives = [0.0 for _ in range(self.number_of_objectives)]
        self.fitness = None
        self.constraints = [0.0 for _ in range(self.number_of_constraints)]
        self.attributes = {}
        self.solution_type = ""
        self.survive_time = 0

    def __eq__(self, solution) -> bool:
        if isinstance(solution, self.__class__):
            return self.variables == solution.variables
        return False

    def __str__(self) -> str:
        return "Solution(variables={},objectives={},constraints={})".format(
            self.variables, self.objectives, self.constraints
        )


class FloatSolution(Solution[float]):
    """Class representing float solutions"""

    def __init__(self,
                 variable_lb: List[float] = [0.0],
                 variable_ub: List[float] = [1.0],
                 number_of_variables: int = 1,
                 number_of_objectives: int = 1,
                 number_of_constraints: int = 0,
                 ):
        super(FloatSolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)
        self.variable_lb = variable_lb
        self.variable_ub = variable_ub
        self.solution_type = "Float"

    def __copy__(self):
        new_solution = FloatSolution(
            self.variable_lb,
            self.variable_ub,
            self.number_of_variables,
            self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.objectives = self.objectives[:]
        new_solution.fitness = self.fitness
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]
        new_solution.attributes = self.attributes.copy()
        return new_solution



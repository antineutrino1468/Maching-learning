# -*- coding: utf-8 -*-
import copy
import numpy as np
from typing import List, TypeVar
from Evolo.config import store
from Evolo.core.algorithm import SwarmRoot
from Evolo.core.problem import Problem
from Evolo.util.evaluator import Evaluator
from Evolo.util.generator import Generator
from Evolo.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")

"""
Module: Basic teaching–learning-based optimization
Creator: 
Zixiang Li, Wuhan University of Science and Technology, https://www.researchgate.net/profile/Zixiang-Li-2, zixiangliwust@gmail.com;
Please contact me (zixiangliwust@gmail.com) freely if you find some mistakes or verify that this module is correct;
Modified or confirmed by the researchers listed as follows[Hoping for 10 researchers to confirm the codes]:
[1] Zixiang Li, Wuhan University of Science and Technology, https://www.researchgate.net/profile/Zixiang-Li-2, zixiangliwust@gmail.com;
"""


class TLBOBase(SwarmRoot[S, R]):
    """
    Basic teaching–learning-based optimization
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/andaviaco/tblo
    [3] Rao, R. V., V. J. Savsani, and D. P. Vakharia. 2011. "Teaching–learning-based optimization: A novel method for constrained mechanical design optimization problems."
    Computer-Aided Design 43 (3):303-15. doi: https://doi.org/10.1016/j.cad.2010.12.015.
    """

    def __init__(self,
                 problem: Problem,
                 pop_size: int,
                 swarm_generator: Generator = store.default_generator,
                 swarm_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        """
        Input parameters:
        pop_size (int): number of population size, default = 100; [2, 10000]
        """
        super(TLBOBase, self).__init__(problem=problem, pop_size=pop_size)
        self.algorithm_name = "Teaching–learning-based optimization"
        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.offsprings_size = 2 * self.pop_size

    def create_solution(self):
        new_solution = self.swarm_generator.create_solution(self.problem)
        new_solution.variables = np.array(new_solution.variables)
        return new_solution

    def selection(self, population: List[S]) -> List[S]:
        return population

    def teacher_phase_reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        position_list = np.array([item.variables for item in self.solutions])
        mean_position = np.mean(position_list, axis=0)
        for j in range(0, self.pop_size):
            # Teaching Phrase
            TF = np.random.randint(1, 3)  # 1 or 2 (never 3)
            offsprings[j].variables = population[j].variables + \
                                      np.random.uniform(0, 1, self.problem.number_of_variables) * (
                                              self.g_best.variables - TF * mean_position)
        return offsprings

    def learner_phase_reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        for j in range(0, self.pop_size):
            idx = np.random.choice(list(set(range(0, self.pop_size)) - {j}))
            if self.comparator.compare(population[j], population[idx]) == 1:
                offsprings[j].variables += np.random.uniform(0, 1, self.problem.number_of_variables) * (
                        population[idx].variables - population[j].variables)
            else:
                offsprings[j].variables += np.random.uniform(0, 1, self.problem.number_of_variables) * (
                        population[j].variables - population[idx].variables)
        return offsprings

    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        return self.replacement_operator.replace(population, offsprings)

    def evolve(self):
        selected_solutions = self.selection(self.solutions)
        offsprings = self.teacher_phase_reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)
        selected_solutions = self.selection(self.solutions)
        offsprings = self.learner_phase_reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)




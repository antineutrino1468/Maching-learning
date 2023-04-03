# -*- coding: utf-8 -*-
import copy
import numpy as np
import math
from typing import List, TypeVar
from Evolo.config import store
from Evolo.core.algorithm import SwarmRoot
from Evolo.core.problem import Problem
from Evolo.core.solution import Solution
from Evolo.util.evaluator import Evaluator
from Evolo.util.generator import Generator
from Evolo.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")

"""
Module: Basic whale optimization algorithm
Creator: 
Zixiang Li, Wuhan University of Science and Technology, https://www.researchgate.net/profile/Zixiang-Li-2, zixiangliwust@gmail.com;
Please contact me (zixiangliwust@gmail.com) freely if you find some mistakes or verify that this module is correct;
Modified or confirmed by the researchers listed as follows[Hoping for 10 researchers to confirm the codes]:
[1] Zixiang Li, Wuhan University of Science and Technology, https://www.researchgate.net/profile/Zixiang-Li-2, zixiangliwust@gmail.com;
"""


class WOABase(SwarmRoot[S, R]):
    """
    Basic whale optimization algorithm
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/7ossam81/EvoloPy, Hossam Faris etc., hossam.faris@ju.edu.jo (H. Faris)
    [3] Mirjalili, Seyedali, and Andrew Lewis. 2016. "The Whale Optimization Algorithm."
    Advances in Engineering Software 95:51-67. doi: https://doi.org/10.1016/j.advengsoft.2016.01.008.
    """

    def __init__(self,
                 problem: Problem,
                 pop_size: int,
                 swarm_generator: Generator = store.default_generator,
                 swarm_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(WOABase, self).__init__(problem=problem, pop_size=pop_size)
        self.algorithm_name = "Whale optimization algorithm"
        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.max_iteration = 1000

    def create_solution(self) -> Solution:
        new_solution = self.swarm_generator.create_solution(self.problem)
        new_solution.variables = np.array(new_solution.variables)
        return new_solution

    def selection(self, population: List[S]) -> List[S]:
        return population

    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        alpha = 2 - self.iterations * (2 / self.max_iteration)
        # Update the Position of search agents
        for j in range(0, self.pop_size):
            r1 = np.random.random()  # r1 is a random number in [0,1]
            r2 = np.random.random()  # r2 is a random number in [0,1]
            a = 2 * alpha * r1 - alpha  # Eq. (2.3) in the paper
            c = 2 * r2  # Eq. (2.4) in the paper
            b = 1  # parameters in Eq. (2.5)
            if np.random.random() < 0.5:
                if np.abs(a) < 1:
                    d_leader = np.abs(c * self.g_best.variables - self.solutions[j].variables)
                    offsprings[j].variables = self.g_best.variables - a * d_leader
                elif np.abs(a) >= 1:
                    idx = np.random.randint(0, self.pop_size)
                    d_idx = np.abs(c * self.solutions[idx].variables - self.solutions[j].variables)
                    offsprings[j].variables = self.solutions[idx].variables - a * d_idx
            else:
                l = - 2 * np.random.random() + 1  # parameters in Eq. (2.5)
                d_leader2 = np.abs(self.g_best.variables - self.solutions[j].variables)  # Eq.(2.5)
                offsprings[j].variables = d_leader2 * math.exp(b * l) * math.cos(
                    l * 2 * math.pi) + self.g_best.variables
        return offsprings

    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        return self.replacement_operator.replace(population, offsprings)




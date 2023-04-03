# -*- coding: utf-8 -*-
import random
import copy
import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar
from Evolo.config import store
from Evolo.core.algorithm import SwarmRoot
from Evolo.core.solution import Solution
from Evolo.core.problem import FloatProblem
from Evolo.util.comparator import Comparator
from Evolo.util.evaluator import Evaluator
from Evolo.util.generator import Generator
from Evolo.core.operator import Crossover, Mutation, Selection
from Evolo.util.termination_criterion import TerminationCriterion
from Evolo.util.neighborhood import Neighborhood
from Evolo.util.distance import EuclideanDistance
from Evolo.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")

"""
Module: Particle swarm optimization
Creator: 
Zixiang Li, Wuhan University of Science and Technology, https://www.researchgate.net/profile/Zixiang-Li-2, zixiangliwust@gmail.com;
Please contact me (zixiangliwust@gmail.com) freely if you find some mistakes or verify that this module is correct;
Modified or confirmed by the researchers listed as follows[Hoping for 10 researchers to confirm the codes]:
[1] Zixiang Li, Wuhan University of Science and Technology, https://www.researchgate.net/profile/Zixiang-Li-2, zixiangliwust@gmail.com;
"""


class PSOBase(SwarmRoot[S, R]):
    """
    Particle swarm optimization
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/jMetal/jMetalPy, Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
    [3] https://github.com/7ossam81/EvoloPy, Hossam Faris etc., hossam.faris@ju.edu.jo (H. Faris)
    [4] Kennedy, J., and R. Eberhart. 1995. Particle swarm optimization. Paper presented at the Proceedings of ICNN'95 - International Conference on Neural Networks, 27 Nov.-1 Dec. 1995.
    """

    def __init__(self,
                 problem: FloatProblem,
                 pop_size: int,
                 swarm_generator: Generator = store.default_generator,
                 swarm_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(PSOBase, self).__init__(problem=problem, pop_size=pop_size)
        self.algorithm_name = "Particle swarm optimization"
        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.c1 = 1.0
        self.c2 = 2.0
        self.r1_min = 0.0
        self.r1_max = 1.0
        self.r2_min = 0.0
        self.r2_max = 1.0
        self.w_min = 0.1
        self.w_max = 0.5
        self.v_max = []
        self.v_min = []
        self.velocity = np.zeros((self.pop_size, self.problem.number_of_variables), dtype=float)
        self.local_best_solutions: List[S] = []
        self.max_iteration = 1000

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.pop_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

        self.local_best_solutions = copy.deepcopy(self.solutions)
        self.v_max = [0.5 * (self.problem.variable_ub[i] - self.problem.variable_lb[i]) for i in
                      range(self.problem.number_of_variables)]
        self.v_min = [-self.v_max[i] for i in range(0, len(self.v_max))]
        for j in range(self.pop_size):
            for i in range(self.problem.number_of_variables):
                self.velocity[j][i] = random.uniform(self.v_min[i], self.v_max[i])

    def update_position(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        wk = self.w_max - self.iterations * ((self.w_max - self.w_min) / self.max_iteration)
        for j in range(self.pop_size):
            r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
            r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
            for i in range(offsprings[j].number_of_variables):
                self.velocity[j][i] = (
                        wk * self.velocity[j][i]
                        + (self.c1 * r1 * (self.local_best_solutions[j].variables[i] - offsprings[j].variables[i]))
                        + (self.c2 * r2 * (self.g_best.variables[i] - offsprings[j].variables[i]))
                )
                if self.velocity[j][i] < self.v_min[i]:
                    self.velocity[j][i] = self.v_min[i]
                if self.velocity[j][i] > self.v_max[i]:
                    self.velocity[j][i] = self.v_max[i]
        for j in range(self.pop_size):
            for i in range(offsprings[j].number_of_variables):
                offsprings[j].variables[i] += self.velocity[j][i]
        return offsprings

    def update_local_best(self, population: List[S]) -> None:
        for j in range(self.pop_size):
            flag = self.comparator.compare(population[j], self.local_best_solutions[j])
            if flag != 1:
                self.local_best_solutions[j] = copy.deepcopy(population[j])

    def evolve(self) -> None:
        self.solutions = self.update_position(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_local_best(self.solutions)



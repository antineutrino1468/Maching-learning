# -*- coding: utf-8 -*-<模块功能已了解>
import copy
import threading
import time
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar
from Evolo.config import store
from Evolo.core.problem import Problem
from Evolo.operator.selection import BestSolutionSelection, WorstSolutionSelection
from Evolo.util.comparator import ObjectiveComparator, IdenticalSolutionsComparator
from Evolo.util.sort_population import SortPopulation
from Evolo.util.replacement import GreedyPopulationReplacement
from Evolo.util.restart import SimpleReplaceDuplicatedSolution
from Evolo.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
module:: algorithm
synopsis: Templates for algorithms.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Algorithm(Generic[S, R], threading.Thread, ABC):
    def __init__(self):
        threading.Thread.__init__(self)
        self.algorithm_name = ""
        self.solutions: List[S] = []
        self.evaluations = 0
        self.iterations = 0
        self.start_time = 0
        self.total_time = 0
        self.swarm_evaluator = store.default_evaluator
        self.observable = store.default_observable

    def create_solution(self) -> S:
        pass

    def evaluate_solution(self, solution) -> S:
        pass

    def create_population(self, pop_size=None) -> List[S]:
        """Creates the initial list of solutions of a metaheuristic."""
        pass

    def evaluate(self, solution_list: List[S]) -> List[S]:
        """Evaluates a solution list."""
        pass

    def initialization(self) -> None:
        self.start_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        self.solutions = self.create_population()
        self.solutions = self.evaluate(self.solutions)

    @abstractmethod
    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        """Initialize the algorithm."""
        pass

    @abstractmethod
    def stopping_condition_is_met(self) -> bool:
        """The stopping condition is met or not."""
        pass

    @abstractmethod
    def evolve(self) -> None:
        """Performs one iteration/evolve of the algorithm's loop."""
        pass

    @abstractmethod
    def update_progress(self) -> None:
        """Update the progress after each iteration."""
        pass

    @abstractmethod
    def get_observable_data(self) -> dict:
        """Get observable data, with the information that will be send to all observers each time."""
        pass

    def run(self) -> None:
        """Execute the algorithm."""
        self.initialization()
        self.after_initialization()
        self.init_progress()
        logger.debug("Running main loop until termination criteria is met")
        while not self.stopping_condition_is_met():
            self.evolve()
            self.after_evolve()
            self.update_progress()

    def after_initialization(self) -> None:
        pass

    def after_evolve(self) -> None:
        pass

    def get_result(self) -> R:
        pass

    def get_name(self) -> str:
        return self.algorithm_name


class SwarmRoot(Algorithm[S, R], ABC):
    def __init__(self, problem: Problem[S], pop_size: int = 100):
        super(SwarmRoot, self).__init__()
        self.problem = problem
        self.pop_size = pop_size
        self.offsprings_size = self.pop_size
        self.p_best, self.g_best = None, None

        self.comparator = store.default_comparator
        self.identical_solutions_comparator = IdenticalSolutionsComparator()
        self.swarm_generator = store.default_generator
        self.swarm_evaluator = store.default_evaluator
        self.termination_criterion = store.default_termination_criteria
        self.best_solution_selection = BestSolutionSelection(comparator=ObjectiveComparator(0))
        self.worst_solution_selection = WorstSolutionSelection(comparator=ObjectiveComparator(0))
        self.sort_population = SortPopulation(comparator=ObjectiveComparator(0))
        self.replacement_operator = GreedyPopulationReplacement(comparator=ObjectiveComparator(0))
        self.restart_operator = SimpleReplaceDuplicatedSolution()
        self.result_archive = None

    def create_solution(self) -> S:
        new_solution = self.swarm_generator.create_solution(self.problem)
        return new_solution

    def evaluate_solution(self, solution) -> S:
        solution = self.problem.evaluate_solution(solution)
        if self.result_archive is not None:
            self.result_archive.add(solution)
        return solution

    def create_population(self, pop_size=None) -> List[S]:
        if pop_size is None:
            return [self.create_solution() for _ in range(self.pop_size)]
        else:
            return [self.create_solution() for _ in range(pop_size)]

    def evaluate(self, solution_list: List[S]):
        solution_list = self.swarm_evaluator.evaluate(solution_list, self.problem)
        if self.result_archive is not None:
            for solution in solution_list:
                self.result_archive.add(solution)
        return solution_list

    def selection(self, population: List[S]) -> List[S]:
        """Select the best-fit individuals for reproduction (parents)."""
        pass

    def reproduction(self, population: List[S]) -> List[S]:
        """Breed new individuals through crossover and mutation operations to give birth to offspring."""
        pass

    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        """Replace least-fit population with new individuals."""
        pass

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def get_observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "TOTAL_TIME": time.time() - self.start_time,
        }

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.pop_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def evolve(self) -> None:
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)

    def update_progress(self) -> None:
        self.evaluations += self.offsprings_size
        self.iterations += 1
        self.total_time = time.time() - self.start_time
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        # print("Evaluations number = " + str(self.evaluations))

    def run(self) -> None:
        """Execute the algorithm."""
        self.initialization()
        self.after_initialization()
        self.init_progress()
        logger.debug("Running main loop until termination criteria is met")
        while not self.stopping_condition_is_met():
            self.evolve()
            self.after_evolve()
            self.update_progress()

    @property
    def label(self) -> str:
        return f"{self.get_name()}.{self.problem.get_name()}"

    def after_initialization(self) -> None:
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        self.g_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))

    def after_evolve(self) -> None:
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        if self.comparator.compare(self.p_best, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.p_best)
        # self.solutions = self.restart_operator.execute(self.solutions)

    def get_result(self) -> R:
        return [self.g_best]



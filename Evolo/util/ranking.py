# -*- coding: utf-8 -*-<模块功能已了解>
from abc import ABC, abstractmethod
from typing import List, TypeVar
from Evolo.util.comparator import (
    Comparator,
    OverallConstraintViolationComparator,
    DominanceComparator,
    SolutionAttributeComparator,
)

S = TypeVar("S")


class Ranking(List[S], ABC):
    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(Ranking, self).__init__()
        self.number_of_comparisons = 0
        self.ranked_sub_lists = []
        self.comparator = comparator

    @abstractmethod
    def compute_ranking(self, solutions: List[S], k: int = None):
        pass

    def get_nondominated(self):
        return self.ranked_sub_lists[0]

    def get_sub_front(self, rank: int):
        if rank >= len(self.ranked_sub_lists):
            raise Exception("Invalid rank: {0}. Max rank: {1}".format(rank, len(self.ranked_sub_lists) - 1))
        return self.ranked_sub_lists[rank]

    def get_number_of_sub_fronts(self):
        return len(self.ranked_sub_lists)

    @classmethod
    def get_comparator(cls) -> Comparator:
        pass


class FastNonDominatedRanking(Ranking[List[S]]):
    """Class implementing the non-dominated ranking of NSGA-II proposed by Deb et al., see [Deb2002]_"""

    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(FastNonDominatedRanking, self).__init__(comparator)
        self.constraint_comparator = OverallConstraintViolationComparator()

    def compute_ranking(self, solutions: List[S], k: int = None):
        """
        Compute ranking of solutions.
        param solutions: Solution list.
        param k: Number of individuals.
        """
        # number of solutions dominating solution ith
        dominating_ith = [0 for _ in range(len(solutions))]
        # list of solutions dominated by solution ith
        ith_dominated = [[] for _ in range(len(solutions))]
        # front[i] contains the list of solutions belonging to front i
        front = [[] for _ in range(len(solutions) + 1)]
        for p in range(len(solutions) - 1):
            for q in range(p + 1, len(solutions)):
                dominance_test_result = self.comparator.compare(solutions[p], solutions[q])
                self.number_of_comparisons += 1
                if dominance_test_result == -1:
                    ith_dominated[p].append(q)
                    dominating_ith[q] += 1
                elif dominance_test_result == 1:
                    ith_dominated[q].append(p)
                    dominating_ith[p] += 1
        for i in range(len(solutions)):
            if dominating_ith[i] == 0:
                front[0].append(i)
                solutions[i].attributes["dominance_ranking"] = 0
        rank = 0
        while len(front[rank]) != 0:
            rank += 1
            for p in front[rank - 1]:
                for q in ith_dominated[p]:
                    dominating_ith[q] -= 1
                    if dominating_ith[q] == 0:
                        front[rank].append(q)
                        solutions[q].attributes["dominance_ranking"] = rank
        self.ranked_sub_lists = [[]] * rank
        for j in range(rank):
            q = [0] * len(front[j])
            for m in range(len(front[j])):
                q[m] = solutions[front[j][m]]
            self.ranked_sub_lists[j] = q
        if k:
            count = 0
            for i, front in enumerate(self.ranked_sub_lists):
                count += len(front)
                if count >= k:
                    self.ranked_sub_lists = self.ranked_sub_lists[: i + 1]
                    break
        return self.ranked_sub_lists

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator("dominance_ranking")


class StrengthRanking(Ranking[List[S]]):
    """Class implementing a ranking scheme based on the strength ranking used in SPEA2."""

    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(StrengthRanking, self).__init__(comparator)

    def compute_ranking(self, solutions: List[S], k: int = None):
        """
        Compute ranking of solutions.
        param solutions: Solution list.
        param k: Number of individuals.
        """
        strength: [int] = [0 for _ in range(len(solutions))]
        raw_fitness: [int] = [0 for _ in range(len(solutions))]
        # strength(i) = | {j | j < - SolutionSet and i dominate j} |
        for i in range(len(solutions)):
            for j in range(len(solutions)):
                if self.comparator.compare(solutions[i], solutions[j]) < 0:
                    strength[i] += 1
        # Calculate the raw fitness:
        # rawFitness(i) = |{sum strength(j) | j <- SolutionSet and j dominate i}|
        for i in range(len(solutions)):
            for j in range(len(solutions)):
                if self.comparator.compare(solutions[i], solutions[j]) == 1:
                    raw_fitness[i] += strength[j]
        max_fitness_value: int = 0
        for i in range(len(solutions)):
            solutions[i].attributes["strength_ranking"] = raw_fitness[i]
            if raw_fitness[i] > max_fitness_value:
                max_fitness_value = raw_fitness[i]
        # Initialize the ranked sublists. In the worst case will be max_fitness_value + 1 different sublists
        self.ranked_sub_lists = [[] for _ in range(max_fitness_value + 1)]
        # Assign each solution to its corresponding front
        for solution in solutions:
            self.ranked_sub_lists[int(solution.attributes["strength_ranking"])].append(solution)
        # Remove empty fronts
        counter = 0
        while counter < len(self.ranked_sub_lists):
            if len(self.ranked_sub_lists[counter]) == 0:
                del self.ranked_sub_lists[counter]
            else:
                counter += 1
        return self.ranked_sub_lists

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator("strength_ranking")

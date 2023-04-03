# -*- coding: utf-8 -*-<模块功能已了解>
from enum import Enum
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar
from Evolo.util.comparator import Comparator, MultiComparator, DominanceComparator
from Evolo.util.ranking import Ranking, FastNonDominatedRanking
from Evolo.util.density_estimator import DensityEstimator, CrowdingDistance

S = TypeVar("S")


class Replacement(Generic[S], ABC):
    def __init__(self):
        pass

    @abstractmethod
    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        pass

    def get_name(self) -> str:
        return "Replacement"


class GreedyPopulationReplacement(Replacement[S]):
    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(GreedyPopulationReplacement, self).__init__()
        self.comparator = comparator

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        result_list = []
        for i in range(0, len(solution_list)):
            if self.comparator.compare(solution_list[i], offspring_list[i]) == -1:
                result_list.append(solution_list[i])
            else:
                result_list.append(offspring_list[i])
        return result_list

    def get_name(self) -> str:
        return "Population greedy replacement"


class JoinPopulationSelectionReplacement(Replacement[S]):
    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(JoinPopulationSelectionReplacement, self).__init__()
        self.comparator = comparator

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        join_population = solution_list + offspring_list
        for i in range(0, len(join_population) - 1):
            for j in range(i + 1, len(join_population)):
                if self.comparator.compare(join_population[i], join_population[j]) == 1:
                    join_population[i], join_population[j] = join_population[j], join_population[i]
        result_list = join_population[0:len(solution_list)]
        return result_list

    def get_name(self) -> str:
        return "Join population selection replacement"


class GreedyPopulationRankingAndDensityEstimatorReplacement(Replacement[S]):
    def __init__(self,
                 ranking: Ranking = FastNonDominatedRanking(DominanceComparator()),
                 density_estimator: DensityEstimator = CrowdingDistance(),
                 ):
        super(GreedyPopulationRankingAndDensityEstimatorReplacement, self).__init__()
        self.ranking = ranking
        self.density_estimator = density_estimator
        self.comparator = MultiComparator([self.ranking.get_comparator(), self.density_estimator.get_comparator()])

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        join_population = solution_list + offspring_list
        size_of_the_result_list = len(solution_list) + len(offspring_list)
        self.ranking.compute_ranking(join_population)
        ranking_id = 0
        result_list = []
        while len(result_list) < size_of_the_result_list:
            current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
            self.density_estimator.compute_density_estimator(current_ranked_solutions)
            if len(current_ranked_solutions) <= (size_of_the_result_list - len(result_list)):
                result_list = result_list + current_ranked_solutions
                ranking_id += 1
        result_list = []
        for i in range(0, len(solution_list)):
            if self.comparator.compare(solution_list[i], offspring_list[i]) == -1:
                result_list.append(solution_list[i])
            else:
                result_list.append(offspring_list[i])
        return result_list

    def get_name(self) -> str:
        return "Greedy population ranking and density estimator replacement"


class RemovalPolicyType(Enum):
    SEQUENTIAL = 1
    ONE_SHOT = 2


class JoinPopulationRankingAndDensityEstimatorReplacement(Replacement[S]):
    def __init__(self,
                 ranking: Ranking = FastNonDominatedRanking(DominanceComparator()),
                 density_estimator: DensityEstimator = CrowdingDistance(),
                 removal_policy=RemovalPolicyType.ONE_SHOT
                 ):
        super(JoinPopulationRankingAndDensityEstimatorReplacement, self).__init__()
        self.ranking = ranking
        self.density_estimator = density_estimator
        self.removal_policy = removal_policy

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        join_population = solution_list + offspring_list
        if self.removal_policy is RemovalPolicyType.SEQUENTIAL:
            result_list = self.sequential_truncation_ranks(join_population, len(solution_list))
            # self.ranking.compute_ranking(join_population)
            # result_list = self.sequential_truncation(0, len(solution_list))
        else:
            result_list = self.one_shot_truncation_ranks(join_population, len(solution_list))
            # self.ranking.compute_ranking(join_population)
            # result_list = self.one_shot_truncation(0, len(solution_list))
        return result_list

    def sequential_truncation_ranks(self, front: List[S], size_of_the_result_list: int) -> List[S]:
        self.ranking.compute_ranking(front)
        ranking_id = 0
        result_list = []
        while len(result_list) < size_of_the_result_list:
            current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
            self.density_estimator.compute_density_estimator(current_ranked_solutions)
            if len(current_ranked_solutions) < (size_of_the_result_list - len(result_list)):
                result_list = result_list + current_ranked_solutions
                ranking_id += 1
            else:
                last_ranked_solutions = []
                for solution in current_ranked_solutions:
                    last_ranked_solutions.append(solution)
                while len(last_ranked_solutions) > (size_of_the_result_list - len(result_list)):
                    self.density_estimator.sort(last_ranked_solutions)
                    del last_ranked_solutions[-1]
                    self.density_estimator.compute_density_estimator(last_ranked_solutions)
                result_list = result_list + last_ranked_solutions
        return result_list

    def one_shot_truncation_ranks(self, front: List[S], size_of_the_result_list: int) -> List[S]:
        self.ranking.compute_ranking(front)
        ranking_id = 0
        result_list = []
        while len(result_list) < size_of_the_result_list:
            current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
            self.density_estimator.compute_density_estimator(current_ranked_solutions)
            if len(current_ranked_solutions) < (size_of_the_result_list - len(result_list)):
                result_list = result_list + current_ranked_solutions
                ranking_id += 1
            else:
                self.density_estimator.sort(current_ranked_solutions)
                for i in range(size_of_the_result_list - len(result_list)):
                    result_list.append(current_ranked_solutions[i])
        return result_list

    def sequential_truncation(self, ranking_id: int, size_of_the_result_list: int) -> List[S]:
        current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
        self.density_estimator.compute_density_estimator(current_ranked_solutions)
        result_list: List[S] = []
        if len(current_ranked_solutions) < size_of_the_result_list:
            result_list.extend(self.ranking.get_sub_front(ranking_id))
            result_list.extend(
                self.sequential_truncation(ranking_id + 1, size_of_the_result_list - len(current_ranked_solutions))
            )
        else:
            for solution in current_ranked_solutions:
                result_list.append(solution)
            while len(result_list) > size_of_the_result_list:
                self.density_estimator.sort(result_list)
                del result_list[-1]
                self.density_estimator.compute_density_estimator(result_list)
        return result_list

    def one_shot_truncation(self, ranking_id: int, size_of_the_result_list: int) -> List[S]:
        current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
        self.density_estimator.compute_density_estimator(current_ranked_solutions)
        result_list: List[S] = []
        if len(current_ranked_solutions) < size_of_the_result_list:
            result_list.extend(self.ranking.get_sub_front(ranking_id))
            result_list.extend(
                self.one_shot_truncation(ranking_id + 1, size_of_the_result_list - len(current_ranked_solutions))
            )
        else:
            self.density_estimator.sort(current_ranked_solutions)
            i = 0
            while len(result_list) < size_of_the_result_list:
                result_list.append(current_ranked_solutions[i])
                i += 1
        return result_list

    def get_name(self) -> str:
        return "Join population ranking and density estimator replacement"

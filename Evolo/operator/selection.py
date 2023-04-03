# -*- coding: utf-8 -*-<模块功能已了解>
import numpy as np
import random
from typing import List, TypeVar
from Evolo.core.operator import Selection
from Evolo.core.solution import Solution
from Evolo.util.comparator import Comparator, DominanceComparator
from Evolo.util.ranking import Ranking, FastNonDominatedRanking
from Evolo.util.density_estimator import DensityEstimator, CrowdingDistance

S = TypeVar("S", bound=Solution)
"""
module:: selection
synopsis: Module implementing selection operators.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class RandomSolutionSelection(Selection[List[S], S]):
    def __init__(self):
        super(RandomSolutionSelection, self).__init__()

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        return random.choice(front)

    def get_name(self) -> str:
        return "Random solution selection"


class NaryRandomSolutionSelection(Selection[List[S], S]):
    def __init__(self, number_of_returned_solutions: int = 1):
        super(NaryRandomSolutionSelection, self).__init__()
        if number_of_returned_solutions < 0:
            raise Exception("The number of solutions to be returned must be positive integer")
        self.number_of_returned_solutions = number_of_returned_solutions

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        if len(front) == 0:
            raise Exception("The front is empty")
        if len(front) < self.number_of_returned_solutions:
            raise Exception("The front contains less elements than required")
        # random_search sampling without replacement
        return random.sample(front, self.number_of_returned_solutions)

    def get_name(self) -> str:
        return "Random solution selection"


class BestSolutionSelection(Selection[List[S], S]):
    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(BestSolutionSelection, self).__init__()
        self.comparator = comparator

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        best_solution = front[0]
        for solution in front[1:]:
            if self.comparator.compare(solution, best_solution) < 0:
                best_solution = solution
        return best_solution

    def return_index(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        best_solution_index = 0
        for idx in range(1, len(front)):
            if self.comparator.compare(front[idx], front[best_solution_index]) < 0:
                best_solution_index = idx
        return best_solution_index

    def get_name(self) -> str:
        return "Best solution selection"


class WorstSolutionSelection(Selection[List[S], S]):
    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(WorstSolutionSelection, self).__init__()
        self.comparator = comparator

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        worst_solution = front[0]
        for solution in front[1:]:
            if self.comparator.compare(solution, worst_solution) > 0:
                worst_solution = solution
        return worst_solution

    def return_index(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        worst_solution_index = 0
        for idx in range(1, len(front)):
            if self.comparator.compare(front[idx], front[worst_solution_index]) > 0:
                worst_solution_index = idx
        return worst_solution_index

    def get_name(self) -> str:
        return "Worst solution selection"


class RouletteWheelSelection(Selection[List[S], S]):
    """Performs roulette wheel selection."""
    __EPS = 1.0e-14

    def __init__(self, objective_index: int = 0):
        super(RouletteWheelSelection).__init__()
        self.objective_index = objective_index

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        score_list = [solution.objectives[self.objective_index] for solution in front]
        score_min = score_list.min()
        score_max = score_list.max()
        score_ptp = score_max - score_min
        probability_list = []
        for i in range(0, len(score_list)):
            probability_list.append((score_list[i] - score_min) / (score_ptp + self.__EPS))
        for i in range(0, len(score_list)):
            probability_list[i] = 1.0 - probability_list[i]
        maximum = sum(probability_list)
        rand = random.uniform(0.0, maximum)
        value = 0.0
        for idx in range(0, len(score_list)):
            value += probability_list[idx]
            if value >= rand:
                return front[idx]
        return front[0]

    def return_index(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        score_list = [solution.objectives[self.objective_index] for solution in front]
        score_min = min(score_list)
        score_max = max(score_list)
        score_ptp = score_max - score_min
        probability_list = []
        for i in range(0, len(score_list)):
            probability_list.append((score_list[i] - score_min) / (score_ptp + self.__EPS))
        for i in range(0, len(score_list)):
            probability_list[i] = 1.0 - probability_list[i]
        maximum = sum(probability_list)
        rand = random.uniform(0.0, maximum)
        value = 0.0
        for idx in range(0, len(score_list)):
            value += probability_list[idx]
            if value >= rand:
                return idx
        return 0

    def return_index_from_scores(self, score_list) -> S:
        score_min = min(score_list)
        score_max = max(score_list)
        score_ptp = score_max - score_min
        probability_list = []
        for i in range(0, len(score_list)):
            probability_list.append((score_list[i] - score_min) / (score_ptp + self.__EPS))
        for i in range(0, len(score_list)):
            probability_list[i] = 1.0 - probability_list[i]
        maximum = sum(probability_list)
        rand = random.uniform(0.0, maximum)
        value = 0.0
        for idx in range(0, len(score_list)):
            value += probability_list[idx]
            if value >= rand:
                return idx
        return 0

    def return_element_from_probabilities(self, probability_list: list, element_set: list) -> S:
        maximum = sum(probability_list)
        rand = random.uniform(0.0, maximum)
        value = 0.0
        for idx in range(0, len(probability_list)):
            value += probability_list[idx]
            if value >= rand:
                return element_set[idx]
        return element_set[0]

    def get_name(self) -> str:
        return "Roulette wheel selection"


class BinaryTournamentSelection(Selection[List[S], S]):
    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(BinaryTournamentSelection, self).__init__()
        self.comparator = comparator

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        if len(front) == 1:
            result = front[0]
        else:
            # Sampling without replacement
            idx1, idx2 = random.sample(range(0, len(front)), 2)
            solution1 = front[idx1]
            solution2 = front[idx2]
            flag = self.comparator.compare(solution1, solution2)
            if flag == -1:
                result = solution1
            elif flag == 1:
                result = solution2
            else:
                result = [solution1, solution2][random.random() < 0.5]
        return result

    def return_index(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        if len(front) == 1:
            result = 0
        else:
            # Sampling without replacement
            idx1, idx2 = random.sample(range(0, len(front)), 2)
            flag = self.comparator.compare(front[idx1], front[idx2])
            if flag == -1:
                result = idx1
            elif flag == 1:
                result = idx2
            else:
                result = [idx1, idx2][random.random() < 0.5]
        return result

    def get_name(self) -> str:
        return "Binary tournament selection"


class BinaryTournament2Selection(Selection[List[S], S]):
    def __init__(self, comparator_list: List[Comparator]):
        super(BinaryTournament2Selection, self).__init__()
        self.comparator_list = comparator_list

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        elif not self.comparator_list:
            raise Exception("The comparators' list is empty")
        if len(front) == 1:
            return front[0]
        else:
            idx1, idx2 = random.sample(range(0, len(front)), 2)
            solution1 = front[idx1]
            solution2 = front[idx2]
            for comparator in self.comparator_list:
                flag = comparator.compare(solution1, solution2)
                if flag == -1:
                    return solution1
                elif flag == 1:
                    return solution2
            if random.random() < 0.5:
                return solution1
            else:
                return solution2

    def return_index(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        elif not self.comparator_list:
            raise Exception("The comparators' list is empty")
        if len(front) == 1:
            return 0
        else:
            idx1, idx2 = random.sample(range(0, len(front)), 2)
            for comparator in self.comparator_list:
                flag = comparator.compare(front[idx1], front[idx2])
                if flag == -1:
                    return idx1
                elif flag == 1:
                    return idx2
            if random.random() < 0.5:
                return idx1
            else:
                return idx2

    def get_name(self) -> str:
        return "Binary tournament selection"


class KwayTournamentSelection:
    def __init__(self, comparator: Comparator = DominanceComparator(),
                 percentage: float = 0.2,
                 number_of_returned_solutions: int = 2):
        super(KwayTournamentSelection, self).__init__()
        self.comparator = comparator
        self.percentage = percentage
        self.number_of_returned_solutions = number_of_returned_solutions

    def execute(self, front=None):
        if 0 < self.percentage < 1:
            number = int(self.percentage * len(front))
        else:
            number = int(self.percentage)
        idx_list = random.sample(range(len(front)), number)
        parents = [front[i] for i in idx_list]
        for j in range(0, len(parents) - 1):
            for k in range(j + 1, len(parents)):
                if self.comparator.compare(parents[j], parents[k]) == 1:
                    solution = parents[j]
                    parents[j] = parents[k]
                    parents[k] = solution
        return parents[:self.number_of_returned_solutions]

    def get_name(self) -> str:
        return "Kway tournament selection"


class DifferentialEvolutionSelection(Selection[List[S], List[S]]):
    def __init__(self):
        super(DifferentialEvolutionSelection, self).__init__()
        self.index_to_exclude = None

    def execute(self, front: List[S]) -> List[S]:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        elif len(front) < 4:
            raise Exception("The front has less than four solutions: " + str(len(front)))
        selected_indexes = random.sample(range(len(front)), 3)
        while self.index_to_exclude in selected_indexes:
            selected_indexes = random.sample(range(len(front)), 3)
        return [front[i] for i in selected_indexes]

    def set_index_to_exclude(self, index: int):
        self.index_to_exclude = index

    def get_name(self) -> str:
        return "Differential evolution selection"


class RankingAndCrowdingDistanceSelection(Selection[List[S], List[S]]):
    def __init__(self, max_population_size: int, dominance_comparator: Comparator = DominanceComparator()):
        super(RankingAndCrowdingDistanceSelection, self).__init__()
        self.max_population_size = max_population_size
        self.dominance_comparator = dominance_comparator
        self.ranking = FastNonDominatedRanking(self.dominance_comparator)
        self.density_estimator = CrowdingDistance()

    def execute(self, front: List[S]) -> List[S]:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        self.ranking.compute_ranking(front)
        ranking_id = 0
        result_list = []
        while len(result_list) < self.max_population_size:
            if len(self.ranking.get_sub_front(ranking_id)) < (self.max_population_size - len(result_list)):
                current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
                self.density_estimator.compute_density_estimator(current_ranked_solutions)
                result_list = result_list + current_ranked_solutions
                ranking_id += 1
            else:
                current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
                self.density_estimator.compute_density_estimator(current_ranked_solutions)
                sorted_sub_front = sorted(current_ranked_solutions, key=lambda x: x.attributes["crowding_distance"],
                                          reverse=True)
                for i in range(self.max_population_size - len(result_list)):
                    result_list.append(sorted_sub_front[i])
        return result_list

    def get_name(self) -> str:
        return "Ranking and crowding distance selection"


class RankingAndDensityEstimatorSelection(Selection[List[S], List[S]]):
    def __init__(self, max_population_size: int,
                 ranking: Ranking = FastNonDominatedRanking(DominanceComparator()),
                 density_estimator: DensityEstimator = CrowdingDistance()
                 ):
        super(RankingAndDensityEstimatorSelection, self).__init__()
        self.max_population_size = max_population_size
        self.ranking = ranking
        self.density_estimator = density_estimator

    def execute(self, front: List[S]) -> List[S]:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        self.ranking.compute_ranking(front)
        ranking_id = 0
        result_list = []
        while len(result_list) < self.max_population_size:
            if len(self.ranking.get_sub_front(ranking_id)) < (self.max_population_size - len(result_list)):
                current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
                self.density_estimator.compute_density_estimator(current_ranked_solutions)
                result_list = result_list + current_ranked_solutions
                ranking_id += 1
            else:
                current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
                self.density_estimator.compute_density_estimator(current_ranked_solutions)
                self.density_estimator.sort(current_ranked_solutions)
                for i in range(self.max_population_size - len(result_list)):
                    result_list.append(current_ranked_solutions[i])
        return result_list

    def get_name(self) -> str:
        return "Ranking and density estimator selection"


class RankingAndFitnessSelection(Selection[List[S], List[S]]):
    def __init__(self, max_population_size: int, reference_point: S,
                 dominance_comparator: Comparator = DominanceComparator()
                 ):
        super(RankingAndFitnessSelection, self).__init__()
        self.max_population_size = max_population_size
        self.dominance_comparator = dominance_comparator
        self.reference_point = reference_point

    def hypesub(self, l, A, actDim, bounds, pvec, alpha, k):
        h = [0 for _ in range(l)]
        Adim = [a[actDim - 1] for a in A]
        indices_sort = sorted(range(len(Adim)), key=Adim.__getitem__)
        S = [A[j] for j in indices_sort]
        pvec = [pvec[j] for j in indices_sort]
        for i in range(1, len(S) + 1):
            if i < len(S):
                extrusion = S[i][actDim - 1] - S[i - 1][actDim - 1]
            else:
                extrusion = bounds[actDim - 1] - S[i - 1][actDim - 1]
            if actDim == 1:
                if i > k:
                    break
                if all(alpha) >= 0:
                    for p in pvec[0:i]:
                        h[p] = h[p] + extrusion * alpha[i - 1]
            else:
                if extrusion > 0:
                    h = [
                        h[j] + extrusion * self.hypesub(l, S[0:i], actDim - 1, bounds, pvec[0:i], alpha, k)[j]
                        for j in range(l)
                    ]
        return h

    def compute_hypervol_fitness_values(self, population: List[S], reference_point: S, k: int):
        points = [ind.objectives for ind in population]
        bounds = reference_point.objectives
        pop_size = len(points)
        if k < 0:
            k = pop_size
        actDim = len(bounds)
        pvec = range(pop_size)
        alpha = []
        for i in range(1, k + 1):
            alpha.append(np.prod([float(k - j) / (pop_size - j) for j in range(1, i)]) / i)
        f = self.hypesub(pop_size, points, actDim, bounds, pvec, alpha, k)
        for i in range(len(population)):
            population[i].attributes["objectives[0]"] = f[i]
        return population

    def execute(self, front: List[S]) -> List[S]:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        ranking.compute_ranking(front)
        ranking_id = 0
        result_list = []
        while len(result_list) < self.max_population_size:
            if len(ranking.get_sub_front(ranking_id)) < self.max_population_size - len(result_list):
                current_ranked_solutions = ranking.get_sub_front(ranking_id)
                result_list = result_list + current_ranked_solutions
                ranking_id += 1
            else:
                current_ranked_solutions = ranking.get_sub_front(ranking_id)
                parameter_K = len(current_ranked_solutions) - (self.max_population_size - len(result_list))
                while parameter_K > 0:
                    current_ranked_solutions = self.compute_hypervol_fitness_values(current_ranked_solutions,
                                                                                    self.reference_point, parameter_K)
                    current_ranked_solutions = sorted(current_ranked_solutions,
                                                      key=lambda x: x.attributes["objectives[0]"], reverse=True)
                    current_ranked_solutions = current_ranked_solutions[:-1]
                    parameter_K = parameter_K - 1
                result_list = result_list + current_ranked_solutions
        return result_list

    def get_name(self) -> str:
        return "Ranking and objectives[0] selection"

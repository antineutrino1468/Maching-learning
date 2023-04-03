# -*- coding: utf-8 -*-<模块功能已了解>
import math
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from Evolo.core.solution import Solution
from Evolo.util.distance import EuclideanDistance
from Evolo.util.constraint_handling import overall_constraint_violation_degree

S = TypeVar("S")


class Comparator(Generic[S], ABC):
    @abstractmethod
    def compare(self, solution1: S, solution2: S) -> int:
        pass


class MultiComparator(Comparator):
    """
    This comparator takes a list of comparators and check all of them iteratively until a
    value != 0 is obtained or the list becomes empty
    """

    def __init__(self, comparator_list: [Comparator]):
        self.comparator_list: [Comparator] = comparator_list

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        for comparator in self.comparator_list:
            flag = comparator.compare(solution1, solution2)
            if flag != 0:
                return flag
        return 0


class OverallConstraintViolationComparator(Comparator):
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        violation_degree_solution_1 = overall_constraint_violation_degree(solution1)
        violation_degree_solution_2 = overall_constraint_violation_degree(solution2)
        if violation_degree_solution_1 < 0 and violation_degree_solution_2 < 0:
            if violation_degree_solution_1 > violation_degree_solution_2:
                result = -1
            elif violation_degree_solution_1 < violation_degree_solution_2:
                result = 1
            else:
                result = 0
        elif violation_degree_solution_1 == 0 and violation_degree_solution_2 < 0:
            result = -1
        elif violation_degree_solution_1 < 0 and violation_degree_solution_2 == 0:
            result = 1
        else:
            result = 0
        return result


class ObjectiveComparator(Comparator):
    def __init__(self, objective_index: int = 0, descending_order: bool = False):
        self.objective_index = objective_index
        self.ascending_order = not descending_order

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        value1 = solution1.objectives[self.objective_index]
        value2 = solution2.objectives[self.objective_index]
        if self.ascending_order:
            if value1 < value2:
                return -1
            elif value1 > value2:
                return 1
            else:
                return 0
        else:
            if value1 < value2:
                return 1
            elif value1 > value2:
                return -1
            else:
                return 0


class EpsilonObjectiveComparator(Comparator):
    def __init__(self, objective_index: int = 0, epsilon: float = 1e-10):
        self.objective_index = objective_index
        self.__EPS = epsilon

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        value1 = solution1.objectives[self.objective_index]
        value2 = solution2.objectives[self.objective_index]
        if value1 / (1 + self.__EPS) < value2:
            return -1
        elif value1 / (1 + self.__EPS) > value2:
            return 1
        else:
            return 0


class EqualSolutionsComparator(Comparator):
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        dominate1 = 0
        dominate2 = 0
        for i in range(len(solution1.objectives)):
            value1 = solution1.objectives[i]
            value2 = solution2.objectives[i]
            if value1 < value2:
                flag = -1
            elif value1 > value2:
                flag = 1
            else:
                flag = 0
            if flag == -1:
                dominate1 = 1
            if flag == 1:
                dominate2 = 1
        if dominate1 == 0 and dominate2 == 0:
            return 0
        elif dominate1 == 1:
            return -1
        elif dominate2 == 1:
            return 1


class EpsilonEqualSolutionComparator(Comparator):
    __EPS = 1e-10

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        if solution1.number_of_variables != solution2.number_of_variables:
            return -1
        euclidean_distance = EuclideanDistance()
        if euclidean_distance.get_distance(solution1.variables, solution2.variables) < self.__EPS:
            return 0
        return -1


class IdenticalSolutionsComparator(Comparator):
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        result = 0
        for i in range(solution1.number_of_variables):
            value1 = solution1.variables[i]
            value2 = solution2.variables[i]
            if value1 < value2:
                result = -1
                break
            elif value1 > value2:
                result = 1
                break
            else:
                result = 0
        return result


class SolutionAttributeComparator(Comparator):
    def __init__(self, key: str, lowest_is_best: bool = True):
        self.key = key
        self.lowest_is_best = lowest_is_best

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        value1 = solution1.attributes.get(self.key)
        value2 = solution2.attributes.get(self.key)
        result = 0
        if value1 is not None and value2 is not None:
            if self.lowest_is_best:
                if value1 < value2:
                    result = -1
                elif value1 > value2:
                    result = 1
                else:
                    result = 0
            else:
                if value1 > value2:
                    result = -1
                elif value1 < value2:
                    result = 1
                else:
                    result = 0
        return result


class RankingAndCrowdingDistanceComparator(Comparator):
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        result = SolutionAttributeComparator("dominance_ranking").compare(solution1, solution2)
        if result == 0:
            result = SolutionAttributeComparator("crowding_distance", lowest_is_best=False).compare(
                solution1, solution2
            )
        return result


class RankingAndKNNDistanceComparator(Comparator):
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        result = SolutionAttributeComparator("dominance_ranking").compare(solution1, solution2)
        if result == 0:
            result = SolutionAttributeComparator("knn_density", lowest_is_best=False).compare(solution1, solution2)
        return result


class DominanceComparator(Comparator):
    def __init__(self, constraint_comparator: Comparator = OverallConstraintViolationComparator()):
        self.constraint_comparator = constraint_comparator

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            raise Exception("The solution1 is None")
        elif solution2 is None:
            raise Exception("The solution2 is None")
        result = self.constraint_comparator.compare(solution1, solution2)
        if result == 0:
            # result = self.__dominance_test(solution1, solution2)
            result = self.dominance_test(solution1.objectives, solution2.objectives)
        return result

    def __dominance_test(self, solution1: Solution, solution2: Solution) -> float:
        best_is_one = 0
        best_is_two = 0
        for i in range(solution1.number_of_objectives):
            value1 = solution1.objectives[i]
            value2 = solution2.objectives[i]
            if value1 != value2:
                if value1 < value2:
                    best_is_one = 1
                if value1 > value2:
                    best_is_two = 1
        if best_is_one > best_is_two:
            result = -1
        elif best_is_two > best_is_one:
            result = 1
        else:
            result = 0
        return result

    @staticmethod
    def dominance_test(vector1: [float], vector2: [float]) -> int:
        result = 0
        for i in range(len(vector1)):
            if vector1[i] > vector2[i]:
                if result == -1:
                    return 0
                result = 1
            elif vector2[i] > vector1[i]:
                if result == 1:
                    return 0
                result = -1
        return result


class EpsilonDominanceComparator(DominanceComparator):
    def __init__(self,
                 epsilon: float,
                 constraint_comparator: Comparator = OverallConstraintViolationComparator(),
                 ):
        super(EpsilonDominanceComparator, self).__init__(constraint_comparator)
        self.__EPS = epsilon

    def compare(self, solution1: Solution, solution2: Solution):
        result = self.constraint_comparator.compare(solution1, solution2)
        if result == 0:
            result = self.__dominance_test(solution1, solution2)
        return result

    def __dominance_test(self, solution1: Solution, solution2: Solution):
        best_is_one = False
        best_is_two = False
        for i in range(solution1.number_of_objectives):
            value1 = math.floor(solution1.objectives[i] / self.__EPS)
            value2 = math.floor(solution2.objectives[i] / self.__EPS)
            if value1 < value2:
                best_is_one = True
                if best_is_two:
                    return 0
            elif value1 > value2:
                best_is_two = True
                if best_is_one:
                    return 0
        if not best_is_one and not best_is_two:
            dist1 = 0.0
            dist2 = 0.0
            for i in range(solution1.number_of_objectives):
                index1 = math.floor(solution1.objectives[i] / self.__EPS)
                index2 = math.floor(solution2.objectives[i] / self.__EPS)
                dist1 += math.pow(solution1.objectives[i] - index1 * self.__EPS, 2.0)
                dist2 += math.pow(solution2.objectives[i] - index2 * self.__EPS, 2.0)
            if dist1 < dist2:
                return -1
            else:
                return 1
        else:
            if best_is_two:
                return 1
            else:
                return -1


class GDominanceComparator(DominanceComparator):
    def __init__(self,
                 reference_point: (),
                 constraint_comparator: Comparator = OverallConstraintViolationComparator(),
                 ):
        super(GDominanceComparator, self).__init__(constraint_comparator)
        self.reference_point = reference_point

    def compare(self, solution1: Solution, solution2: Solution):
        if self.__flag(solution1) > self.__flag(solution2):
            result = -1
        elif self.__flag(solution1) < self.__flag(solution2):
            result = 1
        else:
            result = super(GDominanceComparator, self).compare(solution1, solution2)
        return result

    def __flag(self, solution: Solution):
        result = 1
        for i in range(solution.number_of_objectives):
            if solution.objectives[i] > self.reference_point[i]:
                result = 0
        if result == 0:
            result = 1
            for i in range(solution.number_of_objectives):
                if solution.objectives[i] < self.reference_point[i]:
                    result = 0
        return result

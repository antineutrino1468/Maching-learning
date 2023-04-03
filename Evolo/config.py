# -*- coding: utf-8 -*-<模块功能已了解>
from Evolo.core.observer import Observable
from Evolo.util.comparator import DominanceComparator
from Evolo.util.evaluator import Evaluator, SequentialEvaluator
from Evolo.util.generator import RandomGenerator
from Evolo.util.observable import DefaultObservable
from Evolo.util.termination_criterion import StoppingByEvaluations


class _Store:
    @property
    def default_observable(self) -> Observable:
        return DefaultObservable()

    @property
    def default_evaluator(self) -> Evaluator:
        return SequentialEvaluator()

    @property
    def default_generator(self):
        return RandomGenerator()

    @property
    def default_termination_criteria(self):
        return StoppingByEvaluations(max_evaluations=25000)

    @property
    def default_comparator(self):
        return DominanceComparator()




store = _Store()

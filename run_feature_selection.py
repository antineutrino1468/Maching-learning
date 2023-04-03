from feature_selection_problem import *
from Evolo.util.termination_criterion import StoppingByEvaluations


def select_and_run_algorithm(algorithm_name: str, problem_name: str):
    problem = RegressionPredictionQ1Q2(problem_name)
    max_evaluations = 2000

    """ 
    Population-based algorithms
    """
    if algorithm_name == "PSOBase":
        from Evolo.optimizers.singleobjective.PSO import PSOBase
        algorithm = PSOBase(
            problem=problem,
            pop_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "TLBOBase":
        from Evolo.optimizers.singleobjective.TLBO import TLBOBase
        algorithm = TLBOBase(
            problem=problem,
            pop_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "WOABase":
        from Evolo.optimizers.singleobjective.WOA import WOABase
        algorithm = WOABase(
            problem=problem,
            pop_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )

    algorithm.run()
    result = algorithm.get_result()
    print("Algorithm: " + algorithm.get_name())
    print("Problem: " + problem.get_name())
    print("Solution: " + str(result[0].variables))
    print("Fitness:  " + str(result[0].objectives))
    print("Computing time: " + str(algorithm.total_time))


if __name__ == "__main__":
    boat_name = 'Monohulled Sailboats'
    # boat_name = 'Catamarans'
    algorithm_name_list = [
        # "PSOBase",
        "TLBOBase",
        # "WOABase",
    ]

    for algorithm_name in algorithm_name_list:
        select_and_run_algorithm(algorithm_name, boat_name)
    print("Execution completed")

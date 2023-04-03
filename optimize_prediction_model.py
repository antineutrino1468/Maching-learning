import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import explained_variance_score  # Explained variance regression scoring function
from sklearn.metrics import mean_absolute_error  # Mean absolute error
from sklearn.metrics import mean_squared_error  # Mean Square Error
from sklearn.metrics import mean_squared_log_error  # Mean squared logarithmic error
from sklearn.metrics import median_absolute_error  # Median absolute error
from sklearn.metrics import r2_score  # R2 (coefficient of determination)
from bayes_opt import BayesianOptimization


class OptimizingRandomForestRegressor:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.pbounds = {'n_estimators': (10, 250),#(10, 250)
                        'min_samples_split': (2, 25),#(2, 25)
                        'max_features': (0.1, 0.999),#(0.1, 0.999)
                        'max_depth': (5, 15)}#(5, 15)
        self.params_best = {}

    def output_results_with_initial_values(self):
        model = RandomForestRegressor()
        model = model.fit(train_x, train_y)
        score = model.score(test_x, test_y)
        print('Test set scoring with default parameters：' + str(score))

    def evaluate_solution(self, n_estimators, min_samples_split, max_features, max_depth):
        res = RandomForestRegressor(n_estimators=int(n_estimators),
                                    min_samples_split=int(min_samples_split),
                                    max_features=min(max_features, 0.999),  # float
                                    max_depth=int(max_depth),
                                    random_state=2
                                    ).fit(train_x, train_y).score(test_x, test_y)
        return res

    def output_results_with_best_values(self):
        model = RandomForestRegressor(n_estimators=int(self.params_best['n_estimators']),
                                      min_samples_split=int(self.params_best['min_samples_split']),
                                      max_features=min(self.params_best['max_features'], 0.999),  # float
                                      max_depth=int(self.params_best['max_depth']),
                                      random_state=2)
        model = model.fit(train_x, train_y)
        score = model.score(test_x, test_y)
        print('Optimised test set scores：' + str(score))

    def run(self):
        optimizer = BayesianOptimization(
            f=self.evaluate_solution,
            pbounds=self.pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )
        optimizer.maximize(
            init_points=5,  # Number of steps to perform a random search
            n_iter=25,  # Number of steps to perform Bayesian optimization
        )
        self.params_best = optimizer.max['params']
        print(optimizer.max)
        self.output_results_with_initial_values()
        self.output_results_with_best_values()


if __name__ == "__main__":
    data = pd.read_excel('./Monohulled Sailboats/cleaned_data.xlsx')
    X = data[['Length(ft)', 'Year',
              'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
              'Sail Area (sq ft)', 'Average cargo throughput (tons)',
              'GDP (USD billion)', 'GDP per capita (USD)',
              'logistics costs to GDP%']]
    Y = data['Listing Price (USD)']
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1, random_state=5)

    optimize_RandomForestRegressor = OptimizingRandomForestRegressor(X, Y)
    optimize_RandomForestRegressor.run()

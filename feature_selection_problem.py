import numpy as np
import pandas as pd
import copy
import random
from sklearn import preprocessing
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

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

from Evolo.core.problem import FloatProblem
from Evolo.core.solution import FloatSolution

import shap
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class RegressionPredictionQ1Q2(FloatProblem):
    def __init__(self, boat_name):
        self.boat_name = boat_name
        self.path = "./%s" % self.boat_name
        self.X = None
        self.Y = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        self.output_figure = True

        self.__read_instance_from_file()
        self.number_of_objectives = 1
        self.number_of_variables = len(self.X.columns)
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.variable_lb = [0.0 for _ in range(self.number_of_variables)]
        self.variable_ub = [2.0 for _ in range(self.number_of_variables)]
        FloatSolution.variable_lb = self.variable_lb
        FloatSolution.variable_ub = self.variable_ub

    def __read_instance_from_file(self):
        self.data = pd.read_excel('%s/%s.xlsx' % (self.path, self.boat_name))
        cleaned_data = self.data[['Length \n(ft)', 'Listing Price (USD)', 'Year',
                                  'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
                                  'Sail Area (sq ft)', 'Average cargo throughput (tons)',
                                  'GDP (USD billion)', 'GDP per capita (USD)',
                                  'Average ratio of total logistics costs to GDP']].copy()
        cleaned_data.isnull().sum()
        cleaned_data.dropna(inplace=True)
        cleaned_data.reset_index(inplace=True, drop=True)
        cleaned_data.isnull().sum()
        cleaned_data.columns = ['Length(ft)', 'Listing Price (USD)', 'Year',
                                'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
                                'Sail Area (sq ft)', 'Average cargo throughput (tons)',
                                'GDP (USD billion)', 'GDP per capita (USD)',
                                'logistics costs to GDP%']
        print(cleaned_data.shape)
        print(cleaned_data.dtypes)

        cleaned_data.to_excel('%s/cleaned_data.xlsx' % self.path, index=None)
        lbl = preprocessing.LabelEncoder()
        cleaned_data['Year'] = lbl.fit_transform(cleaned_data['Year'].astype(int))  # Convert the column containing the incorrect data type for the prompt
        # self.X = cleaned_data[['Length(ft)', 'Year', 'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
        #                        'Sail Area (sq ft)', 'Average cargo throughput (tons)',
        #                        'GDP (USD billion)', 'GDP per capita (USD)',
        #                        'logistics costs to GDP%']]
        self.X = cleaned_data[['Length(ft)', 'Year', 'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
                               'Sail Area (sq ft)', 'GDP (USD billion)', 'GDP per capita (USD)']]
        self.Y = cleaned_data['Listing Price (USD)']
        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(self.X, self.Y, test_size=0.1, random_state=5)

    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        solution = self.remedy_solution(solution)
        variables = solution.variables
        selected_column = []
        columns = list(self.train_x.columns)
        print(columns)
        for i in range(0, len(variables)):
            if variables[i] > 1.0:
                selected_column.append(columns[i])
        if len(selected_column) == 0:
            selected_column.append(columns[0])
        train_x_new = self.train_x[selected_column]
        test_x_new = self.test_x[selected_column]
        model = RandomForestRegressor(max_depth=20, n_estimators=1000, random_state=0)
        model.fit(train_x_new, self.train_y)
        y_pred = model.predict(test_x_new)
        # print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(train_x_new), self.train_y)))
        # print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(test_x_new), self.test_y)))
        # print("Mean absolute error:", mean_absolute_error(self.test_y, y_pred))
        r2_score_value = r2_score(self.test_y, y_pred)
        print("r2_score" + str(r2_score_value))
        solution.objectives[0] = -r2_score_value
        solution.fitness = solution.objectives[0]
        return solution

    def get_name(self):
        return "RegressionPredictionQ1Q2"

    def mape(self, actual, pred):
        actual, pred = np.array(actual), np.array(pred)
        return np.mean(np.abs((actual - pred) / actual))

    def plot_learning_curve(self, estimator, title, train_x, train_y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("game num")
        plt.ylabel("score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, train_x, train_y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.savefig('%s/%s.jpg' % (self.path, title))
        plt.legend(loc="best")
        plt.show()
        return plt

    def conduct_MLPRegressor(self):
        print("MLPRegressor regression:")
        model = MLPRegressor(hidden_layer_sizes=10, max_iter=1000).fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "MLPRegressor", self.train_x, self.train_y, ylim=None, cv=None, n_jobs=1)

    def conduct_LinearRegression(self):
        print("LinearRegression regression:")
        model = LinearRegression()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("Mean absolute error:", mean_absolute_error(self.test_y, y_pred))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
            self.plot_learning_curve(model, "LineRegression", self.train_x, self.train_y, ylim=None, cv=cv, n_jobs=1)

    def conduct_DecisionTreeRegressor(self):
        print("DecisionTreeRegressor regression:")
        model = DecisionTreeRegressor(max_depth=50, random_state=0)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "DecisionTreeRegressor", self.train_x, self.train_y, ylim=None, cv=None,
                                     n_jobs=1)

    def conduct_RandomForestRegressor(self):
        print("RandomForestRegressor regression:")
        model = RandomForestRegressor(max_depth=20, n_estimators=1000, random_state=0)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "RandomForestRegressor", self.train_x, self.train_y, ylim=None, cv=None,
                                     n_jobs=1)

    def conduct_LGBMRegressor(self):
        print("LGBMRegressor regression:")
        model = lgb.LGBMRegressor()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "LGBMRegressor", self.train_x, self.train_y, ylim=None, cv=None, n_jobs=1)

    def conduct_XGBRegressor(self):
        print("XGBRegressor regression:")
        model = xgb.XGBRegressor()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "XGBRegressor", self.train_x, self.train_y, ylim=None, cv=None, n_jobs=1)

    def conduct_RidgeCV(self):
        print("RidgeCV regression:")
        model = RidgeCV()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "RidgeCV", self.train_x, self.train_y, ylim=None, cv=None, n_jobs=1)

    def conduct_GradientBoostingRegressor(self):
        print("GradientBoostingRegressor regression:")
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "GradientBoostingRegressor", self.train_x, self.train_y, ylim=None, cv=None,
                                     n_jobs=1)

    def conduct_ExtraTreesRegressor(self):
        print("ExtraTreesRegressor regression:")
        model = ExtraTreesRegressor()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "ExtraTreesRegressor", self.train_x, self.train_y, ylim=None, cv=None,
                                     n_jobs=1)

    def conduct_AdaBoostRegressor(self):
        print("AdaBoostRegressor regression:")
        model = AdaBoostRegressor()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "AdaBoostRegressor", self.train_x, self.train_y, ylim=None, cv=None,
                                     n_jobs=1)

    def conduct_Stacking(self):
        print("Stacking regression:")
        estimators = [('RandomForestRegressor', RandomForestRegressor(max_depth=20, n_estimators=1000, random_state=0)),

                      ('LGBMRegressor', lgb.LGBMRegressor())
                      ]
        # ('DecisionTreeRegressor', ExtraTreesRegressor()),
        reg = StackingRegressor(estimators=estimators,
                                final_estimator=RandomForestRegressor(max_depth=20, n_estimators=1000, random_state=0))
        reg.fit(self.train_x, self.train_y)
        y_pred = reg.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(reg.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(reg.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(reg, "StackingRegressor", self.train_x, self.train_y, ylim=None, cv=None, n_jobs=1)

    def output_sharp_figure(self):
        print("RandomForestRegressor regression and output shapefile:")
        # model = xgb.XGBRegressor()
        model = RandomForestRegressor(max_depth=20, n_estimators=1000, random_state=0)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X)  # Pass in the feature matrix X and calculate the SHAP value
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # Setting the font
        plt.rcParams["axes.unicode_minus"] = False  # Normal display of negative sign
        # summarize the effects of all the features
        shap.summary_plot(shap_values, self.X)
        if self.output_figure:
            plt.show()


class RegressionPredictionQ3(FloatProblem):
    def __init__(self, boat_name):
        self.boat_name = boat_name
        self.path = "./%s" % self.boat_name
        self.cleaned_data = None
        self.X = None
        self.Y = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        self.output_figure = False

        self.__read_instance_from_file()
        self.number_of_objectives = 1
        self.number_of_variables = len(self.X.columns)
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.variable_lb = [0.0 for _ in range(self.number_of_variables)]
        self.variable_ub = [2.0 for _ in range(self.number_of_variables)]
        FloatSolution.variable_lb = self.variable_lb
        FloatSolution.variable_ub = self.variable_ub

    def __read_instance_from_file(self):
        self.data = pd.read_excel('%s/%s.xlsx' % (self.path, self.boat_name))
        cleaned_data = self.data[['Length \n(ft)', 'Listing Price (USD)', 'Year',
                                  'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
                                  'Sail Area (sq ft)', 'Average cargo throughput (tons)',
                                  'GDP (USD billion)', 'GDP per capita (USD)',
                                  'Average ratio of total logistics costs to GDP']].copy()
        cleaned_data.isnull().sum()
        self.index_d = cleaned_data.dropna().index
        cleaned_data.dropna(inplace=True)
        cleaned_data.reset_index(inplace=True, drop=True)
        cleaned_data.isnull().sum()
        cleaned_data.columns = ['Length(ft)', 'Listing Price (USD)', 'Year',
                                'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
                                'Sail Area (sq ft)', 'Average cargo throughput (tons)',
                                'GDP (USD billion)', 'GDP per capita (USD)', 'logistics costs to GDP%']
        cleaned_data.to_excel('%s/cleaned_data.xlsx' % self.path, index=None)
        self.cleaned_data = cleaned_data.copy()
        lbl = preprocessing.LabelEncoder()
        cleaned_data['Year'] = lbl.fit_transform(cleaned_data['Year'].astype(int))  # Convert the column containing the incorrect data type for the prompt
        self.X = cleaned_data[['Length(ft)', 'Year', 'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
                               'Sail Area (sq ft)', 'Average cargo throughput (tons)',
                               'GDP (USD billion)', 'GDP per capita (USD)',
                               'logistics costs to GDP%']]
        self.Y = cleaned_data['Listing Price (USD)']
        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(self.X, self.Y, test_size=0.1, random_state=5)

    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        solution = self.remedy_solution(solution)
        variables = solution.variables
        selected_column = []
        columns = list(self.train_x.columns)
        print(columns)
        for i in range(0, len(variables)):
            if variables[i] > 1.0:
                selected_column.append(columns[i])
        if len(selected_column) == 0:
            selected_column.append(columns[0])
        train_x_new = self.train_x[selected_column]
        test_x_new = self.test_x[selected_column]
        model = RandomForestRegressor(max_depth=20, n_estimators=1000, random_state=0)
        model.fit(train_x_new, self.train_y)
        y_pred = model.predict(test_x_new)
        # print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(train_x_new), self.train_y)))
        # print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(test_x_new), self.test_y)))
        # print("Mean absolute error:", mean_absolute_error(self.test_y, y_pred))
        r2_score_value = r2_score(self.test_y, y_pred)
        print("r2_score" + str(r2_score_value))
        solution.objectives[0] = -r2_score_value
        solution.fitness = solution.objectives[0]
        return solution

    def get_name(self):
        return "RegressionPredictionQ1Q2"

    def mape(self, actual, pred):
        actual, pred = np.array(actual), np.array(pred)
        return np.mean(np.abs((actual - pred) / actual))

    def plot_learning_curve(self, estimator, title, train_x, train_y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("game num")
        plt.ylabel("score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, train_x, train_y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.savefig('%s/%s.jpg' % (self.path, title))
        plt.legend(loc="best")
        plt.show()
        return plt

    def conduct_MLPRegressor(self):
        print("MLPRegressor regression:")
        model = MLPRegressor(hidden_layer_sizes=10, max_iter=1000).fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "MLPRegressor", self.train_x, self.train_y, ylim=None, cv=None, n_jobs=1)

    def conduct_LinearRegression(self):
        print("LinearRegression regression:")
        model = LinearRegression()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("Mean absolute error:", mean_absolute_error(self.test_y, y_pred))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
            self.plot_learning_curve(model, "LineRegression", self.train_x, self.train_y, ylim=None, cv=cv, n_jobs=1)

    def conduct_DecisionTreeRegressor(self):
        print("DecisionTreeRegressor regression:")
        model = DecisionTreeRegressor(max_depth=50, random_state=0)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "DecisionTreeRegressor", self.train_x, self.train_y, ylim=None, cv=None,
                                     n_jobs=1)

    def conduct_RandomForestRegressor(self):
        print("RandomForestRegressor regression:")
        model = RandomForestRegressor(max_depth=20, n_estimators=250, random_state=0)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "RandomForestRegressor", self.train_x, self.train_y, ylim=None, cv=None,
                                     n_jobs=1)

    def conduct_LGBMRegressor(self):
        print("LGBMRegressor regression:")
        model = lgb.LGBMRegressor()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "LGBMRegressor", self.train_x, self.train_y, ylim=None, cv=None, n_jobs=1)

    def conduct_XGBRegressor(self):
        print("XGBRegressor regression:")
        model = xgb.XGBRegressor()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "XGBRegressor", self.train_x, self.train_y, ylim=None, cv=None, n_jobs=1)

    def conduct_RidgeCV(self):
        print("RidgeCV regression:")
        model = RidgeCV()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "RidgeCV", self.train_x, self.train_y, ylim=None, cv=None, n_jobs=1)

    def conduct_GradientBoostingRegressor(self):
        print("GradientBoostingRegressor regression:")
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "GradientBoostingRegressor", self.train_x, self.train_y, ylim=None, cv=None,
                                     n_jobs=1)

    def conduct_ExtraTreesRegressor(self):
        print("ExtraTreesRegressor regression:")
        model = ExtraTreesRegressor()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "ExtraTreesRegressor", self.train_x, self.train_y, ylim=None, cv=None,
                                     n_jobs=1)

    def conduct_AdaBoostRegressor(self):
        print("AdaBoostRegressor regression:")
        model = AdaBoostRegressor()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(model, "AdaBoostRegressor", self.train_x, self.train_y, ylim=None, cv=None,
                                     n_jobs=1)

    def conduct_Stacking(self):
        print("Stacking regression:")
        estimators = [('RandomForestRegressor', RandomForestRegressor(max_depth=20, n_estimators=1000, random_state=0)),
                      ('DecisionTreeRegressor', DecisionTreeRegressor(max_depth=50, random_state=0)),
                      ('LGBMRegressor', lgb.LGBMRegressor())
                      ]
        reg = StackingRegressor(estimators=estimators,
                                final_estimator=RandomForestRegressor(max_depth=20, n_estimators=1000, random_state=0))
        reg.fit(self.train_x, self.train_y)
        y_pred = reg.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(reg.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(reg.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        if self.output_figure:
            self.plot_learning_curve(reg, "StackingRegressor", self.train_x, self.train_y, ylim=None, cv=None, n_jobs=1)

    def output_sharp_figure(self):
        print("RandomForestRegressor regression and output shapefile:")
        # model = xgb.XGBRegressor()
        model = RandomForestRegressor(max_depth=20, n_estimators=1000, random_state=0)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        print("Mean absolute percentage error of the training set:{:.3f}".format(self.mape(model.predict(self.train_x), self.train_y)))
        print("Test set mean absolute percentage error:{:.3f}".format(self.mape(model.predict(self.test_x), self.test_y)))
        print("r2_score", r2_score(self.test_y, y_pred))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X)  # Pass in the feature matrix X and calculate the SHAP value
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # Setting the font
        plt.rcParams["axes.unicode_minus"] = False  # Normal display of negative sign
        # summarize the effects of all the features
        shap.summary_plot(shap_values, self.X)
        if self.output_figure:
            plt.show()

    def process_Q3(self):
        cleaned_data_Hongkong = self.cleaned_data.copy()
        print(cleaned_data_Hongkong.columns)
        print(cleaned_data_Hongkong.head())
        cleaned_data_Hongkong['Average Cargo Throughput (tons)'] = 399200000
        cleaned_data_Hongkong['GDP (bn)'] = 341
        cleaned_data_Hongkong['GDP per capita'] = 45638
        cleaned_data_Hongkong['logistics costs to GDP%'] = 3.3
        X = self.cleaned_data[['Length(ft)', 'Year',
                               'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
                               'Sail Area (sq ft)', 'Average cargo throughput (tons)',
                               'GDP (USD billion)', 'GDP per capita (USD)', 'logistics costs to GDP%']]
        Y = self.cleaned_data['Listing Price (USD)']
        model = RandomForestRegressor(max_depth=20, n_estimators=1000, random_state=0)
        model.fit(X, Y)
        y_pred = model.predict(X)
        print("Mean absolute percentage error:{:.3f}".format(self.mape(y_pred, Y)))
        print("r2_score", r2_score(y_pred, Y))

        DF = pd.DataFrame()
        DF['true_Price'] = Y
        np.set_printoptions(suppress=True)  # Abolition of scientific notation
        pd.set_option('display.float_format', lambda x: '%.1f' % x)
        y_pred = model.predict(cleaned_data_Hongkong[X.columns])
        DF['simulation_HK_Price'] = y_pred
        DF['Country/Region/State'] = self.data.iloc[self.index_d]['Country/Region/State'].to_numpy()
        DF['Make Variant'] = self.data.iloc[self.index_d]['Make Variant'].to_numpy()
        DF.to_excel('%s/DF.xlsx' % self.path, index=None)
        DF.drop_duplicates(subset=['Country/Region/State', 'Make Variant'], keep='first').to_excel(
            '%s/DF1.xlsx' % self.path, index=None)


if __name__ == "__main__":
    boat_name = 'Monohulled Sailboats'
    # boat_name = 'Catamarans'

    regression_prediction = RegressionPredictionQ1Q2(boat_name)
    # regression_prediction.conduct_MLPRegressor()
    # regression_prediction.conduct_LinearRegression()
    # regression_prediction.conduct_DecisionTreeRegressor()
    regression_prediction.conduct_RandomForestRegressor()
    # regression_prediction.conduct_LGBMRegressor()
    # regression_prediction.conduct_XGBRegressor()
    # regression_prediction.conduct_RidgeCV()
    # regression_prediction.conduct_GradientBoostingRegressor()
    # regression_prediction.conduct_ExtraTreesRegressor()
    # regression_prediction.conduct_AdaBoostRegressor()
    # regression_prediction.conduct_Stacking()
    # regression_prediction.output_sharp_figure()
    # regression_prediction.process_Q3()

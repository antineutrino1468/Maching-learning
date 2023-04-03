from sklearn import preprocessing
import pandas as pd
from numpy import *
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msn
import scipy.stats as st
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-darkgrid')
sns.set(style='darkgrid')


class DataProcessing:
    def __init__(self, boat_name):
        self.boat_name = boat_name
        self.path = "./%s" % self.boat_name
        self.data = None
        self.columns = ['Length \n(ft)', 'Listing Price (USD)', 'Year',
                        'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
                        'Sail Area (sq ft)', 'Average cargo throughput (tons)',
                        'GDP (USD billion)', 'GDP per capita (USD)',
                        'Average ratio of total logistics costs to GDP']
        self.output_figure = True

    def getx(self, x):
        return str(x['Make']) + ' ' + str(x['Variant'])

    def tra(self, x):
        try:
            return x.lower().replace(' ', '')
        except:
            return x

    def read_collected_paper(self):
        # Process and merge data
        data1 = pd.read_excel('2023_MCM_Problem_Y_Boats.xlsx', sheet_name=self.boat_name)
        print(data1.head())
        # Sailboat data from：http://www.sailboatdata.com/
        # Economic and throughput data are both for 2019, source: World Bank, International Freight and Trade Association, World Economic Forum
        data2 = pd.read_excel('2023_MCM_Problem_Y_Boats.xlsx', sheet_name='%s_1' % self.boat_name)
        print(data2.head())
        data3 = pd.read_excel('2023_MCM_Problem_Y_Boats.xlsx', sheet_name='%s_2' % self.boat_name)
        print(data3.head())
        for i in ['平均货物吞吐量（吨）', 'GDP（亿美元）', '人均GDP（美元）', '物流总成本占GDP的平均比例']:
            data3[i] = data3[i].apply(lambda x: np.NaN if x == '-' else x)
        data1['Make Variant'] = data1.apply(lambda x: self.getx(x), axis=1)
        print(data1.shape)
        print(data2.shape)
        data1['Make Variant'].nunique()
        data2['型号'].nunique()
        data2 = data2.drop_duplicates(keep='first', subset='型号')
        data2['型号'] = data2['型号'].apply(lambda x: x.replace(' ', ''))
        data1['Make Variant'] = data1['Make Variant'].apply(lambda x: x.replace(' ', ''))
        self.data = pd.merge(data1, data2, how='left', left_on='Make Variant', right_on='型号')
        print(self.data)
        data3['城市/地区'].nunique()
        print(data3['城市/地区'])
        data3['城市/地区'] = data3['城市/地区'].apply(lambda x: self.tra(x))
        self.data['Country/Region/State'] = data1['Country/Region/State'].apply(lambda x: self.tra(x))
        self.data = pd.merge(self.data, data3, how='left', left_on='Country/Region/State', right_on='城市/地区')
        print(self.data)
        print(self.data.columns)
        self.data.columns = ['Make', 'Variant', 'Length \n(ft)', 'Geographic Region',
                             'Country/Region/State', 'Listing Price (USD)', 'Year', 'Make Variant',
                             'Make Variant2', 'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
                             'Sail Area (sq ft)', 'City/Region', 'Average cargo throughput (tons)', 'GDP (USD billion)',
                             'GDP per capita (USD)', 'Average ratio of total logistics costs to GDP']
        self.data = self.data[['Make', 'Variant', 'Length \n(ft)', 'Geographic Region',
                               'Country/Region/State', 'Listing Price (USD)', 'Year', 'Make Variant',
                               'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
                               'Sail Area (sq ft)', 'Average cargo throughput (tons)', 'GDP (USD billion)',
                               'GDP per capita (USD)',
                               'Average ratio of total logistics costs to GDP']]

        lbl = preprocessing.LabelEncoder()
        self.data['Year'] = lbl.fit_transform(self.data['Year'].astype(int))  # Convert the column containing the incorrect data type for the prompt

        self.data.to_excel('%s/%s.xlsx' % (self.path, self.boat_name), index=False)

        msn.matrix(self.data)
        msn.matrix(self.data)

    def plot_bar(self):
        # Draw simple graphs for statistical analysis
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # Used to display negative signs normally
        plt.rcParams['axes.unicode_minus'] = False
        columns = ['Make', 'Make Variant', 'Geographic Region', 'Country/Region/State']
        for column in columns:
            df1 = pd.DataFrame(sorted(Counter(self.data[column]).items(), key=lambda x: x[1], reverse=False),
                               columns=[column, 'number'])
            df1.to_excel('%s/%s.xlsx' % (self.path, column.replace('/', '_')), index=None)
            print(df1)
            n = 10
            plt.figure(figsize=(10, 10), dpi=600)
            plt.barh(df1[column].tail(n), df1['number'].tail(n), align='center', color='green', alpha=0.6)
            plt.title("number")
            plt.xlabel("number")
            plt.ylabel(column)
            plt.savefig('%s/%s.jpg' % (self.path, column.replace('/', '_')))
            if self.output_figure:
                plt.show()

    def plot_box(self):
        columns = ['Length \n(ft)']
        for column in columns:
            plt.figure(figsize=(8, 5), dpi=600)
            plt.boxplot(self.data[column],
                        patch_artist=True,  # Request to fill the box plot with a custom colour, default white fill
                        showmeans=True,  # Show mean values as dots
                        boxprops={'color': 'black', 'facecolor': '#9999ff'},  # Set box properties, fill colour and border colour
                        flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},  # Set the outlier properties, point shape, fill colour and border colour
                        meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},  # Set the properties of the mean point, the shape of the point, the fill colour
                        medianprops={'linestyle': '--', 'color': 'orange'})  # Set the properties of the median line, the type and colour of the line
            plt.title(column.replace('\n', ''))
            plt.xlabel(column.replace('\n', ''))
            plt.ylabel('number')
            plt.savefig('%s/%s.jpg' % (self.path, column.replace('\n', '')))
            if self.output_figure:
                plt.show()

    def plot_hist(self):
        columns = ['Listing Price (USD)', 'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
                   'Sail Area (sq ft)', 'Average cargo throughput (tons)', 'GDP (USD billion)',
                   'GDP per capita (USD)', 'Average ratio of total logistics costs to GDP']
        for column in columns:
            plt.figure(figsize=(8, 5), dpi=600)
            plt.hist(self.data[column],  # Mapping data
                     bins=20,  # Specify the number of bars in the histogram as 20
                     color='steelblue',  # Specify the fill colour
                     edgecolor='k',  # Specify the boundary colour of the histogram
                     label='Histogram')  # Presenting labels for histograms
            plt.title(column)
            plt.xlabel(column)
            plt.ylabel('number')
            plt.savefig('%s/%s.jpg' % (self.path, column))
            if self.output_figure:
                plt.show()

    def plot_box_in_one_figure(self):
        plt.figure(figsize=(80, 60), dpi=75)
        for i in range(len(self.columns)):
            plt.subplot(6, 6, i + 1)
            sns.boxplot(self.data[self.columns[i]], orient='v', width=0.5)
            plt.ylabel(self.columns[i], fontsize=40)
        plt.savefig('%s/box.png' % self.path)
        if self.output_figure:
            plt.show()

    def plot_distribution_in_one_figure(self):
        # Distribution of all fields
        dist_cols = 8
        dist_rows = len(self.data.columns)
        plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))
        i = 1
        for col in self.columns:
            if col == 'Listing Price (USD)':
                continue
            ax = plt.subplot(dist_rows, dist_cols, i)
            ax = sns.kdeplot(self.data[col], color='Red', shade=True)
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax = ax.legend([self.boat_name])
            i += 1
        plt.savefig('%s/distribution.png' % self.path)
        if self.output_figure:
            plt.show()
        # for col in self.columns:
        #     if col == 'Listing Price (USD)':
        #         continue
        #     ax = plt.subplot(dist_rows, dist_cols, i)
        #     ax = sns.kdeplot(self.data[col], color='Red', shade=True)
        #     ax = sns.kdeplot(self.data2[col], color='Blue', shade=True)
        #     ax.set_xlabel(col)
        #     ax.set_ylabel('Frequency')
        #     ax = ax.legend([self.boat_name, 'data2'])
        #     i += 1
        # plt.savefig('%s/distribution.png' % self.path)
        # plt.show()

    def process_response_variable_and_plot(self):
        print(self.data['Listing Price (USD)'].describe())
        # Plug-in price distribution, with 20 above 200 and an average of under 75
        y_p = self.data
        # View the specific frequency of the predicted values
        plt.hist(y_p['Listing Price (USD)'], orientation='vertical', histtype='bar', color='red')
        plt.savefig('%s/price_bar1.png' % self.path)
        plt.show()

        n_train = self.data
        print(n_train)
        y = n_train['Listing Price (USD)']
        fig = sns.kdeplot(y, color='Red', shade=True)
        scatter_fig = fig.get_figure()
        #  The data distribution does not conform to the normal distribution, it is right-skewed data, the regression is more sensitive to the data distribution, if it does not conform to the normal distribution you need to convert the data to an approximately normal distribution
        # Transformation of the data distribution to an approximately normal distribution using the right-skewed transformation function of the logarithm
        n_train['Listing Price (USD)'] = np.log1p(n_train['Listing Price (USD)'])
        fig = sns.kdeplot(n_train['Listing Price (USD)'], color='Red', shade=True)
        scatter_fig = fig.get_figure()
        scatter_fig.savefig('%s/price_normal.png' % self.path, dpi=600)

        y = n_train['Listing Price (USD)']
        plt.figure(1)
        plt.title('Normal')
        fig = sns.distplot(y, kde=False, fit=st.norm)
        scatter_fig = fig.get_figure()
        scatter_fig.savefig('%s/price_normal.png' % self.path, dpi=600)

    def plot_hot_map(self):
        n_train = self.data
        corr = n_train.corr()
        k = len(self.columns)
        col = corr.nlargest(k, 'Listing Price (USD)')['Listing Price (USD)'].index
        cm = np.corrcoef(self.data[col].values.T)
        hm = plt.subplots(figsize=(30, 30))
        hm = sns.heatmap(self.data[col].corr(), annot=True, square=True)
        plt.savefig('%s/hot_map.png' % self.path)
        if self.output_figure:
            plt.show()

    def output_statistic_information(self):
        print(self.data.columns)
        # print(self.data.carCode.value_counts())
        # print(self.data.modelyear.value_counts())

    def clean_data(self):
        cleaned_data = self.data[['Length \n(ft)', 'Listing Price (USD)', 'Year',
                                  'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
                                  'Sail Area (sq ft)', 'Average cargo throughput (tons)',
                                  'GDP (USD billion)', 'GDP per capita (USD)',
                                  'Average ratio of total logistics costs to GDP']].copy()
        cleaned_data.isnull().sum()
        cleaned_data.dropna(inplace=True)
        cleaned_data.reset_index(inplace=True, drop=True)
        cleaned_data.isnull().sum()
        print(cleaned_data.shape)
        print(cleaned_data.dtypes)
        cleaned_data.columns = ['Length(ft)', 'Listing Price (USD)', 'Year',
                                'LWL (ft)', 'Beam (ft)', 'Draft (ft)', 'Displacement (lbs)',
                                'Sail Area (sq ft)', 'Average cargo throughput (tons)',
                                'GDP (USD billion)', 'GDP per capita (USD)',
                                'logistics costs to GDP%']
        cleaned_data.to_excel('%s/cleaned_data.xlsx' % self.path, index=None)


if __name__ == "__main__":
    boat_name = 'Monohulled Sailboats'
    boat_name = 'Catamarans'
    data_processing = DataProcessing(boat_name)
    data_processing.read_collected_paper()
    # data_processing.plot_bar()
    # data_processing.plot_box()
    # data_processing.plot_hist()
    #
    # data_processing.plot_box_in_one_figure()
    # data_processing.plot_distribution_in_one_figure()
    # data_processing.process_response_variable_and_plot()
    # data_processing.plot_hot_map()

    # data_processing.clean_data()

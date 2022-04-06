import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def day_of_year(date):
    days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    d = list(map(int, date.split("-")))
    if d[0] % 400 == 0:
        days[2] += 1
    elif d[0] % 4 == 0 and d[0] % 100 != 0:
        days[2] += 1
    for i in range(1, len(days)):
        days[i] += days[i - 1]
    return days[d[1] - 1] + d[2]

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    city_temp_df = pd.read_csv(filename, parse_dates=True)
    city_temp_df.dropna()
    #city_temp_df = pd.get_dummies(city_temp_df, prefix='Country', columns=['Country'])
    city_temp_df["Date"] = city_temp_df["Date"].apply(day_of_year)

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    print(city_temp_df)
    pd.reset_option('display.max_rows|display.max_columns|display.width')

    return city_temp_df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    city_temp_df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_temp = city_temp_df.loc[city_temp_df['Country'] == 'Israel']
    print(israel_temp)



    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
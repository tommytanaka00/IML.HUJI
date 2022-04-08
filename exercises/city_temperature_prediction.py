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

    # Check for leap years
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
    city_temp_df["DayOfYear"] = city_temp_df["Date"].apply(day_of_year)
    city_temp_df = city_temp_df[city_temp_df['Temp'] > -30]
    city_temp_df = city_temp_df[city_temp_df['Temp'] < 50]

    return city_temp_df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    city_temp_df = load_data("../datasets/City_Temperature.csv")


    # Question 2 - Exploring data for specific country
    COUNTRY = 'Israel'
    israel_temp_df = city_temp_df.loc[city_temp_df['Country'] == COUNTRY]

    scatter_plot_temp_day = px.scatter(x=israel_temp_df['DayOfYear'], y=israel_temp_df['Temp'], color=israel_temp_df['Year'],
                     labels = {'x': "Day of the year}", 'y': "Avg Temperature}"},
                     title=f"{COUNTRY} Average Daily Temperature as a function of the Day Of The Year")
    scatter_plot_temp_day.show()

    grouped_by_month = israel_temp_df.groupby(['Month'])['Temp'].agg(['std']).reset_index()
    bar_plot_std_month = px.bar(grouped_by_month, x='Month', y='std',
           labels={'Month': "Month", 'std': "Standard Deviation"},
           title=f"Standard deviation of average temperatures each month in {COUNTRY}")
    bar_plot_std_month.update_traces(marker_color='green')
    bar_plot_std_month.update_xaxes(dtick=1)
    bar_plot_std_month.show()


    # Question 3 - Exploring differences between countries
    grouped_by_country_month = city_temp_df.groupby(['Country', 'Month'], as_index=False)['Temp'].agg(['mean','std']).reset_index()
    line_plot = px.line(grouped_by_country_month, x='Month', y='mean', error_y='std', color='Country',
                        labels={'Month': "Month", 'mean':'Mean', 'std': "Standard deviation"},
    title = f"{COUNTRY} Average Daily Temperature as a function of the Day Of The Year")

    line_plot.update_xaxes(dtick=1)
    line_plot.show()


    # Question 4 - Fitting model for different values of `k`
    list_of_loss = []
    train_X, train_y, test_X, test_y = split_train_test(israel_temp_df['DayOfYear'], israel_temp_df['Temp'], 0.75)
    for k in range(1, 11):
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(train_X.to_numpy(dtype=float), train_y.to_numpy(dtype=float))
        list_of_loss.append(poly_fit.loss(test_X.to_numpy(dtype=float), test_y.to_numpy(dtype=float)))

    bar_plot_loss_degree = px.bar(x=range(1, 11), y=list_of_loss,
           labels={'x': "Value of k", 'y': "Loss"},
           title="The Loss (test error) recorded when Polynomial Fitting for each degree k")

    bar_plot_loss_degree.update_xaxes(dtick=1)
    bar_plot_loss_degree.show()


    # Question 5 - Evaluating fitted model on different countries
    best_fit_k = int(np.argmin(list_of_loss)) + 1
    poly_fit_best_k = PolynomialFitting(best_fit_k)

    dict_of_loss = {COUNTRY: np.min(list_of_loss)}
    for country in set(city_temp_df["Country"]):
        if country == COUNTRY:
            continue
        country_temp_df = city_temp_df.loc[city_temp_df['Country'] == country]
        train_X, train_y, test_X, test_y = split_train_test(country_temp_df['DayOfYear'], country_temp_df['Temp'], 0.75)
        poly_fit_best_k.fit(train_X.to_numpy(dtype=float), train_y.to_numpy(dtype=float))
        dict_of_loss[country] = (poly_fit_best_k.loss(test_X.to_numpy(dtype=float), test_y.to_numpy(dtype=float)))

    bar_plot_poly_fit = px.bar(x=dict_of_loss.keys(), y=dict_of_loss.values(),
           labels={'x': "Country", 'y': "Loss"},
           title=f"Model loss in each country when fitting with polynomial of degree {best_fit_k}, "
                 f"which was the best fit for {COUNTRY}")
    bar_plot_poly_fit.update_traces(marker_color='orange')
    bar_plot_poly_fit.show()
import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression


from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    house_prices_df = pd.read_csv(filename)

    # remove nulls and duplicates
    house_prices_df = house_prices_df.dropna().drop_duplicates()

    # change the zipcode to int (usually from string)
    house_prices_df["zipcode"] = house_prices_df["zipcode"].astype(int)




    # remove irrelevant information
    for c in ["date", "id", "lat", "long"]:
        house_prices_df = house_prices_df.drop(c, axis='columns')

    # add in only the values that make sense
    for c in ["floors", "bathrooms", "price", "sqft_living", "sqft_lot", "sqft_above", "sqft_living15", "sqft_lot15"]:
        house_prices_df = house_prices_df[house_prices_df[c] > 0]
    for c in ["sqft_basement"]:
        house_prices_df = house_prices_df[house_prices_df[c] >= 0]

    # remove houses older than 300 years old
    house_prices_df = house_prices_df[house_prices_df["yr_built"] > (house_prices_df["yr_built"].max() - 300)]

    # remove ranges that don't make sense
    house_prices_df = house_prices_df[house_prices_df["waterfront"].isin([0, 1])]
    house_prices_df = house_prices_df[house_prices_df["view"].isin(range(5))]
    house_prices_df = house_prices_df[house_prices_df["condition"].isin(range(1, 6))]
    house_prices_df = house_prices_df[house_prices_df["grade"].isin(range(1, 15))]

    # change the year renovated into whether it was recently renovated or not
    # recently renovated means that it was renovated in the past 10 years
    house_prices_df["recently_renovated"] = np.asarray(house_prices_df["yr_renovated"] >= (house_prices_df["yr_renovated"].max() - 30)).astype(int)
    house_prices_df = house_prices_df.drop("yr_renovated", axis='columns')


    house_prices_df["decade_built"] = ((round(house_prices_df["yr_built"] / 10)) * 10).astype(int)
    house_prices_df = house_prices_df.drop("yr_built", axis='columns')

    # get dummy values for zipcode
    # dummies_for_zipcode = pd.get_dummies(house_prices_df["zipcode"])
    house_prices_df = pd.get_dummies(house_prices_df, prefix='zipcode_', columns=['zipcode'])


    # Remove outliers
    house_prices_df = house_prices_df[house_prices_df["bedrooms"] < 16]
    house_prices_df = house_prices_df[house_prices_df["sqft_lot"] < 1250000]
    house_prices_df = house_prices_df[house_prices_df["sqft_lot15"] < 500000]

    #incert intercept
    # house_prices_df.insert(0, 'intercept', 1, True)

    return house_prices_df.drop("price", axis='columns'), house_prices_df.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X = X.loc[:, ~(X.columns.str.contains('^zipcode_', case=False))]
    for col in X:
        rho = np.cov(X[col], y)[0, 1] / (np.std(X[col]) * np.std(y))
        fig = px.scatter(pd.DataFrame({'x': X[col], 'y': y}), x="x", y="y", trendline="ols",
                         title=f"Correlation between {col} Values and Price"
                               f"<br> with Pearson Correlation {rho}",
                         labels={"x": f"{col} Values", "y": "Price"})
        fig.write_image(output_path + "/%s.pearson.correlation.png" % col)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    feature, response = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    if not os.path.exists("./pearson_correlations"):
        os.mkdir("./pearson_correlations")

    feature_evaluation(feature, response, "./pearson_correlations")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(feature, response, 0.75)


    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and std of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    lin_reg = LinearRegression(include_intercept=True)
    NUM_OF_ITERATIONS = 10
    PERCENT_DENSITY = 10
    NUM_OF_PERCENT = int(100 / PERCENT_DENSITY)
    list_of_loss = []
    list_of_mean = []
    list_of_std = []

    test_X_nparray = test_X.to_numpy(dtype=float)
    test_y_nparray = test_y.to_numpy(dtype=float)
    x = np.linspace(PERCENT_DENSITY, 100, PERCENT_DENSITY)
    for p in range(PERCENT_DENSITY, 101, PERCENT_DENSITY):
        list_of_loss.clear()
        for i in range(NUM_OF_ITERATIONS):
            sampled_X = train_X.sample(frac=0.01 * p, random_state=i).to_numpy(dtype=float)
            sampled_y = train_y.sample(frac=0.01 * p, random_state=i).to_numpy(dtype=float)
            lin_reg.fit(sampled_X, sampled_y)
            list_of_loss.append(lin_reg.loss(test_X_nparray, test_y_nparray))

        list_of_mean.append(np.mean(list_of_loss))
        list_of_std.append(np.std(list_of_loss))

    list_of_mean = np.array(list_of_mean)
    list_of_std = np.array(list_of_std)
    std_positive = list_of_mean + 2 * list_of_std
    std_negative = list_of_mean + (-2) * list_of_std

    mean_of_estimator = go.Figure((go.Scatter(x=x, y=list_of_mean, mode="markers+lines",name="Mean Prediction", line=dict(dash="dash"),
                          marker=dict(color="green", opacity=0.8)),

               go.Scatter(x=x, y=std_positive, fill=None, mode="lines",
                          line=dict(color="lightgrey"),
                          showlegend=False),

               go.Scatter(x=x, y=std_negative, fill='tonexty', mode="lines",
                          line=dict(color="lightgrey"),
                          showlegend=False))
                                  )
    mean_of_estimator.update_xaxes(dtick=10)
    mean_of_estimator.update_layout(
        title=r"$\text{Mean and std of Estimator Loss As Function Of Sample Size}$",
        xaxis_title=r"$\text{Sample Percentage}$",
        yaxis_title=r"$\text{MSE Loss}$")
    mean_of_estimator.show()









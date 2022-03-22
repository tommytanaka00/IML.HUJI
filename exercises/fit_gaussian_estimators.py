from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    mean = 10
    variance = 1
    sample_size = 1000

    # Question 1 - Draw samples and print fitted model
    print("Q1")
    uni = UnivariateGaussian()
    samples = np.random.normal(mean, variance, sample_size)
    uni.fit(samples)
    print("(expectation, variance) = (%f, %f)" % (uni.mu_, uni.var_))

    # Question 3 - Plotting Empirical PDF of fitted model
    sorted_samples = np.sort(samples)
    pdfs = uni.pdf(sorted_samples)
    data2 = pd.DataFrame(data={'Sorted Samples': sorted_samples, "Density": pdfs})
    px.scatter(data2, title="Graph of the Probability Density Function",
               x="Sorted Samples", y="Density", height=500).show()

    # Question 2 - Empirically showing sample mean is consistent
    abs_distance = np.empty([99])
    for i in range(1, 100):
        uni.fit(samples[:10 * i:])  # fit first 10*i samples
        abs_distance[i - 1] = abs(uni.mu_ - mean)

    data1 = pd.DataFrame(data={'Samples': range(10, 1000, 10), "Absolute distance": abs_distance})
    px.bar(data1, title="Abs distance between expectation and estimator vs Amount of samples",
           x="Samples", y="Absolute distance", height=500).show()


def test_multivariate_gaussian():
    mean_vector = np.array([0, 0, 4, 0])
    covariant_matrix = np.array([[1, 0.2, 0, 0.5],
                                 [0.2, 2, 0, 0],
                                 [0, 0, 1, 0],
                                 [0.5, 0, 0, 1]])
    sample_size = 1000
    # Question 4 - Draw samples and print fitted model
    print()
    print("Q4")
    multi = MultivariateGaussian()
    samples = np.random.multivariate_normal(mean_vector, covariant_matrix, sample_size)
    multi.fit(samples)
    print("estimated expectation = ", multi.mu_)
    print("estimated covariance matrix = ", multi.cov_)

    # Question 5 - Likelihood evaluation
    RESOLUTION = 200
    rows = np.linspace(-10, 10, RESOLUTION)
    cols = np.linspace(-10, 10, RESOLUTION)

    log_likelihood_array = np.zeros(shape=(RESOLUTION, RESOLUTION))

    mu1 = np.array([rows[0], 0, cols[0], 0])
    max_idx = [0, 0]
    max_log_likelihood = multi.log_likelihood(mu1, covariant_matrix, samples)
    for f1_i in range(RESOLUTION):
        for f2_j in range(RESOLUTION):
            mu1 = np.array([rows[f1_i], 0, cols[f2_j], 0])
            # print(mu1)
            log_likelihood = multi.log_likelihood(mu1, multi.cov_, samples)
            if log_likelihood > max_log_likelihood:
                max_idx[0] = f1_i
                max_idx[1] = f2_j
                max_log_likelihood = log_likelihood
            log_likelihood_array[f1_i][f2_j] = log_likelihood

    fig = go.Figure(go.Heatmap(x=rows, y=cols, z=log_likelihood_array),
                    layout=go.Layout(title="Loglikelihood of [f1, 0, f3, 0] and Sigma", height=600, width=600))

    fig.update_layout(xaxis_title="f1", yaxis_title="f3")
    fig.show()

    # Question 6 - Maximum likelihood
    print()
    print("Q6")
    print("Maximum values for (f1, f3) are (%f, %f)" % (round(rows[max_idx[0]],3), round(cols[max_idx[1]], 3)))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

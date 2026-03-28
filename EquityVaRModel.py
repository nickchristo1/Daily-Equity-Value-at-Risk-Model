# Nicholas Christophides     SBU ID: 113319835

import pandas as pd
import numpy as np
import scipy.stats as stats
import nlopt
from scipy.stats import binomtest
import matplotlib.pyplot as plt

# ******* Data Import and Preparation *******


# Import CSV containing data on the equities that are to be analyzed. The data imported is "AdjustedClose" data.
# Therefore, splits, mergers, and dividends are accounted for.
equity_data = pd.read_csv("/Users/nickchristo/Downloads/Spring 2025/AMS 603/Midterm/EquityData.csv")

# Find dimensions of price data for use in calculating log returns.
rows, columns = equity_data.shape

# Initialize empty numpy array to impute values of log return to.
log_returns = np.zeros((rows - 1, columns))

# Calculate Log Returns.
for i in range(columns):
    for j in range(rows - 1):
        log_returns[j, i] = np.log(equity_data.iloc[j + 1, i] / equity_data.iloc[j, i])


# ******* ARMA GARCH Preparation and Student T Optimization for VaR Calculation *******


def arma(returns, phi, theta):
    # ARMA Process with zero mean
    mu = np.mean(returns)
    r_t = returns - mu
    c = mu / (1.0 - phi)
    T = len(returns)
    eps = np.zeros(T)
    for t in range(1, T):
        forecast = phi * r_t[t - 1] + theta * eps[t - 1]
        eps[t] = r_t[t] - forecast
    # Return predicted value at time T + 1
    prediction = phi * r_t[-1] + theta * eps[-1]
    return prediction + c, eps


def garch(epsilons, psi, beta):
    # Compute the unconditional variance from the original epsilons
    unconditional_variance = np.var(epsilons)
    omega = max((unconditional_variance / (1 - psi - beta)), 1e-6)
    sigma2 = [unconditional_variance]
    for e in epsilons:
        sigma2.append(max(omega + psi * sigma2[-1] + beta * e ** 2, 1e-6))
    nu = [e / np.sqrt(s) for e, s in zip(epsilons, sigma2)]
    return sigma2, nu


# Define a function to optimize psi, beta, theta, and phi
def estimate_parameters(data):
    num_variables = 4
    initial_values = [0.01] * num_variables
    lower_bounds = [-0.99] * num_variables
    upper_bounds = [0.99] * num_variables

    # Initialize optimizer and set bounds and objective
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, num_variables)
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    opt.set_min_objective(lambda params, grad: mle_objective(data, params))
    opt.set_xtol_abs(1e-3)

    # Run optimization
    optimized_params = opt.optimize(initial_values)

    return optimized_params


# Define objective function for optimizing parameters
def mle_objective(data, params):
    phi, theta, psi, beta = params

    # Compute ARMA residuals
    next_return, eps = arma(data, phi, theta)

    # Compute GARCH variances
    sigma2s, innovations = garch(eps, psi, beta)

    # Return negative log-likelihood to minimize
    return -log_likelihood_gaussian(data, sigma2s)


# Define normal log-likelihood function for estimating parameters
def log_likelihood_gaussian(data, sigma2s):
    sigma2s = np.array(sigma2s[0:len(sigma2s) - 1])
    # innovations = np.array(innovations)
    sigma2s += 1e-12  # Small constant to avoid log(0) errors
    log_likelihood = np.sum((-1 / 2) * np.log(2 * np.pi) - (1 / 2) * np.log(1) - (data ** 2) / 2)
    return log_likelihood


# Define objective function for optimizing degrees of freedom and variance based on standardized residuals
def t_objective(params, data):
    nu_param = np.exp(params[0]) + 2  # Ensures nu > 2 (necessary for t-distribution)

    # Compute the negative log likelihood; small constant to avoid issues with log(0)
    nll = -np.sum(np.log(stats.t.pdf(data, df=nu_param, loc=0, scale=1) + 1e-12))
    return nll


# Define function to optimize nu
def fit_t(std_resid):
    # Initial Guesses
    init_param = [np.log(5 - 2)]  # , 0, np.log(1)]

    # Initialize NLOPT optimizer
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 1)

    # Use a lambda function to pass the standardized residuals into our objective.
    opt.set_min_objective(lambda params, grad: t_objective(params, std_resid))
    opt.set_xtol_rel(1e-3)

    try:
        optimized_param = opt.optimize(init_param)
    except Exception as e:
        print("Optimization failed:", e)
        optimized_param = init_param

    # Retrieve the parameters:
    nu_value = np.exp(optimized_param[0]) + 2.01
    return nu_value


# Define a function to allow QQ Plots to be created to test the fit of the model
def qq(distributions, returns):
    model_quantiles = list(sorted(f(r) for f, r in zip(distributions, returns)))
    num_dists = len(model_quantiles)
    empirical_quantiles = [float(k + 1) / float(num_dists + 1) for k in range(num_dists)]
    return list(zip(empirical_quantiles, model_quantiles))


# ******* Calculating VaR Values and Backtesting the Model *******


# Define a function that takes log returns of an asset and returns the 99% and 95% VaR levels as well as
# number of exceedances for the asset
def var95and99(data):
    # Create Arrays to store the 99% and 95% VaR, and to track if the actual loss exceeds these values
    var95 = []
    var95_exceedances = 0
    var99 = []
    var99_exceedances = 0

    # Array to store the t-distributions
    distributions = []

    # Rolling Window ARMA-GARCH Estimation Parameters
    window_size = 250  # 250 periods in each rolling window
    total_periods = len(data)
    num_forecasts = total_periods - window_size

    # Loop over each window, receiving estimations for each period following the window
    for t in range(0, num_forecasts):
        window_data = data[t:t + window_size]

        # Find estimates for phi, theta, psi, and beta
        phi, theta, psi, beta = estimate_parameters(window_data)

        # Run the ARMA(1, 1) function
        next_return, epsilons = arma(window_data, phi, theta)
        epsilons = np.array(epsilons)

        # Run the GARCH(1, 1) function
        sigma2s, innovations = garch(epsilons, psi, beta)

        # Use the ARMA model innovations to optimize nu
        nu = fit_t(innovations)
        scale_param = np.sqrt(sigma2s[-1]) / np.sqrt(nu / (nu - 2))

        # Calculate the VaR values under fitted distribution
        var95.append(stats.t.ppf(0.05, nu, next_return, scale_param))
        var99.append(stats.t.ppf(0.01, nu, next_return, scale_param))

        # Record total exceedances of VaR
        if var95[t] > data[t + window_size]:
            var95_exceedances += 1
        if var99[t] > data[t + window_size]:
            var99_exceedances += 1

        # Create and append a callable t-distribution using the calculated parameters
        # Default arguments in lambda to capture the current values.
        t_dist_callable = lambda x, df=nu, loc=next_return, scale=scale_param: stats.t.cdf(x, df, loc, scale)
        distributions.append(t_dist_callable)

    return var95, var95_exceedances, var99, var99_exceedances, distributions


# ******* Results on Assets *******

# Amount of Forecasts Run
n = len(log_returns[:, 0]) - 250

# Launch Message
print("---------- 95% and 99% VaR Exceedances and Binomial Test for Exceedances to Test Fit of Model ----------\n"
      "Amount of of Forecasts Made for Each Asset: ", n)

# AAPL Summary
AAPL_log_returns = log_returns[:, 0]
AAPL_VaR95, AAPL_VaR95_Exc, AAPL_VaR99, AAPL_VaR99_Exc, AAPL_Distributions = var95and99(AAPL_log_returns)
print("\n\n-----AAPL Summary------\n95% VaR Exceedances:\n", AAPL_VaR95_Exc, "\n99% VaR Exceedances:\n",
      AAPL_VaR99_Exc, "\n")
print(f"AAPL 95% VaR p-value: {binomtest(AAPL_VaR95_Exc, n, 0.05, alternative='two-sided')}")
print(f"AAPL 99% VaR p-value: {binomtest(AAPL_VaR99_Exc, n, 0.01, alternative='two-sided')}")

# MSFT Summary
MSFT_log_returns = log_returns[:, 1]
MSFT_VaR95, MSFT_VaR95_Exc, MSFT_VaR99, MSFT_VaR99_Exc, MSFT_Distributions = var95and99(MSFT_log_returns)
print("\n\n-----MSFT Summary------\n95% VaR Exceedances:\n", MSFT_VaR95_Exc, "\n99% VaR Exceedances:\n", MSFT_VaR99_Exc,
      "\n")
print(f"MSFT 95% VaR p-value: {binomtest(MSFT_VaR95_Exc, n, 0.05, alternative='two-sided')}")
print(f"MSFT 99% VaR p-value: {binomtest(MSFT_VaR99_Exc, n, 0.01, alternative='two-sided')}")

# AMZN Summary
AMZN_log_returns = log_returns[:, 2]
AMZN_VaR95, AMZN_VaR95_Exc, AMZN_VaR99, AMZN_VaR99_Exc, AMZN_Distributions = var95and99(AMZN_log_returns)
print("\n\n-----AMZN Summary------\n95% VaR Exceedances:\n", AMZN_VaR95_Exc, "\n99% VaR Exceedances:\n", AMZN_VaR99_Exc,
      "\n")
print(f"AMZN 95% VaR p-value: {binomtest(AMZN_VaR95_Exc, n, 0.05, alternative='two-sided')}")
print(f"AMZN 99% VaR p-value: {binomtest(AMZN_VaR99_Exc, n, 0.01, alternative='two-sided')}")

# META Summary
META_log_returns = log_returns[:, 3]
META_VaR95, META_VaR95_Exc, META_VaR99, META_VaR99_Exc, META_Distributions = var95and99(META_log_returns)
print("\n\n-----META Summary------\n95% VaR Exceedances:\n", META_VaR95_Exc, "\n99% VaR Exceedances:\n", META_VaR99_Exc,
      "\n")
print(f"META 95% VaR p-value: {binomtest(META_VaR95_Exc, n, 0.05, alternative='two-sided')}")
print(f"META 99% VaR p-value: {binomtest(META_VaR99_Exc, n, 0.01, alternative='two-sided')}")

# V Summary
V_log_returns = log_returns[:, 4]
V_VaR95, V_VaR95_Exc, V_VaR99, V_VaR99_Exc, V_Distributions = var95and99(V_log_returns)
print("\n\n-----V Summary------\n95% VaR Exceedances:\n", V_VaR95_Exc, "\n99% VaR Exceedances:\n", V_VaR99_Exc,
      "\n")
print(f"V 95% VaR p-value: {binomtest(V_VaR95_Exc, n, 0.05, alternative='two-sided')}")
print(f"V 99% VaR p-value: {binomtest(V_VaR99_Exc, n, 0.01, alternative='two-sided')}")

# S&P 500 Summary
SP500_log_returns = log_returns[:, 5]
SP500_VaR95, SP500_VaR95_Exc, SP500_VaR99, SP500_VaR99_Exc, SP500_Distributions = var95and99(SP500_log_returns)
print("\n\n-----S&P 500 Summary-----\n95% VaR Exceedances:\n", SP500_VaR95_Exc, "\n99% VaR Exceedances:\n",
      SP500_VaR99_Exc, "\n")
print(f"S&P 500 95% VaR p-value: {binomtest(SP500_VaR95_Exc, n, 0.05, alternative='two-sided')}")
print(f"S&P 500 99% VaR p-value: {binomtest(SP500_VaR99_Exc, n, 0.01, alternative='two-sided')}")

# Create points for plotting for each asset
AAPL_qq_points = qq(AAPL_Distributions, AAPL_log_returns[250:])
MSFT_qq_points = qq(MSFT_Distributions, MSFT_log_returns[250:])
AMZN_qq_points = qq(AMZN_Distributions, AMZN_log_returns[250:])
META_qq_points = qq(META_Distributions, META_log_returns[250:])
V_qq_points = qq(V_Distributions, V_log_returns[250:])
SP500_qq_points = qq(SP500_Distributions, SP500_log_returns[250:])

# Plot the QQ plot
plt.figure(figsize=(8, 6))


# Plot the results for the equities
def plot_qq(stock_name, qq_points):
    plt.scatter(*zip(*qq_points), color='blue', label="Data Points")  # Plot data points
    min_val = min(map(min, zip(*qq_points)))  # Minimum across both axes
    max_val = max(map(max, zip(*qq_points)))  # Maximum across both axes
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='dashed', label="y = x line")
    plt.ylabel("Model Quantiles")
    plt.xlabel("Empirical (Uniform) Quantiles")
    plt.title(f"QQ Plot for {stock_name}")
    plt.legend()
    plt.grid()
    plt.show()


# List of stocks and corresponding QQ points
stocks = {
    "AAPL": AAPL_qq_points,
    "MSFT": MSFT_qq_points,
    "AMZN": AMZN_qq_points,
    "META": META_qq_points,
    "V": V_qq_points,
    "S&P 500": SP500_qq_points
}

# Generate QQ plots for all stocks
for stock, points in stocks.items():
    plot_qq(stock, points)

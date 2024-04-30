import h5py as h5
import numpy as np
import pandas as pd

from functools import partial

try:
    from scipy.stats import median_absolute_deviation as scimad
except ImportError:
    from scipy.stats import median_abs_deviation as scimad
from tqdm import tqdm


def weighted_linear_regression(X, Y, W=None):
    """
    Parameters
    -----------
    X: (n,) numpy array or list
    Y: (n,) numpy array or list
    W: default to None, (n,) numpy array or list
    Returns
    --------
    best_slope: scalar float,
        Best slope from the least square formula
    best_intercept: scalar float,
        Best intercept from the least square formula
    std_err: scalar float,
        Error on the slope
    """
    X = np.asarray(X)
    if W is None:
        W = np.ones(X.size)
    W_sum = W.sum()
    x_mean = np.sum(W * X) / W_sum
    y_mean = np.sum(W * Y) / W_sum
    x_var = np.sum(W * (X - x_mean) ** 2)
    xy_cov = np.sum(W * (X - x_mean) * (Y - y_mean))
    best_slope = xy_cov / x_var
    best_intercept = y_mean - best_slope * x_mean
    # errors in best_slope and best_intercept
    estimate = best_intercept + best_slope * X
    s2 = sum(estimate - Y) ** 2 / (Y.size - 2)
    s2_intercept = s2 * (1.0 / X.size + x_mean**2 / ((X.size - 1) * x_var))
    s2_slope = s2 * (1.0 / ((X.size - 1) * x_var))
    return best_slope, best_intercept, np.sqrt(s2_slope)


# ----------------------------------------------------------
#           Fit rate ratio vs phase
# ----------------------------------------------------------

def cosine_rate_ratio(x, alpha, phi):
    return 1. + alpha * np.cos(x - phi)

def fit_rate_ratio_vs_phase_bootstrap(x, y, y_err, num_bootstraps=10, objective="l2"):
    """ """
    from scipy.optimize import minimize

    deg2rad = np.pi / 180.0
    x_ = x * deg2rad

    if objective == "l2":
        # l2-norm
        loss = lambda p, obs: np.sum(((1.0 + p[0] * np.cos(x_ - p[1])) - obs) ** 2)
    elif objective == "l1":
        # l1-norm
        loss = lambda p, obs: np.sum(np.abs((1.0 + p[0] * np.cos(x_ - p[1])) - obs))
    elif objective == "negative-log-likelihood":
        # negative log-likelihood
        loss = lambda p, obs: -np.sum(obs * np.log(1.0 + p[0] * np.cos(x_ - p[1])))
    first_guess = (0.0, 0.0)
    bounds = [(0.0, 1.0), (-np.pi, np.pi)]

    inverted_alpha = np.zeros(num_bootstraps, dtype=np.float32)
    inverted_phi = np.zeros(num_bootstraps, dtype=np.float32)
    n = 0
    while n < num_bootstraps:
        # generate random sample assuming that each bin [i] of the histogram
        # is normally distributed with mean y[i] and std y_err[i]
        y_b = np.random.normal(loc=0.0, scale=1.0, size=len(y))
        # noisy y
        y_b = y_b * y_err + y
        # don't allow negative values (impossible)
        y_b = np.maximum(y_b, 0.0)
        # normalized noisy y
        y_b = y_b / np.mean(y_b)
        # first_guess = (0.02 * np.random.random(), 2. * np.pi * np.random.random() - np.pi)
        first_guess = (
            0.05 * np.random.random(),
            np.pi * np.random.random() - np.pi / 2.0,
        )
        optimization_results = minimize(
            loss,
            first_guess,
            args=(y_b),
            bounds=bounds,  # jac="3-point"
        )
        inverted_alpha[n] = optimization_results.x[0]
        inverted_phi[n] = optimization_results.x[1]
        if inverted_alpha[n] == 0:
            continue
        n += 1
    inverted_cos_phi = np.cos(inverted_phi)
    inverted_sin_phi = np.sin(inverted_phi)
    mean_phi = np.arctan2(np.mean(inverted_sin_phi), np.mean(inverted_cos_phi))
    # diff_phi = np.minimum(
    #     np.abs(inverted_phi[:, np.newaxis] - inverted_phi[np.newaxis, :]).flatten(),
    #     np.abs(
    #         2.0 * np.pi + inverted_phi[:, np.newaxis] - inverted_phi[np.newaxis, :]
    #     ).flatten(),
    #     np.abs(
    #         inverted_phi[:, np.newaxis] - inverted_phi[np.newaxis, :] - 2.0 * np.pi
    #     ).flatten(),
    # )
    # mean_alpha = np.sum(cost * inverted_alpha)
    # mean_phi = np.arctan2(
    #     np.sum(cost*inverted_sin_phi), np.sum(cost * inverted_cos_phi)
    #     )
    diff = inverted_phi - mean_phi
    diff_phi = np.min(
        np.stack(
            [
                np.abs(diff),
                np.abs(2.0 * np.pi + diff),
                np.abs(diff - 2.0 * np.pi),
            ],
            axis=1,
        ),
        axis=1,
    )
    model_parameters = {
            "alpha": np.mean(inverted_alpha),
            "phi": mean_phi,
            }
    model_errors = {
            "alpha_err": np.std(inverted_alpha),
            "phi_err": np.mean(diff_phi)
            }
    model_func = partial(
            cosine_rate_ratio,
            alpha=model_parameters["alpha"],
            phi=model_parameters["phi"],
            )
    model = {
            "parameters": model_parameters,
            "errors": model_errors,
            "func": model_func
            }
    return model

# ----------------------------------------------------------
#           Fit rate ratio vs stress
# ----------------------------------------------------------

def linear_func(x, a, b):
    return b + a * x

def fit_rate_ratio_vs_stress_linear(x, y, yerr):
    """Least squares solution for linear regression.

    This routine uses scipy.stats's linregress function.
    """
    from scipy.stats import linregress, t

    slope, intercept, r, p, se = linregress(x, y)

    model_parameters = {
            "slope": slope,
            }
    model_errors = {
            "slope_err": se,
            }
    model_func = partial(
            linear_func,
            a=slope,
            b=intercept,
            )
    model = {
            "parameters": model_parameters,
            "errors": model_errors,
            "func": model_func
            }
    return model

def rate_state(x, Asig_Pa):
    """ """
    return np.exp(x / Asig_Pa)

def fit_rate_ratio_vs_stress_rate_state_bootstrap(x, y, y_err, num_bootstraps=100):
    """
    Fit a rate-state model to data with bootstrapping.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable representing data.

    y : numpy.ndarray
        Dependent variable representing a corresponding response.

    y_err : numpy.ndarray
        Error values associated with the dependent variable y.

    num_bootstraps : int, optional
        The number of bootstrap resamples to generate (default is 100).

    Returns:
    --------
    float
        The median of the optimized parameter for the rate-state model from bootstrapping.

    float
        The robust uncertainty estimate for the parameter.

    Example:
    --------
    >>> import numpy as np

    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> y_err = np.array([0.1, 0.2, 0.15, 0.3, 0.2])

    >>> param_median, param_uncertainty = fit_rate_state_bootstrap(x, y, y_err, num_bootstraps=1000)
    >>> print(param_median, param_uncertainty)
    (1000.0, 100.0)
    """

    from scipy.optimize import minimize_scalar

    bounds = (1.0, 3.0e5)
    W = 1.0 / y_err
    W /= W.sum()
    inverted_Asig_Pa = np.zeros(num_bootstraps, dtype=np.float32)
    for n in range(num_bootstraps):
        # generate random sample assuming that each bin [i] of the histogram
        # is normally distributed with mean y[i] and std y_err[i]
        y_b = np.random.normal(loc=0.0, scale=1.0, size=len(y))
        # noisy y
        y_b = y_b * y_err + y
        # don't allow negative values (impossible)
        y_b = np.maximum(y_b, 0.0)
        y_b = y_b / np.mean(y_b)
        loss = lambda Asig_Pa: np.sum(W * (y_b - rate_state(x, Asig_Pa)) ** 2)
        results = minimize_scalar(loss, bounds=bounds, method="bounded")
        inverted_Asig_Pa[n] = results.x

    model_parameters = {
            "asig_kPa": np.median(inverted_Asig_Pa) / 1000.,
            }
    model_errors = {
            "asig_kPa_err": 1.48 * scimad(inverted_Asig_Pa) / 1000.,
            }
    model_func = partial(
            rate_state,
            Asig_Pa=model_parameters["asig_kPa"] * 1000.,
            )
    model = {
            "parameters": model_parameters,
            "errors": model_errors,
            "func": model_func
            }


    return model


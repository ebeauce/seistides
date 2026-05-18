import h5py as h5
import numpy as np
import pandas as pd
import warnings

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


def _modulo(x, rad=True):
    x = (x + np.pi) % (2.0 * np.pi) - np.pi
    return x


def cosine_rate_ratio(x, alpha, phi, C=1.0, log=False):
    if log:
        return np.log(C * (1.0 + alpha * np.cos(x - phi)))
    else:
        return C * (1.0 + alpha * np.cos(x - phi))


def exp_rate_ratio(x, alpha, phi, C=1.0, log=False):
    phi = _modulo(phi)
    if log and C == 1:
        return alpha * np.cos(x - phi)
    elif log:
        return np.log(C) + alpha * np.cos(x - phi)
    else:
        return C * np.exp(alpha * np.cos(x - phi))


def _check_if_at_bound(p, bounds):
    if len(np.atleast_1d(p)) > 1:
        at_bound = np.any(
            [(p[i] == bounds[i][0]) | (p[i] == bounds[i][1]) for i in range(len(p))]
        )
    else:
        at_bound = (p == bounds[0]) | (p == bounds[1])
    return at_bound


def _analytical_cosine_fit(r, phase):
    """Analytical fit of cosine model based on Fourier transform.

    Parameters
    ----------
    r : array-like
        Relative rate of seismicity.
    phase : array-like
        Tidal phases in radians.

    Returns
    -------
    alpha_0 : float
        Modulus of the 2pi harmonic component.
    phi_0 : float
        Phase of the 2pi harmonic component.
    """
    # harmonic component at frequency of 1 tidal cycle
    dphi = phase[1] - phase[0]
    R = np.sum((r - 1.0) * np.exp(1.0j * phase)) * dphi
    phi_0 = np.angle(R)
    alpha_0 = np.abs(R) / (2.0 * np.pi)  # because FT is defined in terms of phase
    return alpha_0, phi_0


def fit_relative_rate_vs_phase(
    x,
    y,
    y_err,
    num_resamplings=10,
    objective="l2",
    model="cosine",
    invert_norm=False,
    y_err_min=0.0,
    use_err_for_weights=False,
):
    """Fit the relative rate of seismicity as a function of phase with bootstrap uncertainties.

    Parameters
    ----------
    x : array-like
        `num_phases` list or array of tidal phases, in degrees.
    y : array-like
        `num_phases` list or array of relative rate of seismicity.
    y_err : array-like
        `num_phases` list or array of uncertainties on relative rate of seismicity.
    num_resamplings : int, optional
        Number of random samples drawn from N(mean=y, std=y_err) used to propagate
        measurement uncertainties into modeling uncertainties. Defaults to 10.
    objective : str, optional
        Objective function. Either of 'l2', 'l1'. Defaults to `l2`.
    model : str, optional
        Model of relative rate of seismicity. Either of `cosine` or `exp`.
        Defaults to `cosine`.
        - `cosine`: model(x; alpha, phi_0) = 1 + alpha x cos(phi - phi_0)
        - `exp`: model(x; alpha, phi_0) = exp(alpha x cos(phi - phi_0) )
    invert_norm : bool, optional
        If True, the model takes a third parameter, which is an amplitude scaling factor.
        Defaults to False (and should probably be kept False).
    y_err_min : float or numpy.ndarray, optional
        Errors are clipped such that `y_err >= y_err_min`. Defaults to 0.
    use_err_for_weights : bool, optional
        If True, the tidal phases contribute to the objective function inversely
        proportionally to the error in the relative rate of seismicity, that is,
        to `y_err`. Errors are first clipped between the 2.5th and 97.5th percentile
        for numerical stability. Defaults to False.

    Returns
    -------
    dict
        Dictionary with following fields:
        - 'parameters': The values of the inverted parameters.
        - 'errors': The uncertainties on the inverted parameters.
        - 'func': An operator func(x) that returns the values of 
                  the modulation model at phases `x`.
    """
    from scipy.optimize import minimize

    assert model in {"cosine", "exp"}, "model should be either of 'cosine' or 'exp'"
    valid_objective = {"l1", "l2"}
    if objective not in valid_objective:
        raise ValueError(f"Invalid 'objective'. Should be one of {valid_objective}")

    deg2rad = np.pi / 180.0
    x_ = x * deg2rad

    if model == "cosine":
        _model = cosine_rate_ratio
        bounds = [(0.0, 1.0), (-np.pi, np.pi)]
    elif model == "exp":
        _model = exp_rate_ratio
        bounds = [(0.0, 10.0), (-np.pi, np.pi)]
    if invert_norm:
        bounds = bounds + [(0.0, 10.0)]

    y_err = np.maximum(y_err, y_err_min)

    if use_err_for_weights:
        weights = np.clip(
            y_err, a_min=np.percentile(y_err, 2.5), a_max=np.percentile(y_err, 97.5)
        )
        weights = 1.0 / weights
    else:
        weights = np.ones(len(y_err))
    weights /= weights.sum()

    # tried a covariance formalism but did not work
    #cc = np.eye(len(x))
    #for i in range(len(x)):
    #    for k in range(bin_extension):
    #        cc[i, i-k] = 1. - k / bin_extension
    #        cc[i, (i+k)%len(x)] = 1. - k / bin_extension
    #cov = np.diag(
    #        #np.clip(y_err, a_min=np.percentile(y_err, 2.5), a_max=np.percentile(y_err, 97.5))**2
    #        np.clip(y_err, a_min=np.percentile(y_err, 25.), a_max=np.percentile(y_err, 75.))**2
    #        ) @ cc
    #cov_inv = np.linalg.pinv(cov)

    if objective == "l2":
        # l2-norm
        #loss = lambda p, obs: np.sum(weights**2 * (_model(x_, *p) - obs) ** 2)
        loss = lambda p, obs: np.sum(weights * (_model(x_, *p) - obs) ** 2)
        #def loss(p, obs):
        #    res = _model(x_, *p) - obs
        #    return (res[None, :] @ cov_inv @ res[:, None])[0, 0]
    elif objective == "l1":
        # l1-norm
        loss = lambda p, obs: np.sum(weights * np.abs(_model(x_, *p) - obs))
    # elif objective == "negative-log-likelihood":
    #    # negative log-likelihood
    #    loss = lambda p, obs: -np.sum(weights * obs * np.log(_model(x_, *p)))

    inverted_alpha = np.zeros(num_resamplings + 1, dtype=np.float32)
    inverted_phi = np.zeros(num_resamplings + 1, dtype=np.float32)
    if invert_norm:
        inverted_C = np.zeros(num_resamplings + 1, dtype=np.float32)

    method = "L-BFGS-B"
    # --------------------------------
    #      fit original measurement
    success = False
    first_guess = _analytical_cosine_fit(y, x_)
    # first_guess = (0.01, 0.)
    while not success:
        if invert_norm:
            first_guess = first_guess + (np.mean(y),)
        optimization_results = minimize(
            loss,
            first_guess,
            args=(y),
            # method=method,
            bounds=bounds,  # jac="3-point"
        )
        if _check_if_at_bound(optimization_results.x, bounds):
            success = False
        else:
            success = optimization_results.success
        first_guess = (
            0.01 + 0.05 * np.random.random(),
            np.random.uniform(low=-np.pi, high=np.pi),
        )

    inverted_alpha[0] = optimization_results.x[0]
    inverted_phi[0] = optimization_results.x[1]
    if invert_norm:
        inverted_C[0] = optimization_results.x[2]

    first_guess = optimization_results.x

    # --------------------------------
    #      fit perturbed measurements
    n = 0
    # since the process is closer to a log-normal process,
    # estimate std of log-normal process
    y_log_err = np.sqrt(np.log(1. + y_err**2))
    y_log = np.log(y)
    while n < num_resamplings:
        y_b = np.exp(y_log + np.random.normal(loc=0., scale=y_log_err))
        optimization_results = minimize(
            loss,
            first_guess,
            args=(y_b),
            # method=method,
            bounds=bounds,  # jac="3-point"
        )
        if _check_if_at_bound(optimization_results.x, bounds):
            continue
        if optimization_results.success == False:
            continue
        inverted_alpha[1 + n] = optimization_results.x[0]
        inverted_phi[1 + n] = optimization_results.x[1]
        if invert_norm:
            inverted_C[1 + n] = optimization_results.x[2]
        n += 1
    inverted_cos_phi = np.cos(inverted_phi)
    inverted_sin_phi = np.sin(inverted_phi)
    mean_phi = np.arctan2(np.mean(inverted_sin_phi), np.mean(inverted_cos_phi))
    diff = inverted_phi - mean_phi
    # diff_phi = np.min(
    #    np.stack(
    #        [
    #            np.abs(diff),
    #            np.abs(2.0 * np.pi + diff),
    #            np.abs(diff - 2.0 * np.pi),
    #        ],
    #        axis=1,
    #    ),
    #    axis=1,
    # )
    diff_phi2 = np.min(
        np.stack(
            [
                diff**2,
                (2.0 * np.pi + diff) ** 2,
                (diff - 2.0 * np.pi) ** 2,
            ],
            axis=1,
        ),
        axis=1,
    )

    model_parameters = {
        # "alpha": np.mean(inverted_alpha),
        # "phi": mean_phi,
        "alpha": inverted_alpha[0],
        "phi": inverted_phi[0],
    }
    model_errors = {
        "alpha_err": np.std(inverted_alpha),
        # "phi_err": np.mean(diff_phi),
        "phi_err": np.sqrt(np.mean(diff_phi2)),
    }
    if invert_norm:
        model_parameters["C"] = np.mean(inverted_C)
        model_errors["C_err"] = np.std(inverted_C)
    model_func = partial(
        _model,
        alpha=model_parameters["alpha"],
        phi=model_parameters["phi"],
        C=model_parameters["C"] if invert_norm else 1.0,
    )
    model = {"parameters": model_parameters, "errors": model_errors, "func": model_func}
    return model


# ----------------------------------------------------------
#           Fit rate ratio vs stress
# ----------------------------------------------------------


def linear_func(x, a, b):
    return b + a * x


def rate_state_friction(x, Asig_Pa, ln=False):
    """ """
    if ln:
        return x / Asig_Pa
    else:
        return np.exp(x / Asig_Pa)


def fit_relative_rate_vs_stress_linear(
    x,
    y,
    y_err,
    y_err_min=0.0,
    use_err_for_weights=False,
):
    """
    Fit the relative rate of seismicity as a function of stress amplitude with empirical model.

    Parameters:
    -----------
    x : array-like
        `num_phases` list or array of tidal stresses, in Pascal.
    y : array-like
        `num_phases` list or array of relative rate of seismicity.
    y_err : array-like
        `num_phases` list or array of uncertainties on relative rate of seismicity.
    y_err_min : float or numpy.ndarray, optional
        Errors are clipped such that `y_err >= y_err_min`. Defaults to 0.
    use_err_for_weights : bool, optional
        If True, the tidal phases contribute to the objective function inversely
        proportionally to the error in the relative rate of seismicity, that is,
        to `y_err`. Errors are first clipped between the 2.5th and 97.5th percentile
        for numerical stability. Defaults to False.

    Returns:
    --------
    dict
        Dictionary with following fields:
        - 'parameters': The values of the inverted parameters.
        - 'errors': The uncertainties on the inverted parameters.
        - 'func': An operator func(x) that returns the values of 
                  the modulation model at stresses `x`.
    """
    from scipy.stats import linregress, t

    y_err = np.maximum(y_err, y_err_min)
    if use_err_for_weights:
        weights = np.clip(
            y_err, a_min=np.percentile(y_err, 2.5), a_max=np.percentile(y_err, 97.5)
        )
        weights = 1.0 / weights
    else:
        weights = np.ones(len(y_err))
    weights /= weights.sum()

    slope, intercept, se = weighted_linear_regression(x, y, W=weights)

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
    model = {"parameters": model_parameters, "errors": model_errors, "func": model_func}
    return model


def fit_relative_rate_vs_stress_rate_state(
    x,
    y,
    y_err,
    y_err_min=0.0,
    use_err_for_weights=False,
    num_resamplings=10,
):
    """
    Fit the relative rate of seismicity as a function of stress amplitude with bootstrap uncertainties.

    Parameters:
    -----------
    x : array-like
        `num_phases` list or array of tidal stresses, in Pascal.
    y : array-like
        `num_phases` list or array of relative rate of seismicity.
    y_err : array-like
        `num_phases` list or array of uncertainties on relative rate of seismicity.
    num_resamplings : int, optional
        Number of random samples drawn from N(mean=y, std=y_err) used to propagate
        measurement uncertainties into modeling uncertainties. Defaults to 10.
    y_err_min : float or numpy.ndarray, optional
        Errors are clipped such that `y_err >= y_err_min`. Defaults to 0.
    use_err_for_weights : bool, optional
        If True, the tidal phases contribute to the objective function inversely
        proportionally to the error in the relative rate of seismicity, that is,
        to `y_err`. Errors are first clipped between the 2.5th and 97.5th percentile
        for numerical stability. Defaults to False.

    Returns:
    --------
    dict
        Dictionary with following fields:
        - 'parameters': The values of the inverted parameters.
        - 'errors': The uncertainties on the inverted parameters.
        - 'func': An operator func(x) that returns the values of 
                  the modulation model at stresses `x`.
    """
    from scipy.optimize import minimize_scalar
    from scipy.stats import linregress

    bounds = (1.0, 1.0e7)

    y_err = np.maximum(y_err, y_err_min)
    if use_err_for_weights:
        weights = np.clip(
            y_err, a_min=np.percentile(y_err, 2.5), a_max=np.percentile(y_err, 97.5)
        )
        weights = 1.0 / weights
    else:
        weights = np.ones(len(y_err))
    weights /= weights.sum()

    inverted_Asig_Pa = np.zeros(num_resamplings, dtype=np.float32)
    # since the process is closer to a log-normal process,
    # estimate std of log-normal process
    y_log_err = np.sqrt(np.log(1. + y_err**2))
    y_log = np.log(y)
    for n in range(num_resamplings):
        y_log_b = y_log + np.random.normal(loc=0., scale=y_log_err)
        # convert data to log and fit linear function
        # note: inverting for a non-trivial intercept corrects
        # for possible errors when finding the reference rate
        # at sigma=0   :)
        # slope, intercept, r, p, se = linregress(x, np.log(y_b))
        slope, intercept, se = weighted_linear_regression(x, y_log_b, W=weights)
        inverted_Asig_Pa[n] = 1.0 / slope

    model_parameters = {
        # "asig_kPa": np.median(inverted_Asig_Pa) / 1000.0,
        "asig_kPa": np.mean(inverted_Asig_Pa)
        / 1000.0,
    }
    model_errors = {
        "asig_kPa_err": 1.48 * scimad(inverted_Asig_Pa) / 1000.0,
        "asig_kPa_2.5th": np.percentile(inverted_Asig_Pa, 2.5) / 1000.0,
        "asig_kPa_97.5th": np.percentile(inverted_Asig_Pa, 97.5) / 1000.0,
    }
    model_func = partial(
        rate_state_friction,
        Asig_Pa=model_parameters["asig_kPa"] * 1000.0,
    )
    model = {"parameters": model_parameters, "errors": model_errors, "func": model_func}

    return model

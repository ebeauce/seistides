import h5py as h5
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from functools import partial
from scipy.signal import wiener

try:
    from scipy.stats import median_absolute_deviation as scimad
except ImportError:
    from scipy.stats import median_abs_deviation as scimad
from tqdm import tqdm


def estimate_rate_forcingtime_bins(
    cat,
    forcingtime_bins,
    forcing_bin_edges,
    num_bootstraps=100,
    num_std_cutoff=0.0,
):
    num_bins = len(forcing_bin_edges) - 1

    rate_vs_forcing = np.zeros(num_bins)
    rate_vs_forcing_std = np.zeros(num_bins)
    count_vs_forcing = np.zeros(num_bins)
    count_vs_forcing_std = np.zeros(num_bins)
    duration_vs_forcing = np.zeros(num_bins)
    duration_vs_forcing_std = np.zeros(num_bins)

    if num_std_cutoff > 0.0:
        if "forcingtime_bin_rate" in forcingtime_bins:
            r_ = forcingtime_bins["forcingtime_bin_rate"]
        else:
            r_ = (
                forcingtime_bins["forcingtime_bin_count"]
                / forcingtime_bins["forcingtime_bin_duration_sec"]
            )
        mean_ = r_.mean()
        std_ = r_.std()
        anomalous = r_ > mean_ + num_std_cutoff * std_
        forcingtime_bins = forcingtime_bins[~anomalous]

    for i in range(num_bins):
        bin_edge_idx = i + 1
        selected_forcingtime_bin_indexes = np.where(
            (forcingtime_bins["forcing_leftbin_membership"] == bin_edge_idx)
            # & ~anomalous
        )[0]
        # if num_std_cutoff > 0.0:
        #    mean_ = forcingtime_bins.iloc[selected_forcingtime_bin_indexes][
        #        "forcingtime_bin_count"
        #    ].mean()
        #    std_ = forcingtime_bins.iloc[selected_forcingtime_bin_indexes][
        #        "forcingtime_bin_count"
        #    ].std()
        #    anomalous = (
        #        forcingtime_bins.iloc[selected_forcingtime_bin_indexes][
        #            "forcingtime_bin_count"
        #        ]
        #        > mean_ + num_std_cutoff * std_
        #    )
        #    selected_forcingtime_bin_indexes = selected_forcingtime_bin_indexes[
        #        ~anomalous
        #    ]
        rates = np.zeros(num_bootstraps)
        counts = np.zeros(num_bootstraps)
        durations = np.zeros(num_bootstraps)
        for j in range(num_bootstraps):
            indexes_b = np.random.choice(
                selected_forcingtime_bin_indexes,
                len(selected_forcingtime_bin_indexes),
                replace=True,
            )
            # indexes_b = np.random.choice(
            #        selected_forcingtime_bin_indexes,
            #        len(selected_forcingtime_bin_indexes) // 2,
            #        replace=False
            #        )

            rates[j] = (
                forcingtime_bins.iloc[indexes_b]["forcingtime_bin_count"].sum()
                / forcingtime_bins.iloc[indexes_b]["forcingtime_bin_duration_sec"].sum()
            )
            counts[j] = forcingtime_bins.iloc[indexes_b]["forcingtime_bin_count"].sum()
            durations[j] = forcingtime_bins.iloc[indexes_b][
                "forcingtime_bin_duration_sec"
            ].sum()
        variance_stabilized_r = np.sqrt(rates)
        robust_mean_estimate = np.median(
            variance_stabilized_r[variance_stabilized_r > 0.0]
        )
        robust_mean_rate = robust_mean_estimate**2
        rate_vs_forcing[i] = robust_mean_rate
        # rate_vs_forcing[i] = np.mean(rates)
        rate_vs_forcing_std[i] = np.std(rates)
        count_vs_forcing[i] = np.median(counts)
        count_vs_forcing_std[i] = np.std(counts)
        duration_vs_forcing[i] = np.median(durations)
        duration_vs_forcing_std[i] = np.std(durations)

    average_rate = (
        forcingtime_bins["forcingtime_bin_count"].sum()
        / forcingtime_bins["forcingtime_bin_duration_sec"].sum()
    )
    if average_rate == 0.0:
        average_rate = 1.0

    # can re-work to return expected_rate and observed_rate
    output = {
        "relative_rate": rate_vs_forcing / average_rate,
        "relative_rate_err": rate_vs_forcing_std / average_rate,
        "observed_rate": rate_vs_forcing,
        "observed_rate_err": rate_vs_forcing_std,
        "event_count": count_vs_forcing,
        "event_count_err": count_vs_forcing_std,
        "bin_duration": duration_vs_forcing,
        "bin_duration_err": duration_vs_forcing_std,
    }
    return output


def estimate_rate_forcing_bins(
    cat,
    forcing,
    forcing_bin_edges,
    clip_relative_rate_above=None,
    clip_relative_rate_below=None,
):
    """Count number of earthquakes in phase bins.

    Parameters
    -----------
    cat : `pandas.DataFrame`
        Earthquake catalog.
    forcing : `pandas.Series`
        Tidal forcing time series.
    forcing_bin_edges : array-like
        Edges of the forcing bins.

    Returns
    --------
    seismicity_vs_phase : dict
        For each phase component given in 'fields', a dictionary is given
        with the following attributes:
        - "event_count": Histogram of instantaneous phase values at earthquake times.
        - "bins": Bins of the histogram.
        - "expected_rate": Fraction of the time axis covered by each phase
                           bin. For example, some phase values are
                           rare so the associated phase bin covers a small
                           fraction of the time axis and even in the case of
                           tide-independent seismicity one expects
                           significantly less earthquakes in these phase bins.
        - "observed_rate": "phase_hist" normalized by the total number of
                           earthquakes.
    """
    forcing_at_eq = cat[forcing.name].values
    # bin the values of tidal phase at earthquake timings
    (
        count_vs_forcing,
        _,
    ) = np.histogram(
        forcing_at_eq,
        bins=forcing_bin_edges,
    )
    # estimate the expected number of earthquakes in each bin for random seismicity
    n_time_samples = len(forcing)
    tidal_phase_hist, _ = np.histogram(
        forcing,
        bins=forcing_bin_edges,
    )
    # hist/n_times = fraction of time occupied by samples with stress values in given bin
    # (hist/n_times)*ntotal = expected number of events with given stress value if
    #                         seismicity were to be independent of tidal stress
    if n_time_samples > 0:
        expected_rate = tidal_phase_hist / n_time_samples
    else:
        expected_rate = np.zeros_like(tidal_phase_hist)
    observed_rate = count_vs_forcing / np.sum(count_vs_forcing)
    # 0/0 is 0
    invalid = np.isnan(observed_rate) | np.isinf(observed_rate)
    # anticipate 0/0
    zero_by_zero = (observed_rate == 0.0) & (expected_rate == 0.0)
    # estimate the tidal modulation ratio
    relative_rate = np.ones(len(observed_rate), dtype=np.float32)
    relative_rate[~zero_by_zero] = (
        observed_rate[~zero_by_zero] / expected_rate[~zero_by_zero]
    )
    relative_rate[invalid] = 1.0
    if np.any(np.isinf(relative_rate)):
        print("observed rate", observed_rate)
        print("expected rate", expected_rate)
        print("event_count", count_vs_forcing)
        print("data", forcing)
    mean = np.mean(relative_rate)
    if clip_relative_rate_above is not None:
        if mean != 0.0:
            clip_relative_rate_above = clip_relative_rate_above / mean
        relative_rate = np.clip(
            relative_rate, a_min=0.0, a_max=clip_relative_rate_above
        )
    if clip_relative_rate_below is not None:
        if mean != 0.0:
            clip_relative_rate_below = clip_relative_rate_below / mean
        relative_rate = np.clip(
            relative_rate, a_min=clip_relative_rate_below, a_max=relative_rate.max()
        )
    # normalization so that the ratio vector sums up to 1.
    norm = np.mean(relative_rate)
    if norm != 0.0:
        relative_rate /= norm
    relative_rate = relative_rate

    output = {
        "event_count": count_vs_forcing,
        "observed_rate": observed_rate,
        "expected_rate": expected_rate,
        "relative_rate": relative_rate,
    }
    return output


def _weighted_mean(x, weights):
    return np.sum(x * weights, axis=-1)


def _jackknife_bias(x, operator, x_all=None):
    """
    Parameters
    ----------
    x : numpy.ndarray
        (num_bins, num_windows) ndarray.
    """
    pool = np.ones(x.shape[1], dtype=bool)
    bias = np.zeros(x.shape[0], dtype=x.dtype)
    for i in range(x.shape[1]):
        pool[i] = False
        _x_j = x[:, pool]
        bias += operator(_x_j)
        pool[i] = True
    if x_all is None:
        x_all = operator(x)
    bias = float(x.shape[1] - 1) * (bias / float(x.shape[1]) - x_all)
    return bias


def composite_rate_estimate(
    cat,
    forcing,
    forcing_bin_edges,
    window_time,
    window_type="backward",
    short_window_days=3 * 30,
    num_short_windows=8,
    overlap=0.0,
    downsample=0,
    num_bootstrap_for_errors=100,
    aggregate="median",
    wiener_filter=None,
    min_num_events_in_short_window=10,
    min_fraction_of_valid_windows=0.50,
    keep_short_windows=False,
    clip_relative_rate_above=None,
    clip_relative_rate_below=None,
    progress=False,
):
    """Count number of earthquakes in phase bins with bootstrapping analysis.

    Parameters
    -----------
    cat : `pandas.DataFrame`
        The catalog to analyze
    forcing : `pandas.Series`
        Tidal forcing time series.
    forcing_bin_edges : array-like
        Edges of the forcing bins,
    short_window_days : iont, default to 90
        Size of the short time window, in days.
    overlap : float, default to 0.
        Overlap between sliding short time windows.
    num_short_windows : int, default to 8
        Number of sliding short windows over which the statistics is computed.
        The length of the corresponding large window is
        `num_short_windows*(1 - overlap)*short_window_days + overlap*short_window_days`.
        The sliding windows start from `t_end`.
    t_end : string or datetime-like object, default to None
        Reference time from which the short time windows are sliding backward.
        If None, take the latest earthquake timing in `cat`.
    progress : boolean, default to True
        If True, print the progress bar when iterating over the short time windows.
    downsample : int, optional
        If different from 0, the composite rate ratio is downsampled by applying
        average pooling in bins of `downsample` samples. Users may use many bins
        at first to accurately estimate the expected ratio and then downsample
        the observed/expectred ratio to reduce noise. Defaults to 0.
    num_bootstrap_for_errors : int, optional
        The error on the composite rate ratio is computed by estimating the
        composite rate ratio `num_bootstrap_for_errors` times on randomly
        selected short windows.

    """
    from dateutil.relativedelta import relativedelta

    assert window_type in {
        "backward",
        "forward",
    }, "window_type should be either of 'backward' or 'forward'"
    assert aggregate in {
        "mean",
        "median",
        "weighted_mean",
        "svd",
    }, "aggregate should be either of 'mean', 'median' or 'svd'"

    if aggregate == "median":
        # operator = partial(np.ma.median, axis=-1)
        operator = partial(np.median, axis=-1)
        pulling_operator = np.median
    elif aggregate == "mean":
        # operator = partial(np.ma.mean, axis=-1)
        operator = partial(np.mean, axis=-1)
        pulling_operator = np.mean
    elif aggregate == "svd":
        operator = get_singular_vector
        pulling_operator = np.mean

    short_window_dur = relativedelta(days=short_window_days)
    short_window_shift = relativedelta(days=int((1.0 - overlap) * short_window_days))
    min_num_valid_short_windows = int(min_fraction_of_valid_windows * num_short_windows)
    seismicity_vs_forcing_short_win = []

    if window_type == "backward":
        t_end = window_time
        t_start = t_end - short_window_dur
    elif window_type == "forward":
        t_start = window_time
        t_end = t_start + short_window_dur

    # ----------------------------------------------
    #   Estimate the observed and expected rate
    #   of seismicity as a function of 'forcing_name'
    #   in the `num_short_windows` windows.
    # ----------------------------------------------
    disable = False if progress else True
    for n in tqdm(
        range(num_short_windows), desc="Computing in sub-windows", disable=disable
    ):
        # t_start = t_end - short_window_dur
        seismicity_vs_forcing_short_win.append(
            estimate_rate_forcing_bins(
                cat[(cat["origin_time"] > t_start) & (cat["origin_time"] <= t_end)],
                forcing[(forcing.index > t_start) & (forcing.index <= t_end)],
                forcing_bin_edges,
                clip_relative_rate_above=clip_relative_rate_above,
                clip_relative_rate_below=clip_relative_rate_below,
            )
        )
        if window_type == "backward":
            t_end -= short_window_shift
            t_start -= short_window_shift
        elif window_type == "forward":
            t_end += short_window_shift
            t_start += short_window_shift

    # ----------------------------------------------
    #           aggregate results
    # ----------------------------------------------
    seismicity_vs_forcing = {}

    num_events = [
        seismicity_vs_forcing_short_win[i]["event_count"].sum()
        for i in range(num_short_windows)
    ]
    valid_window = np.array(num_events) >= min_num_events_in_short_window
    num_valid_windows = np.sum(valid_window)
    if num_valid_windows == 0:
        print(f"Could not find a single valid window for {forcing.name}")
        return
    if num_valid_windows < min_num_valid_short_windows:
        print(
            f"Not enough valid windows (found {num_valid_windows}, "
            f"required {min_num_valid_short_windows})"
        )
        return

    # define weights proportionally to expected_rate, because expected_rate
    # is itself proportional to the time interval covered by the forcing bin
    weights = np.stack(
        [
            seismicity_vs_forcing_short_win[i]["expected_rate"]
            for i in range(num_short_windows)
            if valid_window[i]
        ],
        axis=-1,
    )
    weights /= np.sum(weights, axis=-1, keepdims=True)
    if aggregate == "weighted_mean":
        operator = partial(_weighted_mean, weights=weights)
        pulling_operator = partial(np.mean, axis=-1)

    for field in seismicity_vs_forcing_short_win[0]:
        # inner loop: all the short-window quantities are aggregated
        #             into a single long-window estimate
        if field == "bins":
            continue
        all_windows = np.stack(
            [
                seismicity_vs_forcing_short_win[i][field]
                for i in range(num_short_windows)
                if valid_window[i]
            ],
            axis=-1,
        )

        if wiener_filter is not None:
            win_indexes = np.arange(all_windows.shape[1])
            np.random.shuffle(win_indexes)
            # robust estimation of noise with MAD
            mad_across_windows = 1.48 * np.median(
                np.abs(all_windows - np.median(all_windows, axis=1, keepdims=True)),
                axis=1,
            )
            # fill in zeros with default value (this shouldn't really matter
            # because cases with mad=0 are cases that are equally zero everywhere)
            mad_across_windows[mad_across_windows == 0.0] = 0.2

            noise_power = np.median(mad_across_windows) ** 2
            # noise_power = 1.48 * np.median(
            #        np.abs(all_windows - np.median(all_windows))
            #        )
            # print(1.48 * np.median(
            #        np.abs(all_windows - np.median(all_windows))
            #        ))

            half_wiener_win = (
                wiener_filter[0] // 2 + wiener_filter[0] % 2,
                wiener_filter[1] // 2 + wiener_filter[1] % 2,
            )
            _padded = np.pad(
                all_windows[:, win_indexes],
                (
                    (half_wiener_win[0], half_wiener_win[0]),
                    (half_wiener_win[1], half_wiener_win[1]),
                ),
                mode="reflect",
            )
            all_windows = wiener(
                _padded, mysize=wiener_filter, noise=noise_power
            )  # [half_wiener_win:-half_wiener_win, :]
            if half_wiener_win[0] > 0:
                all_windows = all_windows[half_wiener_win[0] : -half_wiener_win[0], :]
            if half_wiener_win[1] > 0:
                all_windows = all_windows[:, half_wiener_win[1] : -half_wiener_win[1]]

        if np.sum(np.isnan(all_windows)) > 0:
            breakpoint()

        if keep_short_windows:
            seismicity_vs_forcing[f"all_windows_{field}"] = all_windows

        # ---------------------------------------------------------
        #                  downsample
        if downsample > 0:
            # `expected_rate` and, consequently, `relative_rate` must be
            # estimated in narrow bins. However, the number of bins can
            # afterward be reduced by downsampling (average pooling).
            length = all_windows.shape[0]
            num_windows = all_windows.shape[1]

            assert downsample * (length // downsample) == length

            downsampled_windows = np.zeros(
                (length // downsample, num_windows), dtype=all_windows.dtype
            )
            for i in range(all_windows.shape[1]):
                downsampled_windows[:, i] = pulling_operator(
                    all_windows[:, i].reshape(-1, downsample),
                    axis=-1,
                )
            all_windows = downsampled_windows

        # ---------------------------------------------------------
        #    aggregate short-window observations into one estimate
        if field in {"expected_rate", "bin_duration"}:
            seismicity_vs_forcing[field] = np.sum(all_windows, axis=-1).astype(
                "float32"
            )
            if field == "expected_rate":
                # by construction, this is a pdf so its integral must be 1
                seismicity_vs_forcing[field] /= np.float32(
                    np.sum(seismicity_vs_forcing[field])
                )
            seismicity_vs_forcing[f"{field}_err"] = np.mean(
                all_windows, axis=-1
            ).astype("float32")
        else:
            (
                seismicity_vs_forcing[field],
                seismicity_vs_forcing[f"{field}_err"],
            ) = bootstrap_statistic(
                all_windows, operator, n_bootstraps=num_bootstrap_for_errors
            )
            # if field2 == "relative_rate":
            #    bias = _jackknife_bias(
            #            all_windows, operator, x_all=seismicity_vs_forcing[field1][field2]
            #            )
            #    #print(bias)
            #    seismicity_vs_forcing[field1][field2] -= bias

        # =========================
        #       normalize
        # =========================
        if field == "relative_rate":
            seismicity_vs_forcing[field] /= np.ma.mean(seismicity_vs_forcing[field])
        elif field in {"observed_rate", "expected_rate"}:
            seismicity_vs_forcing[field] /= np.sum(seismicity_vs_forcing[field])
        # ---------------------------------------------------------
    seismicity_vs_forcing["bins"] = forcing_bin_edges
    if downsample > 0:
        # downsample bins if necessary
        seismicity_vs_forcing["bins"] = seismicity_vs_forcing["bins"][::downsample]
    if len(seismicity_vs_forcing) == 0:
        return None
    else:
        return seismicity_vs_forcing


def bootstrap_statistic(x, operator, n_bootstraps=100):
    """
    Estimate the mean and standard deviation of an operator applied to
    bootstrapped samples of data.

    Parameters:
    -----------
    x : numpy.ndarray
        Input data to be bootstrapped.

    operator : callable
        A function that computes a statistic or operation on a data sample.

    n_bootstraps : int, optional
        The number of bootstrap resamples to generate (default is 100).

    Returns:
    --------
    numpy.ndarray
        An array of shape (x.shape[0],) representing the mean of the
        operator's output on bootstrapped samples.

    numpy.ndarray
        An array of shape (x.shape[0],) representing the standard deviation of
        the operator's output on bootstrapped samples.
    """
    if x.ndim > 1:
        instances = np.zeros((n_bootstraps, x.shape[0]), dtype=np.float32)
    else:
        instances = np.zeros(n_bootstraps, dtype=np.float32)
    for i in range(n_bootstraps):
        replica = x[..., np.random.randint(0, x.shape[-1] - 1, size=x.shape[-1])]
        instances[i] = operator(replica)
    return np.mean(instances, axis=0), np.std(instances, axis=0)


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


def fit_stress_hist(x, y, yerr):
    """
    Weighted least squares for linear model.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable representing stress data.

    y : numpy.ndarray
        Dependent variable representing a corresponding response.

    yerr : numpy.ndarray
        Error values associated with the dependent variable y.

    Returns:
    --------
    float
        The intercept of the linear regression model.

    float
        The slope (coefficient) of the linear regression model.

    float
        The standard error of the intercept (default is 0.0).

    float
        The standard error of the slope, representing the uncertainty in the regression coefficient.
    """
    yerr[yerr == 0.0] = max(10.0, 10.0 * yerr.max())
    slope, intercept, slope_err = weighted_linear_regression(x, y, W=(1.0 / yerr) ** 2)
    intercept_err = 0.0
    return intercept, slope, intercept_err, slope_err


def fit_stress_hist_scipy_l1(x, y, yerr):
    """
    Fit a linear regression model using L1 loss optimization.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable representing stress data.

    y : numpy.ndarray
        Dependent variable representing a corresponding response.

    yerr : numpy.ndarray
        Error values associated with the dependent variable y.

    Returns:
    --------
    float
        The intercept of the linear regression model.

    float
        The slope (coefficient) of the linear regression model.

    float
        The standard error of the intercept (default is 0.0).

    float
        The standard error of the slope (default is 0.0).
    """
    from scipy.optimize import minimize

    loss = lambda params: np.sum(np.abs(y - (params[0] * x + params[1])))
    x0 = [0.0, 1.0]

    results = minimize(loss, x0)

    slope, intercept = results.x

    return intercept, slope, 0.0, 0.0


def fit_stress_hist_bootstrap(x, y, y_err, n_bootstrap=100):
    """
    Fit a linear regression model and estimate model uncertainties using bootstrapping.

    The data y are perturbed n_bootstrap times with gaussian noise with standard
    deviation given by y_err. The model parameters are the mean of all
    n_bootstrap models and their uncertainties are the standard deviation of all
    n_bootstrap models.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable representing stress data.

    y : numpy.ndarray
        Dependent variable representing a corresponding response.

    y_err : numpy.ndarray
        Error values associated with the dependent variable y.

    n_bootstrap : int, optional
        The number of bootstrap resamples to generate (default is 100).

    Returns:
    --------
    float
        The mean intercept of the linear regression model from bootstrapping.

    float
        The mean slope (coefficient) of the linear regression model from bootstrapping.

    float
        The standard error of the intercept based on bootstrapping.

    float
        The standard error of the slope based on bootstrapping.
    """
    inverted_intercept = np.zeros(n_bootstrap, dtype=np.float32)
    inverted_slope = np.zeros(n_bootstrap, dtype=np.float32)
    for n in range(n_bootstrap):
        # generate random sample assuming that each bin [i] of the histogram
        # is normally distributed with mean y[i] and std y_err[i]
        y_b = np.random.normal(loc=0.0, scale=1.0, size=len(y))
        # noisy y
        y_b = y_b * y_err + y
        y_b = y_b / np.mean(y_b)
        # don't allow negative values (impossible)
        y_b = np.maximum(y_b, 0.0)
        inverted_slope[n], inverted_intercept[n], _ = weighted_linear_regression(
            x, y_b, W=(1.0 / y_err) ** 2
        )
    return (
        np.mean(inverted_intercept),
        np.mean(inverted_slope),
        np.std(inverted_intercept),
        np.std(inverted_slope),
    )


def fit_stress_corrosion(x, y, y_err):
    """
    Fit a stress-corrosion model to data and return optimized parameters.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable representing stress data.

    y : numpy.ndarray
        Dependent variable representing a corresponding response.

    y_err : numpy.ndarray
        Error values associated with the dependent variable y.

    Returns:
    --------
    numpy.ndarray
        An array of optimized parameters for the stress-corrosion model.
    """
    from scipy.optimize import curve_fit, minimize

    # print(x, y)
    p0 = [2, 1.0e5]
    bounds = [(2.0, 50.0), (1.0e3, 10.0e6)]
    # popt, pcov = curve_fit(stress_corrosion, x, y, p0=p0, bounds=bounds, sigma=y_err)
    # perr = np.sqrt(np.diag(pcov))

    loss = lambda params: np.sum((y - stress_corrosion(x, params[0], params[1])) ** 2)
    results = minimize(loss, p0, bounds=bounds)
    return results.x
    # return *popt, *perr


def fit_rate_state(x, y, y_err):
    """
    Fit a rate-state model to data and return optimized parameters and their uncertainties.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable representing data.

    y : numpy.ndarray
        Dependent variable representing a corresponding response.

    y_err : numpy.ndarray
        Error values associated with the dependent variable y.

    Returns:
    --------
    float
        The optimized parameter for the rate-state model.

    float
        The minimum value within the parameter's uncertainty range.

    float
        The maximum value within the parameter's uncertainty range.
    """
    from scipy.optimize import curve_fit, minimize, minimize_scalar

    # from scipy.stats import chi2

    p0 = [1.0e3]
    bounds = (1.0, 1.0e5)
    # v1
    popt, pcov = curve_fit(rate_state, x, y, p0=p0, bounds=bounds, sigma=y_err)
    perr = np.sqrt(np.diag(pcov))
    Asig_best = popt[0]
    Asig_min = Asig_best - perr[0]
    Asig_max = Asig_best + perr[0]
    # return *popt, *perr
    # # v2
    # W = 1.0 / y_err
    # W /= W.sum()
    # # loss = lambda Asig_Pa: np.sum(W * np.abs(y - rate_state(x, Asig_Pa)))
    # loss = lambda Asig_Pa: np.sum(W * (y - rate_state(x, Asig_Pa))**2)
    # # results = minimize_scalar(loss, bounds=bounds, method="bounded")
    # # return results.x, 0.0
    # Asig_Pa_range = np.linspace(0., 150., 1000)*1000.
    # losses = np.zeros(len(Asig_Pa_range), dtype=np.float32)
    # for i in range(len(losses)):
    #     losses[i] = loss(Asig_Pa_range[i])
    # n_df = len(y) - 1
    # chi2_ = losses / (losses.min() / n_df)
    # CI_66 = chi2.ppf(q=0.66, df=n_df)
    # CI_region = chi2_ < CI_66
    # Asig_min = Asig_Pa_range[CI_region].min()
    # Asig_max = Asig_Pa_range[CI_region].max()
    # Asig_best = Asig_Pa_range[chi2_.argmin()]
    # print(Asig_best/1000., Asig_min/1000., Asig_max/1000.)
    return Asig_best, Asig_min, Asig_max


def fit_rate_state_bootstrap(x, y, y_err, n_bootstrap=100):
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

    n_bootstrap : int, optional
        The number of bootstrap resamples to generate (default is 100).

    Returns:
    --------
    float
        The median of the optimized parameter for the rate-state model from bootstrapping.

    float
        The robust uncertainty estimate for the parameter.
    """

    from scipy.optimize import minimize_scalar

    bounds = (1.0, 3.0e5)
    W = 1.0 / y_err
    W /= W.sum()
    inverted_Asig_Pa = np.zeros(n_bootstrap, dtype=np.float32)
    for n in range(n_bootstrap):
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
    return (
        np.median(inverted_Asig_Pa),
        1.48 * scimad(inverted_Asig_Pa),
    )


# fit ratio obs/exp seismicity vs tidal phase
def fit_semidiurnal_triggering(x, y):
    """ """
    from scipy.optimize import minimize_scalar

    deg2rad = np.pi / 180.0
    y_ = (y - 1.0) / np.std(y)
    x_ = x * deg2rad
    A = 1.0 / np.sqrt(2.0)
    # first: grid search to find the right phase shift
    bs = np.linspace(-180.0, 180.0, 36) * deg2rad
    sum_squared_residuals = np.zeros(len(bs))
    for i, b in enumerate(bs):
        sum_squared_residuals[i] = np.sum((y_ - cos(x_, A, b)) ** 2)
    b = bs[sum_squared_residuals.argmin()]
    # second: first the amplitude that minimize the loss function
    y_ = y - 1.0
    loss = lambda a: np.sum((cos(x_, a, b) - y_) ** 2)
    optimization_results = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
    a = optimization_results.x
    return a, b


def fit_phase_scipy(x, y, y_err):
    """ """
    from scipy.optimize import curve_fit

    p0 = fit_semidiurnal_triggering(x, y)
    deg2rad = np.pi / 180.0
    x_ = x * deg2rad
    # p0 = [0., 0.]
    bounds = ([0.0, -np.pi], [1.0, +np.pi])
    popt, pcov = curve_fit(cos, x_, y, p0=p0, bounds=bounds, sigma=y_err)
    perr = np.sqrt(np.diag(pcov))
    return *popt, *perr


def cos(x, a, b):
    return a * np.cos(x - b)


def stress_corrosion(coulomb_stress_Pa, half_corrosion_index, eq_stress_drop_Pa):
    """ """
    return 1.0 + ((coulomb_stress_Pa / eq_stress_drop_Pa) ** 2) ** half_corrosion_index


def rate_state(x, Asig_Pa):
    """ """
    return np.exp(x / Asig_Pa)


def fit_errors(x, a, b, a_err, b_err):
    df_da = np.cos(x - b)
    df_db = +a * np.sin(x - b)
    err = np.sqrt((df_da * a_err) ** 2 + (df_db * b_err) ** 2)
    return err


def pearson_chi_squared_test(o, e, p):
    """Compute the p-value of `o` being distributed as `e`.

    Compute the p-value that observations `o` are distributed according
    to the theoretical distribution `e`, which has `p`-1 degrees of freedom.
    This is Pearson's chi-squared test.

    """
    from scipy.stats import chi2

    o_norm = o / np.sum(o)
    e_norm = e / np.sum(e)
    chi_squared = np.sum((o_norm - e_norm) ** 2 / e_norm)
    print(f"Chi-2: {chi_squared:.2f}")
    degrees_of_freedom = len(o) - p
    pvalue = chi2.sf(chi_squared, degrees_of_freedom)
    print(f"Critical value at 95% CI: {chi2.ppf(q=0.95, df=len(o)-1):.3f}")
    return pvalue


def kolmogorov_smirnov_test(o, e):
    """Kolmogorov-Smirnov test."""
    o_norm = o / np.sum(o)
    e_norm = e / np.sum(e)
    e_cdf = np.cumsum(e_norm)
    e_cdf /= e_cdf[-1]
    o_cdf = np.cumsum(o_norm)
    o_cdf /= o_cdf[-1]
    KS_statistic = np.abs(o_cdf - e_cdf).max()


def generalized_poisson(k, lbd):
    """Generalization of the poisson distribution for non-integer data.

    Use the gamma function to generalize the factorial function.

    Parameters
    -----------
    k: scalar float or int,
        Number of observed events.
    lbd: scalar float or int,
        Expected number of events.
    Returns
    --------
    p: scalar float,
        The poisson p-value for k given lbd.
    """
    from scipy.special import gamma

    return lbd**k * np.exp(-lbd) / gamma(k + 1)


def loglikelihood_poisson(observed, expectation):
    """ """
    return np.sum(generalized_poisson(observed, expectation))


# --------------------------------------------------------------------
#                       plotting functions
# --------------------------------------------------------------------

# --------------------------------------------------------------------
#                     data loading functions
# --------------------------------------------------------------------


def load_tidal_stress(
    tidal_stress_path,
    fields=["shear_stress", "normal_stress", "coulomb_stress"],
    rate=True,
    t_end="2019-07-04",
    mu=None,
):
    """ """
    tidal_stress = {}
    # load tides (computed by p01_StressFromTidesPlaneStrain.py)
    with h5.File(tidal_stress_path, mode="r") as fin:
        starttime = fin["starttime"][()].decode("utf-8")
        delta = fin["delta_hour"][()] * 3600  # delta in seconds
        if mu is None:
            mu = fin["friction"][()]  # coefficient of friction
        for field in fields:
            if field in fin:
                tidal_stress[field] = fin[field][()]
                n_samples = len(tidal_stress[field])
    if (
        ("coulomb_stress" in fields)
        and ("shear_stress" in fields)
        and ("normal_stress" in fields)
    ):
        tidal_stress["coulomb_stress"] = (
            tidal_stress["shear_stress"] + mu * tidal_stress["normal_stress"]
        )
    tref = np.datetime64(starttime).astype("datetime64[s]").astype("int64")

    # calendar time of stress time series
    time = pd.Series(
        pd.date_range(start=starttime, freq=f"{delta/3600.}h", periods=n_samples)
    )
    tvec_tide = np.arange(0, n_samples) * delta + tref

    # # limit time series to period spanned by seismicity
    selection = tvec_tide <= pd.Timestamp(t_end).timestamp()
    for field in tidal_stress:
        tidal_stress[field] = tidal_stress[field][selection]

    tvec_tide = tvec_tide[selection]
    time = time[selection]

    if rate:
        # compute stressing rates
        dt_c = tvec_tide[1:] - tvec_tide[:-1]
        dt_c = np.hstack((dt_c[0], dt_c))
        for field in list(tidal_stress.keys()):
            tidal_stress[f"{field}_rate"] = np.gradient(tidal_stress[field]) / dt_c

    tidal_stress["time_sec"] = tvec_tide

    tidal_stress = pd.DataFrame(tidal_stress)
    tidal_stress.set_index(time, inplace=True)

    return tidal_stress


def compute_semidiurnal_phase_at_eq(catalog, tidal_stress, fields):
    """Interpolate semidiurnal phase at earthquake timings.

    Notes: Use `compute_instantaneous_phase_at_eq` instead.
    """
    return compute_instantaneous_phase_at_eq(catalog, tidal_stress, fields)


def compute_instantaneous_phase_at_eq(
    catalog, tidal_stress, fields, attach_unravelled_phase=False
):
    """Interpolate instantaneous phase at earthquake timings."""
    eq_timings = catalog.loc[:, "t_eq_s"].values
    for field in fields:
        if f"unravelled_{field}" in tidal_stress:
            # catalog.loc[indexes, field] = (
            #    np.interp(
            #        catalog.loc[indexes, "t_eq_s"].values,
            #        tidal_stress["time_sec"].values,
            #        180.0 + tidal_stress[f"unravelled_{field}"].values,
            #    )
            # ) % 360.0 - 180.0
            catalog = catalog.assign(
                tmp_name=np.interp(
                    eq_timings,
                    tidal_stress["time_sec"].values,
                    180.0 + tidal_stress[f"unravelled_{field}"].values,
                )
                % 360.0
                - 180.0,
            )
            if attach_unravelled_phase:
                # catalog.loc[indexes, f"unravelled_{field}"] = np.interp(
                #    catalog.loc[indexes, "t_eq_s"].values,
                #    tidal_stress["time_sec"].values,
                #    tidal_stress[f"unravelled_{field}"].values,
                # )
                catalog = catalog.assign(
                    tmp_name2=np.interp(
                        eq_timings,
                        tidal_stress["time_sec"].values,
                        tidal_stress[f"unravelled_{field}"].values,
                    ),
                )
                catalog.rename(
                    columns={"tmp_name2": f"unravelled_{field}"}, inplace=True
                )
        else:
            catalog = catalog.assign(
                tmp_name=np.interp(
                    eq_timings,
                    tidal_stress["time_sec"].values,
                    tidal_stress[field].values,
                ),
            )
        catalog.rename(columns={"tmp_name": field}, inplace=True)
    return catalog


# def unravel_phase(phases, degree=True):
#     """
#     """
#     if not degree:
#         phases = np.rad2deg(phases)
#     # 1) differentiate phases
#     dphase = phases[1:] - phases[:-1]
#     dphase_plus_360 = 360. + phases[1:] - phases[:-1]
#     dphase_minus_360 = -360. + phases[1:] - phases[:-1]
#     dphase = np.maximum(
#         dphase, dphase_plus_360, out=dphase, where=np.abs(dphase_plus_360) < np.abs(dphase)
#     )
#     dphase = np.minimum(
#         dphase, dphase_minus_360, out=dphase, where=np.abs(dphase_minus_360) < np.abs(dphase)
#     )
#     if not degree:
#         dphase = np.deg2rad(dphase)
#     # convert dphase to float64 in case it is not
#     # otherwise, error will accumulates in cumsum
#     return phases[0] + np.hstack( (0., np.cumsum(dphase.astype("float64"))) )
#     # return phases[0] + np.hstack( (0., np.cumsum(dphase)) )


def unravel_phase(phases, degree=True):
    """ """
    phases = np.float64(phases)
    if degree:
        return np.unwrap(phases, period=360.0)
    else:
        return np.unwrap(phases)


def compute_stress_at_eq(catalog, tidal_stress, fields):
    """Interpolate stress at earthquake timings."""
    # indexes = catalog.index
    eq_timings = catalog.loc[:, "t_eq_s"].values
    for f in fields:
        catalog = catalog.assign(
            f=np.interp(
                eq_timings,
                tidal_stress["time_sec"].values,
                tidal_stress[f].values,
            ),
        )
        catalog.rename(columns={"f": f}, inplace=True)
    return catalog


def compute_fortnightly(catalog, tidal_stress, full_moons_path):
    """Compute lunar phase and attribute a fortnightly phase to each earthquake."""
    # We use the timings of the full moons to measure the fortnightly cycle.
    full_moons = pd.to_datetime(pd.read_csv(full_moons_path, index_col=0)["full_moons"])
    full_moons = full_moons[(full_moons > "2000-01-01") & (full_moons < "2020-01-01")]
    avg_moon_period_sec = np.mean(
        np.diff(full_moons.values.astype("datetime64[s]").astype("float64"))
    )

    full_moon_time = pd.date_range(
        start=full_moons.min(), end=full_moons.max(), periods=100 * len(full_moons)
    )
    rel_time_sec = full_moon_time.values.astype("datetime64[s]").astype("float64")
    rel_time_sec -= rel_time_sec[0]

    lunar_times = pd.Timestamp(full_moons.min()) + rel_time_sec.astype("timedelta64[s]")
    # lunar_cycle = np.cos(2.0 * np.pi * (rel_time_sec / avg_moon_period_sec))
    # lunar_pos = (360.0 * (rel_time_sec / avg_moon_period_sec)) % 360
    # unravel the lunar phase for easier interpolation
    lunar_pos_unravelled = 360.0 * (rel_time_sec / avg_moon_period_sec)

    # compute the lunar phase at the tide time series times for
    # decomposing catalog into rising and falling fortnightly
    tidal_stress["lunar_phase"] = (
        np.interp(
            tidal_stress["time_sec"],
            lunar_times.astype("datetime64[s]").astype("float64"),
            lunar_pos_unravelled,
        )
        % 360
    )
    tidal_stress["fortnightly_phase"] = (
        2.0 * tidal_stress["lunar_phase"]
    ) % 360 - 180.0
    tidal_stress["rising_fortnightly"] = tidal_stress["fortnightly_phase"] > 0.0
    tidal_stress["falling_fortnightly"] = ~tidal_stress["rising_fortnightly"]

    # compute the lunar phase at the earthquake times for
    # decomposing catalog into rising and falling fortnightly
    catalog["lunar_phase"] = (
        np.interp(
            catalog["t_eq_s"],
            lunar_times.astype("datetime64[s]").astype("float64"),
            lunar_pos_unravelled,
        )
        % 360.0
    )
    catalog["fortnightly_phase"] = (2.0 * catalog["lunar_phase"]) % 360 - 180.0

    catalog["rising_fortnightly"] = catalog["fortnightly_phase"] > 0.0
    catalog["falling_fortnightly"] = ~catalog["rising_fortnightly"]

    # ## Attribute a fortnightly phase to each earthquake
    catalog["fortnightly_phase"] = np.interp(
        catalog["t_eq_s"], tidal_stress["time_sec"], tidal_stress["fortnightly_phase"]
    )


def SVDWF(
    matrix,
    expl_var=0.4,
    max_singular_values=5,
    wiener_filter_colsize=None,
):
    """
    Implementation of the Singular Value Decomposition Wiener Filter (SVDWF)
    described in Moreau et al 2017.

    Parameters
    ----------
    matrix: (n x m) numpy array
        n is the number of events, m is the number of time samples
        per event.
    n_singular_values: scalar float
        Number of singular values to retain in the
        SVD decomposition of matrix.
    max_freq: scalar float, default to cfg.MAX_FREQ_HZ
        The maximum frequency of the data, or maximum target
        frequency, is used to determined the size in the
        time axis of the Wiener filter.

    Returns
    --------
    filtered_data: (n x m) numpy array
        The matrix filtered through the SVD procedure.
    """
    from scipy.linalg import svd
    from scipy.signal import wiener
    import matplotlib.pyplot as plt

    try:
        U, S, Vt = svd(matrix, full_matrices=False)
    except Exception as e:
        print(e)
        print("Problem while computing the svd...!")
        return np.random.normal(loc=0.0, scale=1.0, size=matrix.shape)
    if wiener_filter_colsize is None:
        wiener_filter_colsize = U.shape[0]
    # wiener_filter = [wiener_filter_colsize, int(cfg.SAMPLING_RATE_HZ/max_freq)]
    wiener_filter = [wiener_filter_colsize, 1]
    filtered_data = np.zeros((U.shape[0], Vt.shape[1]), dtype=np.float32)
    # select the number of singular values
    # in order to explain 100xn_singular_values%
    # of the variance of the matrix
    var = np.cumsum(S**2)
    if var[-1] == 0.0:
        # only zeros in matrix
        return filtered_data
    var /= var[-1]
    n_singular_values = np.min(np.where(var >= expl_var)[0]) + 1
    n_singular_values = min(max_singular_values, n_singular_values)
    for n in range(min(U.shape[0], n_singular_values)):
        s_n = np.zeros(S.size, dtype=np.float32)
        s_n[n] = S[n]
        projection_n = np.dot(U, np.dot(np.diag(s_n), Vt))
        if wiener_filter[0] == 1 and wiener_filter[1] == 1:
            # no wiener filtering
            filtered_projection = projection_n
        else:
            # the following application of Wiener filtering is questionable: because each projection in this loop is a projection
            # onto a vector space with one dimension, all the waveforms are colinear: they just differ by an amplitude factor (but same shape).
            filtered_projection = wiener(
                projection_n,
                # mysize=[max(2, int(U.shape[0]/10)), int(cfg.SAMPLING_RATE_HZ/freqmax)]
                mysize=wiener_filter,
            )
        # filtered_projection = projection_n
        if np.isnan(filtered_projection.max()):
            continue
        filtered_data += filtered_projection
    if wiener_filter[0] == 1 and wiener_filter[1] == 1:
        # no wiener filtering
        pass
    else:
        filtered_data = wiener(filtered_data, mysize=wiener_filter)
    # remove nans or infs
    filtered_data[np.isnan(filtered_data)] = 0.0
    filtered_data[np.isinf(filtered_data)] = 0.0
    return filtered_data


def spectral_filtering(x, singular_value_index=0):
    from scipy.linalg import svd

    U, S, Vt = svd(x, full_matrices=False)
    s_n = np.zeros(S.size, dtype=np.float32)
    s_n[singular_value_index] = S[singular_value_index]
    projection_n = np.dot(U, np.dot(np.diag(s_n), Vt))
    return projection_n


def get_singular_vector(x, singular_value_index=0):
    """
    Parameters
    ----------
    x : numpy.ndarray
        (num_bins, num_windows) ndarray.
    """
    from scipy.linalg import svd

    V, S, Ut = svd(x, full_matrices=False)

    singular_vector = V[:, singular_value_index]
    sign = np.sign(np.sum(np.sign(Ut[singular_value_index, :])))
    return sign * singular_vector


def compute_AIC(residuals, num_params):
    """Akaike Information Criterion for least-squares solution."""
    num_samples = len(residuals)
    return (
        num_samples * np.log(2.0 * np.pi)
        + num_samples * np.log(np.sum(residuals**2) / num_samples)
        + num_samples
        + 2 * num_params
    )

import h5py as h5
import numpy as np
import pandas as pd

from functools import partial

try:
    from scipy.stats import median_absolute_deviation as scimad
except ImportError:
    from scipy.stats import median_abs_deviation as scimad
from tqdm import tqdm

FORTNIGHTLY_PHASES = ["rising fortnightly", "falling fortnightly", "entire lunar cycle"]


def bin_eq_tidal_stresses(
    cat,
    tidal_stress,
    nbins=6,
    stress_bins={},
    fields=[
        "shear_stress",
        "normal_stress",
        "coulomb_stress",
        "shear_stressing_rate",
        "normal_stressing_rate",
        "coulomb_stressing_rate",
    ],
):
    """Count number of earthquakes in stress bins.

    Parameters
    -----------
    cat : `pandas.DataFrame`
        Earthquake catalog.
    tidal_stress : `pandas.DataFrame`
        All tidal stress fields.
    nbins : int, optional
        Number of default stress bins if `stress_bins` is not provided.
        Defaults to 6.
    stress_bins : dict, optional
        Dictionary providing the stress bins, e.g.
        `stress_bins["shear_stress_bins"] = [-1000, -800, ..., 800, 1000]`
        Defaults to '{}'.
    fields : list of str, optional
        List of stress components to explore.

    Returns
    --------
    seismicity_vs_stress : dict
        For each stress component given in 'fields', a dictionary is given
        with the following attributes:
        - "stress_hist": Histogram of stress values at earthquake times.
        - "stress_bins": Bins of the histogram.
        - "expected_rate": Fraction of the time axis covered by each stress
                           bin. For example, large tidal stress values are
                           rare so the associated stress bin covers a small
                           fraction of the time axis and even in the case of
                           tide-independent seismicity one expects
                           significantly less earthquakes in these stress bins.
        - "observed_rate": "stress_hist" normalized by the total number of
                           earthquakes.
    """

    seismicity_vs_stress = {}
    for f in fields:
        seismicity_vs_stress[f] = {}
        # values of stress at earthquake timings
        stress_at_eq = cat[f]
        # bin the values of stress at earthquake timings
        if f"{f}_bins" in stress_bins:
            bins = stress_bins[f"{f}_bins"]
        else:
            bins = np.linspace(tidal_stress[f].min(), tidal_stress[f].max(), nbins + 1)
        (
            seismicity_vs_stress[f]["stress_hist"],
            seismicity_vs_stress[f]["stress_bins"],
        ) = np.histogram(
            stress_at_eq,
            bins=bins,
        )
        # estimate the expected number of earthquakes in each bin for random seismicity
        n_total = sum(seismicity_vs_stress[f]["stress_hist"])
        tidal_stress_hist, _ = np.histogram(
            tidal_stress[f],
            bins=seismicity_vs_stress[f]["stress_bins"],
        )
        n_time_samples = np.sum(tidal_stress_hist)
        # hist/n_times = fraction of time occupied by samples with stress values in given bin
        # (hist/n_times)*ntotal = expected number of events with given stress value if
        #                         seismicity were to be independent of tidal stress
        if n_time_samples > 0:
            seismicity_vs_stress[f]["expected_rate"] = (
                tidal_stress_hist / n_time_samples
            )
        else:
            seismicity_vs_stress[f]["expected_rate"] = np.zeros_like(
                tidal_stress_hist
            )
        seismicity_vs_stress[f]["observed_rate"] = seismicity_vs_stress[f][
            "stress_hist"
        ] / np.sum(seismicity_vs_stress[f]["stress_hist"])
        # 0/0 is 0
        seismicity_vs_stress[f]["observed_rate"][
                np.isnan(seismicity_vs_stress[f]["observed_rate"])
                ] = 0.0
        # we study the ratio of the observed rate / expected rate to highlight
        # tidal triggering
        rate_ratio = (
            seismicity_vs_stress[f]["observed_rate"]
            / seismicity_vs_stress[f]["expected_rate"]
        )
        # use special value to indicate no data
        rate_ratio[np.isnan(rate_ratio)] = -10.0
        rate_ratio[np.isinf(rate_ratio)] = -10.0
        # normalization so that the ratio vector sums up to 1.
        norm = np.mean(rate_ratio[rate_ratio != -10.0])
        if norm != 0.0:
            rate_ratio[rate_ratio != -10.0] /= norm
        seismicity_vs_stress[f]["rate_ratio"] = rate_ratio

    return seismicity_vs_stress


def bin_eq_tidal_phases(
    cat,
    tidal_stress,
    nbins=9,
    phase_bins={},
    fields=[
        "instantaneous_phase_shear",
        "instantaneous_phase_normal",
        "instantaneous_phase_coulomb",
    ],
):
    """Count number of earthquakes in phase bins.

    Parameters
    -----------
    cat : `pandas.DataFrame`
        Earthquake catalog.
    tidal_stress : `pandas.DataFrame`
        All tidal stress fields.
    nbins : int, optional
        Number of default instantaneous phase bins if `phase_bins`
        is not provided. Defaults to 9.
    phase_bins : dict, optional
        Dictionary providing the phase bins, e.g.
        `phase_bins["instantaneous_phase_normal_bins"] = [-180, -170, ..., 170, 180]`
        Defaults to '{}'.
    fields : list of str, optional
        List of instantaneous phase components to explore.

    Returns
    --------
    seismicity_vs_phase : dict
        For each phase component given in 'fields', a dictionary is given
        with the following attributes:
        - "phase_hist": Histogram of instantaneous phase values at earthquake times.
        - "phase_bins": Bins of the histogram.
        - "expected_rate": Fraction of the time axis covered by each phase
                           bin. For example, some phase values are
                           rare so the associated phase bin covers a small
                           fraction of the time axis and even in the case of
                           tide-independent seismicity one expects
                           significantly less earthquakes in these phase bins.
        - "observed_rate": "phase_hist" normalized by the total number of
                           earthquakes.
    """
    seismicity_vs_phase = {}
    for f in fields:
        seismicity_vs_phase[f] = {}
    for f in fields:
        phase_at_eq = cat[f].values

        if f"{f}_bins" in phase_bins:
            bins = phase_bins[f"{f}_bins"]
        else:
            bins = np.linspace(tidal_stress[f].min(), tidal_stress[f].max(), nbins + 1)
        # bin the values of tidal phase at earthquake timings
        (
            seismicity_vs_phase[f]["phase_hist"],
            seismicity_vs_phase[f]["phase_bins"],
        ) = np.histogram(
            phase_at_eq,
            bins=bins,
        )
        # estimate the expected number of earthquakes in each bin for random seismicity
        # n_total = sum(seismicity_vs_phase[f]["phase_hist"])
        n_time_samples = len(tidal_stress[f])
        tidal_phase_hist, _ = np.histogram(
            tidal_stress[f],
            bins=seismicity_vs_phase[f]["phase_bins"],
        )
        # hist/n_times = fraction of time occupied by samples with stress values in given bin
        # (hist/n_times)*ntotal = expected number of events with given stress value if
        #                         seismicity were to be independent of tidal stress
        if n_time_samples > 0:
            seismicity_vs_phase[f]["expected_rate"] = (
                tidal_phase_hist / n_time_samples
            )
        else:
            # seismicity_vs_phase[f]["n_expected"] = np.zeros_like(tidal_phase_hist)
            seismicity_vs_phase[f]["expected_rate"] = np.zeros_like(
                tidal_phase_hist
            )
        seismicity_vs_phase[f]["observed_rate"] = (
                seismicity_vs_phase[f]["phase_hist"]
                /
                np.sum(seismicity_vs_phase[f]["phase_hist"])
                )
        # 0/0 is 0
        seismicity_vs_phase[f]["observed_rate"][
                np.isnan(seismicity_vs_phase[f]["observed_rate"])
                ] = 0.0
        seismicity_vs_phase[f]["observed_rate"][
                np.isinf(seismicity_vs_phase[f]["observed_rate"])
                ] = 0.0
        # ratio of observed to expected seismicity = corrected pdf
        rate_ratio = (
            seismicity_vs_phase[f]["observed_rate"] / seismicity_vs_phase[f]["expected_rate"]
        )
        rate_ratio[np.isnan(rate_ratio)] = 1.0
        rate_ratio[np.isinf(rate_ratio)] = 1.0
        # normalization so that the ratio vector sums up to 1.
        norm = np.mean(rate_ratio)
        if norm != 0.0:
            rate_ratio /= norm
        seismicity_vs_phase[f]["rate_ratio"] = rate_ratio
    return seismicity_vs_phase


def composite_rate_ratio_vs_stress(
    cat,
    tidal_stress,
    fortnightly_phase=None,
    nbins=36,
    short_window_days=3 * 30,
    n_short_windows=8,
    overlap=0.0,
    t_end=None,
    progress=True,
    stress_bins={},
    fields=[
        "shear_stress",
        "normal_stress",
        "coulomb_stress",
        "shear_stressing_rate",
        "normal_stressing_rate",
        "coulomb_stressing_rate",
    ],
    downsample=0,
    num_bootstrap_for_errors=100,
):
    """Compute the composite (mean or median) rate ratio vs tidal stress.

    Parameters
    -----------
    cat : `pandas.DataFrame`
        The catalog to analyze
    tidal_stress : `pandas.DataFrame`
        All tidal stress fields.
    nbins : int, optional
        Number of default stress bins if `stress_bins` is not provided.
        Defaults to 6.
    stress_bins : dict, optional
        Dictionary providing the stress bins, e.g.
        `stress_bins["shear_stress_bins"] = [-1000, -800, ..., 800, 1000]`
        Defaults to '{}'.
    fields : list of str, optional
        See `bin_eq_tidal_stresses`.
    overlap : float, optional
        Overlap between sliding short time windows.
    n_short_windows : scalar integer, default to 8
        Number of sliding short windows over which the statistics is computed.
        The length of the corresponding large window is
        `n_short_windows*(1 - overlap)*short_window_days + overlap*short_window_days`.
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

    operator = partial(np.ma.median, axis=-1)

    if fortnightly_phase == FORTNIGHTLY_PHASES[0]:
        # rising fortnightly
        subcat = cat[cat["rising_fortnightly"]]
        subtidal_stress = tidal_stress[tidal_stress["rising_fortnightly"]]
    elif fortnightly_phase == FORTNIGHTLY_PHASES[1]:
        # falling fortnightly
        subcat = cat[cat["falling_fortnightly"]]
        subtidal_stress = tidal_stress[tidal_stress["falling_fortnightly"]]
    elif (fortnightly_phase == FORTNIGHTLY_PHASES[2]) or fortnightly_phase is None:
        # keep all
        subcat = cat
        subtidal_stress = tidal_stress

    short_window_dur = relativedelta(days=short_window_days)
    short_window_shift = relativedelta(days=int((1.0 - overlap) * short_window_days))
    seismicity_vs_stress_short_win = []
    if t_end is None:
        t_end = cat["origin_time"].max()
    disable = False if progress else True
    for n in tqdm(
        range(n_short_windows), desc="Computing in sub-windows", disable=disable
    ):
        t_start = t_end - short_window_dur
        seismicity_vs_stress_short_win.append(
            bin_eq_tidal_stresses(
                subcat[
                    (subcat["origin_time"] > t_start) & (subcat["origin_time"] <= t_end)
                ],
                subtidal_stress[
                    (subtidal_stress.index > t_start) & (subtidal_stress.index <= t_end)
                ],
                nbins=nbins,
                stress_bins=stress_bins,
                fields=fields,
            )
        )
        t_end -= short_window_shift
    # aggregate results
    seismicity_vs_stress = {}
    for field1 in seismicity_vs_stress_short_win[0]:
        seismicity_vs_stress[field1] = {}
        for field2 in seismicity_vs_stress_short_win[0][field1]:
            if field2 != "stress_bins":
                all_windows = np.stack(
                    [
                        seismicity_vs_stress_short_win[i][field1][field2]
                        for i in range(n_short_windows)
                    ],
                    axis=-1,
                )
                if field2 == "expected_rate":
                    seismicity_vs_stress[field1][field2] = np.sum(
                        all_windows, axis=-1
                    ).astype("float32")
                    seismicity_vs_stress[field1][field2] /= np.float32(
                        np.sum(seismicity_vs_stress[field1][field2])
                    )
                elif field2 == "rate_ratio":
                    all_windows = np.ma.masked_array(
                        data=all_windows, mask=all_windows == -10.0
                    )
                    seismicity_vs_stress[field1][field2] = np.ma.median(
                        all_windows, axis=-1
                    )
                    (
                        _,
                        seismicity_vs_stress[field1][f"{field2}_err"],
                    ) = bootstrap_error(all_windows, operator, n_bootstraps=num_bootstrap_for_errors)
                elif field2 in ["stress_hist", "observed_rate"]:
                    seismicity_vs_stress[field1][field2] = np.median(
                        all_windows, axis=-1
                    )
                    #seismicity_vs_stress[field1][f"{field2}_err"] = np.median(
                    #    np.abs(
                    #        all_windows - np.median(all_windows, axis=-1, keepdims=True)
                    #    ),
                    #    axis=-1,
                    #)
                    (
                        _,
                        seismicity_vs_stress[field1][f"{field2}_err"],
                    ) = bootstrap_error(
                            all_windows, operator, n_bootstraps=num_bootstrap_for_errors
                            )
            if (downsample > 0) and field2 in [
                "observed_rate",
                "rate_ratio",
                "stress_hist",
                "expected_rate",
            ]:
                length = len(seismicity_vs_stress[field1][f"{field2}"])
                assert downsample * (length // downsample) == length

                seismicity_vs_stress[field1][f"{field2}"] = np.ma.median(
                    seismicity_vs_stress[field1][f"{field2}"].reshape(-1, downsample),
                    axis=-1,
                )
                if f"{field2}_err" in seismicity_vs_stress[field1]:
                    seismicity_vs_stress[field1][f"{field2}_err"] = np.ma.median(
                        seismicity_vs_stress[field1][f"{field2}_err"].reshape(
                            -1, downsample
                        ),
                        axis=-1,
                    )
            if field2 in ["rate_ratio"]:
                seismicity_vs_stress[field1][f"{field2}"] /= np.ma.mean(
                    seismicity_vs_stress[field1][f"{field2}"]
                )
            elif field2 in ["observed_rate", "expected_rate"]:
                seismicity_vs_stress[field1][f"{field2}"] /= np.ma.sum(
                    seismicity_vs_stress[field1][f"{field2}"]
                )
            else:
                seismicity_vs_stress[field1][
                    "stress_bins"
                ] = seismicity_vs_stress_short_win[0][field1]["stress_bins"]
                if downsample > 0:
                    seismicity_vs_stress[field1]["stress_bins"] = seismicity_vs_stress[
                        field1
                    ]["stress_bins"][::downsample]
    return seismicity_vs_stress


def composite_rate_ratio_vs_phase(
    cat,
    tidal_stress,
    fortnightly_phase=None,
    nbins=36,
    short_window_days=3 * 30,
    n_short_windows=8,
    overlap=0.0,
    t_end=None,
    progress=True,
    phase_bins={},
    fields=[
        "instantaneous_phase_shear",
        "instantaneous_phase_normal",
        "instantaneous_phase_coulomb",
    ],
    downsample=0,
    num_bootstrap_for_errors=100,
    error_fields=[
        "observed_rate",
        "rate_ratio",
        "phase_hist",
                ]
):
    """Count number of earthquakes in phase bins with bootstrapping analysis.

    Parameters
    -----------
    cat : `pandas.DataFrame`
        The catalog to analyze
    tidal_stress : `pandas.DataFrame`
        All tidal stress fields.
    nbins : int, optional
        Number of default instantaneous phase bins if `phase_bins`
        is not provided. Defaults to 9.
    short_window_days : iont, default to 90
        Size of the short time window, in days.
    overlap : float, default to 0.
        Overlap between sliding short time windows.
    n_short_windows : int, default to 8
        Number of sliding short windows over which the statistics is computed.
        The length of the corresponding large window is
        `n_short_windows*(1 - overlap)*short_window_days + overlap*short_window_days`.
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

    operator = partial(np.median, axis=-1)

    if fortnightly_phase == FORTNIGHTLY_PHASES[0]:
        # rising fortnightly
        subcat = cat[cat["rising_fortnightly"]]
        subtidal_stress = tidal_stress[tidal_stress["rising_fortnightly"]]
    elif fortnightly_phase == FORTNIGHTLY_PHASES[1]:
        # falling fortnightly
        subcat = cat[cat["falling_fortnightly"]]
        subtidal_stress = tidal_stress[tidal_stress["falling_fortnightly"]]
    elif (fortnightly_phase == FORTNIGHTLY_PHASES[2]) or fortnightly_phase is None:
        # keep all
        subcat = cat
        subtidal_stress = tidal_stress

    short_window_dur = relativedelta(days=short_window_days)
    short_window_shift = relativedelta(days=int((1.0 - overlap) * short_window_days))
    seismicity_vs_phase_short_win = []
    if t_end is None:
        t_end = cat["origin_time"].max()
    disable = False if progress else True
    for n in tqdm(
        range(n_short_windows), desc="Computing in sub-windows", disable=disable
    ):
        t_start = t_end - short_window_dur
        seismicity_vs_phase_short_win.append(
            bin_eq_tidal_phases(
                subcat[
                    (subcat["origin_time"] > t_start) & (subcat["origin_time"] <= t_end)
                ],
                subtidal_stress[
                    (subtidal_stress.index > t_start) & (subtidal_stress.index <= t_end)
                ],
                phase_bins=phase_bins,
                fields=fields,
                nbins=nbins,
            )
        )
        t_end -= short_window_shift
    # aggregate results
    seismicity_vs_phase = {}
    for field1 in seismicity_vs_phase_short_win[0]:
        seismicity_vs_phase[field1] = {}
        for field2 in seismicity_vs_phase_short_win[0][field1]:
            all_windows = np.stack(
                [
                    seismicity_vs_phase_short_win[i][field1][field2]
                    for i in range(n_short_windows)
                ],
                axis=-1,
            )
            if field2 == "expected_rate":
                seismicity_vs_phase[field1][field2] = np.sum(
                    all_windows, axis=-1
                ).astype("float32")
                seismicity_vs_phase[field1][field2] /= np.float32(
                    np.sum(seismicity_vs_phase[field1][field2])
                )
            if field2 in [
                "observed_rate",
                "rate_ratio",
                "phase_hist",
                        ]:
                # seismicity_vs_phase[field1][f"{field2}"] = np.mean(all_windows, axis=-1)
                seismicity_vs_phase[field1][f"{field2}"] = np.median(
                    all_windows, axis=-1
                )
                # seismicity_vs_phase[field1][f"{field2}_err"] = np.median(
                #     np.abs(
                #         all_windows - np.median(all_windows, axis=-1, keepdims=True)
                #     ),
                #     axis=-1,
                # )
                (
                    _,
                    seismicity_vs_phase[field1][f"{field2}_err"],
                ) = bootstrap_error(all_windows, operator, n_bootstraps=num_bootstrap_for_errors)
            if (downsample > 0) and field2 in [
                "observed_rate",
                "rate_ratio",
                "phase_hist",
                "expected_rate",
            ]:
                length = len(seismicity_vs_phase[field1][f"{field2}"])

                assert downsample * (length // downsample) == length

                seismicity_vs_phase[field1][f"{field2}"] = np.median(
                    seismicity_vs_phase[field1][f"{field2}"].reshape(-1, downsample),
                    axis=-1,
                )
                if f"{field2}_err" in seismicity_vs_phase[field1]:
                    seismicity_vs_phase[field1][f"{field2}_err"] = np.median(
                        seismicity_vs_phase[field1][f"{field2}_err"].reshape(
                            -1, downsample
                        ),
                        axis=-1,
                    )
            if field2 in ["rate_ratio"]:
                seismicity_vs_phase[field1][f"{field2}"] /= np.mean(
                    seismicity_vs_phase[field1][f"{field2}"]
                )
            elif field2 in ["observed_rate", "expected_rate"]:
                seismicity_vs_phase[field1][f"{field2}"] /= np.sum(
                    seismicity_vs_phase[field1][f"{field2}"]
                )
        seismicity_vs_phase[field1]["phase_bins"] = seismicity_vs_phase_short_win[0][
            field1
        ]["phase_bins"]
        if downsample > 0:
            seismicity_vs_phase[field1]["phase_bins"] = seismicity_vs_phase[field1][
                "phase_bins"
            ][::downsample]
    return seismicity_vs_phase


def bootstrap_error(x, operator, n_bootstraps=100):
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

    Example:
    --------
    >>> import numpy as np

    >>> def mean_operator(data):
    ...     return np.mean(data)

    >>> data = np.array([1, 2, 3, 4, 5])
    >>> mean, std = bootstrap_error(data, mean_operator, n_bootstraps=1000)
    >>> print(mean)
    [3.001 3.002 2.999 2.997 3.   ]
    >>> print(std)
    [0.808 0.812 0.825 0.811 0.805]
    """
    if x.ndim > 1:
        instances = np.zeros((n_bootstraps, x.shape[0]), dtype=np.float32)
    else:
        instances = np.zeros(n_bootstraps, dtype=np.float32)
    for i in range(n_bootstraps):
        replica = x[..., np.random.randint(0, x.shape[-1]-1, size=n_bootstraps)]
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

    Example:
    --------
    >>> import numpy as np

    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> yerr = np.array([0.1, 0.2, 0.15, 0.3, 0.2])

    >>> intercept, slope, intercept_err, slope_err = fit_stress_hist(x, y, yerr)
    >>> print(intercept, slope, intercept_err, slope_err)
    (0.08, 2.0, 0.0, 0.03)
    """
    yerr[yerr == 0.0] = max(10.0, 10.0 * yerr.max())
    slope, intercept, slope_err = weighted_linear_regression(x, y, W=(1.0 / yerr) ** 2)
    intercept_err = 0.0
    return intercept, slope, intercept_err, slope_err


def fit_stress_hist_scipy(x, y, yerr):
    """Least squares solution for linear regression.

    This routine uses scipy.stats's linregress function.
    """
    from scipy.stats import linregress, t

    slope, intercept, r, p, se = linregress(x, y)
    return intercept, slope, 0.0, se


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

    Example:
    --------
    >>> import numpy as np

    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> yerr = np.array([0.1, 0.2, 0.15, 0.3, 0.2])

    >>> intercept, slope, intercept_err, slope_err = fit_stress_hist_scipy_l1(x, y, yerr)
    >>> print(intercept, slope, intercept_err, slope_err)
    (0.0, 2.0, 0.0, 0.0)
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

    Example:
    --------
    >>> import numpy as np

    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> y_err = np.array([0.1, 0.2, 0.15, 0.3, 0.2])

    >>> intercept, slope, intercept_err, slope_err = fit_stress_hist_bootstrap(x, y, y_err, n_bootstrap=1000)
    >>> print(intercept, slope, intercept_err, slope_err)
    (0.08, 2.0, 0.01, 0.03)
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

    Example:
    --------
    >>> import numpy as np

    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> y_err = np.array([0.1, 0.2, 0.15, 0.3, 0.2])

    >>> param, param_min, param_max = fit_rate_state(x, y, y_err)
    >>> print(param, param_min, param_max)
    (1000.0, 900.0, 1100.0)
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

    Example:
    --------
    >>> import numpy as np

    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> y_err = np.array([0.1, 0.2, 0.15, 0.3, 0.2])

    >>> param_median, param_uncertainty = fit_rate_state_bootstrap(x, y, y_err, n_bootstrap=1000)
    >>> print(param_median, param_uncertainty)
    (1000.0, 100.0)
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

def fit_semidiurnal_triggering_bootstrap(
        x, y, y_err, n_bootstrap=10, objective="l2"
        ):
    """ """
    from scipy.optimize import minimize

    deg2rad = np.pi / 180.0
    x_ = x * deg2rad

    if objective == "l2":
        # l2-norm
        loss = lambda p, obs: np.sum(((1. + p[0] * np.cos(x_ - p[1])) - obs)**2)
    elif objective == "l1":
        # l1-norm
        loss = lambda p, obs: np.sum(np.abs((1. + p[0] * np.cos(x_ - p[1])) - obs))
    elif objective == "negative-log-likelihood":
        # negative log-likelihood
        loss = lambda p, obs: -np.sum(obs * np.log(1. + p[0] * np.cos(x_ - p[1])))
            # -np.sum(obs * np.log(1. + p[0] * np.cos(x_ - p[1]))),
            # np.array(
            #     [
            #         np.cos(x_ - p[1]) / (1. - p[0] * np.cos(x_ - p[1])),
            #         p[1] * p[0] * np.sin(x_ - p[1]) / (1. - p[0] * np.cos(x_ - p[1]))]
            # )
        # )
    first_guess = (0., 0.)
    bounds = [(0.0, 1.0), (-np.pi, np.pi)]

    inverted_alpha = np.zeros(n_bootstrap, dtype=np.float32)
    inverted_phi = np.zeros(n_bootstrap, dtype=np.float32)
    # cost = np.zeros(n_bootstrap, dtype=np.float32)
    n = 0
    # for n in range(n_bootstrap):
    while n < n_bootstrap:
        # generate random sample assuming that each bin [i] of the histogram
        # is normally distributed with mean y[i] and std y_err[i]
        y_b = np.random.normal(loc=0.0, scale=1., size=len(y))
        # noisy y
        y_b = y_b * y_err + y
        # don't allow negative values (impossible)
        y_b = np.maximum(y_b, 0.0)
        # normalized noisy y
        y_b = y_b / np.mean(y_b)
        # first_guess = (0.02 * np.random.random(), 2. * np.pi * np.random.random() - np.pi)
        first_guess = (0.05 * np.random.random(), np.pi * np.random.random() - np.pi/2.)
        optimization_results = minimize(
            loss, first_guess, args=(y_b), bounds=bounds, #jac="3-point"
        )
        inverted_alpha[n] = optimization_results.x[0]
        inverted_phi[n] = optimization_results.x[1]
        if inverted_alpha[n] == 0:
            # print(optimization_results.success, optimization_results.x)
            continue
        # cost[n] = loss(optimization_results.x, y_b)
        n += 1
    # cost /= cost.sum()
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
        axis=1
        ),
        axis=1
    )

    # alpha_std = np.sqrt(np.sum(cost * (inverted_alpha - mean_alpha)**2))
    # mean_diff = np.sum(cost * diff_phi)
    # print(inverted_alpha)
    # print(cost)
    return (
        np.mean(inverted_alpha),
        mean_phi,
        np.std(inverted_alpha),
        np.mean(diff_phi),
        # mean_alpha,
        # mean_phi,
        # alpha_std,
        # mean_diff,
    )

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
    mu=None
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
        pd.date_range(start=starttime, freq=f"{delta/3600.}H", periods=n_samples)
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
    compute_instantaneous_phase_at_eq(
        catalog, tidal_stress, fields
    )
    

def compute_instantaneous_phase_at_eq(
    catalog, tidal_stress, fields, attach_unravelled_phase=False
):
    """Interpolate instantaneous phase at earthquake timings."""
    indexes = catalog.index
    for field in fields:
        if f"unravelled_{field}" in tidal_stress:
            catalog.loc[indexes, field] = (
                np.interp(
                    catalog.loc[indexes, "t_eq_s"].values,
                    tidal_stress["time_sec"].values,
                    180.0 + tidal_stress[f"unravelled_{field}"].values,
                )
            ) % 360.0 - 180.0
            if attach_unravelled_phase:
                catalog.loc[indexes, f"unravelled_{field}"] = np.interp(
                    catalog.loc[indexes, "t_eq_s"].values,
                    tidal_stress["time_sec"].values,
                    tidal_stress[f"unravelled_{field}"].values,
                )
        else:
            catalog.loc[indexes, field] = np.interp(
                catalog.loc[indexes, "t_eq_s"].values,
                tidal_stress["time_sec"].values,
                tidal_stress[field].values,
            )


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
    if degree:
        return np.unwrap(phases, period=360.0)
    else:
        return np.unwrap(phases)


def compute_stress_at_eq(catalog, tidal_stress, fields):
    """Interpolate stress at earthquake timings."""
    indexes = catalog.index
    for f in fields:
        catalog.loc[indexes, f] = np.interp(
            catalog.loc[indexes, "t_eq_s"].values, tidal_stress["time_sec"].values, tidal_stress[f].values
        )


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

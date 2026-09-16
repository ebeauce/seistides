import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable


def schuster_test(
    times, period, fft_vs_period_interpolator=None, return_trajectory=False, t_ref=0.0
):
    """
    Apply the Schuster test to the time series `times` for the test `period`.

    It computes the log probability that the `times` form a `period`-periodicity
    Schuster walk of length D out of chance.

    Parameters
    ----------
    t_serie : numpy.ndarray
        Time series data
    periods : numpy.ndarray
        Vector containing all the periods to be tested.
    t_ref : float, optional
        Origin time of the time series. Default to 0.
    fft_vs_period_interpolator : func, optional
        Function that interpolates the complex Fourier transform of the time series at
        `periods`. The phase of the Fourier transform is used to determine the phase
        of the Schuster walk. The Fourier transform must be computed on a time series
        that starts at `t_ref`, otherwise the inferred Schuster walk phase will
        be meaningless. Defaults to None.
    return_trajectory : bool, optional
        If True, returns the entire trajectory of the Schuster walk in the complex plane.

    Returns
    -------
    log_prob : numpy.ndarray
        The logarithm of the probability of a random walk
    trajectory_phase : numpy.ndarray
        The phase of the Schuster walk at each input period.
    complex_trajectory : numpy.ndarray, optional
        The entire Schuster walk represented by a series of complex numbers.
        It is returned only if `return_trajectory` is True.
    """
    times = times - t_ref

    times = times[times < period * np.floor(times.max() / period)]

    if fft_vs_period_interpolator is not None:
        # !! t_ref must match the beginning of the time series on which fft was computed !!
        fft_at_T = fft_vs_period_interpolator(period)
        phase_0 = np.angle(fft_at_T)
    else:
        phase_0 = 0.0

    # find phases in cycle of period T
    phases = np.mod(times, period) * 2 * np.pi / period - phase_0
    # build unit vectors on the complex circle
    complex_steps = np.exp(1j * phases)
    # complex_trajectory = np.cumsum(complex_steps)
    complex_trajectory_end = np.sum(complex_steps)

    # Point where the Schuster walk ends
    end_walk = np.array([complex_trajectory_end.real, complex_trajectory_end.imag])
    # Distance from the origin
    D = np.sqrt(np.sum(end_walk**2))
    # angle / phase of the trajectory
    trajectory_phase = np.angle(complex_trajectory_end)
    # Probability to reach the same point by random walk
    log_prob = -(D**2) / len(times)

    if return_trajectory:
        complex_trajectory = np.cumsum(complex_steps)
        return log_prob, trajectory_phase, complex_trajectory
    else:
        return log_prob, trajectory_phase


def _schuster_spectrum(t_serie, periods, t_ref=0.0, fft_vs_period_interpolator=None):
    """
    Compute the logarithm of the probability of a random walk for a given time series and set of periods.

    Parameters
    ----------
    t_serie : numpy.ndarray
        Time series data
    periods : numpy.ndarray
        Vector containing all the periods to be tested.
    t_ref : float, optional
        Origin time of the time series. Default to 0.
    fft_vs_period_interpolator : func, optional
        Function that interpolates the complex Fourier transform of the time series at
        `periods`. The phase of the Fourier transform is used to determine the phase
        of the Schuster walk. The Fourier transform must be computed on a time series
        that starts at `t_ref`, otherwise the inferred Schuster walk phase will
        be meaningless. Defaults to None.

    Returns
    -------
    log_prob : numpy.ndarray
        The logarithm of the probability of a random walk
    trajectory_phase : numpy.ndarray
        The phase of the Schuster walk at each input period.
    """
    log_prob = np.zeros(len(periods))
    trajectory_phase = np.zeros(len(periods))
    n_per = len(periods)  # Number of periods tested
    t_serie = t_serie - t_ref

    for i in range(n_per):
        T = periods[i]

        log_prob[i], trajectory_phase[i] = schuster_test(
            t_serie,
            T,
            fft_vs_period_interpolator=fft_vs_period_interpolator,
            t_ref=t_ref,
        )
    return log_prob, trajectory_phase


def compute_schuster_spectrum(
    time_series, period_min, period_max, fft_vs_period_interpolator=None, t_ref=0.0
):
    """
    Compute the logarithm of the probability of a random walk for a given time series and set of periods.

    Parameters
    ----------
    time_serie : numpy.ndarray
        Time series data.
    period_min : float
        The shortest period to analyze, in the same units as `time_series`.
    period_max : float
        The longest period to analyze, in the same units as `time_series`.
    t_ref : float, optional
        Origin time of the time series. Default to 0.
    fft_vs_period_interpolator : func, optional
        Function that interpolates the complex Fourier transform of the time series at
        `periods`. The phase of the Fourier transform is used to determine the phase
        of the Schuster walk. The Fourier transform must be computed on a time series
        that starts at `t_ref`, otherwise the inferred Schuster walk phase will
        be meaningless. Defaults to None.

    Returns
    -------
    log_prob : numpy.ndarray
        The logarithm of the probability of a random walk
    trajectory_phase : numpy.ndarray
        The phase of the Schuster walk at each input period.
    test_periods : numpy.ndarray
        The periods at which the Schuster test was applied.
    """

    time_series = time_series - time_series.min()
    duration = time_series.max()
    freq_min = 1.0 / period_max
    freq_max = 1.0 / period_min
    test_periods = 1.0 / np.arange(freq_min, freq_max, 1.0 / duration)

    log_p, trajectory_phase = _schuster_spectrum(
        time_series,
        test_periods,
        fft_vs_period_interpolator=fft_vs_period_interpolator,
        t_ref=t_ref,
    )

    return log_p, trajectory_phase, test_periods


def plot_schuster_spectrum(
    log_prob,
    test_periods,
    obs_duration,
    trajectory_phase=None,
    p_min=None,
    figsize=(10, 5),
    **kwargs,
):
    """
    Plot the Schuster spectrum.

    Parameters
    ----------
    log_prob : numpy.ndarray
        The logarithm of the probability of a random walk
    test_periods : numpy.ndarray
        The periods at which the Schuster test was applied.
    obs_duration : float
        The duration of observation (of the catalog) in units of `test_periods`.
    trajectory_phase : numpy.ndarray, optional
        If not None, the phase of the Schuster walk at each input period is used
        to color-code each dot.
    p_min : float, optional
        If not None, `p_min` is used to truncate the y-axis.
    figsize : tuple, optional
        Figure size, in inches. Defaults to (10, 5).

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Figure.
    ax : matplotlib.pyplot.Axis
        The figure axis.
    """

    # Spectrum
    p_val = np.exp(log_prob)
    eps_th = 1

    if trajectory_phase is not None:
        scalar_map = ScalarMappable(
            norm=Normalize(vmin=-np.pi, vmax=np.pi),
            cmap="twilight_shifted",
        )

    fig, ax = plt.subplots(figsize=figsize)

    # Expected value
    expected = eps_th * test_periods / obs_duration
    ax.plot(
        test_periods,
        expected,
        ls=":",
        color="k",
        lw=1.0,
        label="Expected value",
        zorder=1,
    )
    # Points above the expected value: change size and color
    sig_pts = p_val < expected
    test_periods_sig = test_periods[sig_pts]
    p_val_sig = p_val[sig_pts]
    c_diff = np.log10(p_val_sig) - np.log10(eps_th * test_periods_sig / obs_duration)
    c_diff[c_diff < -2] = -2
    c_diff = 2 * c_diff / 10
    # print(c_diff)

    if trajectory_phase is not None:
        color = scalar_map.to_rgba(trajectory_phase[sig_pts])
    else:
        color = "C2"
    ax.scatter(
        test_periods_sig,
        p_val_sig,
        marker="o",
        color=color,
        linewidths=1.0,
        edgecolor="k",
        s=50,
        zorder=2,
        # rasterized=True
    )

    # 99% confidence level
    ax.plot(
        test_periods,
        0.01 * eps_th * test_periods / obs_duration,
        ls="-.",
        color="k",
        lw=1.5,
        label="99% confidence",
        zorder=1,
    )

    # 95% confidence level
    ax.plot(
        test_periods,
        0.05 * eps_th * test_periods / obs_duration,
        ls="--",
        color="k",
        lw=1.5,
        label="95% confidence",
        zorder=1,
    )

    # Points above the 99% confidence level
    sig_pts_99 = p_val < 0.01 * expected
    test_periods_99 = test_periods[sig_pts_99]
    p_val_99 = p_val[sig_pts_99]
    valid = p_val_99 > 0.0
    m_size = (
        5
        - np.log10(p_val_99[valid])
        + np.log10(0.01 * eps_th * test_periods_99[valid] / obs_duration)
    )

    if trajectory_phase is not None:
        color = scalar_map.to_rgba(trajectory_phase[sig_pts_99])
        color = color[valid]
    else:
        color = "C3"
    ax.scatter(
        test_periods_99[valid],
        p_val_99[valid],
        s=50 + m_size,
        color=color,
        linewidths=1.0,
        edgecolor="k",
        zorder=3,
        # rasterized=True
    )

    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Schuster p-value")
    if p_min is None:
        p_min = 1.0e-10
    ax.set_ylim(p_min, 1.0)
    ax.invert_yaxis()

    # plot the rest
    ax.scatter(
        test_periods[~sig_pts],
        p_val[~sig_pts],
        color="dimgrey",
        s=5,
        alpha=0.25,
        marker="o",
        rasterized=True,
    )

    if trajectory_phase is not None:
        pos = ax.get_position()
        cax = fig.add_axes(
            [pos.x1 + kwargs.get("x_offset_cbar", 0.01), pos.y0, 0.02, pos.height]
        )
        plt.colorbar(scalar_map, cax=cax, label="Phase")

        cbar_tick_loc = [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi]
        cbar_tick_lab = [
            r"$-\pi$",
            r"$-\frac{\pi}{2}$",
            r"$0$",
            r"$\frac{\pi}{2}$",
            r"$\pi$",
        ]
        cax.set_yticks(cbar_tick_loc)
        cax.set_yticklabels(cbar_tick_lab)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.legend(loc="upper right")
    return fig, ax


def Schuster_plot(t_serie, periods):
    """
    Plot the Schuster walk for a given time series and set of periods.

    Parameters
    ----------
    t_serie : numpy.ndarray
        Time series data
    periods : numpy.ndarray
        Vector containing all the periods to be tested.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object of the plot
    """
    t_serie = t_serie.flatten()

    n_per = len(periods)  # Number of periods tested

    for i in range(n_per):
        T = periods[i]
        t_serie = t_serie - min(t_serie) + min(t_serie) % T
        # So the time series starts at time 0.
        tlim = max(t_serie) - (max(t_serie) - min(t_serie)) % T
        # To have a round number of cycles.
        # If not it induces artefacts to Schuster.
        t = t_serie[
            t_serie <= tlim
        ]  # selects a round number of cycles from the timeseries.

        phase = t * 2 * np.pi / T  # Phase all the times from the timeseries
        # with respect to the period T.

        end_walk = np.cumsum(np.column_stack((np.cos(phase), np.sin(phase))), axis=0)
        # Point where we end the Schuster walk

        D_end = np.linalg.norm(end_walk[-1])
        # Probability to reach the same point by random walk
        log_prob = -(D_end**2) / len(t)

        fig, ax = plt.subplots()

        #     Plot the walk itself
        ax.plot(
            np.column_stack(([0], end_walk[:, 0])),
            np.column_stack(([0], end_walk[:, 1])),
        )
        ax.scatter(
            np.column_stack(([0], end_walk[:, 0])),
            np.column_stack(([0], end_walk[:, 1])),
            s=15 * np.ones(len(end_walk[:, 0])),
            c=np.floor(t / T),
            marker="o",
            edgecolors="none",
        )

        #     Plot a big dot at the starting and end points
        ax.plot(
            [0, end_walk[-1, 0]],
            [0, end_walk[-1, 1]],
            ".",
            markerfacecolor="r",
            markeredgecolor="k",
            markersize=5,
        )

        #     Plot circles at probabilities 10^(-2) and 10^(-5)
        p = [0.01, 1e-5]  # probability to get there by random walk
        # <=> 1-p: confidence that being that far isn't due to random walk
        theta = np.linspace(0, 2 * np.pi, num=100)

        for j in range(len(p)):
            D = np.sqrt(len(t) * np.log(1 / p[j]))
            x_circ = D * np.cos(theta)
            y_circ = D * np.sin(theta)

            ax.plot(x_circ, y_circ, "--k")
            conf = str(p[j] * 100) + "%"
            ax.text(x_circ[120], y_circ[120], conf, ha="left", va="bottom")

        ax.axis("equal")
    return fig

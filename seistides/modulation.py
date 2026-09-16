import numpy as np
import pandas as pd
import pickle
import warnings

from abc import ABC, abstractmethod
from functools import partial
from . import utils


class Modulationmeter(ABC):
    """A class to measure the modulation of seismicity by some forcing."""

    def __init__(
        self,
        downsample=None,
        catalog=None,
        forcing=None,
        forcing_bins=None,
        window_type="backward",
    ):
        """
        Idea: initialize with a "modulation_parameters" dictionary that is
        systematically passed to self.measure_modulation and a "fit_parameters"
        dictionary that is systematically passsed to self.fit_modualtion.

        """
        assert window_type in {
            "forward",
            "backward",
        }, "window_type should be either of 'backward' or 'forward'."
        if downsample is None:
            downsample = 0
        self.downsample = downsample
        self.set_catalog(catalog)
        self.forcing = forcing
        self.forcing_bins = forcing_bins
        self.window_type = window_type
        self.modulation = {}
        self.model = {}

    @abstractmethod
    def measure_modulation(self, func, window_time, forcing_name, **kwargs):
        """ """
        if self.catalog is None:
            warnings.warn(
                "You need to define the `catalog` attribute. See `set_catalog`."
            )
            return
        if self.forcing is None:
            warnings.warn(
                "You need to define the `forcing` attribute. See `set_forcing`."
            )
            return

        # 'hidden' kwarg for `shuffle_and_measure_modulation`
        catalog = kwargs.get("catalog", None)
        if catalog is None:
            catalog = self.catalog

        ## -----------------------------------------------
        ## here, write func(args, kwargs) appropriately
        ## for example:
        # modulation = func(
        #    catalog,
        #    self.forcing,
        #    window_time,
        #    forcing_name,
        #    short_window_days=self.short_window_days,
        #    num_short_windows=self.num_short_windows,
        #    overlap=self.overlap,
        #    downsample=self.downsample,
        #    **kwargs,
        # )
        ## -----------------------------------------------

        if modulation is None:
            print("Could not estimate seismicity vs forcing!")
            print("(probably because of too few events in the catalog)")
            return

        if forcing_name not in self.modulation:
            self.modulation[forcing_name] = {}
        self.modulation[forcing_name][window_time] = modulation[forcing_name]
        self.modulation[forcing_name][window_time]["midbins"] = self._midbins(
            self.modulation[forcing_name][window_time]["bins"]
        )

    def adjust_forcing_table_to_catalog(self):
        """Adjust forcing table to the catalog's time span."""
        if self.catalog is None:
            warnings.warn(
                "You need to define the `catalog` attribute. See `set_catalog`."
            )
            return
        if self.forcing is None:
            warnings.warn(
                "You need to define the `forcing` attribute. See `set_forcing`."
            )
            return
        cat_t_min = self.catalog["origin_time"].min()
        cat_t_max = self.catalog["origin_time"].max()
        dt_forcing = self.forcing.index[1] - self.forcing.index[0]  # .total_seconds()
        buffer_edges = pd.Timedelta("1d")
        within_catalog = (
            self.forcing.index > cat_t_min - dt_forcing - buffer_edges
        ) & (self.forcing.index < cat_t_max + buffer_edges)
        self.forcing = self.forcing[within_catalog]

    def get_parameter_time_series(self, forcing_name, model_name="model1"):
        """
        Retrieve time series of parameters and errors for a given forcing and model.

        Parameters
        ----------
        forcing_name : str
            The name of the forcing.
        model_name : str, optional
            The name of the model. Default is "model1".

        Returns
        -------
        time : list
            Sorted list of time points.
        parameters : dict
            Dictionary containing parameter time series.
        errors : dict
            Dictionary containing error time series.

        Notes
        -----
        If `forcing_name` is not found in `self.modulation` or `self.model`, a warning is issued,
        and `None` is returned for all outputs.
        """

        if forcing_name not in self.modulation:
            warnings.warn(f"{forcing_name} not in self.modulation")
            return
        if forcing_name not in self.model:
            warnings.warn(f"{forcing_name} not in self.model")
            return
        time = []
        for t in self.model[forcing_name]:
            time.append(t)
        time.sort()
        parameters = {}
        errors = {}
        for param in self.model[forcing_name][time[0]][model_name]["parameters"]:
            parameters[param] = []
        for err in self.model[forcing_name][time[0]][model_name]["errors"]:
            errors[err] = []
        for t in time:
            _mod = self.model[forcing_name][t][model_name]
            for param in _mod["parameters"]:
                parameters[param].append(_mod["parameters"][param])
            for err in _mod["errors"]:
                errors[err].append(_mod["errors"][err])
        for param in _mod["parameters"]:
            parameters[param] = np.asarray(parameters[param])
        for err in _mod["errors"]:
            errors[err] = np.asarray(errors[err])
        return time, parameters, errors

    def get_model_performance_time_series(self, forcing_name, model_name="model1"):
        """
        Retrieve time series of parameters and errors for a given forcing and model.

        Parameters
        ----------
        forcing_name : str
            The name of the forcing.
        model_name : str, optional
            The name of the model. Default is "model1".

        Returns
        -------
        time : list
            Sorted list of time points.
        performance_metrics : dict
            Dictionary containing performance metric time series.

        Notes
        -----
        If `forcing_name` is not found in `self.modulation` or `self.model`, a warning is issued,
        and `None` is returned for all outputs.
        """
        if forcing_name not in self.modulation:
            warnings.warn(f"{forcing_name} not in self.modulation")
            return
        if forcing_name not in self.model:
            warnings.warn(f"{forcing_name} not in self.model")
            return
        time = []
        for t in self.modulation[forcing_name]:
            time.append(t)
        time.sort()
        performance_metrics = {
                "delta_aic": [],
                "rms_residual": [],
                "r2": []
                }
        for t in time:
            _mod = self.model[forcing_name][t][model_name]
            for param in performance_metrics.keys():
                performance_metrics[param].append(_mod[param])
        for param in performance_metrics:
            performance_metrics[param] = np.asarray(performance_metrics[param])
        return time, performance_metrics

    def get_obs_vs_forcing_vs_time(self, forcing_name, obs_name, return_time=True):
        """ """
        time_periods = list(self.modulation[forcing_name].keys())
        time_periods.sort()

        obs = np.stack(
            [self.modulation[forcing_name][tp][obs_name] for tp in time_periods], axis=0
        )

        if return_time:
            return obs, pd.to_datetime(time_periods)
        else:
            return obs

    def get_model_vs_forcing_vs_time(self, forcing_name, obs_name):
        """ """
        time_periods = list(self.model[forcing_name].keys())
        time_periods.sort()

        model = np.stack(
            [self.model[forcing_name][tp][obs_name] for tp in time_periods], axis=0
        )

        return model, pd.to_datetime(time_periods)

    def fit_modulation(
        self,
        func,
        window_time,
        forcing_name,
        model_name="model1",
        quantity="relative_rate",
        **kwargs,
    ):
        """ """
        if forcing_name not in self.model:
            self.model[forcing_name] = {}
        if window_time not in self.model[forcing_name]:
            self.model[forcing_name][window_time] = {}
        self.model[forcing_name][window_time][model_name] = func(
            self.modulation[forcing_name][window_time]["midbins"],
            self.modulation[forcing_name][window_time][quantity],
            self.modulation[forcing_name][window_time][f"{quantity}_err"],
            **kwargs,
        )

    def evaluate_goodnessoffit(
        self, window_time, forcing_name, quantity="relative_rate", model_name="model1"
    ):
        """ """
        if (
            forcing_name not in self.model
            or window_time not in self.model[forcing_name]
            or model_name not in self.model[forcing_name][window_time]
        ):
            print(
                f"Could not find {model_name} at self.model[{forcing_name}][{window_time}]"
            )
            print("Call self.fit_modulation first.")
            return
        obs = self.modulation[forcing_name][window_time][quantity]
        proposed_model = self.model[forcing_name][window_time][model_name]["func"](
            np.deg2rad(self.modulation[forcing_name][window_time]["midbins"])
        )
        residuals_proposed_model = obs - proposed_model
        num_params = len(
            self.model[forcing_name][window_time][model_name]["parameters"]
        )
        residuals_null_hypothesis = obs - 1.0
        self.model[forcing_name][window_time][model_name]["aic"] = utils.compute_AIC(
            residuals_proposed_model, num_params
        )
        self.model[forcing_name][window_time][model_name]["aic_null"] = (
            utils.compute_AIC(residuals_null_hypothesis, 0)
        )
        # if this is positive, the proposed model is preferred over the null hypothesis
        self.model[forcing_name][window_time][model_name]["delta_aic"] = (
            self.model[forcing_name][window_time][model_name]["aic_null"]
            - self.model[forcing_name][window_time][model_name]["aic"]
        )
        self.model[forcing_name][window_time][model_name]["rms_residual"] = np.std(
            residuals_proposed_model
        )
        self.model[forcing_name][window_time][model_name]["variance_reduction"] = (
                100. * (1. - np.sum(residuals_proposed_model**2) / np.sum(obs**2))
                )
        obs_c = obs - obs.mean()
        self.model[forcing_name][window_time][model_name]["r2"] = (
                100. * (1. - np.sum(residuals_proposed_model**2) / np.sum(obs_c**2))
                )

    def plot_modulation(
            self, forcing, window_name, ymin=0.75, ymax=1.25, model_name=None, **kwargs
            ):
        """
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        norm = Normalize(vmin=ymin, vmax=ymax)
        scalar_map = ScalarMappable(
                norm=norm, cmap=kwargs.get("cmap", plt.get_cmap("coolwarm"))
                )

        fig, ax = plt.subplots()
        _mod = self.modulation[forcing][window_name]
        _bins = _mod["bins"]
        _midbins = _mod["midbins"]
        ax.bar(
                _bins[:-1],
                _mod["relative_rate"],
                yerr=_mod["relative_rate_err"],
                width=_bins[1]-_bins[0],
                edgecolor=kwargs.get("edgecolor", "none"),
                align=kwargs.get("align", "edge"),
                color=scalar_map.to_rgba(_mod["relative_rate"]),
                )
        ax.axhline(1.0, lw=0.50, color="k")

        if model_name is not None:
            if window_name in self.model[forcing]:
                _model = self.model[forcing][window_name][model_name]
                fit = _model["func"](np.deg2rad(_midbins))
                ax.plot(_midbins, fit, color="k", label="Model")

        xtick_loc = [-180.0, -90.0, 0, 90.0, 180.0]
        xtick_lab = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]
        ax.set_xticks(xtick_loc)
        ax.set_xticklabels(xtick_lab)

        ax.set_ylabel(r"Relative rate of seismicity")
        ax.set_xlabel(r"Tidal phase, $\phi$ (rad)")

        ax.grid()
        ax.set_ylim(ymin, ymax)

        return fig


    def set_catalog(self, catalog):
        if catalog is None:
            self.catalog = None
        else:
            self.catalog = catalog
            self.catalog.loc[:, "origin_time"] = pd.to_datetime(self.catalog.loc[:, "origin_time"])
            if "event_timing_sec" not in self.catalog:
                self.catalog["event_timing_sec"] = (
                    pd.to_datetime(self.catalog["origin_time"])
                    .values.astype("datetime64[ms]")
                    .astype("float64")
                    / 1000.0
                )

    def set_forcing(self, forcing):
        self.forcing = forcing

    def set_forcing_bins(self, forcing_bins):
        self.forcing_bins = forcing_bins

    def write(self, path):
        with open(path, "wb") as fout:
            pickle.dump(self, fout)

    @classmethod
    def _midbins(cls, bins):
        return (bins[1:] + bins[:-1]) / 2.0

    @classmethod
    def read(cls, path):
        with open(path, "rb") as fin:
            instance = pickle.load(fin)
        return instance


class ModulationmeterForcingTimeBins(Modulationmeter):

    def __init__(
        self,
        forcing_bins=None,
        downsample=None,
        catalog=None,
        forcing=None,
        window_duration_days=None,
        window_type="backward",
    ):
        super().__init__(
            catalog=catalog,
            forcing=forcing,
            forcing_bins=forcing_bins,
            downsample=downsample,
            window_type=window_type,
        )
        self.window_duration_days = window_duration_days

    @property
    def window_duration_sec(self):
        return self.window_duration_days * 24.0 * 3600.0

    def build_forcingtime_bins(self, first_order_correction=False):
        self.forcingtime_bins = {}
        for forcing_name in self.forcing_bins:
            dt = (
                self.forcing[forcing_name].index[1]
                - self.forcing[forcing_name].index[0]
            ).total_seconds()
            if dt > 61.0:
                warnings.warn(
                    f"Your forcing time series is sampled at {dt:.1f}sec. "
                    "Any sampling coarser than 60sec may significantly misestimate "
                    "the duration of each phase-time bin."
                )
            forcing_bin_membership = np.digitize(
                np.clip(
                    self.forcing[forcing_name],
                    a_min=self.forcing_bins[forcing_name].min(),
                    a_max=self.forcing_bins[forcing_name].max(),
                ),
                self.forcing_bins[forcing_name],
            )

            # treat edge cases
            bin_range = (
                self.forcing_bins[forcing_name].max()
                - self.forcing_bins[forcing_name].min()
            )
            if np.abs(bin_range - 360.0) < 0.1:
                cyclic_bins = True
            else:
                cyclic_bins = False
            bin_width = np.abs(
                self.forcing_bins[forcing_name][1] - self.forcing_bins[forcing_name][0]
            )

            # upper edge
            if cyclic_bins:
                if np.random.random() > 0.5:
                    forcing_bin_membership[
                        forcing_bin_membership == len(self.forcing_bins[forcing_name])
                    ] = (len(self.forcing_bins[forcing_name]) - 1)
                else:
                    forcing_bin_membership[
                        forcing_bin_membership == len(self.forcing_bins[forcing_name])
                    ] = 1
            else:
                forcing_bin_membership[
                    forcing_bin_membership == len(self.forcing_bins[forcing_name])
                ] = (len(self.forcing_bins[forcing_name]) - 1)
            # lower edge
            if cyclic_bins:
                if np.random.random() > 0.5:
                    forcing_bin_membership[forcing_bin_membership == 0] = (
                        len(self.forcing_bins[forcing_name]) - 1
                    )
                else:
                    forcing_bin_membership[forcing_bin_membership == 0] = 1
            else:
                forcing_bin_membership[forcing_bin_membership == 0] = 1

            # find when the forcing time series transitions from
            # one bin to another
            # construct jumps_after such that `jumps_after[i]` indicates
            # that the time series jumped from bin to another between
            # time_series[jumps_after[i]-1] and time_series[jumps_after[i]]
            jumps_after = (
                np.abs(forcing_bin_membership[1:] - forcing_bin_membership[:-1]) > 0
            ).astype(bool)
            jumps_after = np.hstack((True, jumps_after))
            jumps_after = np.where(jumps_after)[0]

            # associate the corresponding times
            # the times at `jumps_after` are start times because the bins actually
            # start a little bit before the samples at `jumps_after`
            forcingtime_bin_starttime = self.forcing[forcing_name].index[jumps_after]
            forcingtime_bin_starttime_sec = (
                forcingtime_bin_starttime.values.astype("datetime64[ms]").astype(
                    "float64"
                )
                / 1000.0
            )

            if first_order_correction:
                warnings.warn("first_order_correction feature is not yet operational!")
                _jumps_after = jumps_after[1:]
                _jumps_before = _jumps_after - 1
                if cyclic_bins:
                    _forcing = np.unwrap(
                        self.forcing[forcing_name].astype("float128"), period=360.0
                    )  # .values
                else:
                    _forcing = self.forcing[forcing_name].values
                bin_min = self.forcing_bins[forcing_name].min()
                bin_lower_edge = (
                    bin_min + np.floor((_forcing - bin_min) / bin_width) * bin_width
                )
                bin_upper_edge = bin_lower_edge + bin_width

                dforcing = _forcing[_jumps_after] - _forcing[_jumps_before]
                # gradient = dforcing / dt
                transition_times = np.zeros(len(_jumps_before), dtype=np.float64)
                # positive jumps
                positive = dforcing > 0.0
                time_to_upper_bin = (
                    dt
                    * (
                        bin_upper_edge[_jumps_before[positive]]
                        - _forcing[_jumps_before[positive]]
                    )
                    / dforcing[positive]
                )
                transition_times[positive] = (
                    self.forcing["time_sec"].values[_jumps_before[positive]]
                    + time_to_upper_bin
                    # + np.maximum(0., time_to_upper_bin)
                )
                # negative jumps
                negative = dforcing < 0.0
                time_to_lower_bin = (
                    dt
                    * (
                        bin_lower_edge[_jumps_before[negative]]
                        - _forcing[_jumps_before[negative]]
                    )
                    / dforcing[negative]
                )
                transition_times[negative] = (
                    self.forcing["time_sec"].values[_jumps_before[negative]]
                    + time_to_lower_bin
                    # + np.maximum(0., time_to_lower_bin)
                )
                # if this happens, it means that the numerical precision
                # has reached its limit when unwrapping the phase
                transition_times[dforcing == 0.0] = forcingtime_bin_starttime_sec[1:][
                    dforcing == 0.0
                ]
                transition_times = np.hstack(
                    (forcingtime_bin_starttime_sec[0], transition_times)
                )

                correction = transition_times - forcingtime_bin_starttime_sec

                forcingtime_bin_starttime_sec = transition_times

            forcingtime_bin_duration_sec = (
                forcingtime_bin_starttime_sec[1:] - forcingtime_bin_starttime_sec[:-1]
            )
            forcingtime_bins = {
                "forcing_bin_membership": forcing_bin_membership[jumps_after],
                "forcingtime_bin_starttime_sec": forcingtime_bin_starttime_sec,
                "forcingtime_bin_duration_sec": np.hstack(
                    (forcingtime_bin_duration_sec, np.nan)
                ),
            }
            self.forcingtime_bins[forcing_name] = pd.DataFrame(forcingtime_bins)
            self.forcingtime_bins[forcing_name]["forcing_bin_membership"] = (
                self.forcingtime_bins[forcing_name]["forcing_bin_membership"].astype(
                    "category"
                )
            )

    def count_events_in_forcingtime_bins(self, attach_membership_to_cat=False):

        for forcing_name in self.forcing_bins:
            if (
                self.catalog["event_timing_sec"].values.max()
                > self.forcingtime_bins[forcing_name][
                    "forcingtime_bin_starttime_sec"
                ].max()
            ):
                warnings.warn(
                    "Forcing time series stop before end of catalog! "
                    "Aborting `count_events_in_forcingtime_bins`."
                )
                continue
            if (
                self.catalog["event_timing_sec"].values.min()
                < self.forcingtime_bins[forcing_name][
                    "forcingtime_bin_starttime_sec"
                ].min()
            ):
                warnings.warn(
                    "Forcing time series starts after beginning of catalog! "
                    "Aborting `count_events_in_forcingtime_bins`."
                )
                continue
            # note: we need "- 1" because digitize returns `i` for edge[i-1] < x < edge[i]
            #       but one row of our table gives the start time, duration and eq count of a
            #       given bin
            forcingtime_bin_eq_membership = (
                np.digitize(
                    self.catalog["event_timing_sec"].values,
                    self.forcingtime_bins[forcing_name][
                        "forcingtime_bin_starttime_sec"
                    ],
                )
                - 1
            )
            if attach_membership_to_cat:
                self.catalog[f"membership_{forcing_name}"] = (
                    forcingtime_bin_eq_membership
                )
            forcingtime_bin_values, forcingtime_bin_counts = np.unique(
                forcingtime_bin_eq_membership, return_counts=True
            )
            # initialize
            self.forcingtime_bins[forcing_name]["forcingtime_bin_count"] = 0
            # add counts
            self.forcingtime_bins[forcing_name].loc[
                forcingtime_bin_values, "forcingtime_bin_count"
            ] = forcingtime_bin_counts
            # self.forcingtime_bins[forcing_name].fillna(
            #    {"forcingtime_bin_count": 0}, inplace=True
            # )

            # first row is just here to indicate beginning of first forcingtime bin,
            # but because each row gives stats for bin to the left,
            # the first row has no stat by construction
            # self.forcingtime_bins[forcing_name].loc[0, "forcingtime_bin_count"] = pd.NA
            self.forcingtime_bins[forcing_name].loc[
                self.forcingtime_bins[forcing_name].index[-1], "forcingtime_bin_count"
            ] = pd.NA

    def detect_catalog_gaps(self, event_count_bin_size="1D"):
        """ """
        if self.catalog is None:
            warnings.warn(
                "You need to define the `catalog` attribute. See `set_catalog`."
            )
            return
        warnings.warn(
            "This gap detection routine only works for high seismicity rates."
        )
        event_occ = pd.Series(
            index=pd.to_datetime(self.catalog["origin_time"]),
            data=np.ones(len(self.catalog), dtype=np.int32),
        )
        event_count = (
            event_occ.groupby(pd.Grouper(freq=event_count_bin_size)).sum().sort_index()
        )
        gaps = event_count[event_count == 0]
        gaps = gaps.to_period(event_count_bin_size)

        self.gaps = gaps.index

    def flag_forcingtime_bins_in_gaps(self):
        gaps = pd.IntervalIndex.from_arrays(
            self.gaps.start_time.astype("datetime64[ms]").values.astype("float64")
            / 1000.0,
            self.gaps.end_time.astype("datetime64[ms]").values.astype("float64")
            / 1000.0,
        )
        for forcing_name in self.forcingtime_bins:
            indexes_in = (
                pd.cut(
                    self.forcingtime_bins[forcing_name][
                        "forcingtime_bin_starttime_sec"
                    ],
                    gaps,
                    include_lowest=True,
                )
                .dropna()
                .index
            )
            self.forcingtime_bins[forcing_name]["gaps"] = False
            self.forcingtime_bins[forcing_name].loc[indexes_in, "gaps"] = True

    def remove_gaps(self):
        for forcing_name in self.forcingtime_bins:
            self.forcingtime_bins[forcing_name] = self.forcingtime_bins[forcing_name][
                ~self.forcingtime_bins[forcing_name]["gaps"]
            ]

    def measure_modulation(
        self, window_time, forcing_name, rate_estimate_kwargs={}, **kwargs
    ):
        """ """
        from dateutil.relativedelta import relativedelta

        if self.catalog is None:
            warnings.warn(
                "You need to define the `catalog` attribute. See `set_catalog`."
            )
            return
        if self.forcingtime_bins is None:
            warnings.warn(
                "You need to call `build_forcingtime_bins` and `count_events_in_forcingtime_bins` first."
            )
            return

        # 'hidden' kwarg for `shuffle_and_measure_modulation`
        catalog = kwargs.get("catalog", None)
        if catalog is None:
            catalog = self.catalog

        # -----------------------------------------------
        #                define window
        assert self.window_type in {
            "backward",
            "forward",
        }, "window_type should be either of 'backward' or 'forward'"
        window_time = pd.Timestamp(window_time)
        if self.window_type == "backward":
            t_end = window_time
            t_start = t_end - relativedelta(days=self.window_duration_days)
        elif self.window_type == "forward":
            t_start = window_time
            t_end = t_start + relativedelta(days=self.window_duration_days)
        # -----------------------------------------------

        # -----------------------------------------------
        subcat = catalog[
            (catalog["origin_time"] > t_start) & (catalog["origin_time"] <= t_end)
        ]
        subforcingtime_bins = self.forcingtime_bins[forcing_name]
        subforcingtime_bins = subforcingtime_bins[
            (subforcingtime_bins["forcingtime_bin_starttime_sec"] > t_start.timestamp())
            & (
                subforcingtime_bins["forcingtime_bin_starttime_sec"]
                <= t_end.timestamp()
            )
        ]

        modulation = utils.estimate_rate_forcingtime_bins(
            subcat,
            subforcingtime_bins,
            self.forcing_bins[forcing_name],
            **rate_estimate_kwargs,
        )

        if modulation is None:
            print("Could not estimate seismicity vs forcing!")
            print("(probably because of too few events in the catalog)")
            return

        if forcing_name not in self.modulation:
            self.modulation[forcing_name] = {}
        self.modulation[forcing_name][window_time] = modulation
        self.modulation[forcing_name][window_time]["bins"] = self.forcing_bins[forcing_name]
        self.modulation[forcing_name][window_time]["midbins"] = self._midbins(
            self.modulation[forcing_name][window_time]["bins"]
        )



class ModulationmeterMultiWindows(Modulationmeter):

    def __init__(
        self,
        short_window_days=None,
        num_short_windows=None,
        overlap=None,
        downsample=None,
        catalog=None,
        forcing=None,
        forcing_bins=None,
        window_type="backward",
    ):
        """
        Idea: initialize with a "modulation_parameters" dictionary that is
        systematically passed to self.measure_modulation and a "fit_parameters"
        dictionary that is systematically passsed to self.fit_modualtion.

        """
        super().__init__(
            catalog=catalog,
            forcing=forcing,
            downsample=downsample,
            window_type=window_type,
            forcing_bins=forcing_bins,
        )
        self.short_window_days = short_window_days
        self.num_short_windows = num_short_windows
        self.overlap = overlap

    @property
    def large_window_days(self):
        return (
            self.num_short_windows * (1.0 - self.overlap) * self.short_window_days
            + self.overlap * self.short_window_days
        )

    def update_window_parameters(
        self,
        short_window_days=None,
        num_short_windows=None,
        overlap=None,
    ):
        """
        Not sure this function would make sense...
        """
        return

    def measure_modulation(
        self, window_time, forcing_name, rate_estimate_kwargs={}, **kwargs
    ):
        """ """
        if self.catalog is None:
            warnings.warn(
                "You need to define the `catalog` attribute. See `set_catalog`."
            )
            return
        if self.forcing is None:
            warnings.warn(
                "You need to define the `forcing` attribute. See `set_forcing`."
            )
            return

        # 'hidden' kwarg for `shuffle_and_measure_modulation`
        catalog = kwargs.get("catalog", None)
        if catalog is None:
            catalog = self.catalog

        func = partial(utils.estimate_rate_forcing_bins, **rate_estimate_kwargs)

        modulation = utils.composite_rate_estimate(
            catalog,
            self.forcing[forcing_name],
            self.forcing_bins[forcing_name],
            window_time,
            func,
            short_window_days=self.short_window_days,
            num_short_windows=self.num_short_windows,
            overlap=self.overlap,
            downsample=self.downsample,
            **kwargs,
        )

        if modulation is None:
            print("Could not estimate seismicity vs forcing!")
            print("(probably because of too few events in the catalog)")
            return

        if forcing_name not in self.modulation:
            self.modulation[forcing_name] = {}
        self.modulation[forcing_name][window_time] = modulation
        self.modulation[forcing_name][window_time]["midbins"] = self._midbins(
            self.modulation[forcing_name][window_time]["bins"]
        )


class ModulationmeterMultiWindowForcingTimeBins(
    ModulationmeterForcingTimeBins, ModulationmeterMultiWindows
):

    def __init__(
        self,
        short_window_days=None,
        num_short_windows=None,
        overlap=None,
        downsample=None,
        catalog=None,
        forcing=None,
        forcing_bins=None,
        window_type="backward",
    ):
        super().__init__(
            catalog=catalog,
            forcing=forcing,
            downsample=downsample,
            window_type=window_type,
            forcing_bins=forcing_bins,
        )
        self.short_window_days = short_window_days
        self.num_short_windows = num_short_windows
        self.overlap = overlap

    def measure_modulation(
        self, window_time, forcing_name, rate_estimate_kwargs={}, **kwargs
    ):
        """ """
        if self.catalog is None:
            warnings.warn(
                "You need to define the `catalog` attribute. See `set_catalog`."
            )
            return
        if self.forcing is None:
            warnings.warn(
                "You need to define the `forcing` attribute. See `set_forcing`."
            )
            return

        # 'hidden' kwarg for `shuffle_and_measure_modulation`
        catalog = kwargs.get("catalog", None)
        if catalog is None:
            catalog = self.catalog

        func = partial(utils.estimate_rate_forcingtime_bins, **rate_estimate_kwargs)

        modulation = utils.composite_rate_estimate(
            catalog,
            self.forcingtime_bins[forcing_name],
            self.forcing_bins[forcing_name],
            window_time,
            func,
            mode="forcingtime",
            short_window_days=self.short_window_days,
            num_short_windows=self.num_short_windows,
            overlap=self.overlap,
            downsample=self.downsample,
            **kwargs,
        )

        if modulation is None:
            print("Could not estimate seismicity vs forcing!")
            print("(probably because of too few events in the catalog)")
            return

        if forcing_name not in self.modulation:
            self.modulation[forcing_name] = {}

        self.modulation[forcing_name][window_time] = modulation
        self.modulation[forcing_name][window_time]["midbins"] = self._midbins(
            self.modulation[forcing_name][window_time]["bins"]
        )


# --------------------------------------------------------------------
class ShuffledModulationmeter(Modulationmeter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replica_counts = {}
        self.original_catalog = self.catalog.copy()

    def randomize_catalog(self, method="block-shuffle", **kwargs):
        """ """
        if self.catalog is None:
            warnings.warn(
                "You need to define the `catalog` attribute. See `set_catalog`."
            )
            return
        if method == "block-shuffle":
            self._block_shuffle_catalog(**kwargs)
        elif method == "random":
            self._random_catalog()
        else:
            warnings.warn("'method' should be one of ['block-shuffle', 'random'].")
            return

    def _random_catalog(self):
        """ """
        tmin = self.original_catalog["event_timing_sec"].min()
        tmax = self.original_catalog["event_timing_sec"].max()
        num_events = len(self.original_catalog)
        event_timing_sec = tmin + (tmax - tmin) * np.random.random(size=num_events)
        self.random_catalog = pd.DataFrame(
            {"event_timing_sec": event_timing_sec, "origin_time": pd.to_datetime(event_timing_sec, unit="s")}
        )
        self.random_catalog.sort_values("event_timing_sec", inplace=True)

    def _block_shuffle_catalog(self, num_events_per_block=100):
        """ """
        # differentiate times
        wt = np.diff(self.original_catalog["event_timing_sec"])
        # wt = np.hstack((wt.mean(), wt))
        # attribute bloc membership
        indexes = np.arange(1, len(self.original_catalog))
        block_membership = indexes // num_events_per_block
        num_blocks = block_membership.max() + 1
        block_indexes = np.arange(num_blocks)
        blocks = [np.where(block_membership == i)[0] for i in block_indexes]
        # shuffle blocs
        np.random.shuffle(block_indexes)
        wt_shuffled = np.hstack([0.0] + [wt[blocks[i]] for i in block_indexes])
        event_timing_sec = np.cumsum(wt_shuffled) + self.original_catalog["event_timing_sec"].min()
        if not hasattr(self, "random_catalog"):
            self.random_catalog = self.original_catalog[
                ["event_timing_sec", "origin_time"]
            ].copy()
        self.random_catalog["origin_time"] = pd.to_datetime(event_timing_sec, unit="s")
        self.random_catalog["event_timing_sec"] = event_timing_sec
        self.random_catalog.sort_values("event_timing_sec", inplace=True)

    def shuffle_and_measure_modulation(
        self, window_time, forcing_names, randomization_kwargs={}, **kwargs
    ):
        """ """
        self.randomize_catalog(**randomization_kwargs)
        self.random_catalog.sort_values("event_timing_sec", inplace=True)
        self.catalog = self.random_catalog
        self.count_events_in_forcingtime_bins()

        for forcing_name in np.atleast_1d(forcing_names):
            if not forcing_name in self.replica_counts:
                self.replica_counts[forcing_name] = 0
            self.replica_counts[forcing_name] += 1
            rep_count = self.replica_counts[forcing_name]

            self.measure_modulation(window_time, forcing_name, **kwargs)

            self.modulation[forcing_name][f"replica{rep_count}"] = self.modulation[
                forcing_name
            ][window_time]
            del self.modulation[forcing_name][window_time]


class ShuffledModulationmeterMultiWindowForcingTimeBins(
    ModulationmeterMultiWindowForcingTimeBins, ShuffledModulationmeter
):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

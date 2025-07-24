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
        self.catalog = catalog
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

        ## -----------------------------------------------
        ## here, write func(args, kwargs) appropriately
        ## for example:
        # modulation = func(
        #    self.catalog,
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
        """Adjust forcing table to the catalog's time span.
        """
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
        dt_forcing = (self.forcing.index[1] - self.forcing.index[0])#.total_seconds()
        within_catalog = (
                (self.forcing.index > cat_t_min - dt_forcing)
                & (self.forcing.index < cat_t_max + dt_forcing)
                )
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
        for t in self.modulation[forcing_name]:
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
        performance_metrics = {"delta_aic": [], "rms_residual": []}
        for t in time:
            _mod = self.model[forcing_name][t][model_name]
            performance_metrics["delta_aic"].append(_mod["delta_aic"])
            performance_metrics["rms_residual"].append(_mod["rms_residual"])
        return time, performance_metrics

    def get_obs_vs_forcing_vs_time(self, forcing_name, obs_name):
        """
        """
        time_periods = list(
                self.modulation[forcing_name].keys()
                )
        time_periods.sort()

        obs = np.stack(
                [
                    self.modulation[forcing_name][tp][obs_name]
                    for tp in time_periods
                 ],
                axis=0
                )

        return obs, pd.to_datetime(time_periods)

    def get_model_vs_forcing_vs_time(self, forcing_name, obs_name):
        """
        """
        time_periods = list(
                self.model[forcing_name].keys()
                )
        time_periods.sort()

        model = np.stack(
                [
                    self.model[forcing_name][tp][obs_name]
                    for tp in time_periods
                 ],
                axis=0
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

    def evaluate_aic(
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

    def set_catalog(self, catalog):
        self.catalog = catalog

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
        short_window_days=None,
        num_short_windows=None,
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

    def build_forcingtime_bins(self):
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
            forcing_leftbin_membership = np.digitize(
                np.clip(
                    self.forcing[forcing_name],
                    a_min=self.forcing_bins[forcing_name].min(),
                    a_max=self.forcing_bins[forcing_name].max(),
                ),
                self.forcing_bins[forcing_name],
            )
            # arbitrary choice: when value is exactly at the bin edge,
            # we assign it to the previous bin
            forcing_leftbin_membership[
                forcing_leftbin_membership == len(self.forcing_bins[forcing_name])
            ] = (len(self.forcing_bins[forcing_name]) - 1)
            # find when the forcing time series transitions from
            # one bin to another
            jumps_right = (
                np.abs(forcing_leftbin_membership[1:] - forcing_leftbin_membership[:-1])
                > 0
            ).astype(bool)
            jumps_right = np.hstack([True, jumps_right])
            jumps_right = np.where(jumps_right)[0]
            # associate the corresponding times
            forcingtime_bin_time_edges = self.forcing[forcing_name].index[jumps_right]
            forcingtime_bin_time_edges_sec = (
                forcingtime_bin_time_edges.values.astype("datetime64[ms]").astype(
                    "float64"
                )
                / 1000.0
            )
            forcingtime_bin_duration_sec = (
                forcingtime_bin_time_edges_sec[1:] - forcingtime_bin_time_edges_sec[:-1]
            )
            forcingtime_bins = {
                "forcing_leftbin_membership": np.hstack(
                    (pd.NA, forcing_leftbin_membership[jumps_right][:-1])
                ),
                "forcingtime_bin_time_edges_sec": forcingtime_bin_time_edges_sec,
                "forcingtime_bin_duration_sec": np.hstack(
                    (np.nan, forcingtime_bin_duration_sec)
                ),
            }
            self.forcingtime_bins[forcing_name] = pd.DataFrame(forcingtime_bins)

    def count_events_in_forcingtime_bins(self, attach_membership_to_cat=False):
        for forcing_name in self.forcing_bins:
            if (
                self.catalog["t_eq_s"].values.max()
                > self.forcingtime_bins[forcing_name][
                    "forcingtime_bin_time_edges_sec"
                ].max()
            ):
                warnings.warn(
                    "Forcing time series stop before end of catalog! "
                    "Aborting `count_events_in_forcingtime_bins`."
                )
                continue
            if (
                self.catalog["t_eq_s"].values.min()
                < self.forcingtime_bins[forcing_name][
                    "forcingtime_bin_time_edges_sec"
                ].min()
            ):
                warnings.warn(
                    "Forcing time series starts after beginning of catalog! "
                    "Aborting `count_events_in_forcingtime_bins`."
                )
                continue
            forcingtime_bin_eq_membership = np.digitize(
                self.catalog["t_eq_s"].values,
                self.forcingtime_bins[forcing_name]["forcingtime_bin_time_edges_sec"],
            )
            if attach_membership_to_cat:
                self.catalog[f"membership_{forcing_name}"] = (
                    forcingtime_bin_eq_membership
                )
            forcingtime_bin_values, forcingtime_bin_counts = np.unique(
                forcingtime_bin_eq_membership, return_counts=True
            )
            self.forcingtime_bins[forcing_name].loc[
                forcingtime_bin_values, "forcingtime_bin_count"
            ] = forcingtime_bin_counts
            self.forcingtime_bins[forcing_name].fillna(
                {"forcingtime_bin_count": 0}, inplace=True
            )

            # first row is just here to indicate beginning of first forcingtime bin,
            # but because each row gives stats for bin to the left,
            # the first row has no stat by construction
            self.forcingtime_bins[forcing_name].loc[0, "forcingtime_bin_count"] = pd.NA

    def detect_catalog_gaps(self, event_count_bin_size="1D"):
        """ """
        if self.catalog is None:
            warnings.warn(
                "You need to define the `catalog` attribute. See `set_catalog`."
            )
            return
        warnings.warn("This gap detection routine only works for high seismicity rates.")
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
                        "forcingtime_bin_time_edges_sec"
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
        subcat = self.catalog[
            (self.catalog["origin_time"] > t_start)
            & (self.catalog["origin_time"] <= t_end)
        ]
        subforcingtime_bins = self.forcingtime_bins[forcing_name]
        subforcingtime_bins = subforcingtime_bins[
            (
                subforcingtime_bins["forcingtime_bin_time_edges_sec"]
                > t_start.timestamp()
            )
            & (
                subforcingtime_bins["forcingtime_bin_time_edges_sec"]
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
        self.modulation[forcing_name][window_time]["midbins"] = self._midbins(
            self.forcing_bins[forcing_name]
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

        func = partial(utils.estimate_rate_forcing_bins, **rate_estimate_kwargs)

        modulation = utils.composite_rate_estimate(
            self.catalog,
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

        func = partial(utils.estimate_rate_forcingtime_bins, **rate_estimate_kwargs)

        modulation = utils.composite_rate_estimate(
            self.catalog,
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

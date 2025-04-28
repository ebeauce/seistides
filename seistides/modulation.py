import numpy as np
import pandas as pd
import pickle
import warnings

from abc import ABC, abstractmethod
from . import utils


class Modulationmeter(ABC):
    """A class to measure the modulation of seismicity by some forcing."""

    def __init__(
        self,
        downsample=None,
        catalog=None,
        forcing=None,
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
        self.downsample = downsample
        self.catalog = catalog
        self.forcing = forcing
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

    def fit_modulation(
        self, func, window_time, forcing_name, model_name="model1",
        quantity="rate_ratio", **kwargs
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
            self, window_time, forcing_name,
            quantity="rate_ratio", model_name="model1"
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
            downsample=downsample,
            window_type=window_type,
        )
        self.forcing_bins = forcing_bins
        self.window_duration_days = window_duration_days

    @property
    def window_duration_sec(self):
        return self.window_duration_days * 24.0 * 3600.0

    def set_forcing_bins(self, forcing_bins):
        self.forcing_bins = forcing_bins

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
                self.forcing[forcing_name], self.forcing_bins[forcing_name]
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
            forcingtime_bin_eq_membership = np.digitize(
                self.catalog["t_eq_s"].values,
                self.forcingtime_bins[forcing_name]["forcingtime_bin_time_edges_sec"],
            )
            if attach_membership_to_cat:
                self.catalog[f"membership_{forcing_name}"] = forcingtime_bin_eq_membership
            forcingtime_bin_values, forcingtime_bin_counts = np.unique(
                forcingtime_bin_eq_membership, return_counts=True
            )
            #breakpoint()
            self.forcingtime_bins[forcing_name].loc[
                forcingtime_bin_values, "forcingtime_bin_count"
            ] = forcingtime_bin_counts
            #self.forcingtime_bins[forcing_name].loc[:, "forcingtime_bin_count"].fillna(
            #    0, inplace=True
            #)
            self.forcingtime_bins[forcing_name].fillna(
                    {"forcingtime_bin_count": 0}, inplace=True
                    )

            # first row is just here to indicate beginning of first forcingtime bin,
            # but because each row gives stats for bin to the left,
            # the first row has no stat by construction
            self.forcingtime_bins[forcing_name].loc[0, "forcingtime_bin_count"] = pd.NA

    def measure_modulation(self, func, window_time, forcing_name, **kwargs):
        """ """
        from dateutil.relativedelta import relativedelta

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
        modulation = func(
            subcat,
            subforcingtime_bins,
            self.forcing_bins[forcing_name],
            num_bootstraps=kwargs.get("num_bootstraps", 100),
            num_std_cutoff=kwargs.get("num_std_cutoff", 0.)
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

        modulation = func(
            self.catalog,
            self.forcing,
            window_time,
            forcing_name,
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
        self.modulation[forcing_name][window_time] = modulation[forcing_name]
        self.modulation[forcing_name][window_time]["midbins"] = self._midbins(
            self.modulation[forcing_name][window_time]["bins"]
        )

import os

HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "")

import numpy as np
import pandas as pd
import geopandas as geopd
import sys

from obspy.geodetics.base import calc_vincenty_inverse
from cartopy.geodesic import Geodesic
from shapely import geometry


def within_box(catalog, pol):
    """Return a boolean array indicating whether the event is within the box."""
    return np.array(
        [
            pol.contains(geometry.Point(lon, lat))
            for lon, lat in zip(catalog["longitude"], catalog["latitude"])
        ]
    )


def select_subregion(catalog, region, width_km=None, dist_to_fault=None):

    #                   define regions
    # ----------------------------------------------------------

    G = Geodesic()

    # ----------------------------------------------------
    # define the box along the foreshock fault
    foreshock = geometry.Polygon(
        np.array(
            [
                [-117.75046177118023, 35.61703733731494],
                [-117.52936123349164, 35.79765725555746],
                [-117.3728593157097, 35.67034118024011],
                [-117.59431167478111, 35.48971743550447],
            ]
        )
    )
    # ----------------------------------------------------

    # ----------------------------------------------------
    # define the box along the garlock fault
    garlock = geometry.Polygon(
        np.array(
            [
                [-118.03276185660877, 35.34694377133767],
                [-117.73281857743964, 35.486942658247045],
                [-117.66721950746593, 35.39304806597723],
                [-117.96727596657672, 35.2530469992298],
            ]
        )
    )
    # ----------------------------------------------------

    COSO = {
        "latitudes": [35.98, 35.98, 36.06, 36.06],
        "longitudes": [-117.75, -117.83, -117.83, -117.75],
    }
    coso = geometry.Polygon(np.stack((COSO["longitudes"], COSO["latitudes"]), axis=1))

    FZ_NORTH_END = {
        "latitudes": [35.95, 35.95, 36.15, 36.15],
        "longitudes": [-117.60, -117.74, -117.74, -117.60],
    }
    fz_north_end = geometry.Polygon(
        np.stack((FZ_NORTH_END["longitudes"], FZ_NORTH_END["latitudes"]), axis=1)
    )

    # ----------------------------------------------------------
    #            select subset of earthquake catalog
    # ----------------------------------------------------------
    geocat = geopd.GeoDataFrame(
        catalog, geometry=geopd.points_from_xy(catalog.longitude, catalog.latitude)
    )

    if region in {"fault_zone", "foreshock_zone"}:
        selection = dist_to_fault.loc[catalog.index, "distance_to_fault"] < width_km
        if region == "foreshock_zone":
            selection2 = geocat.within(foreshock)
            selection = selection & selection2
    elif region == "garlock":
        selection = geocat.within(garlock)
    elif region == "coso":
        selection = within_box(catalog, coso)
    elif region == "north":
        selection = within_box(catalog, fz_north_end)
    catalog = catalog[selection].copy()

    print(f"There are {len(catalog)} earthquakes in the requested box.")

    return catalog

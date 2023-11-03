import numpy as np
from shapely.geometry import Polygon

def cross_section_v2(lon1, lat1, lon2, lat2, orthogonal_dist):
    """Deprecated!
    """
    print("cross_section_v2 is deprecated! Use cross_section instead.")
    return cross_section(
            lon1, lat1, lon2, lat2, orthogonal_dist
            )

def cross_section(lon1, lat1, lon2, lat2, orthogonal_dist):
    """Define a box along a given cross-section axis.

    Parameters
    ----------
    lon1: scalar float
        Longitude of the axis' end #1.
    lat1: scalar float
        Latitude of the axis' end #1.
    lon2: scalar float
        Longitude of the axis' end #2.
    lat2: scalar float
        Latitude of the axis' end #2.
    orthogonal_dist: scalar float
        Distance, in km, defining the width of the box taken around
        the cross-section axis. The width is 2 times `orthogonal_dist`.

    Returns
    ---------
    cs_geometry: dictionary
        Dictionary with information on the geometry of the cross-section.
        `cs_geometry['parallel']` and `cs_geometry['orthogonal']` are the
        cross-section parallel and orthogonal unit vectors.
        `cs_geometry['lon1/2']` and `cs_geometry['lat1/2']` are the longitude
        and latitude of the start and end points.
        `cs_geometry['dist_per_lon/lat']` are the distances, in km, per degree
        of longitude or latitude, assuming a spherical earth.
        `cs_geometry['corners']` are the lon/lat of the four corners of the box
        defined around the cross-section (see `orthogonal_dist`).
        `cs_geometry['box']` is the `shapely.geometry.Polygon` instance with
        geometry defined by `corners`. It has built-in methods to test whether
        a point is inside or outside the box.
    """
    from cartopy.geodesic import Geodesic
    from obspy.geodetics.base import calc_vincenty_inverse

    geodesic = Geodesic()

    # geometrical constants
    d2r = np.pi / 180.0
    r2d = 1.0 / d2r
    R_earth = 6371.0  # km
    # initialize results
    cs_geometry = {}
    cs_geometry["lon1"] = lon1
    cs_geometry["lat1"] = lat1
    cs_geometry["lon2"] = lon2
    cs_geometry["lat2"] = lat2
    # take (lon1, lat1) as the origin of our reference frame
    # The distance spanned by 1 degree in longitude at lat1 is:
    alpha = np.pi - lat1 * d2r
    r = R_earth * np.sin(alpha)
    # distance per degree in longitude (and not per radian!)
    cs_geometry["dist_per_lon"] = r * d2r
    # distance per degree in latitude
    cs_geometry["dist_per_lat"] = R_earth * d2r
    # build the vector that is colinear to the section
    # (lon1, lat1) to (lon2, lat2)
    S = np.array(
        [
            (cs_geometry["lon2"] - cs_geometry["lon1"]) * cs_geometry["dist_per_lon"],
            (cs_geometry["lat2"] - cs_geometry["lat1"]) * cs_geometry["dist_per_lat"],
        ]
    )
    S_length = np.sqrt(np.sum(S**2))
    # make it a unit vector
    S /= S_length
    S_orth = np.array([-S[1], S[0]])
    cs_geometry["parallel"] = S
    cs_geometry["orthogonal"] = S_orth
    ## compute the orthogonal distance to the section, i.e. scalar product
    ## |y_2| = sqrt(||y||**2 - y_1**2)
    # distance_to_section = np.abs(meta_db['CS_orthogonal'].values)
    dist_to_degrees = np.array(
        [1.0 / cs_geometry["dist_per_lon"], 1.0 / cs_geometry["dist_per_lat"]]
    )
    dist, az, baz = calc_vincenty_inverse(lat1, lon1, lat2, lon2)
    lon_1, lat_1, _ = np.asarray(
            geodesic.direct([cs_geometry["lon1"], cs_geometry["lat1"]], az-90., 1000.*orthogonal_dist)
            )[0]
    lon_4, lat_4, _ = np.asarray(
            geodesic.direct([cs_geometry["lon1"], cs_geometry["lat1"]], az+90., 1000.*orthogonal_dist)
            )[0]
    lon_2, lat_2, _ = np.asarray(
            geodesic.direct([cs_geometry["lon2"], cs_geometry["lat2"]], az-90., 1000.*orthogonal_dist)
            )[0]
    lon_3, lat_3, _ = np.asarray(
            geodesic.direct([cs_geometry["lon2"], cs_geometry["lat2"]], az+90., 1000.*orthogonal_dist)
            )[0]
    corner_1 = (lon_1, lat_1)
    corner_2 = (lon_2, lat_2)
    corner_3 = (lon_3, lat_3)
    corner_4 = (lon_4, lat_4)

    cs_geometry["corners"] = np.vstack((corner_1, corner_2, corner_3, corner_4))
    cs_geometry["box"] = Polygon(np.vstack([corner_1, corner_2, corner_3, corner_4]))
    return cs_geometry

def project_onto_cs(cs_geometry, longitudes, latitudes):
    """Project the points onto the cross-section.

    Parameters
    -----------
    cs_geometry: dictionary
        Dictionary returned by `cross_section`.
    longitudes: (n_events,) `numpy.ndarray`
        Longitudes of the earthquakes to project along the cross-section.
    latitudes: (n_events), `numpy.ndarray`
        Latitudes of the earthquakes to project along the cross-section.

    Returns
    ---------
    coords: (n_events, 2) `numpy.ndarray`
        Coordinates in the cross-section frame. `coords[:, 0]` are the
        coordinates along the cross-section, and `coords[:, 1]` are the
        coordinates perpendicular to the cross-section.
    """
    # build the vector that is colinear to the section
    # (lon1, lat1) to (lon2, lat2)
    S = np.array(
        [
            (cs_geometry["lon2"] - cs_geometry["lon1"]) * cs_geometry["dist_per_lon"],
            (cs_geometry["lat2"] - cs_geometry["lat1"]) * cs_geometry["dist_per_lat"],
        ]
    )
    S_length = np.sqrt(np.sum(S**2))
    # make it a unit vector
    S /= S_length
    S_orth = np.array([-S[1], S[0]])
    # build array of cartesian coordinates
    X = np.zeros((len(longitudes), 2), dtype=np.float32)
    X[:, 0] = (longitudes - cs_geometry["lon1"]) * cs_geometry["dist_per_lon"]
    X[:, 1] = (latitudes - cs_geometry["lat1"]) * cs_geometry["dist_per_lat"]
    # project coords
    coords_parallel = np.sum(X * S[np.newaxis, :], axis=-1)
    coords_orthogonal = np.sum(X * S_orth[np.newaxis, :], axis=-1)
    return np.stack([coords_parallel, coords_orthogonal], axis=1)


def compute_longlat_along_axis(cs_geometry, step_lon=0.05, step_lat=0.05):
    """
    Compute longitudes/latitudes along the cross-sections
    for labeling the distance axis with geographic coordinates
    as well.

    Parameters
    -----------
    cs_geometry: dictionary
        Output of `cross_section`.

    Returns
    --------
    output: dictionary
        `output['xtickpos_lon/lat']` is the position, in the cross-section
        coordinate system, of the lon/lat ticks along the cross-section axis.
        `output['xticklabels_lon/lat']` are the longitudes or latitudes at
        each of the ticks given in `output['xtickpos_lon/lat']`.
    """
    d2r = np.pi / 180.0
    r2d = 1.0 / d2r
    R_earth = 6371.0  # km
    lon_1, lon_2 = cs_geometry["lon1"], cs_geometry["lon2"]
    lat_1, lat_2 = cs_geometry["lat1"], cs_geometry["lat2"]
    # take (lon1, lat1) as the origin of our reference frame
    # The distance spanned by 1 degree in longitude at lat1 is:
    alpha = np.pi - lat_1 * d2r
    r = R_earth * np.sin(alpha)
    # distance per degree in longitude (and not per radian!)
    dist_per_lon = r * d2r
    # distance per degree in latitude
    dist_per_lat = R_earth * d2r
    # longitude tick positions
    lon_start = np.ceil(lon_1 / step_lon) * step_lon
    lon_end = np.floor(lon_2 / step_lon) * step_lon
    longitudes = np.arange(lon_start, lon_end + 0.001, step_lon)
    # latitude tick positions
    lat_start = np.ceil(lat_1 / step_lat) * step_lat
    lat_end = np.floor(lat_2 / step_lat) * step_lat
    latitudes = np.arange(lat_start, lat_end + 0.001, step_lat)
    # simple linear interpolation to get the corresponding latitudes on the axis
    if lon_2 == lon_1:
        # doesn't matter which value we fill the array with
        corresponding_lats = np.ones(len(longitudes)) * lat_start
    else:
        corresponding_lats = lat_1 + (lat_2 - lat_1) / (lon_2 - lon_1) * (
            longitudes - lon_1
        )
    # simple linear interpolation to get the corresponding longitudes on the axis
    if lat_2 == lat_1:
        # doesn't matter which value we fill the array with
        corresponding_lons = np.ones(len(latitudes)) * lon_start
    else:
        corresponding_lons = lon_1 + (lon_2 - lon_1) / (lat_2 - lat_1) * (
            latitudes - lat_1
        )
    # convert longitudes/latitudes to position vectors
    X_lon = np.column_stack(
        (
            (longitudes - lon_1) * dist_per_lon,
            (corresponding_lats - lat_1) * dist_per_lat,
        )
    )
    X_lat = np.column_stack(
        (
            (corresponding_lons - lon_1) * dist_per_lon,
            (latitudes - lat_1) * dist_per_lat,
        )
    )
    # convert the positions to distances along the
    # CS axis in km
    xtickpos_lon = np.sum(X_lon * cs_geometry["parallel"], axis=-1)
    xtickpos_lat = np.sum(X_lat * cs_geometry["parallel"], axis=-1)
    output = {}
    output["xtickpos_lon"] = xtickpos_lon
    output["xtickpos_lat"] = xtickpos_lat
    output["xticklabels_lon"] = longitudes
    output["xticklabels_lat"] = latitudes
    return output

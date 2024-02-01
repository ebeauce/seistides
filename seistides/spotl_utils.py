#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:32:00 2022

"""

import os
import pandas as pd
import numpy as np


def run_Ertid_potential(lat, lon, year1, day1, year2, day2, delta_hours):
    os.system("rm run_ertid.sh")
    f = open("run_ertid.sh", "a")
    f.write("#!/bin/csh\n")
    f.write("ertid << EOF\n")
    f.write(str(year1) + " " + str(day1) + " 0\n")  # time zeros
    f.write(str(year2) + " " + str(day2) + " 0\n")  # time end
    f.write(str(delta_hours) + "\n")  # time interval
    f.write("t\n")
    f.write(str(lat) + "\n")  # time interval
    f.write(str(lon) + "\n")  # time interval
    f.write("-1\n")
    f.write("0\n")
    f.write("0\n")
    f.write("potential\n")
    f.write("EOF\n")
    f.close()

    os.system("sh run_ertid.sh")


# calculate strain with ertid
def run_Ertid(
    station_lat,
    station_lon,
    year1,
    jul_day1,
    year2,
    jul_day2,
    delta_hours,
    azimuth1=0.0,
    azimuth2=270.0,
    azimuth3=315.0,
    files_basename="solid_earth_tides_Ridgecrest",
):
    """Call `ertid` from SPOTL.

    Write a cshell scripts that calls `ertid` with the parameters determined
    by this function's arguments and hard-coded parameters:
    - line 1: location of shell
    - line 2: command line calling ertid with the arguments up to EOF
    - line 3: start time in year,day,hour
    - line 4: end time in year,day,hour
    - line 5: sample interval, in hours
    - line 6: t, for theoretical tides (m, for Munk-Cartwright coefficients)
    - line 7: latitude of the virtual strainmeter
    - line 8: longitude of the virtual strainmeter
    - line 9: 0, for 0 gravity tides (up to 1)
    - line 10: 0, for 0 tilt tides (up to 2)
    - line 11: 3, for 3 strain tides (up to 3)
    - line 12: strainmeter azimuth 1 = 0 degree
    - line 13: strainmeter azimuth 2 = 90 degrees
    - line 14: strainmeter azimuth 3 = 45 degrees
    - line 15: output filename 1 = ertid0
    - line 16: output filename 2 = ertid90
    - line 17: output filename 3 = ertid45
    - line 18: EOF (end of file), signals end of program arguments

    The outputs are strain time series written in text files, one for each
    virtual strainmeter given by its azimuth.


    Parameters
    ----------
    station_lat: float
        Latitude, in decimal degrees, of the station, that is, the
        point where strain is computed.
    station_lon: float
        Longitude, in decimal degrees, of the station, that is, the
        point where strain is computed.
    year1: int
        Year of start time.
    jul_day1: int
        Julian day of start time.
    year2: int
        Year of end time.
    jul_day2: int
        Julian day of end time.
    delta_hours: float
        Sample time in hours.
    azimuth1: float, default to 0
        Azimuth, angle from north in degrees, of first channel.
    azimuth2: float, default to 270
        Azimuth, angle from north in degrees, of second channel.
    azimuth3: float, default to 315
        Azimuth, angle from north in degrees, of third channel.
    files_basename: string, default to 'solid_earth_tides_Ridgecrest'
        Basename of the output files.
    """
    os.system("rm run_ertid.sh")
    fin = open("run_ertid.sh", "a")
    fin.write("#!/bin/csh\n")
    fin.write("ertid << EOF\n")
    fin.write(str(year1) + " " + str(jul_day1) + " 0\n")  # start time
    fin.write(str(year2) + " " + str(jul_day2) + " 0\n")  # end time
    fin.write(str(delta_hours) + "\n")  # sample interval
    fin.write("t\n")  #
    fin.write(str(station_lat) + "\n")  # "station" latitude
    fin.write(str(station_lon) + "\n")  # "station" longitude
    fin.write("0\n")
    fin.write("0\n")
    fin.write("3\n")
    fin.write(f"{azimuth1:.0f}\n")
    fin.write(f"{azimuth2:.0f}\n")
    fin.write(f"{azimuth3:.0f}\n")
    fin.write(f"{files_basename}_az{azimuth1:.0f}.txt\n")
    fin.write(f"{files_basename}_az{azimuth2:.0f}.txt\n")
    fin.write(f"{files_basename}_az{azimuth3:.0f}.txt\n")
    fin.write("EOF\n")
    fin.close()

    os.system("sh run_ertid.sh")

def ocean_load_example1(
    station_name,
    station_longitude,
    station_latitude,
    station_elevation_m,
    year_start,
    julday_start,
    n_periods,
    sample_time_sec,
    hour_start=0,
    minute_start=0,
    second_start=0,
    working_dir="/home/eric/software/SPOTL/spotl/working",
    azimuth1=0.0,
    azimuth2=270.0,
    azimuth3=315.0,
    files_basename="ocean_load_Ridgecrest_test",
    earth_green_function="gr.gbaver.wef.p02.ce",
    tidal_components=["o1", "p1"],
):
    """Compute strain from ocean tides.

    Compute the strain produced by the oceanic tides using a regional,
    west coast model and a global model.

    Parameters
    -----------
    azimuth1: float, default to 0
        Azimuth, angle from north in degrees, of first channel.
    azimuth2: float, default to 270
        Azimuth, angle from north in degrees, of second channel.
    azimuth3: float, default to 315
        Azimuth, angle from north in degrees, of third channel.
    files_basename: string, default to 'solid_earth_tides_Ridgecrest'
        Basename of all the files produced by SPOTL's routines.

    """
    import glob
    from time import sleep

    # keep current working directory in memory for later
    cwd = os.getcwd()
    # go to target working dir
    os.chdir(working_dir)
    # list of models used in each sub-region
    models = ["osu.usawest.2010", "got4p7.2004"]
    polygons = ["poly1", "poly2"]
    # write the input shell file
    with open(files_basename + ".csh", "w") as fin:
        fin.write("#!/bin/csh\n")
        # tides on US West coast
        fin.write("polymake << EOF > poly1.tmp\n")
        fin.write("+ osu.usawest.2010\n")
        fin.write("EOF\n")
        # global tides
        fin.write("polymake << EOF > poly2.tmp\n")
        fin.write("- osu.usawest.2010\n")
        fin.write("EOF\n")
        for i, comp in enumerate(tidal_components):
            for poly, model in zip(polygons, models):
                fin.write(
                    f"nloadf {station_name} {station_latitude} {station_longitude} "
                    f"{station_elevation_m} {comp}.{model} {earth_green_function} "
                    f"l {poly}.tmp > {files_basename}_{poly}_{comp}.txt\n"
                )
            # combine all polygons
            fin.write(
                    f"cat {files_basename}_{polygons[0]}_{comp}.txt "
                    f"{files_basename}_{polygons[1]}_{comp}.txt | "
                    f"loadcomb c > {files_basename}_tmp1.txt\n"
                    )
            print(f"Adding {comp} to {files_basename}_all_components.txt...")
            # add this tidal component to others
            if i == 0:
                fin.write(
                        f"cat {files_basename}_tmp1.txt > "
                        f"{files_basename}_all_components.txt\n"
                        )
            else:
                fin.write(
                        f"cat {files_basename}_tmp1.txt >> "
                        f"{files_basename}_all_components.txt\n"
                        )
        # write the harmonic constants for extensional strain at given azimuths
        fin.write(
            f"harprp l {azimuth1} < {files_basename}_all_components.txt > "
            f"harprp_out_{files_basename}_az{azimuth1:.0f}.txt\n"
        )
        fin.write(
            f"harprp l {azimuth2} < {files_basename}_all_components.txt > "
            f"harprp_out_{files_basename}_az{azimuth2:.0f}.txt\n"
        )
        fin.write(
            f"harprp l {azimuth3} < {files_basename}_all_components.txt > "
            f"harprp_out_{files_basename}_az{azimuth3:.0f}.txt\n"
        )
        # use the harmonic constants to compute the time series
        # of tidal (nano)strain
        fin.write(
            f"hartid {year_start:d} {julday_start:d} {hour_start:d} "
            f"{minute_start:d} {second_start:d} {n_periods:d} {sample_time_sec} "
            f"< harprp_out_{files_basename}_az{azimuth1:.0f}.txt "
            f"> tidal_series_{files_basename}_az{azimuth1:.0f}.txt\n"
        )
        fin.write(
            f"hartid {year_start:d} {julday_start:d} {hour_start:d} "
            f"{minute_start:d} {second_start:d} {n_periods:d} {sample_time_sec} "
            f"< harprp_out_{files_basename}_az{azimuth2:.0f}.txt "
            f"> tidal_series_{files_basename}_az{azimuth2:.0f}.txt\n"
        )
        fin.write(
            f"hartid {year_start:d} {julday_start:d} {hour_start:d} "
            f"{minute_start:d} {second_start:d} {n_periods:d} {sample_time_sec} "
            f"< harprp_out_{files_basename}_az{azimuth3:.0f}.txt "
            f"> tidal_series_{files_basename}_az{azimuth3:.0f}.txt\n"
        )
    # ready to run the script!
    os.system(f"sh {files_basename}.csh")
    # save all full file names
    filenames = glob.glob(f"*{files_basename}*")
    folder = os.getcwd()
    # go back to initial working directory and move files there
    os.chdir(cwd)
    for fn in filenames:
        os.system(f"mv {os.path.join(folder, fn)} .")
    print("Done!")

def ocean_load_example2(
    station_name,
    station_longitude,
    station_latitude,
    station_elevation_m,
    year_start,
    julday_start,
    n_periods,
    sample_time_sec,
    hour_start=0,
    minute_start=0,
    second_start=0,
    working_dir="/home/eric/software/SPOTL/spotl/working",
    azimuth1=0.0,
    azimuth2=270.0,
    azimuth3=315.0,
    files_basename="ocean_load_Ridgecrest_test",
    earth_green_function="gr.gbaver.wef.p02.ce",
    tidal_components=["k1", "m2", "s2", "n2"],
):
    """Compute strain from ocean tides.

    Compute the strain produced by the oceanic tides using a local
    model of the Cortez sea, a regional west coast model and a global model.

    Parameters
    -----------
    azimuth1: float, default to 0
        Azimuth, angle from north in degrees, of first channel.
    azimuth2: float, default to 270
        Azimuth, angle from north in degrees, of second channel.
    azimuth3: float, default to 315
        Azimuth, angle from north in degrees, of third channel.
    files_basename: string, default to 'solid_earth_tides_Ridgecrest'
        Basename of all the files produced by SPOTL's routines.

    """
    import glob
    from time import sleep

    # keep current working directory in memory for later
    cwd = os.getcwd()
    # go to target working dir
    os.chdir(working_dir)
    # list of models used in each sub-region
    models = ["cortez.1976", "osu.usawest.2010", "got4p7.2004"]
    polygons = ["poly1", "poly2", "poly3"]
    # write the input shell file
    with open(files_basename + ".csh", "w") as fin:
        fin.write("#!/bin/csh\n")
        # tides in Gulf of California (Sea of Cortez)
        fin.write("polymake << EOF > poly1.tmp\n")
        fin.write("+ cortez.1976\n")
        fin.write("EOF\n")
        # tides on US West coast
        fin.write("polymake << EOF > poly2.tmp\n")
        fin.write("- cortez.1976\n")
        fin.write("+ osu.usawest.2010\n")
        fin.write("EOF\n")
        # global tides
        fin.write("polymake << EOF > poly3.tmp\n")
        fin.write("- cortez.1976\n")
        fin.write("- osu.usawest.2010\n")
        fin.write("EOF\n")
        for i, comp in enumerate(tidal_components):
            for poly, model in zip(polygons, models):
                fin.write(
                    f"nloadf {station_name} {station_latitude} {station_longitude} "
                    f"{station_elevation_m} {comp}.{model} {earth_green_function} "
                    f"l {poly}.tmp > {files_basename}_{poly}_{comp}.txt\n"
                )
            # combine all polygons
            fin.write(
                    f"cat {files_basename}_{polygons[0]}_{comp}.txt "
                    f"{files_basename}_{polygons[1]}_{comp}.txt | "
                    f"loadcomb c > {files_basename}_tmp1.txt\n"
                    )
            fin.write(
                    f"cat {files_basename}_tmp1.txt "
                    f"{files_basename}_{polygons[2]}_{comp}.txt | "
                    f"loadcomb c > {files_basename}_tmp2.txt\n"
                    )
            print(f"Adding {comp} to {files_basename}_all_components.txt...")
            # add this tidal component to others
            if i == 0:
                fin.write(
                        f"cat {files_basename}_tmp2.txt > "
                        f"{files_basename}_all_components.txt\n"
                        )
            else:
                fin.write(
                        f"cat {files_basename}_tmp2.txt >> "
                        f"{files_basename}_all_components.txt\n"
                        )
        # write the harmonic constants for extensional strain at given azimuths
        fin.write(
            f"harprp l {azimuth1} < {files_basename}_all_components.txt > "
            f"harprp_out_{files_basename}_az{azimuth1:.0f}.txt\n"
        )
        fin.write(
            f"harprp l {azimuth2} < {files_basename}_all_components.txt > "
            f"harprp_out_{files_basename}_az{azimuth2:.0f}.txt\n"
        )
        fin.write(
            f"harprp l {azimuth3} < {files_basename}_all_components.txt > "
            f"harprp_out_{files_basename}_az{azimuth3:.0f}.txt\n"
        )
        # use the harmonic constants to compute the time series
        # of tidal (nano)strain
        fin.write(
            f"hartid {year_start:d} {julday_start:d} {hour_start:d} "
            f"{minute_start:d} {second_start:d} {n_periods:d} {sample_time_sec} "
            f"< harprp_out_{files_basename}_az{azimuth1:.0f}.txt "
            f"> tidal_series_{files_basename}_az{azimuth1:.0f}.txt\n"
        )
        fin.write(
            f"hartid {year_start:d} {julday_start:d} {hour_start:d} "
            f"{minute_start:d} {second_start:d} {n_periods:d} {sample_time_sec} "
            f"< harprp_out_{files_basename}_az{azimuth2:.0f}.txt "
            f"> tidal_series_{files_basename}_az{azimuth2:.0f}.txt\n"
        )
        fin.write(
            f"hartid {year_start:d} {julday_start:d} {hour_start:d} "
            f"{minute_start:d} {second_start:d} {n_periods:d} {sample_time_sec} "
            f"< harprp_out_{files_basename}_az{azimuth3:.0f}.txt "
            f"> tidal_series_{files_basename}_az{azimuth3:.0f}.txt\n"
        )
    # ready to run the script!
    os.system(f"sh {files_basename}.csh")
    # save all full file names
    filenames = glob.glob(f"*{files_basename}*")
    folder = os.getcwd()
    # go back to initial working directory and move files there
    os.chdir(cwd)
    for fn in filenames:
        os.system(f"mv {os.path.join(folder, fn)} .")
    print("Done!")

#def ocean_load(
#    station_name,
#    station_longitude,
#    station_latitude,
#    station_elevation_m,
#    year_start,
#    julday_start,
#    n_periods,
#    sample_time_sec,
#    hour_start=0,
#    minute_start=0,
#    second_start=0,
#    working_dir="/home/eric/software/SPOTL/spotl/working",
#    azimuth1=0.0,
#    azimuth2=270.0,
#    azimuth3=315.0,
#    files_basename="ocean_load_Ridgecrest_test",
#    global_ocean_model="got4p7.2004",
#    earth_green_function="gr.gbaver.wef.p02.ce",
#    tidal_components=["o1", "p1", "k1", "m2", "s2"],
#):
#    """Compute strain from ocean tides.
#
#    Parameters
#    -----------
#    azimuth1: float, default to 0
#        Azimuth, angle from north in degrees, of first channel.
#    azimuth2: float, default to 270
#        Azimuth, angle from north in degrees, of second channel.
#    azimuth3: float, default to 315
#        Azimuth, angle from north in degrees, of third channel.
#    files_basename: string, default to 'solid_earth_tides_Ridgecrest'
#        Basename of all the files produced by SPOTL's routines.
#
#    """
#    import glob
#    from time import sleep
#
#    # keep current working directory in memory for later
#    cwd = os.getcwd()
#    # go to target working dir
#    os.chdir(working_dir)
#    # write the input shell file
#    with open(files_basename + ".csh", "w") as fin:
#        fin.write("#!/bin/csh\n")
#        fin.write("polymake << EOF > poly.tmp\n")
#        fin.write("- cortez.1976\n")
#        fin.write("EOF\n")
#        for i, comp in enumerate(tidal_components):
#            fin.write(
#                f"nloadf {station_name} {station_latitude} {station_longitude} "
#                f"{station_elevation_m} {comp}.{global_ocean_model} {earth_green_function} "
#                f"l poly.tmp > {files_basename}_no_Gulf_{comp}.txt\n"
#            )
#            fin.write(
#                f"nloadf {station_name} {station_latitude} {station_longitude} "
#                f"{station_elevation_m} {comp}.cortez.1976 {earth_green_function} "
#                f"l poly.tmp > {files_basename}_only_Gulf_{comp}.txt\n"
#            )
#            fin.write(
#                f"cat {files_basename}_no_Gulf_{comp}.txt "
#                f"{files_basename}_only_Gulf_{comp}.txt | "
#                f"loadcomb c > {files_basename}_{comp}.txt\n"
#            )
#
#            if i == 0:
#                os.system(f"cp {files_basename}_{comp}.txt {files_basename}.txt\n")
#            else:
#                fin.write(
#                    f"cat {files_basename}_{comp}.txt {files_basename}.txt "
#                    f"| loadcomb c > {files_basename}.txt\n"
#                )
#        # write the harmonic constants for extensional strain at given azimuths
#        fin.write(
#            f"harprp l {azimuth1} < {files_basename}.txt > "
#            f"harprp_out_{files_basename}_az{azimuth1:.0f}.txt\n"
#        )
#        fin.write(
#            f"harprp l {azimuth2} < {files_basename}.txt > "
#            f"harprp_out_{files_basename}_az{azimuth2:.0f}.txt\n"
#        )
#        fin.write(
#            f"harprp l {azimuth3} < {files_basename}.txt > "
#            f"harprp_out_{files_basename}_az{azimuth3:.0f}.txt\n"
#        )
#        # use the harmonic constants to compute the time series
#        # of tidal (nano)strain
#        fin.write(
#            f"hartid {year_start:d} {julday_start:d} {hour_start:d} "
#            f"{minute_start:d} {second_start:d} {n_periods:d} {sample_time_sec} "
#            f"< harprp_out_{files_basename}_az{azimuth1:.0f}.txt "
#            f"> tidal_series_{files_basename}_az{azimuth1:.0f}.txt\n"
#        )
#        fin.write(
#            f"hartid {year_start:d} {julday_start:d} {hour_start:d} "
#            f"{minute_start:d} {second_start:d} {n_periods:d} {sample_time_sec} "
#            f"< harprp_out_{files_basename}_az{azimuth2:.0f}.txt "
#            f"> tidal_series_{files_basename}_az{azimuth2:.0f}.txt\n"
#        )
#        fin.write(
#            f"hartid {year_start:d} {julday_start:d} {hour_start:d} "
#            f"{minute_start:d} {second_start:d} {n_periods:d} {sample_time_sec} "
#            f"< harprp_out_{files_basename}_az{azimuth3:.0f}.txt "
#            f"> tidal_series_{files_basename}_az{azimuth3:.0f}.txt\n"
#        )
#    # ready to run the script!
#    os.system(f"sh {files_basename}.csh")
#    # save all full file names
#    filenames = glob.glob(f"*{files_basename}*")
#    folder = os.getcwd()
#    # go back to initial working directory and move files there
#    os.chdir(cwd)
#    for fn in filenames:
#        os.system(f"mv {os.path.join(folder, fn)} .")
#    print("Done!")


# get stress from rosette formula
def getStress(strike, rake, dip):

    # put angles in rad
    s_strike = np.deg2rad(strike)
    s_rake = np.deg2rad(rake)
    s_dip = np.deg2rad(dip)

    # load the strain
    N = pd.read_csv("ertid0").to_numpy().flatten()
    E = pd.read_csv("ertid90").to_numpy().flatten()
    NE = pd.read_csv("ertid45").to_numpy().flatten()

    S = NE - (N + E) / 2

    # Elastic parameteres
    G = 30e9  # Shear
    poi = 0.25  #  # Poisson
    PWM = 2 * G * (1 - poi) / (1 - 2 * poi)  # P-wave modulus

    # get the stress
    exy = S
    Ee = (
        N * np.sin(s_strike) ** 2
        + E * np.cos(s_strike) ** 2
        - 2 * exy * np.sin(s_strike) * np.cos(s_strike)
    )
    Ss = (E - N) * np.sin(s_strike) * np.cos(s_strike) + exy * (
        np.cos(s_strike) ** 2 - np.sin(s_strike) ** 2
    )

    # Normal Strain
    sigma = Ee * np.sin(s_dip)

    # Shear Strain
    # In strike-dip-rake system, where x1 is the strike direction, a rake of
    # zero is left lateral
    tau = Ss * np.cos(s_rake)

    # ertid returns nanostrain
    ss = tau * G * 1e-9
    fns = sigma * PWM * 1e-9
    return ss, fns, N, E, NE

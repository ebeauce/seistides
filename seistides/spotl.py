import os
HOME = os.path.expanduser("~")
ROOT = HOME

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
def solid_earth_tides(
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
    files_basename="solid_earth_tides_Japan",
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
    files_basename: string, default to 'solid_earth_tides_Japan'
        Basename of the output files.
    """
    os.system("rm run_ertid.sh")
    f = open("run_ertid.sh", "a")
    f.write("#!/bin/csh\n")
    f.write("ertid << EOF\n")
    f.write(str(year1) + " " + str(jul_day1) + " 0\n")  # start time
    f.write(str(year2) + " " + str(jul_day2) + " 0\n")  # end time
    f.write(str(delta_hours) + "\n")  # sample interval
    f.write("t\n")  #
    f.write(str(station_lat) + "\n")  # "station" latitude
    f.write(str(station_lon) + "\n")  # "station" longitude
    f.write("0\n")
    f.write("0\n")
    f.write("3\n")
    f.write(f"{azimuth1:.0f}\n")
    f.write(f"{azimuth2:.0f}\n")
    f.write(f"{azimuth3:.0f}\n")
    f.write(f"{files_basename}_az{azimuth1:.0f}.txt\n")
    f.write(f"{files_basename}_az{azimuth2:.0f}.txt\n")
    f.write(f"{files_basename}_az{azimuth3:.0f}.txt\n")
    f.write("EOF\n")
    f.close()

    os.system("sh run_ertid.sh")

def ocean_load(
    station_name,
    station_longitude,
    station_latitude,
    station_elevation_m,
    year_start,
    julday_start,
    n_periods,
    sample_time_sec,
    local_models,
    hour_start=0,
    minute_start=0,
    second_start=0,
    working_dir=os.path.join(ROOT, "software/SPOTL/spotl/working"),
    azimuth1=0.0,
    azimuth2=270.0,
    azimuth3=315.0,
    files_basename="ocean_load_Japan_test",
    earth_green_function="gr.gbaver.wef.p02.ce",
    tidal_components=["k1", "m2", "s2", "n2"],
    global_model="got4p7.2004",
):
    """
    Compute strain from ocean tides using SPOTL routines.

    This function generates a C-shell script to automate the calculation of 
    ocean tide loading using the SPOTL (Some Programs for Ocean Tide Loading) 
    software package. It handles polygon creation for local/global models, 
    harmonic constants calculation, and time series generation for three 
    different azimuths.

    Parameters
    ----------
    station_name : str
        Name of the virtual station.
    station_longitude : float
        Longitude of the station in decimal degrees.
    station_latitude : float
        Latitude of the station in decimal degrees.
    station_elevation_m : float
        Elevation of the station in meters.
    year_start : int
        Starting year for the tidal time series (e.g., 2023).
    julday_start : int
        Starting Julian day (day of year) for the time series.
    n_periods : int
        Number of samples to generate in the time series.
    sample_time_sec : float or int
        Sampling interval in seconds.
    local_models : list of str
        List of names of local ocean tide models to be used (e.g., ["naoregional.1999"]).
    hour_start : int, optional
        Starting hour for the time series (0-23). Default is 0.
    minute_start : int, optional
        Starting minute for the time series (0-59). Default is 0.
    second_start : int, optional
        Starting second for the time series (0-59). Default is 0.
    working_dir : str, optional
        Directory where SPOTL routines and model files are located. 
        Defaults to a subpath within the ROOT directory.
    azimuth1 : float, optional
        Azimuth (angle from North in degrees) of the first channel. Default is 0.0.
    azimuth2 : float, optional
        Azimuth (angle from North in degrees) of the second channel. Default is 270.0.
    azimuth3 : float, optional
        Azimuth (angle from North in degrees) of the third channel. Default is 315.0.
    files_basename : str, optional
        Prefix for all temporary and output files generated by the script. 
        Default is "ocean_load_Japan_test".
    earth_green_function : str, optional
        Filename of the Earth's Green function to be used. 
        Default is "gr.gbaver.wef.p02.ce".
    tidal_components : list of str, optional
        List of tidal constituents to include (e.g., ["m2", "s2"]). 
        Default is ["k1", "m2", "s2", "n2"].
    global_model : str, optional
        The global ocean tide model to use outside of local model areas. 
        Default is "got4p7.2004".

    Notes
    -----
    The function performs the following steps:
    1. Changes the directory to `working_dir`.
    2. Constructs a `.csh` script that calls `polymake`, `nloadf`, `loadcomb`, 
       `harprp`, and `hartid`.
    3. Executes the script using `os.system`.
    4. Moves the resulting files back to the original working directory.

    Requires the SPOTL software suite to be installed and accessible in the 
    system path or specified working directory.
    """
    import glob
    from time import sleep

    # keep current working directory in memory for later
    cwd = os.getcwd()
    # go to target working dir
    os.chdir(working_dir)
    # list of models used in each sub-region
    #local_models = ["naoregional.1999", "osu.chinasea.2010"]
    models = local_models + [global_model]
    polygons = []
    combined_file = f"{files_basename}_all_components.txt"
    # write the input shell file
    with open(files_basename + ".csh", "w") as fin:
        fin.write("#!/bin/csh\n")

        # -------------------------------------------
        #          DEFINE POLYGONS
        # -------------------------------------------
        local_models_comp = {}
        polygons_comp = {}
        for c, comp in enumerate(tidal_components):
            # check which models support the tidal component
            local_models_comp[comp] = []
            for mod in local_models:
                if os.path.isfile(os.path.join(working_dir, f"{comp}.{mod}")):
                    local_models_comp[comp].append(mod)

            # polygons for local models
            polygons_comp[comp] = []
            for i, mod1 in enumerate(local_models_comp[comp]):
                # build polygon
                poly_name = f"poly{i+1}_{comp}"
                fin.write(f"polymake << EOF > {poly_name}.tmp\n")
                for j in range(i):
                    # if i==0, this loop doesn't do anything
                    fin.write(f"- {models[j]}\n")
                fin.write(f"+ {mod1}\n")
                fin.write("EOF\n")
                # add polygon to list
                polygons_comp[comp].append(poly_name)
            # polygon for global model
            poly_name = f"poly_global_{comp}"
            fin.write(f"polymake << EOF > {poly_name}.tmp\n")
            for i, mod in enumerate(local_models_comp[comp]):
                fin.write(f"- {mod}\n")
            # add global polygon to list
            polygons_comp[comp].append(poly_name)
            fin.write("EOF\n")
            # -------------------------------------------


            # -------------------------------------------
            #   COMPUTE LOAD FOR EACH MODEL, COMPONENT AND POLYGON
            # -------------------------------------------
            for poly_name, model in zip(
                    polygons_comp[comp], local_models_comp[comp] + [global_model]
                    ):
                # compute load for comp with model in poly
                fin.write(
                    f"nloadf {station_name} {station_latitude} {station_longitude} "
                    f"{station_elevation_m} {comp}.{model} {earth_green_function} "
                    f"l {poly_name}.tmp > {files_basename}_{poly_name}_{comp}.txt\n"
                )

            # combine all polygons
            for i in range(len(polygons_comp[comp])-1):
                if i == 0:
                    file1 = f"{files_basename}_{polygons_comp[comp][i]}_{comp}.txt"
                else:
                    file1 = out_file
                file2 = f"{files_basename}_{polygons_comp[comp][i+1]}_{comp}.txt"
                out_file = f"{files_basename}_tmp{i+1}.txt"
                fin.write(
                        f"cat {file1} {file2} | loadcomb c > {out_file}\n"
                        )

            # add this tidal component to others
            print(f"Adding {comp} to {combined_file}...")
            if os.path.isfile(combined_file):
                fin.write(f"cat {out_file} > {combined_file}\n")
            else:
                fin.write(f"cat {out_file} >> {combined_file}\n")

        #print("=====================================")
        #print(polygons_comp)
        #print(local_models_comp)
        #print("=====================================")

        # write the harmonic constants for extensional strain at given azimuths
        fin.write(
            f"harprp l {azimuth1} < {combined_file} > "
            f"harprp_out_{files_basename}_az{azimuth1:.0f}.txt\n"
        )
        fin.write(
            f"harprp l {azimuth2} < {combined_file} > "
            f"harprp_out_{files_basename}_az{azimuth2:.0f}.txt\n"
        )
        fin.write(
            f"harprp l {azimuth3} < {combined_file} > "
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
    sleep(0.25)
    # ready to run the script!
    os.system(f"sh {files_basename}.csh")
    sleep(0.5)
    # save all full file names
    filenames = glob.glob(f"*{files_basename}*")
    folder = os.getcwd()
    # go back to initial working directory and move files there
    os.chdir(cwd)
    for fn in filenames:
        os.system(f"mv {os.path.join(folder, fn)} .")
    print("Done!")

def ocean_height(
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
    working_dir=os.path.join(ROOT, "software/SPOTL/spotl/working"),
    files_basename="ocean_load_Japan_test",
    tidal_components=["k1", "m2", "s2", "n2"],
    model="got4p7.2004"
        ):
    """
    """
    import glob
    from time import sleep

    # keep current working directory in memory for later
    cwd = os.getcwd()
    # go to target working dir
    os.chdir(working_dir)
    # write the input shell file
    with open(files_basename + ".csh", "w") as fin:
        fin.write("#!/bin/csh\n")
        first_cmd_line = True
        for i, comp in enumerate(tidal_components):
            # write load file for ocean height
            symb = ">" if first_cmd_line else ">>"
            if not os.path.isfile(os.path.join(working_dir, f"{comp}.{model}")):
                print(f"Component {comp} not available in {model}!")
                continue
            fin.write(
                    f"oclook {comp}.{model} {station_latitude} {station_longitude} "
                    f"o {symb} {files_basename}_ocean_height.txt\n"
                    )
            first_cmd_line = False
        # write ocean height
        fin.write(
            f"cat {files_basename}_ocean_height.txt | "
            f"harprp o > harprp_out_{files_basename}_ocean_height.txt\n "
        )
        # use the harmonic constants to compute the time series
        # of tidal (nano)strain
        fin.write(
            f"hartid {year_start:d} {julday_start:d} {hour_start:d} "
            f"{minute_start:d} {second_start:d} {n_periods:d} {sample_time_sec} "
            f"< harprp_out_{files_basename}_ocean_height.txt "
            f"> tidal_series_{files_basename}_ocean_height.txt\n"
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

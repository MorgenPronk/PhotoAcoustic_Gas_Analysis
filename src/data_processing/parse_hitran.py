## parse_hitran.py

import json
import pandas as pd
import traceback
import os

def read_header(header_file):
    """
    Reads the HITRAN .header file and extracts column positions and discriptions.

    :param header_file:
        header_file (str): Path to the .header file.

    :returns dict:
        dict: A dictionary with column order positions and descriptions.
    """
    try:
        with open(header_file, 'r') as f:
            header = json.load(f)
        return {
            "columns": header["order"],
            "positions": header["position"],
            "formats": header["format"]
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"Header file {header_file} not found")
    except json.JSONDecodeError:
        raise ValueError(f"Header file {header_file} is not a valid JSON file.")

def parse_data_file(data_file, header_info):
    """
    Parses the HITRAN .data file using fixed-width form

    :param data_file: (str) Path to the .data file.
    :param header_info: (dict) Parsed header info with column names, positions, and formats

    :return: pd.DataFrame: Parsed HITRAN data
    """

    # Create fixed-width positions
    positions = sorted(header_info["positions"].items(), key=lambda x: x[1])
    start_positions = [pos[1] for pos in positions]
    end_positions = start_positions[1:] + [None] # None for the last column
    column_names = [pos[0] for pos in positions]

    # Parse file line by line
    data = []
    try:
        with open(data_file, 'r') as f:
            for line in f:
                row = {
                    col: line[start:end].strip()
                    for col, start, end in zip(column_names, start_positions, end_positions)
                }
                data.append(row)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file {data_file} not found")

    # Create DataFrame
    df = pd.DataFrame(data)

    # Convert numerical fields to proper types
    numeric_fields = ['nu', 'sw', 'gamma_air', 'gamma_self', 'elower', 'n_air', 'delta_air']
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce')

    return df

def union_wavenumber_ranges(wavenumber_ranges):
    """
    Unions overlapping wavenumber ranges from seperate light sources. This doesn't take into account a buffer amount, so ensure that buffers are added before this step

    :param wavenumber_ranges: (list[tuple]): List of (min_wavenumber, max_wavenumber) ranges in cm^-1
    :param buffers: (list[int,float] or int,float): buffers assocated with the ranges. Either one buffer for all ranges, or a buffer for each range
    :return: (list[tuple]) List of the merged (union) of the ranges
    """
    # Sort ranges by their start values
    sorted_ranges = sorted(wavenumber_ranges, key = lambda x: x[0])

    # Merge overlapping ranges
    merged_ranges = []
    current_range = sorted_ranges[0]
    for next_range in sorted_ranges[1:]:
        # check if the end of current range is in the next range and merge if they are
        if current_range[1] >= next_range[0]:
            current_range = (current_range[0], max(current_range[1], next_range[1]))
        else:
            merged_ranges.append(current_range)
            current_range = next_range

    # add the last range
    merged_ranges.append(current_range)

    return merged_ranges

def filter_hitran_data(data, wavenumber_ranges, intensity_threshold=1e-20, buffers=5):
    """
    Filters HITRAN data for wavenumber range and intensity threshold. Note that because one is inversely related to the
    other, that the minimum of one is the maximum of the other. This function automatically switches the min and max.

    :param data: (pd.DataFrame) Parsed HITRAN data
    :param wavenumber_ranges: list[tuple] (min_wavenumber, max_wavenumber) in cm^-1
    :param intensity_threshold: (float) Minimum line intensity to include
    :param buffers: (float or list[float]) Extra range to account for line broadening
    :return: pd.DataFrame Filtered HITRAN data
    """
    # Check the buffers for valid input
    if isinstance(buffers, (int, float)):
        buffers = [buffers] * len(wavenumber_ranges)  # Apply a single buffer to all ranges
    elif len(buffers) != len(wavenumber_ranges):
        raise ValueError("There must be one buffer applied for all ranges, or one for each range.")

    # Apply buffers to the ranges
    buffed_wavenumber_ranges = [
        (rng_min - buffer, rng_max + buffer)
        for (rng_min, rng_max), buffer in zip(wavenumber_ranges, buffers)
    ]

    # Merge overlapping ranges from different light sources
    modified_wavenumber_ranges = union_wavenumber_ranges(buffed_wavenumber_ranges)

    # Apply filters for each range
    filtered_data = pd.DataFrame()
    for min_nu, max_nu in modified_wavenumber_ranges:
        filtered_instance = data[
            (data['nu'] >= min_nu) & (data['nu'] <= max_nu) & (data['sw'] > intensity_threshold)
        ]
        filtered_data = pd.concat([filtered_data, filtered_instance], ignore_index=True)

    return filtered_data

def parse_hitran(file_base, wavenumber_ranges, intensity_threshold=1e-20, buffer=5):
    """
    Parses and preprocesses HITRAN .data and .header files

    :param file_base: (str) Base path to the HITRAN file pair (e.g. "C2H2")
    :param wavenumber_range: (tuple) (min_wavenumber, max_wavenumber) in cm^-1
    :param intensity_threshold: (float) Minimum line intensity to include
    :param buffer: (float) Extra range to account for line broadening

    :return: (pd.DataFrame) Preprocessed HITRAN data
    """
    header_file = f"{file_base}.header"
    data_file = f"{file_base}.data"

    # Step 1: Read header
    header_info = read_header(header_file)

    # Step 2: Parse data file
    raw_data = parse_data_file(data_file, header_info)

    # Step 3: Filter data
    filtered_data = filter_hitran_data(raw_data, wavenumber_ranges, intensity_threshold, buffer)

    return filtered_data

def wavelength_to_wavenumber(wavelength_ranges):
    """
    Converts lists of wavelengths (usually the unit given for lightsources) and gives them back in wavenumbers (units
    used in HITRAN and other databases). Typically, this will be used for converting the ranges of practical data that will
    be parsed from a database, like HITRAN, to use for calculations.
    :param wavelength_range: (list[tuple]) list of wavelength range.
    :return: wavenumber_range (list[tuple]) list of wavnumber range.
    """
    if any(n <= 0 for rng in wavelength_ranges for n in rng):
        raise ValueError("Negative or zero wavelengths are not permitted.")
    return [(1e4 / rng[1], 1e4 / rng[0]) for rng in wavelength_ranges]

if __name__ == '__main__':
    # Example Usage
    file_base = '../../data/HITRAN/C2H2' # If running it from the project folder the file base is likely './data/HITRAN/C2H2'
    light_source_emission = [(15, 18), (20, 25)] # light source with 15 to 18 nm and 20 to 25 for optical 3 dB Bandwidth
    wavenumber_range = wavelength_to_wavenumber(light_source_emission)
    intensity_threshold = 1e-20 # Example intensity threshold
    buffers = 50 # buffer of 50 cm^-1

    # Parse and filter HITRAN data
    try:
        filtered_data = parse_hitran(file_base, wavenumber_range, intensity_threshold, buffers)
        print("Filtered Data:")
        print(filtered_data.columns)
        print(filtered_data.head()) # Display dataframe
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()


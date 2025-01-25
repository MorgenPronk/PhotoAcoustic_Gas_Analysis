import json
import pandas as pd
import traceback
import os
from configparser import ConfigParser
from src.utils import resolve_path

# Load configuration
config = ConfigParser()
config.read(resolve_path("config.ini"))

# Debugging the path resolution
print("Resolved path to config.ini:", resolve_path("config.ini"))

# Resolve HITRAN directory
HITRAN_DIR = resolve_path(config["paths"]["HITRAN_dir"])
if not os.path.exists(HITRAN_DIR):
    raise FileNotFoundError(f"HITRAN directory not found: {HITRAN_DIR}")

# Debugging the resolved HITRAN_DIR
print("Resolved HITRAN_DIR:", HITRAN_DIR)

def read_header(header_file):
    """
    Reads the HITRAN .header file and extracts column positions and descriptions.

    :param header_file: Path to the .header file.
    :returns: A dictionary with column order positions and descriptions.
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
    Parses the HITRAN .data file using fixed-width format.

    :param data_file: Path to the .data file.
    :param header_info: Parsed header info with column names, positions, and formats.
    :return: pd.DataFrame with the parsed HITRAN data.
    """
    positions = sorted(header_info["positions"].items(), key=lambda x: x[1])
    start_positions = [pos[1] for pos in positions]
    end_positions = start_positions[1:] + [None]
    column_names = [pos[0] for pos in positions]

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

    df = pd.DataFrame(data)

    # Convert numerical fields to proper types
    numeric_fields = ['nu', 'sw', 'gamma_air', 'gamma_self', 'elower', 'n_air', 'delta_air']
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce')

    return df

def union_wavenumber_ranges(wavenumber_ranges):
    """
    Unions overlapping wavenumber ranges.

    :param wavenumber_ranges: List of (min_wavenumber, max_wavenumber) ranges in cm^-1.
    :return: List of the merged (union) ranges.
    """
    sorted_ranges = sorted(wavenumber_ranges, key=lambda x: x[0])
    merged_ranges = []
    current_range = sorted_ranges[0]

    for next_range in sorted_ranges[1:]:
        if current_range[1] >= next_range[0]:
            current_range = (current_range[0], max(current_range[1], next_range[1]))
        else:
            merged_ranges.append(current_range)
            current_range = next_range

    merged_ranges.append(current_range)
    return merged_ranges

def filter_hitran_data(data, wavenumber_ranges, intensity_threshold=1e-20, buffers=5):
    """
    Filters HITRAN data for wavenumber range and intensity threshold.

    :param data: Parsed HITRAN data as a pd.DataFrame.
    :param wavenumber_ranges: List of (min_wavenumber, max_wavenumber) ranges in cm^-1.
    :param intensity_threshold: Minimum line intensity to include.
    :param buffers: Extra range to account for line broadening.
    :return: Filtered HITRAN data as a pd.DataFrame.
    """
    if isinstance(buffers, (int, float)):
        buffers = [buffers] * len(wavenumber_ranges)
    elif len(buffers) != len(wavenumber_ranges):
        raise ValueError("There must be one buffer applied for all ranges, or one for each range.")

    buffed_wavenumber_ranges = [
        (rng_min - buffer, rng_max + buffer)
        for (rng_min, rng_max), buffer in zip(wavenumber_ranges, buffers)
    ]

    modified_wavenumber_ranges = union_wavenumber_ranges(buffed_wavenumber_ranges)
    filtered_data = pd.DataFrame()

    for min_nu, max_nu in modified_wavenumber_ranges:
        filtered_instance = data[
            (data['nu'] >= min_nu) & (data['nu'] <= max_nu) & (data['sw'] > intensity_threshold)
        ]
        filtered_data = pd.concat([filtered_data, filtered_instance], ignore_index=True)

    return filtered_data

def parse_hitran(file_base, wavenumber_ranges, intensity_threshold=1e-20, buffer=5):
    """
    Parses and preprocesses HITRAN .data and .header files.

    :param file_base: Base path to the HITRAN file pair (e.g., "CO2").
    :param wavenumber_ranges: List of (min_wavenumber, max_wavenumber) ranges in cm^-1.
    :param intensity_threshold: Minimum line intensity to include.
    :param buffer: Extra range to account for line broadening.
    :return: Preprocessed HITRAN data as a pd.DataFrame.
    """
    header_file = f"{file_base}.header"
    data_file = f"{file_base}.data"

    # Resolve paths
    header_file = resolve_path(header_file)
    data_file = resolve_path(data_file)

    # Read and parse files
    header_info = read_header(header_file)
    raw_data = parse_data_file(data_file, header_info)
    filtered_data = filter_hitran_data(raw_data, wavenumber_ranges, intensity_threshold, buffer)

    return filtered_data

def wavelength_to_wavenumber(wavelength_ranges):
    """
    Converts wavelength ranges to wavenumber ranges.

    :param wavelength_ranges: List of wavelength ranges [(min, max)] in microns.
    :return: List of wavenumber ranges [(min, max)] in cm^-1.
    """
    if any(n <= 0 for rng in wavelength_ranges for n in rng):
        raise ValueError("Negative or zero wavelengths are not permitted.")
    return [(1e4 / rng[1], 1e4 / rng[0]) for rng in wavelength_ranges]

if __name__ == '__main__':
    file_base = os.path.join(HITRAN_DIR, 'CO2')
    light_source_emission = [(2.5, 3.0), (4.0, 4.5)]
    wavenumber_range = wavelength_to_wavenumber(light_source_emission)
    intensity_threshold = 1e-20
    buffers = 50

    try:
        filtered_data = parse_hitran(file_base, wavenumber_range, intensity_threshold, buffers)
        print("Filtered Data:")
        print(filtered_data.columns)
        print(filtered_data.head())
        print(f"Dataframe shape: {filtered_data.shape}")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

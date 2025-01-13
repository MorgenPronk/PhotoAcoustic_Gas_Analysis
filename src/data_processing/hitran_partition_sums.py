"""
hitran_partition_sums.py

This module provides functionality for loading, interpolating, and computing molecular partition sums
from HITRAN's Total Internal Partition Sums (TIPS) data. Partition sums are critical for temperature
corrections in spectroscopic line intensity calculations, particularly when using HITRAN data.

Context:
- HITRAN provides precomputed partition sums for molecules and isotopologues at various temperatures.
- These sums are essential for correcting line intensities for non-reference temperatures.

Features:
1. Load partition sum data from HITRAN's TIPS dataset.
2. Interpolate partition sums for arbitrary temperatures.
3. Compute the partition function ratio ( Q(T_{\text{ref}})/Q(T) ) for temperature corrections.

Dependencies:
- Requires the HITRAN TIPS data (`QTpy` directory) available from the HITRAN online supplemental section.
- Assumes the partition sum data is stored in `.QTpy` pickle files, one per isotopologue.
"""

import pickle
import os
from scipy.interpolate import interp1d

def load_partition_sums(molecule_id, isotopologue_id, qtpy_dir='QTpy'):
    """
    Load partition sums for a specific molecule and isotopologue from HITRAN TIPS data.

    :param molecule_id: ID of the molecule (string, e.g., '1' for H2O).
    :param isotopologue_id: ID of the isotopologue (string, e.g., '161' for H2O).
    :param qtpy_dir: Directory containing QTpy pickle files.
    :return: Dictionary mapping temperature (int) to partition sum (float).
    """
    file_path = os.path.join(qtpy_dir, f"{molecule_id}_{isotopologue_id}.QTpy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Partition sum file not found: {file_path}")

    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def interpolate_partition_sum(qt_dict, temperature):
    """
    Interpolate the partition sum for a given temperature using linear interpolation.

    :param qt_dict: Dictionary mapping temperature (int) to partition sum (float).
    :param temperature: Desired temperature (float).
    :return: Interpolated partition sum (float).
    """
    temperatures = sorted(qt_dict.keys(), key=int)  # Ensure keys are sorted numerically
    temps = [int(t) for t in temperatures]
    partition_sums = [float(qt_dict[str(t)]) for t in temps]

    # Interpolation function
    interp_func = interp1d(temps, partition_sums, kind='linear', fill_value="extrapolate")
    return float(interp_func(temperature))


def compute_partition_function_ratio(molecule_id, isotopologue_id, temperature, reference_temperature, qtpy_dir='QTpy'):
    """
    Compute the partition function ratio Q(T_ref)/Q(T) for temperature corrections.

    :param molecule_id: ID of the molecule (string, e.g., '1' for H2O).
    :param isotopologue_id: ID of the isotopologue (string, e.g., '161' for H2O).
    :param temperature: Desired temperature (float).
    :param reference_temperature: Reference temperature (float, e.g., 296 K).
    :param qtpy_dir: Directory containing QTpy pickle files.
    :return: Partition function ratio Q(T_ref)/Q(T) (float).
    """
    # Load partition sums
    qt_dict = load_partition_sums(molecule_id, isotopologue_id, qtpy_dir)

    # Interpolate partition sums
    q_t = interpolate_partition_sum(qt_dict, temperature)
    q_t_ref = interpolate_partition_sum(qt_dict, reference_temperature)

    # Compute and return the ratio
    return q_t_ref / q_t

# Example molecule and isotopologue IDs
if __name__ == "__main__":
    molecule_id = '1'  # H2O
    isotopologue_id = '8'  # H2O isotopologue

    # Desired temperature and reference temperature
    temperature = 300.0  # Current temperature (K)
    reference_temperature = 296.0  # Reference temperature (K)

    # Compute the partition function ratio
    qtpy_dir = '../../data/TIPS2021/QTpy'  # Directory containing partition sum data
    q_ratio = compute_partition_function_ratio(molecule_id, isotopologue_id, temperature, reference_temperature, qtpy_dir)

    print(f"Partition function ratio Q({reference_temperature})/Q({temperature}): {q_ratio}")
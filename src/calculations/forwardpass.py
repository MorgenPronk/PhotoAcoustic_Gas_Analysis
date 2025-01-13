## forwardpass.py

import torch
# import numpy as np
import pandas as pd
from src.data_processing.parse_hitran import *
from src.data_processing.hitran_partition_sums import compute_partition_function_ratio

# TODO: Light_source_range might be good if it could take wavelength values or wavenumber
# TODO: Make it generic so it can get integrate with getting data from a database and not a file or directory
def load_hitran_data(species, light_source_ranges, intensity_threshold=1e-20, buffers=5):
    """
    Load HITRAN data for a given species based on light source parameters. Only loads areas near the light source for
    efficiency.

    :param species: Name of the species (e.g., 'CO2', 'CH4').
    :param light_source_ranges: list of tuples specifying the ranges of the light sources. e.g (min, max).
        **THIS IS CURRENTLY IN WAVELENGTH (microns)**
    :param intensity_threshold: Minimum line intensity to include.
    :param buffers: Extra range to account for line broadening.
    :return: pd.DataFrame with preprocessed HITRAN data for the species.
    """

    # Parse and filter HITRAN data
    try:
        file_base = f'../../data/HITRAN/{species}'  # If running it from the project folder the file base is likely './data/HITRAN/C2H2'
        wavenumber_ranges = wavelength_to_wavenumber(light_source_ranges)
        filtered_data = parse_hitran(file_base, wavenumber_ranges, intensity_threshold, buffers)
        return filtered_data
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()


from src.data_processing.hitran_partition_sums import compute_partition_function_ratio

def compute_absorptivity(hitran_data, concentrations, light_source, pressure=1.0, temperature=296, T_ref=296):
    """
    Compute the total absorptivity for the mixture with temperature correction.

    :param hitran_data: Dict of {species: pd.DataFrame} with HITRAN data.
    :param concentrations: Dict of {species: concentration} (mol/L).
    :param light_source: Dict with light source properties (wavelength range, intensity).
    :param pressure: Total pressure (atm).
    :param temperature: Temperature (K).
    :param T_ref: Reference temperature (K). Default is 296 K.
    :return: torch.Tensor of total absorptivity across wavelengths.
    """
    c2 = 1.4388  # Second radiation constant in cm K
    wavelengths = light_source['wavelengths']
    wavenumbers = 1e4 / wavelengths  # Convert wavelengths (microns) to wavenumbers (cm^-1)

    absorptivity = torch.zeros_like(wavenumbers)

    for gas, data in hitran_data.items():
        concentration = concentrations[gas]
        try:
            # HITRAN columns
            nu = torch.tensor(data['nu'].values, dtype=torch.float32)  # Wavenumbers
            S_ref = torch.tensor(data['sw'].values, dtype=torch.float32)  # Line intensities at T_ref
            E_l = torch.tensor(data['elower'].values, dtype=torch.float32)  # Lower state energy
            gamma_air = torch.tensor(data['gamma_air'].values, dtype=torch.float32)
            gamma_self = torch.tensor(data['gamma_self'].values, dtype=torch.float32)

            # Molecule and isotopologue IDs
            molec_id = str(data['molec_id'].iloc[0])
            iso_id = str(data['local_iso_id'].iloc[0])

            # Compute Q(T_ref)/Q(T) using the partition sum logic
            Q_ratio = compute_partition_function_ratio(
                molecule_id=molec_id,
                isotopologue_id=iso_id,
                temperature=temperature,
                reference_temperature=T_ref,
                qtpy_dir='..\\..\\data\\TIPS2021\\QTpy'
            )

            # Temperature correction for line intensity
            T_correction = (
                torch.exp(-c2 * E_l / temperature) / torch.exp(-c2 * E_l / T_ref)
            ) * (
                (1 - torch.exp(-c2 * nu / temperature)) / (1 - torch.exp(-c2 * nu / T_ref))
            )
            S = S_ref * Q_ratio * T_correction

            # Broadening and line shape
            gamma = gamma_air * (1 - concentration) + gamma_self * concentration
            for i, nu_center in enumerate(nu):
                f = (gamma[i] / ((wavenumbers - nu_center) ** 2 + gamma[i] ** 2)) / torch.pi
                absorptivity += S[i] * f * concentration

        except KeyError as e:
            print(f"Missing expected column in HITRAN data for {gas}: {e}")
            continue

    return absorptivity




def calculate_pressure_signal(absorptivity, light_source, environmental_params):
    """
    Placeholder: Calculate the pressure signal based on absorptivity.

    :param absorptivity: torch.Tensor of total absorptivity across wavelengths.
    :param light_source: Dict with light source properties (wavelength range, intensity).
    :param environmental_params: Dict with environmental parameters (e.g., beta, rho, Cp).
    :return: torch.Tensor of pressure signal over time.
    """
    # Placeholder: Use absorptivity to calculate the pressure signal
    return torch.zeros(1)  # Single value or array for time-dependent signal


def voltage_from_pressure(pressure_signal, calibration_params):
    """
    Placeholder: Convert pressure signal to microphone voltage.

    :param pressure_signal: torch.Tensor of predicted pressure signals.
    :param calibration_params: Dict with calibration parameters for microphone.
    :return: torch.Tensor of predicted microphone voltage.
    """
    # Placeholder: Linear scaling for simplicity
    return calibration_params['sensitivity'] * pressure_signal

def max_and_min(light_wavelengths):
    """
    Takes the discrete data for lightsource wavelengths and gives the min and max values

    :param light_wavelengths: torch.Tensor of wavelengths (1 x 2)
    :return: tuple (max, min) of wavelengths
    """
    return (float(light_wavelengths.min()), float(light_wavelengths.max()))

def compute_total_error(predicted_voltages, measured_voltages):
    """
    Compute the total error between predicted and measured voltages.

    :param predicted_voltages: List of predicted voltages for each light source.
    :param measured_voltages: List of measured voltages for each light source.
    :return: Scalar torch.Tensor of the combined error.
    """
    predicted = torch.tensor(predicted_voltages)
    measured = torch.tensor(measured_voltages)
    return torch.linalg.norm(predicted - measured)


def forward_pass(concentrations, gases, light_sources, measured_voltages, pressure=1.0, temperature=296):
    """
    Forward pass for the numerical solver, generalized for N light sources and N gases.

    :param concentrations: Dict of {species: concentration} (mol/L).
    :param gases: List of gases in the mixture (e.g., ['CO2', 'CH4']).
    :param light_sources: List of light source properties (wavelength range, intensity).
    :param measured_voltages: List of measured microphone voltages (one per light source).
    :param pressure: Total pressure (atm). Default is 1 atm.
    :param temperature: Temperature (K). Default is 296 K.
    :return: Scalar torch.Tensor of the combined error across all light sources.
    """
    if len(light_sources) != len(gases):
        raise ValueError("Number of light sources must match the number of gases in the mixture.")
    if len(light_sources) != len(measured_voltages):
        raise ValueError("Number of light sources must match the number of voltage measurements.")

    # Step 1: Load HITRAN data for each gas
    # hitran_data is going to be structured like:
    #  {"gas_formula": pd.DataFrame_filtered_data_join_of_lightsources, ... }
    hitran_data = {}
    for i, gas in enumerate(gases):
        try:
            # For each gas, we need to find the data that overlaps with each lightsource
            # light_source_ranges is going to be the list of min - max tuples that represent the range for each light
            #  source. The parse_hitran() function used by load_hitran_data() expects a list of tuples anyways.
            light_source_ranges = []
            for light_source in light_sources:
                wavenumber_min_max_tuple = max_and_min(light_source['wavelengths'])
                light_source_ranges.append(wavenumber_min_max_tuple)
            hitran_data[gas] = load_hitran_data(
                gas, light_source_ranges,
                light_sources[i]['threshold'],
                light_sources[i]['buffers']
            )
        except Exception as e:
            print(f"Error loading HITRAN data for gas {gas}: {e}")
            continue
    #print(f"hitran_data:\n {hitran_data}") # Debugging
    #print(f"hitran columns available:\n {list(hitran_data.items())[0][1].columns}") # debugging

    # Step 2: Compute predicted voltage for each light source
    predicted_voltages = []
    for i, light_source in enumerate(light_sources):
        absorptivity = compute_absorptivity(hitran_data, concentrations, light_source, pressure, temperature)
        pressure_signal = calculate_pressure_signal(absorptivity, light_source, {})
        predicted_voltage = voltage_from_pressure(pressure_signal, {'sensitivity': 1.0})
        predicted_voltages.append(predicted_voltage)

    # Step 3: Compute the combined error
    error = compute_total_error(predicted_voltages, measured_voltages)
    return error

# Example usage
if __name__ == "__main__":
    # Example list of gases
    gases = ['CO2', 'CH4']

    # Initial guess for concentrations (mol/L)
    concentrations = {
        'CO2': 0.01,  # Initial concentration of CO2
        'CH4': 0.005,  # Initial concentration of CH4
    }

    # Example light sources
    # Careful of the format that we need the wavelengths in

    # TODO: There is going to be an error if we have empty dataframes. I.e. A lightsource range doesn't overlap with a chemicals line-by-line data.
    light_sources = [
        {
            'name': 'Light Source 1',
            'wavelengths': torch.linspace(3.000, 3.500, 500),  # Wavelength range in microns
            'intensity': torch.ones(500),  # Placeholder: Uniform intensity
            'threshold': 1e-20,
            'buffers': 5
        },
        {
            'name': 'Light Source 2',
            'wavelengths': torch.linspace(4.000, 4.500, 500),  # Wavelength range in microns
            'intensity': torch.ones(500),  # Placeholder: Uniform intensity
            'threshold': 1e-20,
            'buffers': 5
        },
    ]

    # Example measured microphone voltages (one per light source)
    measured_voltages = [0.1, 0.05]  # Placeholder: Simulated measurements

    # Perform forward pass
    error = forward_pass(concentrations, gases, light_sources, measured_voltages)

    # Output the combined error
    print("Combined error across light sources:", error)


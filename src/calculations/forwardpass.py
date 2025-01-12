## forwardpass.py

import torch
# import numpy as np
import pandas as pd
from src.data_processing.parse_hitran import *

# TODO: Light_source_emission might be good if it could take wavelength values or wavenumber
# TODO: Make it generic so it can get integrate with getting data from a database and not a file or directory
def load_hitran_data(species, light_source_range, intensity_threshold=1e-20, buffers=5):
    """
    Load HITRAN data for a given species based on light source parameters. Only loads areas near the light source for
    efficiency.

    :param species: Name of the species (e.g., 'CO2', 'CH4').
    :param light_source_range: tuple specifying the range of the light source. e.g (min, max).
        **THIS IS CURRENTLY IN WAVELENGTH (microns)**
    :param intensity_threshold: Minimum line intensity to include.
    :param buffers: Extra range to account for line broadening.
    :return: pd.DataFrame with preprocessed HITRAN data for the species.
    """

    # Parse and filter HITRAN data
    try:
        file_base = f'../../data/HITRAN/{species}'  # If running it from the project folder the file base is likely './data/HITRAN/C2H2'
        wavenumber_range = wavelength_to_wavenumber(light_source_emission)
        filtered_data = parse_hitran(file_base, wavenumber_range, intensity_threshold, buffers)
        return filtered_data
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

def compute_absorptivity(hitran_data, concentrations, light_source, pressure=1.0, temperature=296):
    """
    Compute the total absorptivity for the mixture.

    :param hitran_data: Dict of {species: pd.DataFrame} with HITRAN data.
    :param concentrations: Dict of {species: concentration} (mol/L).
    :param light_source: Dict with light source properties (wavelength range, intensity).
    :param pressure: Total pressure (atm).
    :param temperature: Temperature (K).
    :return: np.array of total absorptivity across wavelengths.
    """
    wavelengths = light_source['wavelengths']
    absorptivity = np.zeros_like(wavelengths)  # Placeholder for actual computation

    for gas, data in hitran_data.items():
        # Compute gas-specific contributions to absorptivity
        # TODO: Implement actual absorptivity calculations based on HITRAN data
        pass

    return absorptivity


def calculate_pressure_signal(absorptivity, light_source, environmental_params):
    """
    Placeholder: Calculate the pressure signal based on absorptivity.

    :param absorptivity: np.array of total absorptivity across wavelengths.
    :param light_source: Dict with light source properties (wavelength range, intensity).
    :param environmental_params: Dict with environmental parameters (e.g., beta, rho, Cp).
    :return: np.array of pressure signal over time.
    """
    # Placeholder: Use absorptivity to calculate the pressure signal
    return np.zeros(1)  # Single value or array for time-dependent signal


def voltage_from_pressure(pressure_signal, calibration_params):
    """
    Placeholder: Convert pressure signal to microphone voltage.

    :param pressure_signal: np.array of predicted pressure signals.
    :param calibration_params: Dict with calibration parameters for microphone.
    :return: np.array of predicted microphone voltage.
    """
    # Placeholder: Linear scaling for simplicity
    return calibration_params['sensitivity'] * pressure_signal

def max_and_min(light_wavelengths):
    """
    Takes the discrete data for lightsource wavelengths and gives the min and max values

    :param light_wavelengths: np.array (1 x 2)
    :return: tuple (max, min)
    """
    return (light_wavelengths.min(), light_wavelengths.max())

def compute_total_error(predicted_voltages, measured_voltages):
    """
    Compute the total error between predicted and measured voltages.

    :param predicted_voltages: List of predicted voltages for each light source.
    :param measured_voltages: List of measured voltages for each light source.
    :return: Scalar error value.
    """
    predicted = np.array(predicted_voltages)
    measured = np.array(measured_voltages)
    return np.linalg.norm(predicted - measured)


def forward_pass(concentrations, gases, light_sources, measured_voltages, pressure=1.0, temperature=296):
    """
    Forward pass for the numerical solver, generalized for N light sources and N gases.

    :param concentrations: Dict of {species: concentration} (mol/L).
    :param gases: List of gases in the mixture (e.g., ['CO2', 'CH4']).
    :param light_sources: List of light source properties (wavelength range, intensity).
    :param measured_voltages: List of measured microphone voltages (one per light source).
    :param pressure: Total pressure (atm). Default is 1 atm.
    :param temperature: Temperature (K). Default is 296 K.
    :return: float, the combined error across all light sources.
    """
    if len(light_sources) != len(gases):
        raise ValueError("Number of light sources must match the number of gases in the mixture.")
    if len(light_sources) != len(measured_voltages):
        raise ValueError("Number of light sources must match the number of voltage measurements.")

    # Step 1: Load HITRAN data for each gas
    hitran_data = {}
    for i, gas in enumerate(gases):
        try:
            wavenumber_ranges = max_and_min(light_sources[i]['wavelengths'])
            hitran_data[gas] = load_hitran_data(gas, wavenumber_ranges,
                                                light_sources[i]['threshold'],
                                                light_sources[i]['buffers'])
        except Exception as e:
            print(f"Error loading HITRAN data for gas {gas}: {e}")
            continue

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
    light_sources = [
        {
            'name': 'Light Source 1',
            'wavelengths': np.linspace(4500, 5000, 500),  # Wavelength range in nm
            'intensity': np.ones(500),  # Placeholder: Uniform intensity
            'threshold': 1e-20,
            'buffers': 5
        },
        {
            'name': 'Light Source 2',
            'wavelengths': np.linspace(4000, 4500, 500),  # Wavelength range in nm
            'intensity': np.ones(500),  # Placeholder: Uniform intensity
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

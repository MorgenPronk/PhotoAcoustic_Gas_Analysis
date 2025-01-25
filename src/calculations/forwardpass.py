import torch
import pandas as pd
from configparser import ConfigParser
import os
from src.utils import resolve_path
from src.data_processing.parse_hitran import parse_hitran, wavelength_to_wavenumber
from src.data_processing.hitran_partition_sums import compute_partition_function_ratio


# Debugging the path resolution
print("Resolved path to config.ini:", resolve_path("config.ini"))

# Load configuration
config = ConfigParser()
config.read(resolve_path("config.ini"))

# Validate and resolve paths
TIPS2021_DIR = resolve_path(config["paths"]["TIPS2021_dir"])
HITRAN_DIR = resolve_path(config["paths"]["HITRAN_dir"])

# Debugging the resolved paths
print("Resolved TIPS2021_DIR:", TIPS2021_DIR)
print("Resolved HITRAN_DIR:", HITRAN_DIR)

if not os.path.exists(TIPS2021_DIR):
    raise FileNotFoundError(f"TIPS2021 directory not found: {TIPS2021_DIR}")
if not os.path.exists(HITRAN_DIR):
    raise FileNotFoundError(f"HITRAN directory not found: {HITRAN_DIR}")

def load_hitran_data(species, light_source_ranges, intensity_threshold=1e-20, buffers=5):
    """
    Load HITRAN data for a given species based on light source parameters. Only loads areas near the light source for efficiency.

    :param species: Name of the species (e.g., 'CO2', 'CH4').
    :param light_source_ranges: List of tuples specifying the ranges of the light sources. e.g., [(min, max)].
        **This is expected to be in wavelengths (microns)**.
    :param intensity_threshold: Minimum line intensity to include.
    :param buffers: Extra range to account for line broadening.
    :return: pd.DataFrame with preprocessed HITRAN data for the species.
    """
    try:
        file_base = os.path.join(HITRAN_DIR, species)
        wavenumber_ranges = wavelength_to_wavenumber(light_source_ranges)
        filtered_data = parse_hitran(file_base, wavenumber_ranges, intensity_threshold, buffers)
        return filtered_data
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

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
        # If no data is available in this region, the add zero absorptivitiy for the gas
        if data.empty:
            continue

        concentration = concentrations.get(gas, 0.0) #If a gas doesn't have any contribution in this region then we will give it zero.
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
                qtpy_dir=TIPS2021_DIR
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
    return torch.zeros(1)  # Placeholder for pressure signal calculation

def voltage_from_pressure(pressure_signal, calibration_params):
    """
    Placeholder: Convert pressure signal to microphone voltage.

    :param pressure_signal: torch.Tensor of predicted pressure signals.
    :param calibration_params: Dict with calibration parameters for microphone.
    :return: torch.Tensor of predicted microphone voltage.
    """
    return calibration_params['sensitivity'] * pressure_signal

def max_and_min(light_wavelengths):
    """
    Takes the discrete data for light source wavelengths and gives the min and max values.

    :param light_wavelengths: torch.Tensor of wavelengths (1 x 2).
    :return: Tuple (min, max) of wavelengths.
    """
    return float(light_wavelengths.min()), float(light_wavelengths.max())

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
    hitran_data = {}
    for gas in gases:
        light_source_ranges = [max_and_min(ls['wavelengths']) for ls in light_sources]
        hitran_data[gas] = load_hitran_data(gas, light_source_ranges)

    # Step 2: Compute predicted voltage for each light source
    predicted_voltages = []
    for i, light_source in enumerate(light_sources):
        absorptivity = compute_absorptivity(hitran_data, concentrations, light_source, pressure, temperature)
        pressure_signal = calculate_pressure_signal(absorptivity, light_source, {})
        predicted_voltage = voltage_from_pressure(pressure_signal, {'sensitivity': 1.0})
        predicted_voltages.append(predicted_voltage)

    # Step 3: Compute the combined error
    return compute_total_error(predicted_voltages, measured_voltages)

if __name__ == "__main__":
    # Example usage
    gases = ['CO2', 'CH4']
    concentrations = {'CO2': 0.01, 'CH4': 0.005}
    light_sources = [
        {'name': 'Light Source 1', 'wavelengths': torch.linspace(3.0, 3.5, 500), 'intensity': torch.ones(500)},
        {'name': 'Light Source 2', 'wavelengths': torch.linspace(4.0, 4.5, 500), 'intensity': torch.ones(500)}
    ]
    measured_voltages = [0.1, 0.05]

    error = forward_pass(concentrations, gases, light_sources, measured_voltages)
    print("Combined error across light sources:", error)

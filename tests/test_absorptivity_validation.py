import torch
from hapi import *
from src.calculations.forwardpass import compute_absorptivity
from src.data_processing.parse_hitran import parse_hitran
import matplotlib.pyplot as plt
from configparser import ConfigParser
from src.utils import resolve_path
import os

# Load paths from config
config = ConfigParser()
config.read(resolve_path("config.ini"))

TIPS2021_DIR = resolve_path(config["paths"]["TIPS2021_dir"])
HITRAN_DIR = resolve_path(config["paths"]["HITRAN_dir"])

def test_absorptivity_against_hapi():
    """
    Validate the custom compute_absorptivity function against HAPI's absorptionCoefficient_Voigt.
    """

    # Step 1: Initialize HAPI and fetch HITRAN data
    db_begin('HITRAN_Data')
    numin = 2260  # Lower bound of the wavenumber range (cm⁻¹)
    numax = 2400  # Upper bound of the wavenumber range (cm⁻¹)
    fetch('CO2', 2, 1, numin=numin, numax=numax)  # CO2 molecule, isotopologue 1

    # Step 2: Define conditions
    temperature = 296  # Temperature in K
    pressure = 1.0  # Pressure in atm
    path_length = 1.0  # Path length in cm
    concentration = 0.01  # Mole fraction of CO2
    wavenumber_range = [numin, numax]  # Wavenumber range (cm⁻¹)
    wavenumber_step = 0.01  # Step size for wavenumber grid
    diluent = {'air': 0.8, 'self': 0.2}  # Example gas mixture

    # Step 3: Calculate absorption coefficients using HAPI
    nu, coef_hapi = absorptionCoefficient_Voigt(
        SourceTables='CO2',
        Environment={'T': temperature, 'p': pressure},
        WavenumberRange=wavenumber_range,
        WavenumberStep=wavenumber_step,
        Diluent=diluent
    )

    # Step 4: Calculate absorptivity using custom function
    # Parse HITRAN data for custom calculation
    hitran_data = parse_hitran(
        file_base=os.path.join(HITRAN_DIR, 'CO2'),
        wavenumber_ranges=[(numin, numax)]
    )

    # Convert wavenumber grid to wavelengths (in microns)
    light_source_wavelengths = 1e4 / torch.tensor(nu)  # Convert cm⁻¹ to microns

    # Define light source properties
    light_source = {'wavelengths': light_source_wavelengths}

    # Compute absorptivity with custom function
    absorptivity_custom = compute_absorptivity(
        hitran_data={'CO2': hitran_data},
        concentrations={'CO2': concentration},
        light_source=light_source,
        pressure=pressure,
        temperature=temperature
    )

    # Step 5: Validate Results
    # Ensure matching dimensions
    assert absorptivity_custom.shape == torch.tensor(coef_hapi).shape, \
        "Shape mismatch between custom and HAPI results."

    # Compute maximum difference
    max_diff = torch.max(torch.abs(absorptivity_custom - torch.tensor(coef_hapi)))
    assert max_diff < 1e-3, f"Absorptivity mismatch too large: max_diff={max_diff.item()}"

    # Step 6: Visualize Results
    plt.figure(figsize=(10, 6))
    plt.plot(nu, coef_hapi, label='HAPI Absorption Coefficient', linestyle='--')
    plt.plot(nu, absorptivity_custom.detach().numpy()*93, label='Custom Absorptivity')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Absorptivity')
    plt.legend()
    plt.title('Absorptivity Comparison: HAPI vs Custom')
    plt.grid()
    plt.show()

    print("Validation passed: Custom absorptivity matches HAPI within tolerance.")

if __name__ == "__main__":
    test_absorptivity_against_hapi()

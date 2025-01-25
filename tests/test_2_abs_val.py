import torch
from hapi import *
from src.calculations.forwardpass import compute_absorptivity
from src.data_processing.parse_hitran import parse_hitran
import matplotlib.pyplot as plt
from configparser import ConfigParser
from src.utils import resolve_path, HITRAN_DICT
import os

# Load paths from config
config = ConfigParser()
config.read(resolve_path("config.ini"))

TIPS2021_DIR = resolve_path(config["paths"]["TIPS2021_dir"])
HITRAN_DIR = resolve_path(config["paths"]["HITRAN_dir"])

# Get HITRAN Molecule index list
idx_mol_dict = HITRAN_DICT()


def get_hitran_molecule_id(chemical_formula):
    """
    Get the HITRAN molecule ID for a given chemical formula.

    Args:
        chemical_formula (str): The simplified chemical formula (e.g., "H2O", "CO2").

    Returns:
        int: The HITRAN molecule ID corresponding to the formula.

    Raises:
        ValueError: If the molecule is not found in the HITRAN database.
    """
    mol_idx_dict = {formula: mol_id for mol_id, formula in idx_mol_dict.items()}
    #print(mol_idx_dict)
    mol_id = mol_idx_dict[chemical_formula]
    return int(mol_id)

def fetch_and_parse_hitran(molecule, isotopologue, numin, numax, wavenumber_step, hitran_dir):
    """
    Fetch and parse HITRAN data for a given molecule and wavenumber range.
    """
    db_begin('HITRAN_Data')
    #print(molecule)
    mol_id = get_hitran_molecule_id(molecule)
    fetch(molecule, mol_id, isotopologue, numin=numin, numax=numax)

    hitran_data = parse_hitran(
        file_base=os.path.join(hitran_dir, molecule),
        wavenumber_ranges=[(numin, numax)]
    )
    return hitran_data


def calculate_hapi_absorption(molecule, numin, numax, wavenumber_step, temperature, pressure, diluent):
    """
    Calculate absorption coefficients using HAPI.
    """
    nu, coef_hapi = absorptionCoefficient_Voigt(
        SourceTables=molecule,
        Environment={'T': temperature, 'p': pressure},
        WavenumberRange=[numin, numax],
        WavenumberStep=wavenumber_step,
        Diluent=diluent
    )
    return nu, coef_hapi


def calculate_custom_absorptivity(hitran_data, concentration, light_source_wavelengths, pressure, temperature):
    """
    Calculate absorptivity using the custom function.
    """
    light_source = {'wavelengths': light_source_wavelengths}
    absorptivity = compute_absorptivity(
        hitran_data=hitran_data,
        concentrations=concentration,
        light_source=light_source,
        pressure=pressure,
        temperature=temperature
    )
    return absorptivity


def test_absorptivity(molecule, isotopologue, numin, numax, wavenumber_step, temperature, pressure, path_length,
                      concentration, diluent):
    """
    Validate the custom compute_absorptivity function against HAPI's absorptionCoefficient_Voigt.
    """
    # Fetch and parse HITRAN data
    hitran_data = fetch_and_parse_hitran(molecule, isotopologue, numin, numax, wavenumber_step, HITRAN_DIR)

    # Calculate HAPI absorption coefficients
    nu, coef_hapi = calculate_hapi_absorption(molecule, numin, numax, wavenumber_step, temperature, pressure, diluent)

    # Convert wavenumber grid to wavelengths (in microns)
    light_source_wavelengths = 1e4 / torch.tensor(nu)  # Convert cm⁻¹ to microns

    # Calculate custom absorptivity
    absorptivity_custom = calculate_custom_absorptivity(
        hitran_data={molecule: hitran_data},
        concentration={molecule: concentration},
        light_source_wavelengths=light_source_wavelengths,
        pressure=pressure,
        temperature=temperature
    )

    # Validate Results
    assert absorptivity_custom.shape == torch.tensor(coef_hapi).shape, \
        "Shape mismatch between custom and HAPI results."

    max_diff = torch.max(torch.abs(absorptivity_custom - torch.tensor(coef_hapi)))
    assert max_diff < 1e-3, f"Absorptivity mismatch too large: max_diff={max_diff.item()}"

    # Visualize Results
    plt.figure(figsize=(10, 6))
    plt.plot(nu, coef_hapi, label='HAPI Absorption Coefficient', linestyle='--')
    plt.plot(nu, absorptivity_custom.detach().numpy() * path_length, label='Custom Absorptivity')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Absorptivity')
    plt.legend()
    plt.title(f'Absorptivity Comparison: HAPI vs Custom ({molecule})')
    plt.grid()
    plt.show()

    print("Validation passed: Custom absorptivity matches HAPI within tolerance.")


if __name__ == "__main__":
    # Example test case
    test_absorptivity(
        molecule='H2O',
        isotopologue=1,
        numin=2260,
        numax=2400,
        wavenumber_step=0.01,
        temperature=296,
        pressure=1.0,
        path_length=93,
        concentration=0.01,
        diluent={'air': 0.8, 'self': 0.2}
    )

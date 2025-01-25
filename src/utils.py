## utils.py

import os


def get_project_root():
    """
    Returns the absolute path to the project root.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))


def resolve_path(relative_path):
    """
    Resolves a relative path to an absolute path, based on the project root.

    :param relative_path: (str) The relative path to resolve.
    :return: (str) The absolute path.
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)

def HITRAN_DICT():
    HITRAN_MOLECULE_LIST = {
        1: "H2O",
        2: "CO2",
        3: "O3",
        4: "N2O",
        5: "CO",
        6: "CH4",
        7: "O2",
        8: "NO",
        9: "SO2",
        10: "NO2",
        11: "NH3",
        12: "HNO3",
        13: "OH",
        14: "HF",
        15: "HCl",
        16: "HBr",
        17: "HI",
        18: "ClO",
        19: "OCS",
        20: "H2CO",
        21: "HOCl",
        22: "N2",
        23: "HCN",
        24: "CH3Cl",
        25: "H2O2",
        26: "C2H2",
        27: "C2H6",
        28: "PH3",
        29: "COF2",
        30: "SF6",
        31: "H2S",
        32: "HCOOH",
        33: "HO2",
        34: "O",
        35: "ClONO2",
        36: "NO+",
        37: "HOBr",
        38: "C2H4",
        39: "CH3OH",
        40: "CH3Br",
        41: "CH3CN",
        42: "CF4",
        43: "C4H2",
        44: "HC3N",
        45: "H2",
        46: "CS",
        47: "SO3",
        48: "C2N2",
        49: "COCl2",
        50: "SO",
        51: "CH3F",
        52: "GeH4",
        53: "CS2",
        54: "CH3I",
        55: "NF3"
    }

    return HITRAN_MOLECULE_LIST
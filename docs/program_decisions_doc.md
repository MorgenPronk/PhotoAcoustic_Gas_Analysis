# Project Decisions
So far what we are trying to do is that the program is given a 


## Data Processing
### parse_hitran.py

This module has to do with getting the right information from a data source. Currently that is HITRAN data and header files, but in the future that will be likely a bigger database.

Getting the right data includes taking the only the part of the data that we want to do calculations with. This means only taking some data near the light source addmission areas. Since there will be multiple data sources, we will have to take that into account.

For multiple light sources and parsing the data from HITRAN, I decided to union the ranges before filtering the data. This means at this step we lose disctinction that there is a overlap and what range is what lightsource, but I don't think this is a problem at this stage because we are merely trying to extract the correct subset of information for the calculations. The integrations will happen individually for each light source and effects from overlapping light sources will need to summed together later.

The focus currently is prototyping and speed of calculations since this is theoretically going to be a program potentially working on an edge device. Because of this I choose to take all of the data from each molecules .data file (the file that comes directly from HITRAN) and convert the relevant columns () into dataframes even though we really only need to look at areas near the wavelengths emitted by the light source. This makes things easier and simpiler to parse and filter instead of working with a text-based .data file directly.

## Calculations

### forwardpass.py

Need to make it usable in pytorch, instead of numpy or anything else

This is the high level structure for the forward pass of the calculations. What we have to do for this is the following:

- Take in the inputs:
  - Light source data - discrete measurements or a function of the distribution (intensity as a function of wavelength)
  - Gas mixture constituents and initial concentration guess
    - It needs a way to also get information about the gases like density given temperature and pressure
- Load the HITRAN data required to calculate the absorptivity spectrums
  - This requires the light source, because we only want to take HITRAN data relevant to the light source
- Calculate the absorptivity

# Theory and Assumptions for `compute_absorptivity` in `forwardpass.py`

## Overview
The `compute_absorptivity` function calculates the total absorptivity of a gas mixture across a range of wavenumbers. This involves combining the contributions of individual gas species and correcting for temperature, pressure, and line broadening effects. The function is designed for use with HITRAN spectroscopic data.

---

## Theory and Mathematical Framework

### Line Intensity Correction
The intensity of a spectral line at temperature \( T \) is calculated using a reference intensity \( S_{\text{ref}} \) provided by HITRAN at a reference temperature \( T_{\text{ref}} \) (typically 296 K). The relationship is given by:

\[
S(T) = S_{\text{ref}} \cdot \frac{Q(T_{\text{ref}})}{Q(T)} \cdot \left( \frac{\exp\left(-\frac{c_2 E_l}{T}\right)}{\exp\left(-\frac{c_2 E_l}{T_{\text{ref}}}\right)} \right) \cdot \frac{1 - \exp\left(-\frac{c_2 \nu}{T}\right)}{1 - \exp\left(-\frac{c_2 \nu}{T_{\text{ref}}}\right)}
\]

#### Terms:
- \( S(T) \): Line intensity at temperature \( T \) (cm\(^{-1}\)).
- \( S_{\text{ref}} \): Reference line intensity at \( T_{\text{ref}} \).
- \( Q(T) \): Partition sum at temperature \( T \).
- \( Q(T_{\text{ref}}) \): Partition sum at reference temperature \( T_{\text{ref}} \).
- \( E_l \): Lower state energy of the transition (cm\(^{-1}\)).
- \( \nu \): Wavenumber of the spectral line (cm\(^{-1}\)).
- \( c_2 \): Second radiation constant (\( c_2 = 1.4388 \) cm K).

#### Temperature Corrections:
1. **Partition Function Ratio \( Q(T_{\text{ref}})/Q(T) \):**
   - Corrects for differences in the population of molecular energy levels between \( T \) and \( T_{\text{ref}} \).
2. **Exponential Energy Term:**
   - Accounts for the Boltzmann distribution of molecular states.
3. **Stimulated Emission Term:**
   - Accounts for differences in stimulated emission probabilities at \( T \) and \( T_{\text{ref}} \).

---

### Line Shape and Broadening
The shape of an absorption line is modeled using a Lorentzian function, which accounts for pressure and self-broadening effects:

\[
f(\nu) = \frac{\gamma}{\pi \left[ (\nu - \nu_0)^2 + \gamma^2 \right]}
\]

#### Terms:
- \( f(\nu) \): Line shape function centered at \( \nu_0 \) (cm\(^{-1}\)).
- \( \nu_0 \): Central wavenumber of the transition (cm\(^{-1}\)).
- \( \gamma \): Line width (half-width at half-maximum, HWHM), determined by:
  \[
  \gamma = \gamma_{\text{air}} \cdot (1 - x) + \gamma_{\text{self}} \cdot x
  \]
  where \( x \) is the mole fraction of the gas.

#### Assumptions:
- Line shape is purely Lorentzian (does not include Doppler or Voigt broadening). THIS CAN BE INCLUDED IN THE FUTURE POTENTIALLY TO IMPROVE ACCURACY
- Broadening coefficients \( \gamma_{\text{air}} \) and \( \gamma_{\text{self}} \) are assumed constant for a given pressure and temperature.

---

### Total Absorptivity
The total absorptivity for a given gas across all transitions is calculated as:

\[
\alpha(\nu) = \sum_i S_i \cdot f_i(\nu) \cdot C
\]

#### Terms:
- \( \alpha(\nu) \): Total absorptivity at wavenumber \( \nu \) (unitless).
- \( S_i \): Line intensity for the \( i \)-th transition (cm\(^{-1}\)).
- \( f_i(\nu) \): Line shape function for the \( i \)-th transition (cm).
- \( C \): Concentration of the gas (mol/L).

For a mixture of gases, the total absorptivity is the sum of contributions from each gas.

---

## Simplifications and Assumptions

1. **Lorentzian Line Shape:**
   - The function assumes a Lorentzian line shape for simplicity, neglecting Doppler broadening and Voigt profiles. This is valid for pressure-dominated broadening.

2. **Single Temperature for All Species:**
   - The temperature \( T \) is assumed uniform across all gas species in the mixture.

3. **Neglect of Line Mixing:**
   - Line mixing effects, which alter line shapes in dense gas mixtures, are not included.

4. **Constant Broadening Coefficients:**
   - Broadening coefficients (\( \gamma_{\text{air}} \) and \( \gamma_{\text{self}} \)) are assumed constant for a given pressure and temperature.

5. **No Isotopic Abundance Corrections:**
   - The function assumes that line intensities provided by HITRAN are already scaled by isotopic abundances.

6. **Finite Partition Sum Data:**
   - Partition sums are interpolated from discrete temperature values provided in HITRAN’s TIPS data.

---

## Potential Improvements
1. **Voigt Profile:**
   - Implement Voigt profiles to account for both Doppler and pressure broadening.
2. **Line Mixing:**
   - Include effects of line mixing for dense gas mixtures.
3. **Dynamic Partition Sums:**
   - Use dynamic calculation of partition sums for temperatures beyond the HITRAN-provided range.
4. **Pressure and Temperature Dependence of Broadening:**
   - Incorporate temperature and pressure dependence of \( \gamma_{\text{air}} \) and \( \gamma_{\text{self}} \).

# Notes on laser
potentially 30 nanometer difference with 40*C shift in temp

---

## References
1. HITRAN Database Documentation: [HITRANonline](https://hitran.org).
2. Gordon, I.E., et al. "The HITRAN2020 molecular spectroscopic database." Journal of Quantitative Spectroscopy and Radiative Transfer (2021).
3. Gamache, R.R., et al. "Total internal partition sums for the HITRAN2020 database." JQSRT (2021).


### Project Structure:

Project Structure so far:

project/
│
├── .venv/                      # Virtual environment (not usually checked into Git)
│
├── data/                       # Project data
│   ├── experimental/           # Experimental data (inputs)
│   ├── HITRAN/                 # HITRAN database files
│   ├── outputs/                # Generated outputs (e.g., results, logs)
│   └── processed/              # Intermediate processed data (optional)
│
├── docs/                       # Documentation
│   ├── API.md                  # API documentation
│   ├── user_guide.md           # User guide
│   └── design_notes.md         # Optional: Design considerations
│
├── src/                        # Source code
│   ├── calculations/           # Core computational logic (e.g., forward pass)
│   │   ├── forwardpass.py      # Forward pass and related calculations
│   │   └── objective.py        # Error/objective function for optimization
│   │
│   ├── data_processing/        # Data processing logic
│   │   ├── hitran_loader.py    # HITRAN-specific data parsing
│   │   └── preprocess.py       # General preprocessing logic
│   │
│   ├── solvers/                # Numerical solvers and optimization logic
│   │   ├── optimization.py     # Optimization routines (e.g., gradient descent)
│   │   └── nonlinear_solver.py # Optional: Nonlinear system solvers
│   │
│   ├── visualization/          # Plotting and visualization scripts
│   │   ├── plot_results.py     # General-purpose plotting
│   │   └── plot_comparisons.py # Visualization of predictions vs measurements
│   │
│   └── utils/                  # Utility functions (optional)
│       ├── config.py           # Shared configurations/constants
│       └── logging.py          # Logging utilities
│
├── tests/                      # Unit and integration tests
│   ├── test_forwardpass.py     # Tests for forwardpass.py
│   ├── test_optimization.py    # Tests for optimization routines
│   ├── test_data_processing.py # Tests for data processing
│   └── ...                     # Additional test files
│
├── main.py                     # Entry point for running the program
│
├── .gitignore                  # Git ignore file
├── README.md                   # Project overview and instructions
└── requirements.txt            # Python dependencies

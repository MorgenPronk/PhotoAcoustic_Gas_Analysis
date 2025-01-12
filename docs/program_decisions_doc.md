# Project Decisions
So far what we are trying to do is that the program is given a 


## Data Processing
### parse_hitran.py

This module has to do with getting the right information from a data source. Currently that is HITRAN data and header files, but in the future that will be likely a bigger database.

Getting the right data includes taking the only the part of the data that we want to do calculations with. This means only taking some data near the light source addmission areas. Since there will be multiple data sources, we will have to take that into account.

For multiple light sources and parsing the data from HITRAN, I decided to union the ranges before filtering the data. This means at this step we lose disctinction that there is a overlap and what range is what lightsource, but I don't think this is a problem at this stage because we are merely trying to extract the correct subset of information for the calculations. The integrations will happen individually for each light source and effects from overlapping light sources will need to summed together later.

The focus currently is prototyping and speed of calculations since this is theoretically going to be a program potentially working on an edge device. Because of this I choose to take all of the data from each molecules .data file (the file that comes directly from HITRAN) and convert the relevant columns () into dataframes even though we really only need to look at areas near the wavelengths emitted by the light source. This makes things easier and simpiler to parse and filter instead of working with a text-based .data file directly.

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

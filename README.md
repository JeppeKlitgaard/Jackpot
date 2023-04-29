# Computing Project – Part II Physics

Overview:
- `report.ipynb` holds the report and should be the first point of entry
- `workbench.ipynb` holds the application that runs the simulations. It can be run manually or parameterised using `papermill`.
- `jackpot` holds the Python code
- `notebooks` holds a few notebooks used to analyse data
- `parameters` holds parameters used to run the simulations on a server. This is done using `tmux` or `screen` in combination with `papermill`
- `bin` holds a few utility bash scripts to run the simulations
- `experiments` holds the obtained experimental data (except serialised arrays, which are omitted due to GitHub storage constraints)

## Installation

As detailed in the report, if the reader wants to  run _all_ cells in the notebook, they will need to install the project dependencies. These are:
- A CPython 3.11 installation (can be installed easily using `pyenv`, recommended)
- Python dependencies listed in `pyproject.toml` that can be installed using `poetry`
- By default only CPU libraries are installed. Follow JAX install instructions if wanting to execute on TPU's or GPU's

If not wishing to install the dependencies required for the simulation, only parts of the report notebook can be run, assuming up-to-date versions of standard Scientific Python tools are available.

## Notes
- The primary repository for this project is a private repository and holds the true git history. This repository also uses `git lfs`, likely making it inappropriate as a hand-in repository. This is the reason why this repository holds very little git history, though rest assured that commits were made along the way.
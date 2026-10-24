# Assignment 1

This repository contains the first assignment for Dlytica Batch 16. The project is a practical exercise in three areas:

- Git and version control fundamentals
- NumPy array operations and data manipulation
- Pandas-based cleaning and analysis of a messy dataset

The assignment uses a synthetic student performance dataset so you can practice real-world data cleaning without needing a live database.

## Project structure

- `answers/01_git_theory.md` – written answers and explanations for the Git theory section
- `answers/02_numpy.py` – NumPy exercises and example outputs
- `answers/02_pandas.ipynb` – Jupyter notebook for the pandas/data-cleaning part
- `data/make_data.py` – script that generates the raw dataset
- `data/scores_raw.csv` – generated raw data file
- `outputs/` – summary files and outputs produced from analysis
- `requirements.txt` – project dependencies

## Prerequisites

Before you begin, make sure you have:

- Python 3.10 or newer
- Git installed on your machine
- Jupyter Notebook or JupyterLab installed (for the notebook part)

## 1) Create a virtual environment

Open a terminal in the project folder and run one of the following commands.

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

To leave the environment later, run:

```bash
deactivate
```

## 2) Install dependencies

From the project root, install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs NumPy, Pandas, Jupyter, and other utilities needed by the assignment.

## 3) Generate the dataset

To create the raw data file used in the assignment, run:

```bash
python data/make_data.py
```

This script creates `data/scores_raw.csv` with synthetic student records. The generated data includes:

- student information
- city names in inconsistent formats
- missing marks/attendance values
- duplicate rows
- invalid mark values used for practice checks

If you want to regenerate the file, simply run the same command again.

## 4) Run each part of the project

### Part 1: Git theory

This part is documentation-based. Open the file:

```bash
answers/01_git_theory.md
```

Read it in VS Code, GitHub, or another Markdown viewer. No Python execution is required for this section.

### Part 2: NumPy exercises

Run the NumPy script from the repository root:

```bash
python answers/02_numpy.py
```

This script demonstrates:

- array shapes and dimensions
- indexing and slicing
- broadcasting
- NaN handling
- vectorized vs loop-based computations

### Part 3: Pandas notebook

Start Jupyter and open the notebook:

```bash
jupyter notebook answers/02_pandas.ipynb
```

Or, if you prefer JupyterLab:

```bash
jupyter lab answers/02_pandas.ipynb
```

Then run all notebook cells in order. The notebook cleans and standardizes the generated dataset, handles missing values, removes duplicates, and produces output summaries.

## Expected outputs

After running the analysis pipeline, you should see files under `outputs/` such as summary outputs generated from the cleaned dataset.

## Quick start summary

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python data/make_data.py
python answers/02_numpy.py
jupyter notebook answers/02_pandas.ipynb
```

This project is designed to help you practice data handling, analysis, and Git workflow concepts in a single assignment.

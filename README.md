# Microfluidic Nanoparticle Particle-Size Prediction

This project analyzes a microfluidic nanoparticle dataset and trains regression models to predict hydrodynamic particle size from four experimental inputs:

- `AIE_conc` — AIE concentration
- `F127_conc` — F127 concentration
- `TFR` — Total flow rate
- `FRR` — Flow rate ratio

The main workflow is implemented in the Jupyter notebook:

`microfluidic_particle_size_ml_project.ipynb`

The notebook performs data preprocessing, exploratory data analysis, model comparison, cross-validation, feature-importance analysis, and result visualization.

## Project files

Keep these files in the same project folder:

```text
microfluidic_particle_size_project/
│
├── .venv/                                  # Local Python virtual environment
├── aie_python_ready_template.xlsx          # Input dataset
├── microfluidic_particle_size_ml_project.ipynb
├── microfluidic_particle_size_ml_project.py
├── microfluidic_particle_size_ml_project.html
├── particle_size_results_summary.md
├── particle_size_gradient_boosting_model.joblib
└── particle_size_figures/
```

The notebook expects the Excel dataset to be in the same folder as the notebook.

## Python version

This project has been tested with:

```text
Python 3.13.5
```

Other recent Python 3 versions may also work, but using the same version is the easiest way to reproduce the setup.

## 1. Open the project in VS Code

Install the following VS Code extensions:

- Python
- Jupyter

Then open VS Code and select:

```text
File -> Open Folder
```

Choose the `microfluidic_particle_size_project` folder.

Do not open only the notebook file by itself. Opening the whole folder makes relative file paths work correctly.

## 2. Create a virtual environment

Open a VS Code terminal:

```text
Terminal -> New Terminal
```

Make sure the terminal is inside the project directory.

### Windows

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should begin with something similar to:

```text
(.venv)
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install the required packages

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
python -m pip install numpy pandas matplotlib scikit-learn openpyxl joblib jupyter ipykernel
```

If your computer is configured to use a broken or unavailable package mirror, force the official PyPI index:

```bash
python -m pip install --index-url https://pypi.org/simple numpy pandas matplotlib scikit-learn openpyxl joblib jupyter ipykernel
```

## 4. Register the Jupyter kernel

Run:

```bash
python -m ipykernel install --user --name particle-size-ml --display-name "Python 3.13 - Particle Size ML"
```

This makes the virtual environment available as a Jupyter kernel in VS Code.

## 5. Select the correct Python interpreter

In VS Code press:

```text
Ctrl + Shift + P
```

Search for:

```text
Python: Select Interpreter
```

Choose the interpreter inside:

```text
.venv\Scripts\python.exe
```

On macOS/Linux, choose:

```text
.venv/bin/python
```

## 6. Open the Jupyter notebook

Open:

```text
microfluidic_particle_size_ml_project.ipynb
```

At the top-right of the notebook, click:

```text
Select Kernel
```

Choose either:

```text
Python 3.13 - Particle Size ML
```

or the `.venv` Python 3.13 environment.

To confirm that the correct environment is being used, run:

```python
import sys
print(sys.executable)
```

The path should point to the `.venv` directory inside this project.

## 7. Test the environment

Run the following cell:

```python
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import openpyxl
import joblib

print("Environment is working.")
print("Python:", sys.version)
print("scikit-learn:", sklearn.__version__)
```

If this runs without errors, the environment is ready.

## 8. Run the complete analysis

The recommended approach for the first run is to execute the notebook cell-by-cell using:

```text
Shift + Enter
```

Once everything works, you can use:

```text
Run All
```

The notebook will perform the following steps:

```text
Load Excel dataset
        ↓
Clean and preprocess the data
        ↓
Generate descriptive statistics
        ↓
Perform correlation analysis
        ↓
Create exploratory plots
        ↓
Compare regression models
        ↓
Evaluate models with cross-validation
        ↓
Select the best tested regression model
        ↓
Generate out-of-fold predictions
        ↓
Calculate R² and MAE
        ↓
Analyze permutation feature importance
        ↓
Save figures and model outputs
```

## 9. Target variable used in this project

The workbook contains multiple particle-size measurements:

- `initial_size_nm`
- `mean_size_nm`
- `final_size_nm`

This project currently uses:

```python
TARGET_SOURCE = "initial_size_nm"
```

and internally renames it to:

```text
Particle_Size
```

The initial DLS measurement was selected because it is the particle-size measurement most directly associated with the formulation and microfluidic conditions.

If the assignment specifically requires the final DLS size instead, change:

```python
TARGET_SOURCE = "initial_size_nm"
```

to:

```python
TARGET_SOURCE = "final_size_nm"
```

and rerun the notebook from the beginning.

## 10. Run the standalone Python script

The same analysis can also be executed without Jupyter.

With the virtual environment activated, run:

```bash
python microfluidic_particle_size_ml_project.py
```

For coursework or presentation purposes, the Jupyter notebook is recommended because it keeps the code, outputs, figures, and explanation together.

## 11. Main generated outputs

The notebook writes figures to:

```text
particle_size_figures/
```

Expected figures include:

```text
01_particle_size_distribution.png
02_correlation_heatmap.png
03_1_AIE_conc_vs_particle_size.png
03_2_F127_conc_vs_particle_size.png
03_3_TFR_vs_particle_size.png
03_4_FRR_vs_particle_size.png
04_model_comparison_mae.png
05_observed_vs_predicted.png
06_feature_importance.png
```

The trained model is saved as:

```text
particle_size_gradient_boosting_model.joblib
```

## 12. Current model results

Using the current target definition and cleaned dataset:

- Valid experiments used: `135`
- Best tested model: Gradient Boosting Regressor with Huber loss
- 5-fold cross-validated R²: approximately `0.440 ± 0.328`
- 5-fold cross-validated MAE: approximately `22.37 ± 8.05 nm`
- Strongest predictive parameter: `FRR`

The model identifies FRR as the most important predictor according to permutation importance.

Feature importance should be interpreted as predictive importance, not proof of causation.

## Reproducibility

The notebook uses a fixed random seed:

```python
RANDOM_STATE = 42
```

This helps make model comparison and cross-validation results reproducible.

Because the dataset is relatively small, performance can still vary depending on the validation samples. For this reason, model performance is evaluated with cross-validation rather than relying only on one train/test split.


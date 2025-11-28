# Optimal Fertilizer Prediction

Machine learning pipeline for predicting **optimal fertilizer recommendations** based on soil properties, environmental conditions, and crop type using multiple approaches including model training and an interactive GUI application.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Data Flow & Architecture](#data-flow--architecture)
- [Local Setup (Linux / macOS)](#local-setup-linux--macos)
- [Local Setup (Windows)](#local-setup-Windows)
- [Running the Notebooks](#running-the-notebooks)
- [Model Approaches](#model-approaches)
- [GUI Application](#gui-application)
- [Running the Notebook in Google Colab](#running-the-notebook-in-google-colab)
- [Results](#results)

---

## Project Overview

This project builds an **Machine learning pipeline** for predicting optimal fertilizer recommendations based on:
- Soil properties (Nitrogen, Phosphorous, Potassium levels)
- Environmental conditions (Temperature, Humidity, Moisture)
- Crop and soil types

The project implements **two distinct approaches**:

1. **Approach 1: Ensemble with Hyperparameter Tuning**
   - Uses XGBoost, LightGBM, and CatBoost
   - Comprehensive EDA with green-themed visualizations
   - RandomizedSearchCV for hyperparameter optimization
   - StandardScaler preprocessing with pipelines

2. **Approach 2: XGBoost with Feature Engineering**
   - Focused XGBoost implementation
   - Advanced feature engineering with derived parameters
   - Stratified K-Fold cross-validation with bagging
   - GPU acceleration support
   - Custom MAP@3 (Mean Average Precision at 3) evaluation

3. **GUI Application: Interactive Fertilizer Recommender**
   - User-friendly Tkinter-based desktop application
   - Real-time predictions using trained ensemble models
   - Input validation for environmental and soil parameters
   - Weighted ensemble of XGBoost, LightGBM, and CatBoost
   - Instant top-3 fertilizer recommendations

> ℹ️ **Note:** The main dataset is stored using **Git LFS**.  
> You **must** have Git LFS installed and configured, a plain `git clone` will not download the dataset.
---

## Features

- End-to-end pipeline from raw CSV → trained models → submission files
- Three complementary approaches: two for model training and one interactive GUI
- Comprehensive exploratory data analysis with insights cards
- Multiple ML algorithms (XGBoost, LightGBM, CatBoost)
- Advanced feature engineering and categorical encoding
- Cross-validation with bagging for robust predictions
- GPU acceleration support (Approach 2)
- MAP@3 metric for multi-class recommendation evaluation
- Interactive desktop GUI application for real-time predictions

---

## Technology Stack

- Python **3.11 and above**
- **Machine Learning Libraries**:
  - XGBoost – gradient boosting with GPU support
  - LightGBM – light gradient boosting
  - CatBoost – categorical boosting
- **Data Science Stack**:
  - Pandas – data manipulation
  - NumPy – numerical operations
  - scikit-learn – preprocessing & model selection
- **Visualization**:
  - Matplotlib – static plots
  - Seaborn – statistical visualizations
  - Plotly – interactive charts
- **GUI**:
  - Tkinter – desktop application interface
  - Joblib – model serialization and loading
- **Environment**:
  - Jupyter Lab – notebook workflow

---

## Repository Structure

```text
OptimalFertilizer/
├─ README.md                              # Project documentation (this file)
├─ requirements.txt                       # Python dependencies
├─ setup.sh                               # Automated setup script for Mac/Linux
├─ setup_windows.ps1                      # Automated setup script for Windows
├─ OptimalFertilizer_Approach1.ipynb     # Ensemble approach with EDA
├─ OptimalFertilizer_Approach2.ipynb     # XGBoost with feature engineering
├─ OptimalFertilizer_GUI.ipynb           # Interactive GUI application
├─ data/
│   ├─ test_ieee.csv                       # Original training data from ieee
│   ├─ train_kaggle.csv                    # Training dataset from kaggle
│   └─ test_kaggle.csv                     # Test dataset from kaggle
├─ models/
    ├─ best_xgb.pkl                       # Trained XGBoost model
    ├─ best_lgb.pkl                       # Trained LightGBM model
    ├─ best_cat.pkl                       # Trained CatBoost model
    └─ label_encoder.pkl                  # Label encoder for fertilizer names
```

---

## Data Flow & Architecture

### Input Data
- `data/train_kaggle.csv` – Competition training dataset
- `data/test_kaggle.csv` – Competition test dataset
- `data/train_ieee.csv` – Additional training data for augmentation

### Features
**Numerical Features:**
- Temperature (°C)
- Humidity (%)
- Moisture (%)
- Nitrogen content (ppm)
- Phosphorous content (ppm)
- Potassium content (ppm)

**Categorical Features:**
- Soil Type
- Crop Type

**Target:**
- Fertilizer Name (7 classes for multi-class classification)

### Output
> Note: The Jupyter notebook writes outputs to PROJECT_HOME.  
- Submission CSV files with top-3 fertilizer recommendations per sample
- Model evaluation metrics (MAP@3)

---

## Local Setup (Linux / macOS)

### Prerequisites

- Git
- Python **3.11.14** available as `python3.11`  
  (via pyenv, Homebrew, or system package manager)

### 1. Clone the repository

```bash
git clone <your_repo_url>.git
cd OptimalFertilizer
```

### 2. Make the setup script executable

```bash
chmod +x setup.sh
```

### 3. Run the setup script

```bash
./setup.sh
```

The script will:
- Create a `.venv` virtual environment using Python 3.11
- Install all dependencies from `requirements.txt`
- Install an IPython kernel named `fertilizer_py311`
- Launch Jupyter Lab in the project directory

---

### Manual Setup (<span style="color: red;">**Alternative to `setup.sh`**</span>)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python -m ipykernel install --user \
  --name "fertilizer_py311" \
  --display-name "Python 3.11 (Fertilizer)"

jupyter lab
```

---

## Local Setup (Windows)

### Prerequisites

- Git
- Python **3.11.14** available as `python` or `py -3.11`  
   (from python.org installer, Microsoft Store, or Chocolatey)

### 1. Clone the repository

```
powershell
git lfs clone <your_repo_url>.git
cd OptimalFertilizer
```

### 2. Run the setup script

```
powershell
.\setup_windows.ps1
 ```
Alternate command
```
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

The script will:
- Create a `.venv` virtual environment using Python 3.11
- Install all dependencies from `requirements.txt`
- Install an IPython kernel named `fertilizer_py311`
- Launch Jupyter Lab in the project directory

---

### Manual Setup (<span style="color: red;">**Alternative to `setup_windows.ps1`**</span>)

```powershell
# Create virtual environment
python -m venv .venv
# or if you have multiple Python versions:
# py -3.11 -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Install Jupyter kernel
python -m ipykernel install --user --name "fertilizer_py311" --display-name "Python 3.11 (Fertilizer)"

# Launch Jupyter Lab
jupyter lab
```

**Note:** If you encounter execution policy issues with PowerShell scripts, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---


## Running the Notebooks

### 1. Activate the virtual environment

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Start Jupyter Lab
**Linux/macOS/Windows:**
```bash
jupyter lab
```

**Note:** The command is the same across all platforms once the virtual environment is activated.


### 3. Select your approach

**For Approach 1 (Ensemble with EDA):**
- Open `OptimalFertilizer_Approach1.ipynb`
- Features comprehensive EDA with green-themed visualizations
- Uses ensemble of XGBoost, LightGBM, and CatBoost
- Includes hyperparameter tuning with RandomizedSearchCV

**For Approach 2 (XGBoost with Feature Engineering):**
- Open `OptimalFertilizer_Approach2.ipynb`
- Features advanced feature engineering
- Uses XGBoost with stratified K-fold + bagging
- Supports GPU acceleration

**For GUI Application:**
- Open `OptimalFertilizer_GUI.ipynb`
- Requires trained models in `models/` directory
- Run Approach 1 first to generate model files
- Creates interactive desktop application with Tkinter
- Provides real-time fertilizer recommendations

### 4. Configure paths (if needed)

Both notebooks use:
```python
PROJECT_HOME = os.getcwd()
TRAIN_PATH = PROJECT_HOME + "/data/train_kaggle.csv"
TEST_PATH = PROJECT_HOME + "/data/test_kaggle.csv"
ORIGINAL_PATH = PROJECT_HOME + "/data/train_ieee.csv"
```

### 5. Run all cells

- Execute cells sequentially
- Monitor training progress and validation scores
- Submission files will be generated in `output_dir/`

---

## Model Approaches

### Approach 1: Ensemble with Comprehensive EDA

**Highlights:**
- **Exploratory Data Analysis**: Green-themed visualizations with insight cards
- **Models**: XGBoost, LightGBM, CatBoost
- **Preprocessing**: StandardScaler, Label Encoding, One-Hot Encoding
- **Validation**: RepeatedStratifiedKFold cross-validation
- **Hyperparameter Tuning**: RandomizedSearchCV
- **Feature Engineering**: NPK ratios, moisture bands

**Best suited for:**
- Understanding data patterns and distributions
- Comparing multiple algorithms
- Systematic hyperparameter optimization

### Approach 2: XGBoost with Advanced Feature Engineering

**Highlights:**
- **Feature Engineering**: Derived parameters using weighted combinations
- **Model**: XGBoost with GPU support
- **Validation**: Stratified K-Fold (5 folds) with bagging (2 bags per fold)
- **Data Augmentation**: Random oversampling of original dataset
- **Categorical Handling**: Native XGBoost categorical support
- **Evaluation**: Custom MAP@3 (Mean Average Precision at 3)
- **Optimization**: 
  - Early stopping with 100-round patience
  - Random parameter perturbation per bag
  - Histogram-based tree method for speed

**Best suited for:**
- Production-ready pipelines
- GPU-accelerated training
- Maximum prediction accuracy

**MAP@3 Metric:**
```python
MAP@3 = (1/N) × Σ(hit@1/1 + hit@2/2 + hit@3/3)
```
Where hit@k = 1 if true label is in top-k predictions, else 0

---

## GUI Application

### Interactive Fertilizer Recommendation System

The GUI notebook provides a user-friendly desktop application for real-time fertilizer recommendations.

**Features:**
- **Input Fields**:
  - Temperature (0-60°C)
  - Humidity (0-100%)
  - Moisture (0-100%)
  - Nitrogen content (0-150 ppm)
  - Potassium content (0-150 ppm)
  - Phosphorous content (0-150 ppm)
  - Soil Type (Sandy, Clay, Loamy, Black, Red)
  - Crop Type (Barley, Wheat, Maize, Cotton, Sugarcane, Millets, Ground Nut, Tobacco)

- **Input Validation**: Real-time validation of numeric ranges
- **Ensemble Prediction**: Weighted combination of three models:
  - CatBoost (40%)
  - XGBoost (35%)
  - LightGBM (25%)
- **Output**: Top fertilizer recommendation

### Prerequisites for GUI

1. **Train models first**:
   ```bash
   # Run Approach 1 notebook to generate models
   jupyter lab OptimalFertilizer_Approach1.ipynb
   ```

2. **Ensure models directory exists**:
   ```bash
   mkdir -p models/
   ```

3. **Required model files** (generated by Approach 1):
   - `models/best_xgb.pkl`
   - `models/best_lgb.pkl`
   - `models/best_cat.pkl`
   - `models/label_encoder.pkl`

### Running the GUI

1. **Activate virtual environment**:
   
   **Linux/macOS:**
   ```bash
   source .venv/bin/activate
   ```
   
   **Windows (Command Prompt):**
   ```cmd
   .venv\Scripts\activate.bat
   ```
   
   **Windows (PowerShell):**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Launch Jupyter and open GUI notebook**:
   ```bash
   jupyter lab OptimalFertilizer_GUI.ipynb
   ```
   
   **Note:** The Jupyter Lab command is the same across all platforms once the virtual environment is activated.

3. **Run the cell** to launch the GUI application

4. **Use the application**:
   - Fill in all environmental and soil parameters
   - Click "Predict Fertilizer"
   - View the recommended fertilizer

### GUI Screenshot Flow

1. Input environmental parameters (temperature, humidity, moisture)
2. Input soil nutrients (N, P, K levels)
3. Select soil and crop types from dropdowns
4. Click "Predict Fertilizer" button
5. View instant recommendation

---

## Running the Notebook in Google Colab

### For Approach 1:

1. **Upload to Colab**
   - Go to [Google Colab](https://colab.research.google.com)
   - Upload `OptimalFertilizer_Approach1.ipynb`

2. **Install dependencies**
   ```python
   !pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm catboost
   ```

3. **Upload data files**
   - Upload `train_kaggle.csv`, `test_kaggle.csv`, `train_ieee.csv` via file browser

4. **Run all cells**

### For Approach 2:

1. **Upload to Colab**
   - Upload `OptimalFertilizer_Approach2.ipynb`

2. **Install dependencies**
   ```python
   !pip install pandas numpy xgboost scikit-learn
   ```

3. **Enable GPU (for acceleration)**
   - Runtime → Change runtime type → Hardware accelerator → GPU (T4)

4. **Upload data files**
   - Upload CSVs via file browser

5. **Adjust paths in notebook**
   ```python
   PROJECT_HOME = "/content"
   TRAIN_PATH = PROJECT_HOME + "/train_kaggle.csv"
   TEST_PATH = PROJECT_HOME + "/test_kaggle.csv"
   ORIGINAL_PATH = PROJECT_HOME + "/train_ieee.csv"
   ```

6. **Run all cells**

---

## Results

### Approach 1
- **Output**: `Approach1.csv`
- **Score**: MAP@3 = 0.34556
- **Models**: Ensemble of XGBoost, LightGBM, CatBoost
- **Validation**: Multiple algorithms with hyperparameter tuning

### Approach 2
- **Output**: `Approach2.csv`
- **Score**: MAP@3 = 0.3772
- **Model**: XGBoost with 5-fold CV × 2 bags = 10 models
- **Training**: GPU-accelerated with categorical support

---

## Extending the Project

**Potential improvements:**
- Add SHAP or LIME for model interpretability
- Implement stacking with meta-learner
- Add more derived features (climate indices, nutrient ratios)
- Hyperparameter optimization with Optuna
- Create interactive dashboard for predictions
- Add model monitoring and drift detection

---

> 💡 Contributions, issues, and feature requests are welcome.  
> Feel free to open an issue or submit a pull request!

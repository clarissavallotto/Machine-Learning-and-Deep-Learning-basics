# ML-code.py

## Overview

`ML-code.py` is an end-to-end, progressive reference script covering foundational and intermediate **Machine Learning** concepts in Python using `pandas`, `scikit-learn`, and `xgboost`. The script takes you from basic exploratory data analysis and single decision trees through model validation, bias-variance optimization, ensemble methods, automated preprocessing pipelines, $k$-fold cross-validation, gradient boosting, and target leakage diagnosis.

---

## Prerequisites & Dependencies

This project is designed for use in a [**Google Colab**](https://colab.research.google.com/) environment.

To execute `ML-code.py`, you need **Python 3.8+** and the following third-party libraries:

| Library | Purpose in Script |
| :--- | :--- |
| `pandas` | Data ingestion (`read_csv`), DataFrame manipulation, summary statistics, and CSV export. |
| `scikit-learn` | Preprocessing (`SimpleImputer`, `OrdinalEncoder`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`), splitting/validation (`train_test_split`, `cross_val_score`), metrics (`mean_absolute_error`), and models (`DecisionTreeRegressor`, `RandomForestRegressor`, `RandomForestClassifier`). |
| `xgboost` | Gradient boosted decision trees (`XGBRegressor`) with early stopping support. |

### Installation

Install all required packages via `pip`:

```bash
pip install pandas scikit-learn xgboost
```

---

## Environment Setup & Data Extraction

At the very start of `ML-code.py`, the built-in `zipfile` module extracts `data.zip` into a local `data/` directory:

```python
import zipfile

with zipfile.ZipFile("data.zip", "r") as zip_ref:
    zip_ref.extractall("data")
```

Ensure `data.zip` is present in your working directory before running the script.

---

## Datasets Used

The script utilizes **four CSV datasets** extracted from `data.zip`:

| Dataset Filename | Domain / Description | Target Variable | Sections Used |
| :--- | :--- | :--- | :--- |
| `melb_data.csv` | Melbourne Housing Snapshot containing property attributes (`Rooms`, `Bathroom`, `Landsize`, `BuildingArea`, `YearBuilt`, `Distance`, `Lattitude`, `Longtitude`, and categorical features). | `Price` (Regression) | Intro to ML (Modules 1–6), Intermediate ML (Missing Values, Categorical Variables, Pipelines, Cross-Validation, XGBoost) |
| `train.csv` | Ames Housing training dataset containing 79 explanatory variables (numerical and categorical). | `SalePrice` (Regression) | ML Competitions, Intermediate ML Introduction |
| `test.csv` | Ames Housing test dataset (omits `SalePrice`, retains `Id`) used to generate competition predictions. | N/A (Inference) | ML Competitions, Intermediate ML Introduction |
| `AER_credit_card_data.csv` | Credit card application dataset containing applicant demographics, financial history, and post-approval card usage metrics (`yes`/`no` boolean strings). | `card` (Binary Classification) | Intermediate ML (Data Leakage) |

---

## Module-by-Module Breakdown

### Part 1: Intro to Machine Learning

#### 1. How Models Work
- **Core Concept:** Establishes the fundamental supervised learning workflow: **Load Data** $\rightarrow$ **Select Features ($X$) & Target ($y$)** $\rightarrow$ **Define Model** $\rightarrow$ **Fit Model** $\rightarrow$ **Predict**.
- **Implementation:** Initializes `DecisionTreeRegressor(random_state=1)` on `melb_data.csv` using 5 core predictors (`Rooms`, `Bathroom`, `Landsize`, `Lattitude`, `Longtitude`) to predict `Price`, printing the first five predictions.

#### 2. Basic Data Exploration
- **Structural Inspection:** Audits dataset dimensions and schema using `.head()`, `.tail()`, `.head(10)`, `.shape`, `.columns`, and `.info()`.
- **Statistical Profiling:** Computes global `.describe()`, `.mean(numeric_only=True)`, `.median(numeric_only=True)`, `.std(numeric_only=True)`, `.min(numeric_only=True)`, and `.max(numeric_only=True)`.
- **Completeness & Target Analysis:** Checks non-null counts (`.count()`) and missing values (`.isnull().sum()`), followed by targeted summary statistics for `melbourne_data["Price"]`.

#### 3. Your First Machine Learning Model
- **Data Cleaning:** Removes rows containing any missing values via `melbourne_data.dropna(axis=0)`.
- **Reproducibility:** Demonstrates setting `random_state=1` in `DecisionTreeRegressor` to guarantee deterministic splits across runs, verifying predictions against `X.head()`.

#### 4. Model Validation
- **Evaluation Metric:** Introduces **Mean Absolute Error (MAE)** via `sklearn.metrics.mean_absolute_error`.
- **In-Sample vs. Out-of-Sample Comparison:**
  - **Part A (In-Sample):** Trains and evaluates `DecisionTreeRegressor()` on the exact same dataset, demonstrating why in-sample MAE is misleadingly low.
  - **Part B (Out-of-Sample):** Splits the data via `train_test_split(X, y, random_state=0)` and evaluates predictions on unseen validation records (`val_X`, `val_y`), revealing true generalization error.

#### 5. Underfitting and Overfitting
- **Bias-Variance Tradeoff:** Controls tree complexity using `max_leaf_nodes`.
- **Automated Model Selection:** Defines a helper function `get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)` and loops over `candidate_max_leaf_nodes = [5, 50, 500, 5000]`. Automatically tracks `best_tree_size` corresponding to `best_mae`, then trains and validates `final_model` using that optimal leaf count.

#### 6. Random Forests
- **Ensemble Learning:** Replaces a single decision tree with `RandomForestRegressor(random_state=1)` on the same 7-feature Melbourne split (`Rooms`, `Bathroom`, `Landsize`, `BuildingArea`, `YearBuilt`, `Lattitude`, `Longtitude`) to reduce variance without manual depth tuning.
- **Model Introspection:** Prints the model representation and inspects the default ensemble size (`forest_model.n_estimators`).

#### 7. Machine Learning Competitions
- **Competition Pipeline:** Trains `RandomForestRegressor(n_estimators=100, random_state=1)` on `train.csv` using 7 structural predictors (`LotArea`, `YearBuilt`, `1stFlrSF`, `2ndFlrSF`, `FullBath`, `BedroomAbvGr`, `TotRmsAbvGrd`).
- **Submission Export:** Predicts `SalePrice` on `test.csv` and writes `submission.csv` (`Id`, `SalePrice`) without row indices.

---

### Part 2: Intermediate Machine Learning

#### 1. Introduction (Full Pipeline Baseline)
- **All-Feature Modeling:** Expands beyond a small manual subset of features to use **all** predictors in `train.csv` (dropping only `SalePrice` and `Id`).
- **Automated Preprocessing:**
  - `SimpleImputer(strategy="median")` for numerical columns.
  - `Pipeline` of `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder(handle_unknown="ignore")` for object/categorical columns.
  - `ColumnTransformer` + `RandomForestRegressor(n_estimators=100, random_state=0)` bundled into `my_pipeline` to generate an updated `submission.csv`.

#### 2. Missing Values
- **Experimental Setup:** Uses an 80/20 train-validation split (`random_state=0`) on numerical predictors of `melb_data.csv` and evaluates three strategies with a 10-tree `RandomForestRegressor` via `score_dataset()`:
  1. **Approach 1 — Drop Columns:** Drops all columns containing missing values in `X_train` (`cols_with_missing`).
  2. **Approach 2 — Imputation:** Replaces missing values with training column means using `SimpleImputer()`.
  3. **Approach 3 — Imputation + Missing Indicators:** Appends binary boolean indicator columns (`<col>_was_missing`) prior to mean imputation so the model retains information about missingness patterns.
- **Comparison:** Aggregates all three MAE scores into a sorted summary `DataFrame` and prints per-column missing value counts.

#### 3. Categorical Variables
- **Feature Filtering:** Drops columns with missing values, then retains numerical columns (`int64`, `float64`) and low-cardinality categorical columns (`nunique() < 10` and `dtype == 'object'`).
- **Experimental Setup:** Benchmarks three categorical encoding methods using a 100-tree `RandomForestRegressor`:
  1. **Approach 1 — Drop Categorical Variables:** Trains exclusively on numerical features (`select_dtypes(exclude=['object'])`).
  2. **Approach 2 — Ordinal Encoding:** Maps unique categories to integers via `OrdinalEncoder()`.
  3. **Approach 3 — One-Hot Encoding:** Expands categories into binary indicator columns via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` and concatenates them with numerical features.
- **Comparison:** Outputs a sorted `DataFrame` ranking all three encoding strategies by validation MAE.

#### 4. Pipelines
- **Leakage-Free Workflow:** Encapsulates preprocessing and modeling into a single `scikit-learn` `Pipeline` (`my_pipeline`) on the 80/20 Melbourne housing split:
  - **Numerical Transformer:** `SimpleImputer(strategy='constant')` (fills missing numerical entries with `0`).
  - **Categorical Transformer:** Sequential `SimpleImputer(strategy='most_frequent')` and `OneHotEncoder(handle_unknown='ignore')`.
  - **ColumnTransformer:** Applies both transformers in parallel to `numerical_cols` and `categorical_cols`.
  - **Estimator:** `RandomForestRegressor(n_estimators=100, random_state=0)`, evaluated with `mean_absolute_error(y_valid, preds)`.

#### 5. Cross-Validation
- **$k$-Fold Evaluation:** Uses `cross_val_score(..., cv=5, scoring='neg_mean_absolute_error')` across the full `melb_data.csv` dataset (without dropping missing values beforehand) using a `Pipeline` of `SimpleImputer()` and `RandomForestRegressor(n_estimators=50, random_state=0)`.
- **Performance Distribution:** Multiplies negative MAE scores by `-1` and reports all 5 fold scores alongside their mean (`average_mae`), minimum, maximum, and standard deviation.

#### 6. XGBoost
- **Gradient Boosting:** Trains `XGBRegressor` on `melb_data.csv` (`Rooms`, `Distance`, `Landsize`, `BuildingArea`, `YearBuilt`) using `n_estimators=1000`, `learning_rate=0.05`, `early_stopping_rounds=5`, `n_jobs=4`, and `random_state=0`.
- **Early Stopping & Inspection:** Passes `eval_set=[(X_valid, y_valid)]` with `verbose=False` during `.fit()`, evaluates validation MAE, and inspects `my_model.best_iteration` and `my_model.best_iteration + 1` (optimal tree count).

#### 7. Data Leakage
- **Target Leakage Diagnosis:** Loads `AER_credit_card_data.csv` (parsing `'yes'`/`'no'` as booleans) and evaluates a `RandomForestClassifier(n_estimators=100, random_state=0)` pipeline with 5-fold cross-validation (`scoring='accuracy'`).
- **Root-Cause Analysis:** Compares zero-expenditure rates between rejected applicants (`~y`) and accepted cardholders (`y`), demonstrating that `expenditure` (and related post-approval features) leaks future target outcomes.
- **Remediation:** Drops leaky predictors (`['expenditure', 'share', 'active', 'majorcards']`) to produce `X2` and compares initial inflated accuracy against realistic post-cleanup cross-validation accuracy.

---

## Generated Output Artifacts

Running `ML-code.py` creates the following files and directories on disk:

1. `data/` — Extracted directory containing `melb_data.csv`, `train.csv`, `test.csv`, and `AER_credit_card_data.csv`.
2. `submission.csv` — Kaggle competition prediction CSV with columns `Id` and `SalePrice` (generated in *Intro to ML: Machine Learning Competitions* and overwritten in *Intermediate ML: Introduction* by the full preprocessing pipeline).

---

## Execution & Path Configuration Notes

- **Google Colab File Paths:** All `pd.read_csv()` calls reference `/content/data/data/<filename>.csv`. If running locally outside of Google Colab, update these paths to match your local extraction path (for example, `data/<filename>.csv` or `data/data/<filename>.csv` depending on how `data.zip` is structured internally).
- **Independent Sections:** Each major module re-imports its required libraries and reloads its dataset independently, allowing you to run individual sections in isolation (such as Jupyter Notebook cells) once `data.zip` has been extracted.

---

# DL-code.py

## Overview

`DL-code.py` is an end-to-end, progressive reference script covering foundational **Deep Learning** concepts in Python using **TensorFlow** and **Keras**. The script walks step-by-step from the mathematical mechanics of a single linear neuron and multi-layer perceptrons (MLPs) to training with Stochastic Gradient Descent (Adam), preventing overfitting via Early Stopping, stabilizing deep networks with Dropout and Batch Normalization, and solving binary classification problems with sigmoid outputs and binary cross-entropy.

---

## Prerequisites & Dependencies

This project is designed for use in a [**Google Colab**](https://colab.research.google.com/) environment.

To run `DL-code.py`, you need **Python 3.8+** and the following libraries:

| Library | Purpose in Script |
| :--- | :--- |
| `tensorflow` / `keras` | Building `Sequential` neural networks, defining layers (`Input`, `Dense`, `Dropout`, `BatchNormalization`), configuring callbacks (`EarlyStopping`), compiling optimizers/losses, and training/inference. |
| `pandas` | Data loading (`read_csv`), train/validation splitting (`sample`, `drop`), manual Min-Max normalization, missing-column filtering, and training history analysis. |
| `matplotlib` | Configuring plot themes (`seaborn-v0_8-whitegrid`), typography (`plt.rc`), and rendering training/validation loss and accuracy curves via `pandas.DataFrame.plot()`. |

### Installation

Install all required packages via `pip`:

```bash
pip install tensorflow pandas matplotlib
```

---

## Environment Setup & Data Extraction

At the beginning of `DL-code.py`, the standard-library `zipfile` module extracts `data.zip` into a local `data/` directory:

```python
import zipfile

with zipfile.ZipFile("data.zip", "r") as zip_ref:
    zip_ref.extractall("data")
```

Ensure `data.zip` is located in your current working directory prior to executing the script.

---

## Datasets Used

The script utilizes **two tabular datasets** extracted from `data.zip`:

| Dataset Filename | Domain / Description | Target Variable | Task Type | Modules Used |
| :--- | :--- | :--- | :--- | :--- |
| `red-wine.csv` | Physicochemical properties of red wine samples (e.g., acidity, sugar, chlorides, sulphates, alcohol). | `quality` (Continuous/Ordinal score) | Regression (`loss='mae'`) | Modules 3, 4, and 5 |
| `ion.csv` | Johns Hopkins Ionosphere radar dataset containing 34 numerical signal attributes and a categorical radar return label (`'good'` or `'bad'`). | `Class` (`0` = `'good'`, `1` = `'bad'`) | Binary Classification (`loss='binary_crossentropy'`) | Module 6 |

---

## Module-by-Module Breakdown

### 1. A Single Neuron

* **Mathematical Foundations:**
  * **Single-Input Linear Unit:** Computes $y = w \cdot x + b$ manually in Python (`w = 2.5`, `x = 5`, `b = 90`) to illustrate how a single neuron scales an input by a weight and shifts it by a bias.
  * **Multi-Input Linear Unit:** Computes $y = w_0 x_0 + w_1 x_1 + w_2 x_2 + b$ across three nutritional features (`Sugars`, `Fiber`, `Protein`).
* **Keras Implementation:**
  * Constructs a single-neuron `keras.Sequential` model using `layers.Dense(units=1, input_shape=[3])`.
  * Demonstrates the modern, recommended alternative using an explicit `keras.Input(shape=(3,))` layer followed by `layers.Dense(units=1)`.
  * Inspects layer shapes and trainable parameter counts ($3 \text{ weights} + 1 \text{ bias} = 4 \text{ parameters}$) via `model.summary()`.

---

### 2. Deep Neural Networks

* **From Linear Models to Non-Linear Depth:**
  * Contrasts a single-layer linear model (`keras.Input(shape=(2,))` $\rightarrow$ `layers.Dense(units=1)`) with a multi-layer deep neural network capable of learning non-linear relationships.
* **Hidden Layers & Activation Functions:**
  * Stacks two hidden `Dense` layers (`units=4` and `units=3`) equipped with Rectified Linear Unit (`activation='relu'`) activations, followed by a linear output layer (`layers.Dense(units=1)`) without an activation function for regression.
  * Shows both inline `input_shape=[2]` syntax and explicit `keras.Input(shape=(2,))` syntax, ending with `model.summary()`.

---

### 3. Stochastic Gradient Descent

* **Data Preparation & Leakage-Free Scaling (`red-wine.csv`):**
  * Splits the dataset into 70% training (`df_train`) and 30% validation (`df_valid`) using `red_wine.sample(frac=0.7, random_state=0)`.
  * Computes feature-wise minimums (`min_`) and maximums (`max_`) **exclusively on `df_train`**, then applies Min-Max scaling $(x - \min) / (\max - \min)$ to both `df_train` and `df_valid` to map values into $[0, 1]$.
  * Separates predictors (`X_train`, `X_valid`) from the `quality` target (`y_train`, `y_valid`).
* **Model Architecture & Training:**
  * Builds a high-capacity 3-hidden-layer MLP (`512` $\rightarrow$ `512` $\rightarrow$ `512` $\rightarrow$ `1` units) with `'relu'` activations.
  * Compiles with the **Adam** optimizer (`optimizer='adam'`) and **Mean Absolute Error** loss (`loss='mae'`).
  * Trains for `epochs=10` with `batch_size=256`, passing `validation_data=(X_valid, y_valid)`.
* **Diagnostics & Inference:**
  * Converts `history.history` into a `pandas.DataFrame` (`history_df`), plots training loss and combined training/validation loss curves, and generates sample predictions on `X_valid`.

---

### 4. Overfitting and Underfitting

* **Preventing Overfitting with Early Stopping:**
  * Reuses the normalized 70/30 `red-wine.csv` split and the $3 \times 512$ ReLU architecture.
  * Configures `callbacks.EarlyStopping` with:
    * `min_delta=0.001`: Minimum validation loss improvement required to qualify as progress.
    * `patience=20`: Number of consecutive epochs without improvement allowed before halting training.
    * `restore_best_weights=True`: Automatically rolls back model weights to the epoch that achieved the lowest `val_loss`.
* **Long-Horizon Training & Analysis:**
  * Trains for up to `epochs=500` (`batch_size=256`, `verbose=0`) with `callbacks=[early_stopping]`.
  * Extracts `minimum_validation_loss`, computes `best_epoch = history_df['val_loss'].idxmin() + 1`, prints total epochs completed before stopping, plots learning curves, and runs inference on `X_valid`.

---

### 5. Dropout and Batch Normalization

* **Visualization Styling:**
  * Configures `matplotlib` with the `'seaborn-v0_8-whitegrid'` style, `autolayout=True`, and bold axis titles/labels.
* **Regularization & Training Stabilization (`red-wine.csv`):**
  * Uses the raw (unscaled) 70/30 `red-wine.csv` split to demonstrate how **Batch Normalization** stabilizes unnormalized inputs internally.
  * Builds a wide 3-block network (`1024` units per block), where each block follows the pattern:
    $$\text{Dense}(1024, \text{relu}) \rightarrow \text{Dropout}(\text{rate}=0.3) \rightarrow \text{BatchNormalization}()$$
    ending with a single linear output unit `layers.Dense(1)`.
* **Training & Evaluation:**
  * Compiles with `optimizer='adam'` and `loss='mae'`, trains for `epochs=100` (`batch_size=256`, `verbose=0`), plots `loss` vs. `val_loss`, reports the minimum validation MAE, and outputs predictions.

---

### 6. Binary Classification

* **Target Encoding & Preprocessing (`ion.csv`):**
  * Loads `ion.csv` (`index_col=0`) and maps string labels in `'Class'` (`'good'` $\rightarrow$ `0`, `'bad'` $\rightarrow$ `1`).
  * Performs a 70/30 train-validation split (`random_state=0`) and applies training-derived Min-Max scaling to $[0, 1]$.
  * Removes constant/zero-variance columns (which produce `NaN` after division by `max_ - min_ = 0`) using `df_train.columns[df_train.notna().any()]` across both splits.
* **Classification Architecture & Loss:**
  * Builds a compact MLP (`4` $\rightarrow$ `4` $\rightarrow$ `1` units) using `'relu'` in the hidden layers and `activation='sigmoid'` in the output layer to squash outputs into probabilities $p \in [0, 1]$.
  * Compiles with `optimizer='adam'`, `loss='binary_crossentropy'`, and `metrics=['binary_accuracy']`.
* **Training, Learning Curves & Thresholding:**
  * Trains for up to `epochs=1000` (`batch_size=512`, `verbose=0`) with `EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)`.
  * Plots loss and accuracy curves starting from epoch 5 (`history_df.loc[5:, ...]`) to skip initial transient spikes, prints best validation loss and accuracy, and thresholds predicted probabilities (`y_prob >= 0.5`) into binary integer class predictions (`y_pred`).

---

## Execution & Path Configuration Notes

* **Google Colab File Paths:** All `pd.read_csv()` calls point to `/content/data/data/<filename>.csv`. If running locally outside of Google Colab, update these paths to match your local extraction directory (e.g., `data/data/red-wine.csv` or `data/red-wine.csv`).
* **Plotting in Non-Interactive Environments:** The script calls `DataFrame.plot()` without explicit `plt.show()` statements (standard in Jupyter/Colab notebooks). If running `DL-code.py` as a standalone terminal script, append `import matplotlib.pyplot as plt; plt.show()` after plot calls to display figures.
* **Independent Sections:** Every module re-imports its required libraries and reloads its dataset independently, allowing each section to be executed standalone as a notebook cell once `data.zip` is extracted.

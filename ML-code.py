import zipfile

with zipfile.ZipFile("data.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# ============================================================
# INTRO TO MACHINE LEARNING: HOW DOES MODELS WORK
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd

from sklearn.tree import DecisionTreeRegressor


# ------------------------------------------------------------
# 2. LOAD THE DATASET
# ------------------------------------------------------------

# Read historical house sales data
# Replace the file path with the actual dataset location

home_data = pd.read_csv("/content/data/data/melb_data.csv")


# ------------------------------------------------------------
# 3. SELECT FEATURES
# ------------------------------------------------------------

# Input variables used to predict house prices

features = [
    "Rooms",
    "Bathroom",
    "Landsize",
    "Lattitude",
    "Longtitude"
]

# X contains the independent variables (features)
X = home_data[features]


# ------------------------------------------------------------
# 4. SELECT TARGET
# ------------------------------------------------------------

# Target variable: house sale price

y = home_data["Price"]


# ------------------------------------------------------------
# 5. CREATE THE DECISION TREE MODEL
# ------------------------------------------------------------

# Initialise a Decision Tree Regressor

model = DecisionTreeRegressor(
    random_state=1
)


# ------------------------------------------------------------
# 6. TRAIN / FIT THE MODEL
# ------------------------------------------------------------

# Learn patterns from the historical data

model.fit(X, y)


# ------------------------------------------------------------
# 7. GENERATE PREDICTIONS
# ------------------------------------------------------------

# Predict prices using the trained model

predictions = model.predict(X)

# Display the first five predictions
print(predictions[:5])


# ============================================================
# INTRO TO MACHINE LEARNING: BASIC DATA EXPLORATION
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd


# ------------------------------------------------------------
# 2. LOAD THE DATASET
# ------------------------------------------------------------

# Store the file path
melbourne_file_path = (
    "/content/data/data/melb_data.csv"
)

# Read the CSV file into a DataFrame
melbourne_data = pd.read_csv(melbourne_file_path)


# ------------------------------------------------------------
# 3. EXPLORE THE DATASET
# ------------------------------------------------------------

# Display the first five rows
print("FIRST FIVE ROWS:")
print(melbourne_data.head())

# Display the last five rows
print("\nLAST FIVE ROWS:")
print(melbourne_data.tail())

# Display the first ten rows
print("\nFIRST TEN ROWS:")
print(melbourne_data.head(10))

# Display the dimensions (rows, columns)
print("\nDATASET DIMENSIONS:")
print(melbourne_data.shape)

# Display column names
print("\nCOLUMN NAMES:")
print(melbourne_data.columns)

# Display data types and non-null counts
print("\nDATASET INFORMATION:")
melbourne_data.info()


# ------------------------------------------------------------
# 4. SUMMARY STATISTICS
# ------------------------------------------------------------

# Descriptive statistics for numerical columns
print("\nDESCRIPTIVE STATISTICS:")
print(melbourne_data.describe())

# Mean of numerical columns
print("\nMEAN:")
print(melbourne_data.mean(numeric_only=True))

# Median of numerical columns
print("\nMEDIAN:")
print(melbourne_data.median(numeric_only=True))

# Standard deviation
print("\nSTANDARD DEVIATION:")
print(melbourne_data.std(numeric_only=True))

# Minimum values
print("\nMINIMUM:")
print(melbourne_data.min(numeric_only=True))

# Maximum values
print("\nMAXIMUM:")
print(melbourne_data.max(numeric_only=True))


# ------------------------------------------------------------
# 5. MISSING VALUES
# ------------------------------------------------------------

# Count non-missing values in each column
print("\nNON-MISSING VALUES:")
print(melbourne_data.count())

# Count missing values in each column
print("\nMISSING VALUES:")
print(melbourne_data.isnull().sum())


# ------------------------------------------------------------
# 6. ANALYSE A SPECIFIC COLUMN
# ------------------------------------------------------------

# Descriptive statistics for house prices
print("\nHOUSE PRICE STATISTICS:")
print(melbourne_data["Price"].describe())

# Average house price
print("\nAVERAGE HOUSE PRICE:")
print(melbourne_data["Price"].mean())

# Median house price
print("\nMEDIAN HOUSE PRICE:")
print(melbourne_data["Price"].median())

# Maximum house price
print("\nMAXIMUM HOUSE PRICE:")
print(melbourne_data["Price"].max())


# ============================================================
# INTRO TO MACHINE LEARNING: YOUR FIRST MACHINE LEARNING MODEL
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


# 2. LOAD DATASET
melbourne_file_path = '/content/data/data/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)


# 3. INSPECT AVAILABLE COLUMNS
print("Available columns:")
print(melbourne_data.columns)


# 4. HANDLE MISSING VALUES
# Remove observations containing at least one missing value.
melbourne_data = melbourne_data.dropna(axis=0)

print("\nDataset dimensions after removing missing values:")
print(melbourne_data.shape)


# 5. SELECT THE PREDICTION TARGET (y)
# Price is the variable that the model must predict.
y = melbourne_data.Price


# 6. SELECT THE FEATURES (X)
# These variables will be used to predict house prices.
melbourne_features = [
    'Rooms',
    'Bathroom',
    'Landsize',
    'Lattitude',
    'Longtitude'
]

X = melbourne_data[melbourne_features]


# 7. INSPECT FEATURES AND TARGET
print("\nFeature columns:")
print(X.columns)

print("\nFeature summary statistics:")
print(X.describe())

print("\nFirst five observations:")
print(X.head())

print("\nTarget summary statistics:")
print(y.describe())

print("\nFeature dimensions:", X.shape)
print("Target dimensions:", y.shape)


# 8. DEFINE THE MODEL
# random_state ensures reproducibility.
melbourne_model = DecisionTreeRegressor(random_state=1)


# 9. FIT THE MODEL
# Learn patterns from the features (X) and target (y).
melbourne_model.fit(X, y)


# 10. GENERATE PREDICTIONS
# Demonstrate predictions using the first five observations.
sample_houses = X.head()

predictions = melbourne_model.predict(sample_houses)


# 11. DISPLAY RESULTS
print("\nMaking predictions for the following 5 houses:")
print(sample_houses)

print("\nThe predictions are:")
print(predictions)


# ============================================================
# INTRO TO MACHINE LEARNING: MODEL VALIDATION
# ============================================================


# 1. IMPORT LIBRARIES
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# 2. LOAD DATASET
melbourne_file_path = '/content/data/data/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)


# 3. HANDLE MISSING VALUES
filtered_melbourne_data = melbourne_data.dropna(axis=0)


# 4. SELECT TARGET (y)
y = filtered_melbourne_data.Price


# 5. SELECT FEATURES (X)
melbourne_features = [
    'Rooms',
    'Bathroom',
    'Landsize',
    'BuildingArea',
    'YearBuilt',
    'Lattitude',
    'Longtitude'
]

X = filtered_melbourne_data[melbourne_features]


# ============================================================
# PART A: IN-SAMPLE MODEL VALIDATION
# ============================================================

# 6. DEFINE MODEL
melbourne_model = DecisionTreeRegressor()

# 7. FIT MODEL USING THE ENTIRE DATASET
melbourne_model.fit(X, y)

# 8. PREDICT USING TRAINING DATA
predicted_home_prices = melbourne_model.predict(X)

# 9. CALCULATE IN-SAMPLE MAE
in_sample_mae = mean_absolute_error(
    y,
    predicted_home_prices
)

print("In-sample MAE:", in_sample_mae)


# ============================================================
# PART B: OUT-OF-SAMPLE MODEL VALIDATION
# ============================================================

# 10. SPLIT DATA INTO TRAINING AND VALIDATION SETS
train_X, val_X, train_y, val_y = train_test_split(
    X,
    y,
    random_state=0
)

# 11. DEFINE MODEL
melbourne_model = DecisionTreeRegressor()

# 12. FIT MODEL USING TRAINING DATA ONLY
melbourne_model.fit(train_X, train_y)

# 13. GENERATE VALIDATION PREDICTIONS
val_predictions = melbourne_model.predict(val_X)

# 14. CALCULATE VALIDATION MAE
val_mae = mean_absolute_error(
    val_y,
    val_predictions
)

print("Validation MAE:", val_mae)


# ============================================================
# PART C: COMPARE MODEL PERFORMANCE
# ============================================================

print("\nMODEL PERFORMANCE COMPARISON")

print("In-sample MAE:", in_sample_mae)
print("Validation MAE:", val_mae)



# ============================================================
# INTRO TO MACHINE LEARNING: UNDERFITTING AND OVERFITTING
# ============================================================


# 1. IMPORT LIBRARIES
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# 2. LOAD DATASET
melbourne_file_path = '/content/data/data/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)


# 3. HANDLE MISSING VALUES
filtered_melbourne_data = melbourne_data.dropna(axis=0)


# 4. SELECT TARGET (y)
y = filtered_melbourne_data.Price


# 5. SELECT FEATURES (X)
melbourne_features = [
    'Rooms',
    'Bathroom',
    'Landsize',
    'BuildingArea',
    'YearBuilt',
    'Lattitude',
    'Longtitude'
]

X = filtered_melbourne_data[melbourne_features]


# 6. SPLIT DATA INTO TRAINING AND VALIDATION SETS
train_X, val_X, train_y, val_y = train_test_split(
    X,
    y,
    random_state=0
)


# ============================================================
# 7. CREATE REUSABLE MAE FUNCTION
# ============================================================

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    # Define model
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        random_state=0
    )

    # Train model
    model.fit(train_X, train_y)

    # Generate validation predictions
    preds_val = model.predict(val_X)

    # Calculate validation MAE
    mae = mean_absolute_error(val_y, preds_val)

    return mae


# ============================================================
# 8. COMPARE DIFFERENT TREE SIZES
# ============================================================

candidate_max_leaf_nodes = [5, 50, 500, 5000]

for max_leaf_nodes in candidate_max_leaf_nodes:

    my_mae = get_mae(
        max_leaf_nodes,
        train_X,
        val_X,
        train_y,
        val_y
    )

    print(
        "Max leaf nodes: %d \t Mean Absolute Error: %d"
        % (max_leaf_nodes, my_mae)
    )


# ============================================================
# 9. IDENTIFY THE BEST TREE SIZE AUTOMATICALLY
# ============================================================

best_mae = float('inf')
best_tree_size = None

for max_leaf_nodes in candidate_max_leaf_nodes:

    mae = get_mae(
        max_leaf_nodes,
        train_X,
        val_X,
        train_y,
        val_y
    )

    if mae < best_mae:
        best_mae = mae
        best_tree_size = max_leaf_nodes

print("\nBEST MODEL CONFIGURATION")
print("Best maximum leaf nodes:", best_tree_size)
print("Lowest validation MAE:", best_mae)


# ============================================================
# 10. TRAIN FINAL MODEL
# ============================================================

final_model = DecisionTreeRegressor(
    max_leaf_nodes=best_tree_size,
    random_state=0
)

final_model.fit(train_X, train_y)


# ============================================================
# 11. GENERATE FINAL VALIDATION PREDICTIONS
# ============================================================

final_predictions = final_model.predict(val_X)

final_mae = mean_absolute_error(
    val_y,
    final_predictions
)

print("\nFINAL MODEL VALIDATION")
print("Final validation MAE:", final_mae)



# ============================================================
# INTRO TO MACHINE LEARNING: RANDOM FORESTS
# ============================================================


# 1. IMPORT LIBRARIES
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# 2. LOAD DATASET
melbourne_file_path = '/content/data/data/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)


# 3. HANDLE MISSING VALUES
melbourne_data = melbourne_data.dropna(axis=0)


# 4. SELECT TARGET (y)
y = melbourne_data.Price


# 5. SELECT FEATURES (X)
melbourne_features = [
    'Rooms',
    'Bathroom',
    'Landsize',
    'BuildingArea',
    'YearBuilt',
    'Lattitude',
    'Longtitude'
]

X = melbourne_data[melbourne_features]


# 6. SPLIT DATA INTO TRAINING AND VALIDATION SETS
train_X, val_X, train_y, val_y = train_test_split(
    X,
    y,
    random_state=0
)


# ============================================================
# 7. DEFINE RANDOM FOREST MODEL
# ============================================================

forest_model = RandomForestRegressor(
    random_state=1
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

forest_model.fit(train_X, train_y)


# ============================================================
# 9. GENERATE VALIDATION PREDICTIONS
# ============================================================

melb_preds = forest_model.predict(val_X)


# ============================================================
# 10. EVALUATE MODEL
# ============================================================

forest_mae = mean_absolute_error(
    val_y,
    melb_preds
)

print("Random Forest MAE:", forest_mae)


# ============================================================
# 11. DISPLAY MODEL INFORMATION
# ============================================================

print("\nRandom Forest model:")
print(forest_model)

print("\nNumber of trees:")
print(forest_model.n_estimators)



# ============================================================
# INTRO TO MACHINE LEARNING: MACHINE LEARNING COMPETITIONS
# ============================================================


# 1. IMPORT LIBRARIES
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# 2. LOAD DATASETS
train_data = pd.read_csv(
    '/content/data/data/train.csv'
)

test_data = pd.read_csv(
    '/content/data/data/test.csv'
)


# 3. SELECT TARGET
y = train_data.SalePrice


# 4. SELECT FEATURES
features = [
    'LotArea',
    'YearBuilt',
    '1stFlrSF',
    '2ndFlrSF',
    'FullBath',
    'BedroomAbvGr',
    'TotRmsAbvGrd'
]

X = train_data[features]

X_test = test_data[features]


# 5. DEFINE RANDOM FOREST MODEL
model = RandomForestRegressor(
    n_estimators=100,
    random_state=1
)


# 6. TRAIN MODEL
model.fit(X, y)


# 7. GENERATE TEST PREDICTIONS
predictions = model.predict(X_test)


# 8. CREATE SUBMISSION DATAFRAME
output = pd.DataFrame({
    'Id': test_data.Id,
    'SalePrice': predictions
})


# 9. EXPORT SUBMISSION FILE
output.to_csv(
    'submission.csv',
    index=False
)


# 10. INSPECT SUBMISSION
print(output.head())

print("\nSubmission dimensions:")
print(output.shape)

print("\nSubmission saved as submission.csv")


# ============================================================
# INTERMEDIATE MACHINE LEARNING: INTRODUCTION
# ============================================================

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


# 1. LOAD TRAINING AND TEST DATASETS
# ------------------------------------------------------------

train_data = pd.read_csv("/content/data/data/train.csv")
test_data = pd.read_csv("/content/data/data/test.csv")


# 2. SEPARATE TARGET FROM FEATURES
# ------------------------------------------------------------

y = train_data["SalePrice"]

X = train_data.drop(columns=["SalePrice"])


# 3. REMOVE ID COLUMN FROM PREDICTORS
# ------------------------------------------------------------

test_ids = test_data["Id"]

X = X.drop(columns=["Id"])
test_data = test_data.drop(columns=["Id"])


# 4. IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# ------------------------------------------------------------

categorical_cols = X.select_dtypes(
    include=["object"]
).columns

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns


# 5. NUMERICAL PREPROCESSING
# ------------------------------------------------------------

numerical_transformer = SimpleImputer(
    strategy="median"
)


# 6. CATEGORICAL PREPROCESSING
# ------------------------------------------------------------

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# 7. COMBINE PREPROCESSING
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


# 8. DEFINE MODEL
# ------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)


# 9. CREATE PIPELINE
# ------------------------------------------------------------

my_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# 10. TRAIN MODEL
# ------------------------------------------------------------

my_pipeline.fit(X, y)


# 11. GENERATE PREDICTIONS
# ------------------------------------------------------------

predictions = my_pipeline.predict(test_data)


# 12. CREATE KAGGLE SUBMISSION
# ------------------------------------------------------------

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": predictions
})

submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")
print(submission.head())


# ============================================================
# INTERMEDIATE MACHINE LEARNING: MISSING VALUES
# ============================================================

# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


# 2. LOAD AND PREPARE DATA
# ------------------------------------------------------------

# Load Melbourne Housing dataset
data = pd.read_csv(
    '/content/data/data/melb_data.csv'
)

# Define target variable
y = data['Price']

# Select numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Split into training and validation datasets
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=0
)


# 3. DEFINE MODEL EVALUATION FUNCTION
# ------------------------------------------------------------

def score_dataset(X_train, X_valid, y_train, y_valid):

    # Define Random Forest model
    model = RandomForestRegressor(
        n_estimators=10,
        random_state=0
    )

    # Train model
    model.fit(X_train, y_train)

    # Generate predictions
    preds = model.predict(X_valid)

    # Calculate MAE
    return mean_absolute_error(y_valid, preds)


# 4. IDENTIFY COLUMNS WITH MISSING VALUES
# ------------------------------------------------------------

cols_with_missing = [
    col for col in X_train.columns
    if X_train[col].isnull().any()
]


# 5. APPROACH 1 - DROP COLUMNS
# ------------------------------------------------------------

# Remove columns containing missing values
reduced_X_train = X_train.drop(
    cols_with_missing,
    axis=1
)

reduced_X_valid = X_valid.drop(
    cols_with_missing,
    axis=1
)

# Evaluate model
mae_approach_1 = score_dataset(
    reduced_X_train,
    reduced_X_valid,
    y_train,
    y_valid
)


# 6. APPROACH 2 - IMPUTATION
# ------------------------------------------------------------

# Define imputer (mean strategy by default)
my_imputer = SimpleImputer()

# Fit on training data and transform
imputed_X_train = pd.DataFrame(
    my_imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

# Transform validation data using training statistics
imputed_X_valid = pd.DataFrame(
    my_imputer.transform(X_valid),
    columns=X_valid.columns,
    index=X_valid.index
)

# Evaluate model
mae_approach_2 = score_dataset(
    imputed_X_train,
    imputed_X_valid,
    y_train,
    y_valid
)


# 7. APPROACH 3 - IMPUTATION + MISSING INDICATORS
# ------------------------------------------------------------

# Copy original datasets
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Create Boolean indicators for missing values
for col in cols_with_missing:

    X_train_plus[col + '_was_missing'] = (
        X_train_plus[col].isnull()
    )

    X_valid_plus[col + '_was_missing'] = (
        X_valid_plus[col].isnull()
    )

# Define imputer
my_imputer = SimpleImputer()

# Fit and transform training data
imputed_X_train_plus = pd.DataFrame(
    my_imputer.fit_transform(X_train_plus),
    columns=X_train_plus.columns,
    index=X_train_plus.index
)

# Transform validation data
imputed_X_valid_plus = pd.DataFrame(
    my_imputer.transform(X_valid_plus),
    columns=X_valid_plus.columns,
    index=X_valid_plus.index
)

# Evaluate model
mae_approach_3 = score_dataset(
    imputed_X_train_plus,
    imputed_X_valid_plus,
    y_train,
    y_valid
)


# 8. COMPARE RESULTS
# ------------------------------------------------------------

results = pd.DataFrame({
    'Approach': [
        'Drop columns',
        'Imputation',
        'Imputation + indicators'
    ],
    'MAE': [
        mae_approach_1,
        mae_approach_2,
        mae_approach_3
    ]
})

# Sort by MAE, from lowest to highest
results = results.sort_values(
    by='MAE',
    ascending=True
)

print("\nMODEL PERFORMANCE COMPARISON")
print(results)


# 9. INVESTIGATE MISSING VALUES
# ------------------------------------------------------------

# Dataset dimensions
print("\nTRAINING DATA DIMENSIONS")
print(X_train.shape)

# Missing values by column
missing_val_count_by_column = X_train.isnull().sum()

# Display columns with missing values only
print("\nMISSING VALUES BY COLUMN")
print(
    missing_val_count_by_column[
        missing_val_count_by_column > 0
    ]
)


# ============================================================
# INTERMEDIATE MACHINE LEARNING: CATEGORICAL VARIABLES
# ============================================================


# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder
)


# 2. LOAD DATA
# ------------------------------------------------------------

data = pd.read_csv(
    '/content/data/data/melb_data.csv'
)

# Separate target and predictors
y = data['Price']
X = data.drop(['Price'], axis=1)

# Split into training and validation datasets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=0
)


# 3. REMOVE COLUMNS WITH MISSING VALUES
# ------------------------------------------------------------

# Identify columns with missing values in training data
cols_with_missing = [
    col for col in X_train_full.columns
    if X_train_full[col].isnull().any()
]

# Remove the same columns from both datasets
X_train_full = X_train_full.drop(
    cols_with_missing,
    axis=1
)

X_valid_full = X_valid_full.drop(
    cols_with_missing,
    axis=1
)


# 4. SELECT CATEGORICAL AND NUMERICAL VARIABLES
# ------------------------------------------------------------

# Categorical variables with fewer than 10 unique values
low_cardinality_cols = [
    cname
    for cname in X_train_full.columns
    if (
        X_train_full[cname].nunique() < 10
        and X_train_full[cname].dtype == 'object'
    )
]

# Numerical variables
numerical_cols = [
    cname
    for cname in X_train_full.columns
    if X_train_full[cname].dtype in ['int64', 'float64']
]

# Combine selected columns
my_cols = low_cardinality_cols + numerical_cols

# Create selected datasets
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


# 5. IDENTIFY CATEGORICAL COLUMNS
# ------------------------------------------------------------

s = X_train.dtypes == 'object'

object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


# 6. DEFINE MODEL EVALUATION FUNCTION
# ------------------------------------------------------------

def score_dataset(X_train, X_valid, y_train, y_valid):

    # Define Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=0
    )

    # Train model
    model.fit(X_train, y_train)

    # Generate predictions
    preds = model.predict(X_valid)

    # Calculate MAE
    return mean_absolute_error(y_valid, preds)


# 7. APPROACH 1 - DROP CATEGORICAL VARIABLES
# ------------------------------------------------------------

# Keep numerical columns only
drop_X_train = X_train.select_dtypes(
    exclude=['object']
)

drop_X_valid = X_valid.select_dtypes(
    exclude=['object']
)

# Evaluate model
mae_approach_1 = score_dataset(
    drop_X_train,
    drop_X_valid,
    y_train,
    y_valid
)


# 8. APPROACH 2 - ORDINAL ENCODING
# ------------------------------------------------------------

# Copy original datasets
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Define ordinal encoder
ordinal_encoder = OrdinalEncoder()

# Fit and transform training data
label_X_train[object_cols] = ordinal_encoder.fit_transform(
    X_train[object_cols]
)

# Transform validation data
label_X_valid[object_cols] = ordinal_encoder.transform(
    X_valid[object_cols]
)

# Evaluate model
mae_approach_2 = score_dataset(
    label_X_train,
    label_X_valid,
    y_train,
    y_valid
)


# 9. APPROACH 3 - ONE-HOT ENCODING
# ------------------------------------------------------------

# Define one-hot encoder
OH_encoder = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

# Fit and transform training categorical variables
OH_cols_train = pd.DataFrame(
    OH_encoder.fit_transform(X_train[object_cols]),
    index=X_train.index
)

# Transform validation categorical variables
OH_cols_valid = pd.DataFrame(
    OH_encoder.transform(X_valid[object_cols]),
    index=X_valid.index
)

# Remove original categorical columns
num_X_train = X_train.drop(
    object_cols,
    axis=1
)

num_X_valid = X_valid.drop(
    object_cols,
    axis=1
)

# Combine numerical and encoded categorical columns
OH_X_train = pd.concat(
    [num_X_train, OH_cols_train],
    axis=1
)

OH_X_valid = pd.concat(
    [num_X_valid, OH_cols_valid],
    axis=1
)

# Ensure all column names are strings
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

# Evaluate model
mae_approach_3 = score_dataset(
    OH_X_train,
    OH_X_valid,
    y_train,
    y_valid
)


# 10. COMPARE RESULTS
# ------------------------------------------------------------

results = pd.DataFrame({
    'Approach': [
        'Drop categorical variables',
        'Ordinal encoding',
        'One-hot encoding'
    ],
    'MAE': [
        mae_approach_1,
        mae_approach_2,
        mae_approach_3
    ]
})

# Sort from lowest to highest MAE
results = results.sort_values(
    by='MAE',
    ascending=True
)

print("\nMODEL PERFORMANCE COMPARISON")
print(results)


# ============================================================
# INTERMEDIATE MACHINE LEARNING: PIPELINES
# ============================================================


# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# 2. LOAD DATA
# ------------------------------------------------------------

data = pd.read_csv(
    '/content/data/data/melb_data.csv'
)

# Separate target from predictors
y = data['Price']
X = data.drop(['Price'], axis=1)


# 3. SPLIT DATA
# ------------------------------------------------------------

X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=0
)


# 4. IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# ------------------------------------------------------------

# Categorical columns with low cardinality
categorical_cols = [
    cname for cname in X_train_full.columns
    if (
        X_train_full[cname].nunique() < 10
        and X_train_full[cname].dtype == 'object'
    )
]

# Numerical columns
numerical_cols = [
    cname for cname in X_train_full.columns
    if X_train_full[cname].dtype in ['int64', 'float64']
]

# Keep selected columns
my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


# 5. NUMERICAL PREPROCESSING
# ------------------------------------------------------------

# Replace missing numerical values with 0
numerical_transformer = SimpleImputer(
    strategy='constant'
)


# 6. CATEGORICAL PREPROCESSING
# ------------------------------------------------------------

categorical_transformer = Pipeline(
    steps=[
        (
            'imputer',
            SimpleImputer(strategy='most_frequent')
        ),
        (
            'onehot',
            OneHotEncoder(handle_unknown='ignore')
        )
    ]
)


# 7. COMBINE PREPROCESSING
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)


# 8. DEFINE MODEL
# ------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)


# 9. CREATE PIPELINE
# ------------------------------------------------------------

my_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ]
)


# 10. FIT PIPELINE
# ------------------------------------------------------------

my_pipeline.fit(X_train, y_train)


# 11. GENERATE PREDICTIONS
# ------------------------------------------------------------

preds = my_pipeline.predict(X_valid)


# 12. EVALUATE MODEL
# ------------------------------------------------------------

score = mean_absolute_error(y_valid, preds)

print('MAE:', score)


# ============================================================
# INTERMEDIATE MACHINE LEARNING: CROSS-VALIDATION
# ============================================================


# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score


# 2. LOAD DATA
# ------------------------------------------------------------

data = pd.read_csv(
    '/content/data/data/melb_data.csv'
)


# 3. SELECT FEATURES AND TARGET
# ------------------------------------------------------------

cols_to_use = [
    'Rooms',
    'Distance',
    'Landsize',
    'BuildingArea',
    'YearBuilt'
]

# Explanatory variables
X = data[cols_to_use]

# Target variable
y = data['Price']


# 4. DEFINE PREPROCESSING AND MODEL PIPELINE
# ------------------------------------------------------------

my_pipeline = Pipeline(
    steps=[
        (
            'preprocessor',
            SimpleImputer()
        ),
        (
            'model',
            RandomForestRegressor(
                n_estimators=50,
                random_state=0
            )
        )
    ]
)


# 5. PERFORM CROSS-VALIDATION
# ------------------------------------------------------------

scores = -1 * cross_val_score(
    my_pipeline,
    X,
    y,
    cv=5,
    scoring='neg_mean_absolute_error'
)


# 6. DISPLAY INDIVIDUAL MAE SCORES
# ------------------------------------------------------------

print("MAE scores:")
print(scores)


# 7. CALCULATE AVERAGE MAE
# ------------------------------------------------------------

average_mae = scores.mean()

print("\nAverage MAE score:")
print(average_mae)


# 8. ADDITIONAL PERFORMANCE STATISTICS
# ------------------------------------------------------------

print("\nMinimum MAE:")
print(scores.min())

print("\nMaximum MAE:")
print(scores.max())

print("\nStandard deviation:")
print(scores.std())


# ============================================================
# INTERMEDIATE MACHINE LEARNING: XGBOOST
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

# Read dataset
data = pd.read_csv(
    '/content/data/data/melb_data.csv'
)

# Select predictors
cols_to_use = [
    'Rooms',
    'Distance',
    'Landsize',
    'BuildingArea',
    'YearBuilt'
]

X = data[cols_to_use]

# Select target
y = data.Price

# Split into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    random_state=0
)


# ============================================================
# 3. INITIALISE XGBOOST MODEL
# ============================================================

my_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    early_stopping_rounds=5,
    n_jobs=4,
    random_state=0
)


# ============================================================
# 4. TRAIN MODEL
# ============================================================

my_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)


# ============================================================
# 5. GENERATE PREDICTIONS
# ============================================================

predictions = my_model.predict(X_valid)


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

mae = mean_absolute_error(y_valid, predictions)

print("Mean Absolute Error:", mae)


# ============================================================
# 7. INSPECT OPTIMAL NUMBER OF TREES
# ============================================================

# Best boosting iteration selected through early stopping
print("Best iteration:", my_model.best_iteration)

# Number of boosting rounds used for the best iteration
print("Best number of trees:", my_model.best_iteration + 1)


# ============================================================
# INTERMEDIATE MACHINE LEARNING: DATA LEAKAGE
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# ============================================================
# 2. LOAD DATA
# ============================================================

data = pd.read_csv(
    '/content/data/data/AER_credit_card_data.csv',
    true_values=['yes'],
    false_values=['no']
)


# ============================================================
# 3. SELECT TARGET AND PREDICTORS
# ============================================================

# Target variable
y = data.card

# Predictors
X = data.drop(['card'], axis=1)

# Dataset dimensions
print("Number of rows:", X.shape[0])

# First observations
print(X.head())


# ============================================================
# 4. INITIALISE RANDOM FOREST PIPELINE
# ============================================================

my_pipeline = make_pipeline(
    RandomForestClassifier(
        n_estimators=100,
        random_state=0
    )
)


# ============================================================
# 5. INITIAL MODEL EVALUATION
# ============================================================

cv_scores = cross_val_score(
    my_pipeline,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

print(
    "Initial cross-validation accuracy:",
    cv_scores.mean()
)


# ============================================================
# 6. INVESTIGATE POTENTIAL TARGET LEAKAGE
# ============================================================

# Separate expenditure by target class
expenditures_cardholders = X.expenditure[y]

expenditures_noncardholders = X.expenditure[~y]

# Percentage of rejected applicants with zero expenditure
print(
    'Rejected applicants with zero expenditure: %.2f'
    % (expenditures_noncardholders == 0).mean()
)

# Percentage of accepted applicants with zero expenditure
print(
    'Accepted applicants with zero expenditure: %.2f'
    % (expenditures_cardholders == 0).mean()
)


# ============================================================
# 7. REMOVE POTENTIALLY LEAKY PREDICTORS
# ============================================================

potential_leaks = [
    'expenditure',
    'share',
    'active',
    'majorcards'
]

X2 = X.drop(
    potential_leaks,
    axis=1
)


# ============================================================
# 8. RE-EVALUATE MODEL
# ============================================================

cv_scores = cross_val_score(
    my_pipeline,
    X2,
    y,
    cv=5,
    scoring='accuracy'
)

print(
    "Cross-validation accuracy after removing potential leaks:",
    cv_scores.mean()
)


# ============================================================
# 9. COMPARE RESULTS
# ============================================================

print("\nMODEL COMPARISON")

print(
    "Initial accuracy:",
    100 * cross_val_score(
        my_pipeline,
        X,
        y,
        cv=5,
        scoring='accuracy'
    ).mean()
)

print(
    "Accuracy without potential leaks:",
    100 * cv_scores.mean()
)

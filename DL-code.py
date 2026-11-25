import zipfile

with zipfile.ZipFile("data.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# ============================================================
# INTRO TO DEEP LEARNING: A SINGLE NEURON
# ============================================================

# 1. IMPORT LIBRARIES

from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 2. LINEAR UNIT: ONE INPUT
# ============================================================

# Define input, weight and bias
x = 5
w = 2.5
b = 90

# Calculate output
y = w * x + b

print("Predicted calories:", y)


# ============================================================
# 3. LINEAR UNIT: MULTIPLE INPUTS
# ============================================================

# Define three input features
x0 = 5       # Sugars
x1 = 3       # Fiber
x2 = 4       # Protein

# Define corresponding weights
w0 = 2.5
w1 = 1.2
w2 = 3.0

# Define bias
b = 90

# Calculate output
y = w0*x0 + w1*x1 + w2*x2 + b

print("Predicted output:", y)


# ============================================================
# 4. CREATE A LINEAR NEURAL NETWORK
# ============================================================

# Sequential model with one Dense layer
model = keras.Sequential([
    layers.Dense(
        units=1,
        input_shape=[3]
    )
])


# ============================================================
# 5. ALTERNATIVE: EXPLICIT INPUT LAYER
# ============================================================

model = keras.Sequential([
    keras.Input(shape=(3,)),
    layers.Dense(units=1)
])


# ============================================================
# 6. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# INTRO TO DEEP LEARNING: DEEP NEURAL NETWORKS
# ============================================================

# 1. IMPORT LIBRARIES

from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 2. DEFINE A SIMPLE LINEAR MODEL
# ============================================================

linear_model = keras.Sequential([
    keras.Input(shape=(2,)),
    layers.Dense(units=1)
])


# ============================================================
# 3. DEFINE A DEEP NEURAL NETWORK
# ============================================================

model = keras.Sequential([

    # First hidden layer
    layers.Dense(
        units=4,
        activation='relu',
        input_shape=[2]
    ),

    # Second hidden layer
    layers.Dense(
        units=3,
        activation='relu'
    ),

    # Output layer
    layers.Dense(
        units=1
    )
])


# ============================================================
# 4. ALTERNATIVE: EXPLICIT INPUT LAYER
# ============================================================

model = keras.Sequential([

    # Input layer
    keras.Input(shape=(2,)),

    # First hidden layer
    layers.Dense(
        units=4,
        activation='relu'
    ),

    # Second hidden layer
    layers.Dense(
        units=3,
        activation='relu'
    ),

    # Linear output layer
    layers.Dense(
        units=1
    )
])


# ============================================================
# 5. DISPLAY MODEL ARCHITECTURE
# ============================================================

model.summary()


# ============================================================
# INTRO TO DEEP LEARNING: STOCHASTIC GRADIENT DESCENT
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers


# 2. LOAD DATASET
red_wine = pd.read_csv('/content/data/data/red-wine.csv')


# 3. SPLIT TRAINING AND VALIDATION DATA
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)


# 4. SCALE DATA TO THE INTERVAL [0, 1]
# Calculate scaling parameters using training data only
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)

# Apply the same scaling parameters to both datasets
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)


# 5. SEPARATE FEATURES AND TARGET
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)

y_train = df_train['quality']
y_valid = df_valid['quality']


# 6. IDENTIFY NUMBER OF INPUT FEATURES
input_shape = [X_train.shape[1]]

print("Training shape:", X_train.shape)
print("Input features:", input_shape)


# 7. BUILD NEURAL NETWORK
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=1)
])


# 8. COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='mae'
)


# 9. TRAIN MODEL
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10
)


# 10. CONVERT TRAINING HISTORY TO DATAFRAME
history_df = pd.DataFrame(history.history)

print(history_df)


# 11. PLOT TRAINING LOSS
history_df['loss'].plot(
    title='Training Loss',
    xlabel='Epoch',
    ylabel='MAE'
)


# 12. PLOT TRAINING AND VALIDATION LOSS
history_df[['loss', 'val_loss']].plot(
    title='Training and Validation Loss',
    xlabel='Epoch',
    ylabel='MAE'
)


# 13. GENERATE PREDICTIONS
predictions = model.predict(X_valid)

print(predictions[:5])


# ============================================================
# INTRO TO DEEP LEARNING: OVERFITTING AND UNDERFITTING
# ============================================================


# 1. IMPORT LIBRARIES
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers, callbacks


# 2. LOAD DATASET
red_wine = pd.read_csv('/content/data/data/red-wine.csv')


# 3. SPLIT TRAINING AND VALIDATION DATA
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)


# 4. SCALE DATA TO [0, 1]
# Calculate scaling parameters from training data only
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)

# Apply the same parameters to both datasets
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)


# 5. SEPARATE FEATURES AND TARGET
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)

y_train = df_train['quality']
y_valid = df_valid['quality']


# 6. DEFINE EARLY STOPPING CALLBACK
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True
)


# 7. BUILD NEURAL NETWORK
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1)
])


# 8. COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='mae'
)


# 9. TRAIN MODEL WITH EARLY STOPPING
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping],
    verbose=0
)


# 10. CONVERT TRAINING HISTORY TO DATAFRAME
history_df = pd.DataFrame(history.history)


# 11. PLOT TRAINING AND VALIDATION LOSS
history_df[['loss', 'val_loss']].plot(
    title='Training and Validation Loss',
    xlabel='Epoch',
    ylabel='MAE'
)


# 12. CALCULATE MINIMUM VALIDATION LOSS
minimum_validation_loss = history_df['val_loss'].min()

print("Minimum validation loss:", minimum_validation_loss)


# 13. IDENTIFY BEST EPOCH
best_epoch = history_df['val_loss'].idxmin() + 1

print("Best epoch:", best_epoch)
print("Epochs actually completed:", len(history_df))


# 14. GENERATE PREDICTIONS
predictions = model.predict(X_valid)

print(predictions[:5])



# ============================================================
# INTRO TO DEEP LEARNING: DROPOUT AND BATCH NORMALIZATION
# ============================================================


# 1. IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


# 2. CONFIGURE MATPLOTLIB
plt.style.use('seaborn-v0_8-whitegrid')

plt.rc('figure', autolayout=True)

plt.rc(
    'axes',
    labelweight='bold',
    labelsize='large',
    titleweight='bold',
    titlesize=18,
    titlepad=10
)


# 3. LOAD DATASET
red_wine = pd.read_csv('/content/data/data/red-wine.csv')


# 4. SPLIT TRAINING AND VALIDATION DATA
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)


# 5. SEPARATE FEATURES AND TARGET
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)

y_train = df_train['quality']
y_valid = df_valid['quality']


# 6. BUILD NEURAL NETWORK
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),

    layers.Dense(1024, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.BatchNormalization(),

    layers.Dense(1024, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.BatchNormalization(),

    layers.Dense(1024, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.BatchNormalization(),

    layers.Dense(1)
])


# 7. COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='mae'
)


# 8. TRAIN MODEL
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=100,
    verbose=0
)


# 9. CONVERT TRAINING HISTORY TO DATAFRAME
history_df = pd.DataFrame(history.history)


# 10. PLOT TRAINING AND VALIDATION LOSS
history_df[['loss', 'val_loss']].plot(
    title='Training and Validation Loss',
    xlabel='Epoch',
    ylabel='MAE'
)


# 11. CALCULATE MINIMUM VALIDATION LOSS
print(
    "Minimum validation loss:",
    history_df['val_loss'].min()
)


# 12. GENERATE PREDICTIONS
predictions = model.predict(X_valid)

print(predictions[:5])


# ============================================================
# INTRO TO DEEP LEARNING: BINARY CLASSIFICATION
# ============================================================


# 1. IMPORT LIBRARIES
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers


# 2. LOAD DATASET
ion = pd.read_csv(
    '/content/data/data/ion.csv',
    index_col=0
)


# 3. CONVERT CATEGORICAL TARGET INTO NUMERIC LABELS
df = ion.copy()

df['Class'] = df['Class'].map({
    'good': 0,
    'bad': 1
})


# 4. SPLIT TRAINING AND VALIDATION DATA
df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)


# 5. SCALE DATA TO [0, 1]
# Calculate scaling parameters using training data
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)

# Apply the same scaling parameters to both datasets
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)


# 6. REMOVE EMPTY COLUMNS
# Identify columns containing at least one non-missing value
valid_columns = df_train.columns[df_train.notna().any()]

# Apply the same columns to training and validation data
df_train = df_train[valid_columns].copy()
df_valid = df_valid[valid_columns].copy()


# 7. SEPARATE FEATURES AND TARGET
X_train = df_train.drop('Class', axis=1)
X_valid = df_valid.drop('Class', axis=1)

y_train = df_train['Class']
y_valid = df_valid['Class']


# 8. IDENTIFY NUMBER OF INPUT FEATURES
print("Training shape:", X_train.shape)
print("Number of features:", X_train.shape[1])


# 9. BUILD NEURAL NETWORK
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(4, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


# 10. COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)


# 11. DEFINE EARLY STOPPING
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
)


# 12. TRAIN MODEL
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0
)


# 13. CONVERT TRAINING HISTORY TO DATAFRAME
history_df = pd.DataFrame(history.history)


# 14. PLOT TRAINING AND VALIDATION LOSS
history_df.loc[5:, ['loss', 'val_loss']].plot(
    title='Training and Validation Loss'
)


# 15. PLOT TRAINING AND VALIDATION ACCURACY
history_df.loc[5:, [
    'binary_accuracy',
    'val_binary_accuracy'
]].plot(
    title='Training and Validation Accuracy'
)


# 16. CALCULATE BEST VALIDATION METRICS
best_validation_loss = history_df['val_loss'].min()

best_validation_accuracy = (
    history_df['val_binary_accuracy'].max()
)

print("Best Validation Loss: {:.4f}".format(
    best_validation_loss
))

print("Best Validation Accuracy: {:.4f}".format(
    best_validation_accuracy
))


# 17. GENERATE PROBABILITY PREDICTIONS
y_prob = model.predict(X_valid)


# 18. CONVERT PROBABILITIES INTO CLASS PREDICTIONS
y_pred = (y_prob >= 0.5).astype(int)

print("First five probabilities:")
print(y_prob[:5])

print("First five predicted classes:")
print(y_pred[:5])

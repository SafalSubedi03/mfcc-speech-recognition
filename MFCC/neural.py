import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("mfcc_dataset.csv")


# ==========================================================
# FEATURES AND LABELS
# ==========================================================

X = df.iloc[:, 1:].values     # mfcc_1 to mfcc_13
y = df.iloc[:, 0].values      # labels


# ==========================================================
# LABEL ENCODING
# ==========================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# Convert to one-hot vectors
y_categorical = to_categorical(y_encoded)


# ==========================================================
# NORMALIZATION
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_categorical,
    test_size=0.2,
    random_state=42
)


# ==========================================================
# BUILD NEURAL NETWORK
# ==========================================================

model = Sequential()

# Input Layer + Hidden Layer
model.add(Dense(64, activation='relu', input_shape=(13,)))

# Hidden Layer
model.add(Dense(32, activation='relu'))

# Output Layer
model.add(Dense(y_categorical.shape[1], activation='softmax'))


# ==========================================================
# COMPILE MODEL
# ==========================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ==========================================================
# TRAIN MODEL
# ==========================================================

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2
)


# ==========================================================
# EVALUATE MODEL
# ==========================================================

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy*100:.2f}%")


# ==========================================================
# SAVE MODEL
# ==========================================================

model.save("mfcc_classifier.h5")

print("Model saved successfully.")
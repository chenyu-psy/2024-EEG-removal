# -*- coding: utf-8 -*-

"""
@author: Chenyu Li
@desp: 
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm


# Step 1: Load and preprocess data

# Directory where .fif files are stored
data_dir = Path.cwd() / "epochs"

X_list = []
y_list = []

# Load each .fif file and extract data and labels
for fif_file in data_dir.glob("*_cleaned_epo.fif"):
    epochs = mne.read_epochs(fif_file, preload=True)

    # Extract data and labels (assuming events have been coded correctly)
    data = epochs.get_data(picks=[i for i in epochs.ch_names if i != 'Status'])  # shape: (n_epochs, n_channels, n_times)
    labels = epochs.events[:, -1]  # Assuming last column has category codes

    # Append to the overall dataset
    X_list.append(data)
    y_list.append(labels)

# Convert lists to numpy arrays
X = np.concatenate(X_list, axis=0)  # Shape: (n_epochs, n_channels, n_times)
y = np.concatenate(y_list, axis=0)  # Shape: (n_epochs,)

# Reshape the data for MVPA
n_samples, n_channels, n_times = X.shape
X_reshaped = X.reshape(n_samples, n_channels * n_times)  # Shape: (n_epochs, n_channels * n_times)

# Step 2: Train a classifier
classifier = SVC(kernel='linear', probability=True)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True,)
predictions = []
true_labels = []

def train_and_predict(train_index, test_index):
    X_train, X_test = X_reshaped[train_index], X_reshaped[test_index]
    y_train, y_test = y[train_index], y[test_index]

    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    return preds, y_test

# Use tqdm to visualize progress
results = Parallel(n_jobs=4)(
    delayed(train_and_predict)(train_index, test_index) 
    for train_index, test_index in tqdm(cv.split(X_reshaped, y), total=cv.n_splits, desc="Cross-Validation Progress")
)

# Combine results
for preds, y_test in results:
    predictions.extend(preds)
    true_labels.extend(y_test)

# Step 3: Create a confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)


# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['letter', 'character', 'animals', 'foods', 'tools', 'faces'],
            yticklabels=['letter', 'character', 'animals', 'foods', 'tools', 'faces'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Step 4: Visualize weights of the classifier
classifier.fit(X_reshaped, y)  # Fit on the full dataset to get feature importance
weights = classifier.coef_

# Reshape weights for visualization
weights_reshaped = weights.reshape(n_channels, -1)  # Adjust as necessary

# Plot the weights as a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(weights_reshaped, cmap='coolwarm', xticklabels=range(weights_reshaped.shape[1]),
            yticklabels=[f'Channel {i}' for i in range(n_channels)])
plt.title('Weights of the SVM Classifier per Channel')
plt.xlabel('Time Points')
plt.ylabel('Channels')
plt.show()

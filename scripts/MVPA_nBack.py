import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# Step 1: Load and preprocess data
data_dir = Path.cwd() / "epochs"
X_list = []
y_list = []

for fif_file in data_dir.glob("*_cleaned_epo.fif"):
    epochs = mne.read_epochs(fif_file, preload=True)
    data = epochs.get_data(picks=[i for i in epochs.ch_names if i != 'Status'])
    labels = epochs.events[:, -1]  
    X_list.append(data)
    y_list.append(labels)

if not X_list or not y_list:
    raise ValueError("No data was loaded. Ensure the directory and file naming conventions are correct.")

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
n_samples, n_channels, n_times = X.shape
X_reshaped = X.reshape(n_samples, n_channels * n_times)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)

# Step 2: Initialize the SVM with class_weight balanced
classifier = SVC(kernel='linear', probability=True, class_weight='balanced')
cv = StratifiedKFold(n_splits=5, shuffle=True)
predictions = []
true_labels = []

def train_and_predict(train_index, test_index):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Print class distribution for debugging
    print("Training set class distribution:", np.bincount(y_train))
    print("Test set class distribution:", np.bincount(y_test))

    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    return preds, y_test

# Parallel processing
results = Parallel(n_jobs=5)(
    delayed(train_and_predict)(train_index, test_index) 
    for train_index, test_index in cv.split(X_scaled, y)
)


# Combine results
for fold in results:
    predictions.extend(fold[0])
    true_labels.extend(fold[1])

# Step 3: Create a confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['letter', 'character', 'animals', 'foods', 'tools', 'faces'],
            yticklabels=['letter', 'character', 'animals', 'foods', 'tools', 'faces'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# -*- coding: utf-8 -*-

"""
@author: Chenyu Li
@description: MVPA for nBack

The AUC (Area Under the Curve) is a measure of the ability of a classifier to distinguish between classes. It is used to evaluate the performance of a model. The value of AUC ranges from 0 to 1:

- **AUC = 1**: Perfect classifier. The model can perfectly distinguish between all positive and negative examples.
- **AUC = 0.5**: Random classifier. The model has no discriminative ability, equivalent to random guessing.
- **AUC < 0.5**: Worse than random guessing. This usually indicates that the model is making predictions in the opposite direction.
- **AUC > 0.5**: Indicates a good level of separability. The closer the AUC is to 1, the better the model's performance in distinguishing between positive and negative classes.

For multi-class problems, the AUC is often computed as a weighted average across classes (e.g., using the `ovr` strategy).
"""

#%% Import libraries
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from pkg.mvpa import MVPAMultiClassClassifier
import re


#%% Fit the data for each subject
data_dir = Path.cwd() / "epochs"
decoding_categories = ["character","foods","faces"]
epoch_time = [0,0.5]
sample_rate = 500
time_window = 0.05
significance_threshold = 0.05
results_all = []

for fif_file in data_dir.glob("*_cleaned_epo.fif"):
    # Get the subject ID from the file name
    filename = fif_file.stem
    sub = re.findall(r"\d+", filename)[0]

    # Load the data
    epochs = mne.read_epochs(fif_file, preload=True)

    # Preprocess the data
    epochs.drop_channels(['Status'])  # Drop the Status channel
    epochs = epochs[decoding_categories]
    epochs = epochs.crop(tmin=epoch_time[0], tmax=epoch_time[1])

    data = epochs.get_data()  # Get the data
    labels = epochs.events[:, -1]  # Get the labels

    # Reshape the data to nTimeWindow
    nTimeWindow = int((epoch_time[1] - epoch_time[0]) / time_window)
    data = data[:, :, :int(sample_rate*(epoch_time[1]-epoch_time[0]))]
    nSample_per_window = int(time_window * sample_rate)
    data_agg = data.reshape(data.shape[0], data.shape[1], nTimeWindow, nSample_per_window).mean(axis=-1)

    # Concatenate the data
    X = data_agg
    y = labels

    # Run F-test to determine the importance of each channel
    X_channel_reshaped = X.mean(axis=-1)
    f_values, p_values = f_classif(X_channel_reshaped, y)

    # Set a significance threshold (e.g., p < 0.05)
    used_channels = p_values < significance_threshold

    # Filter the features based on their significance
    X_filtered = X[:, used_channels, :]

    # Reshape the data
    n_samples, n_channels, n_times = X_filtered.shape
    X_reshaped = X_filtered.reshape(n_samples, n_channels * n_times)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    # Initialize the logistic regression classifier with OneVsRest strategy
    # base_classifier = LogisticRegression(penalty='l2', max_iter=1000)
    # classifier = OneVsRestClassifier(base_classifier)
    classifier = LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs')

    # Parallel processing
    model_fit = MVPAMultiClassClassifier(classifier)
    results = model_fit.cross_validate(X_scaled, y, cv=5, n_jobs=5)
    auc_list = results["auc"]
    cma_list = results["cma"]

    # Mean AUC
    mean_auc = np.mean(results["auc"], axis=0)
    mean_cma = np.mean(results["cma"], axis=0)

    # feature importance
    feature_importance = model_fit.calculate_feature_importance()

    # save the classifier
    save_path = Path.cwd() / "classifiers" / f"{sub}_classifier_nBack_logit.pkl"
    model_fit.save_classifier(save_path)

    # Save the results
    results_all.append((sub, mean_cma, mean_auc, feature_importance, used_channels))
    

#%% Report the feature selection
selected_channels = []

for _, _, _, feature_importance, used_channels in results_all:
    selected_channels.append(used_channels)

selected_channels = np.mean(selected_channels, axis=0)

# plot the selected channels
plt.figure(figsize=(20, 5))
plt.bar(np.arange(len(selected_channels)), selected_channels)
plt.xticks(np.arange(len(selected_channels)), epochs.ch_names, rotation=90)
plt.xlabel('Channel')
plt.ylabel('Frequency of use')
plt.title('Average frequency of use')
plt.show()

# save the selected channels
save_path = Path.cwd() / "figures" / "selected_channels_nBack_logit.pdf"
plt.savefig(save_path)


#%% Report all AUCs
Table_cat_aus = pd.DataFrame(columns=decoding_categories)

for i in range(len(decoding_categories)):

    category = decoding_categories[i]
    cat_auc_list = []
    
    for sub, _, auc,_,_ in results_all:

        Table_cat_aus.loc[sub, category] = auc[i]

print("\nAUC Scores per category:")
print(Table_cat_aus)


#%% Create a confusion matrix and plot it
cma_list = []

for _, cma, _, _, _ in results_all:
    cma_list.append(cma)

acc_mat_mean = np.mean(cma_list, axis=0)

# Plot the confusion matrix
tick_labels = decoding_categories
plt.rcParams.update({'font.size': 15})  # Set default font size to 15
plt.figure(figsize=(10, 8))
sns.heatmap(acc_mat_mean, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=tick_labels,
            yticklabels=tick_labels)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

#%% Plot the feature importance

importance_list = []

for _, _, _, importance, _ in results_all:
    importance_list.append(importance)

importance_mean = np.mean(importance_list, axis=0)

importance_reshape = importance_mean.reshape(n_channels, n_times)
# importance_transpose = importance_reshape.T

# Calculate the mean values along the time and channel dimensions
importance_time_mean = np.mean(importance_reshape, axis=0)  # Mean along channels for each time window
importance_channel_mean = np.mean(importance_reshape, axis=1)  # Mean along time for each channel

# Create the JointGrid for combined plotting
g = sns.JointGrid(height=15, ratio=4, xlim=(0, n_times-1), ylim=(0, n_channels-1))

# Plot the heatmap in the center (main plot)
heatmap_ax = g.ax_joint
sns.heatmap(importance_reshape, ax=heatmap_ax, cmap='Blues', cbar=False,
            xticklabels=[round(i * time_window, 2) for i in range(n_times)],
            yticklabels=epochs.ch_names)

heatmap_ax.set_xlabel('Time Window')
heatmap_ax.set_ylabel('Channel')

# Plot the average feature importance along the top (time axis)
time_mean_ax = g.ax_marg_x
sns.lineplot(x=np.arange(n_times), y=importance_time_mean, ax=time_mean_ax, color='blue')
time_mean_ax.set_title('Mean Feature Importance per Time Window')

# Plot the average feature importance along the left (channel axis)
channel_mean_ax = g.ax_marg_y
sns.lineplot(y=np.arange(n_channels), x=importance_channel_mean, ax=channel_mean_ax, color='blue',orient="y")
channel_mean_ax.invert_yaxis()  # To match the heatmap orientation

# Adjust the layout and save the plot
plt.tight_layout()
save_path = Path.cwd() / "figures" / "feature_importance_jointgrid_nBack_logit.pdf"
plt.savefig(save_path)

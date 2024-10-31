# -*- coding: utf-8 -*-
"""
@author: Chenyu Li
@desp: preprocessing EEG data for nBack task
"""

import mne
from pathlib import Path
from autoreject import AutoReject
import matplotlib
matplotlib.use('Qt5Agg') # switch to GUI backend

#%% Parameters

# file path
data_path = Path.cwd() / "data/EEG_data"
epoch_path = Path.cwd() / "epochs"
ar_path = Path.cwd() / "ar"
ica_path = Path.cwd() / "ica"

# Subject ID
sub='24102801'


#%% Step 1: Load data
datafile = data_path / f'remcode{sub}.bdf'
raw = mne.io.read_raw_bdf(datafile, preload=True)

#%% Step 2: Set montage (Electrode Locations))
biosemi_montage = mne.channels.make_standard_montage('biosemi64')
# biosemi_montage.plot(show_names=True)
raw.set_montage(biosemi_montage, on_missing='warn')

# re-reference
ref_channels = ['EXG1','EXG5'] # left and right
raw_ref, _ = mne.set_eeg_reference(raw, ref_channels)

# remove the raw object from memory
del raw

# mark bad channels (if the channels are bad all the time. Otherwise, use the artifact rejection)
raw_ref.plot()
# print(raw_ref.info['bads'])



#%% Step 3: Downsample and filter the extreme high and low frequencies
raw_ref_ds = raw_ref.copy().resample(500)

# filter the extreme high and low frequencies
raw_ref_ds.filter(l_freq=.1, h_freq=40.)

# remove the raw_ref objects from memory
del raw_ref

#%% Step 5: Epoching

# find the events from the data
events = mne.find_events(raw_ref_ds)

# Replace with actual event mapping
event_id = {
    # "fixation": 1,
    "letter": 11,
    "character": 12,
    "animals": 13,
    "foods": 14,
    "tools": 15,
    "faces": 16,}  

# create epochs
epochs = mne.Epochs(
    raw_ref_ds, 
    events, event_id, 
    tmin=-0.3, tmax=1.7, 
    baseline=(None, 0), preload=True)

# remove the raw_ref_ds object from memory
del raw_ref_ds

# drop the channels
epochs64 = epochs.copy()
epochs64.drop_channels([f"EXG{i+1}" for i in range(8)])

# save the epochs
savefile = epoch_path / f'remcode{sub}_raw_epo.fif'
epochs64.save(savefile, overwrite=True)

# epochs64.plot()

#%% Step 6: Use the AutoReject package to detect and reject bad epochs

# create the AutoReject object
ar = AutoReject(
    n_jobs=4, 
    picks=[i for i in range(64)],
    n_interpolate= [1, 4, 8])

# fit the model and get the reject log
epochs64_ar, reject_log = ar.fit_transform(epochs64, return_log=True) # The bad channels are automatically removed from the model fitting.

# print the rejected epochs
reject_log.plot("horizontal",aspect="auto")

#Save the reject log
savefile = ar_path / f'remcode{sub}_ar_log.npz'
reject_log.save(savefile, overwrite=True)

# interpolate bad channels
epochs64_int = epochs64_ar.copy()
epochs64_int.interpolate_bads(reset_bads=True)

# plot the cleaned epochs
epochs64_int.plot()



#%% Step 7: ICA for artifact removal
ica = mne.preprocessing.ICA(n_components=0.95)
ica.fit(epochs64_int.copy())

# Automatic detection of EOG
eog_indices, eog_scores = ica.find_bads_eog(epochs,["EXG2",'EXG3',"EXG4"])

# plot components
ica.plot_components()
ica.plot_sources(epochs64_int)
ica.plot_properties(epochs64_int, picks=[0, 14])

# remove artifact components
ica.exclude = [0, 2, 13]
reconst_epochs = epochs64_int.copy()
ica.apply(reconst_epochs)

# save the ica object
savefile = ica_path / f'remcode{sub}_ica.fif'
ica.save(savefile, overwrite=True)

# compare before and after removing ICAs
epochs.plot()
reconst_epochs.plot()



#%% Step 8: Save cleaned epochs
savefile = epoch_path / f'remcode{sub}_cleaned_epo.fif'
reconst_epochs.save(savefile, overwrite=True)




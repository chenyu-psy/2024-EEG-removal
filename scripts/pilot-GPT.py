import mne
import numpy as np
from pathlib import Path
from autoreject import AutoReject
import matplotlib
matplotlib.use('Qt5Agg') # switch to GUI backend

#%% Parameters

# file path
data_path = Path.cwd() / "data"
epoch_path = Path.cwd() / "epochs"

# EEG parameters
total_chan_num = 72
eeg_chan_num = 64

# Subject ID
sub='100301'


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

#%% Step 3: Downsample
raw_ref_ds = raw_ref.copy().resample(500)

#%% Step 4: Filtering
raw_ref_ds.filter(l_freq=.1, h_freq=40.)

#%% Step 5: Epoching
events = mne.find_events(raw_ref_ds)
event_id = {
    "letter": 1,
    "character": 2,
    "animal": 3,
    "fruit": 4,
    "furniture": 5,
    "face": 7}  # Replace with actual event mapping
# epochs = mne.Epochs(raw_ref_ds, events, event_id, tmin=-1.3, tmax=0, baseline=(None, 0), preload=True)
epochs = mne.Epochs(
    raw_ref_ds, 
    events, event_id, 
    tmin=-1.3, tmax=0, 
    baseline=(None, 0), preload=True)

# drop the channels
epochs64 = epochs.copy()
epochs64.drop_channels([f"EXG{i+1}" for i in range(8)])

# epochs.plot()

#%% Step 6: Use the AutoReject package to detect and reject bad epochs
ar = AutoReject()
epochs64 = ar.fit_transform(epochs64)
epochs64.plot()

# Print bad epochs
print(f'Detected bad epochs: {epochs64.info["bads"]}')

#%% Step 7: ICA for artifact removal
ica = mne.preprocessing.ICA(eeg_chan_num-len(epochs.info['bads']))
ica.fit(epochs64.copy())

# plot components
# epochs.plot(title='epochs before ICA')
ica.plot_components()
ica.plot_sources(epochs64)
ica.plot_properties(epochs64, picks=[0])

# remove artifact components
ica.exclude = [0, 5, 7, 11, 13]
reconst_epochs = epochs.copy()
ica.apply(reconst_epochs)

# compare before and after removing ICAs
epochs.plot()
reconst_epochs.plot()
# or use the automatic detection
eog_indices, eog_scores = ica.find_bads_eog(epochs,["EXG2",'EXG3',"EXG4"])


#%% Step 8: Save cleaned data
raw.save('cleaned_data.fif', overwrite=True)

#%% Step 9: Visualize
raw.plot(n_channels=30, scalings='auto')




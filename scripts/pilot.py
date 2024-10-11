import mne
import numpy as np
from pathlib import Path
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

#%% 1 load the data
datafile = data_path / f'remcode{sub}.bdf'
raw = mne.io.read_raw_bdf(datafile, preload=True)

#%% 2 inspect data
# raw.plot(duration=5, n_channels=72)

#%% 3 add channel locations
biosemi_montage = mne.channels.make_standard_montage('biosemi64')
# biosemi_montage.plot(show_names=True)
raw.set_montage(biosemi_montage, on_missing='warn')

# plot to check sensor
# raw.plot_sensor()

#%% 4 re-reference and plot
ref_channels = ['EXG1','EXG5'] # left and right
raw_ref, _ = mne.set_eeg_reference(raw, ref_channels)
raw_ref.plot(duration=5, n_channels=72)

#%% 5 downsample to 500 hz (reducing sampling rate)
raw_ref_ds = raw_ref.copy().resample(500)


#%% 6 get events after downsampling
events = mne.find_events(raw_ref_ds)
## if you wanna correct event labels
# events[events[:, 2] == 129, 2] = 1


#%% 6 bandpass filter [0.1,40] except for EXTRA channels
l_freq = 0.1 # low frequency
h_freq = 40 # high frequency
pick_ch = [i for i in range(eeg_chan_num)]
raw_ref_ds_fil = raw_ref_ds.copy().filter( l_freq=l_freq,h_freq = h_freq,picks = pick_ch)
#raw_ref_ds_fil.plot(duration=5, n_channels=72)

# plot to check the filtered data
# raw.plot_psd(area_mode='range', tmax=10.0,picks = np.arange(0,eeg_chan_num,dtype = int).tolist(), average=False)
# raw_ref_ds_fil.plot_psd(area_mode='range', tmax=10.0,picks = np.arange(0,eeg_chan_num,dtype = int).tolist(), average=False)

#%% 7 epoch
# create event code dictionary bins
event_dict = {
    "letter": 1,
    "character": 2,
    "animal": 3,
    "fruit": 4,
    "furniture": 5,
    "face": 7}
timings = {
    "letter": 1.05,
    "character": 1.05,
    "animal": 1.05,
    "fruit": 1.05,
    "furniture": 1.05,
    "face": 1.05
        }

# visualize events
#fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw_ref_ds_fil.info['sfreq'],first_samp=raw_ref_ds_fil.first_samp)

## epoch
# epoch_dict = {'letter': 1}  
# because the triger was sent after the trial, we need to use [-1.5,0]
# once we fix the issue, we can use [-0.3,1.05]
int_time = [-1.5,0] 
epochs = mne.Epochs(raw_ref_ds_fil, events,event_id=event_dict, tmin=int_time[0], tmax=int_time[1], preload=True, baseline= None)
epochs.save(epoch_path /f'{sub}_beforedrop_epo.fif',overwrite='True')


#%% 8 ICA only eeg channels
pick_ch =[i for i in range(eeg_chan_num)]
ica = mne.preprocessing.ICA(eeg_chan_num) #-len(epochs.info['bads']
ica.fit(epochs.copy().pick(picks=pick_ch))

# plot components
# epochs.plot(title='epochs before ICA')
ica.plot_components()
ica.plot_sources(epochs)
ica.plot_properties(epochs, picks=np.arange(10,20))

# remove artifact components
ica.exclude = [0,1,2,3,8]
reconst_epochs = epochs.copy()
ica.apply(reconst_epochs)

# compare before and after removing ICAs
epochs.plot()
reconst_epochs.plot()
# or use the automatic detection
ica.find_bads_eog(epochs,['EXG3'])

# reconst_epochs.interpolate_bads()

#%% 8. manually drop epochs and indexs for undropped epochs
# epochs_afterdrop = epochs.copy()
# epochs_afterdrop.plot()
# saveidx = np.where(np.isin(epochs.selection, epochs_afterdrop.selection))[0]
# #save
# epochs_afterdrop.save(datapath+'/'+i+'/'+i+'_afterdrop-epo.fif',overwrite='True')
# sub_info = {
#         "sub": i,
#         "epochs": epochs_afterdrop,
#         "trial2save": saveidx,
#     }
# import pickle
# with open(subpath +'/'+i+".pickle", 'wb') as handle:
#         pickle.dump(sub_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

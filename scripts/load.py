import mne
import os
import numpy as np
import glob
import pandas as pd
from pathlib import Path


sub='chenyu09191'
datafilepath = Path.cwd() / "data" / f'{sub}.bdf'

# 1.load the data
raw = mne.io.read_raw_bdf(datafilepath,preload=True)

# 2 inspect data
# raw.plot(duration=5, n_channels=72)

# 3 add channel locations
biosemi_montage = mne.channels.make_standard_montage('biosemi64')
# biosemi_montage.plot(show_names=True)
raw.set_montage(biosemi_montage, on_missing='warn')



# plot to check sensor
# raw.plot_sensor()

# 4 re-reference and plot
ref_channels = ['EXG1','EXG5'] # left and right
raw_ref,_ = mne.set_eeg_reference(raw, ref_channels)
raw_ref.plot(duration=5, n_channels=72)

# 5 downsample to 500 hz
raw_ref_ds = raw_ref.copy().resample(500)


# 6 get events after downsampling
events = mne.find_events(raw_ref_ds)
## if you wanna correct event labels
# events[events[:, 2] == 129, 2] = 1


# 6 bandpass filter [0.1,40] except for EXTRA channels
l_freq = 0.1
h_freq = 40
eeg_chan_num =64
pick_ch =np.arange(0,eeg_chan_num,dtype = int).tolist()
raw_ref_ds_fil = raw_ref_ds.copy().filter( l_freq=l_freq,h_freq = h_freq,picks = pick_ch)
#raw_ref_ds_fil.plot(duration=5, n_channels=72)

# plot to check the filtered data
# raw.plot_psd(area_mode='range', tmax=10.0,picks = np.arange(0,eeg_chan_num,dtype = int).tolist(), average=False)
# raw_ref_ds_fil.plot_psd(area_mode='range', tmax=10.0,picks = np.arange(0,eeg_chan_num,dtype = int).tolist(), average=False)

# 7 epoch
# create event code dictionary bins
event_dict = {
        'other': 0,
        'letter':1,
        'character':2,
        'object':3}

timings = {
            'other': 0,
            'letter':1.05,
            'character':1.05,
            'object':1.05
        }

# visualize events
#fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw_ref_ds_fil.info['sfreq'],first_samp=raw_ref_ds_fil.first_samp)

## epoch
epoch_dict = {'letter': 1}
int_time = [-0.3,1.15]
#epochs = mne.Epochs(raw_ref, events, event_id=event_dict, tmin=-0.2, tmax=0.5,reject=reject_criteria, preload=True)
epochs = mne.Epochs(raw_ref_ds_fil, events,event_id=epoch_dict, tmin=int_time[0], tmax=int_time[1], preload=True, baseline= None)
epochs.save(datapath + '/' + i + '/' + i + '-_beforedrop_epo.fif',overwrite='True')


# 8 ICA only eeg channels
# eeg_chan_num =64
# pick_ch =np.arange(0,eeg_chan_num,dtype = int).tolist()
# ica = mne.preprocessing.ICA(eeg_chan_num) #-len(epochs.info['bads']
# ica.fit(epochs.copy().pick(picks=pick_ch))
        # # #
# # # # plot components
# # # # epochs.plot(title='epochs before ICA')
# ica.plot_components()
# # # # ica.plot_sources(epochs)
# ica.plot_properties(epochs, picks=np.arange(10,20))
#
# # remove artifact components
# ica.exclude = [0,1,2,3,8]
# reconst_epochs = epochs.copy()
# ica.apply(reconst_epochs)

# compare before and after removing ICAs
# epochs.plot()
# reconst_epochs.plot()
# or use the automatic detection
#ica.find_bads_eog(epochs,['EXG3'])

# reconst_epochs.interpolate_bads()

# 8. manually drop epochs and indexs for undropped epochs
epochs_afterdrop = epochs.copy()
epochs_afterdrop.plot()
saveidx = np.where(np.isin(epochs.selection, epochs_afterdrop.selection))[0]
#save
epochs_afterdrop.save(datapath+'/'+i+'/'+i+'_afterdrop-epo.fif',overwrite='True')
sub_info = {
        "sub": i,
        "epochs": epochs_afterdrop,
        "trial2save": saveidx,
    }
import pickle
with open(subpath +'/'+i+".pickle", 'wb') as handle:
        pickle.dump(sub_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

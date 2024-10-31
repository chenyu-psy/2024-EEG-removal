# -*- coding: utf-8 -*-

"""
@author: Ziyao
@file: MVPA_training.py
@time: 03/14/23
@desp: train classifier in the localizer task, based on zoe 2021
"""
from sklearn.svm import SVC
import mne_connectivity
import os
import mne
import numpy as np
import time
import pandas as pd
import pickle
from tqdm import trange

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV


from mne.time_frequency import (tfr_multitaper, tfr_stockwell, tfr_morlet,
                                tfr_array_morlet)

from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedShuffleSplit, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, Vectorizer)
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import LeaveOneGroupOut


def MVPA_decoding(data, labels, info, f_name, all_time_points, timeWindows, sfreq, clf, cv,cat):
    '''
    parameters:
        'data': epochs, shape is (n_trial, n_channel, n_times)
        'labels': labels for all data, shape is n_trial
        'info': data information for scaler
        'f_name': the filename you want to save your results
        'all_time_points': array-like including all time points in the data
        'timeWindows': the time window you want to decode, it can be multiple windows
                        We treat different time windows as separate samples
        'sfreq': sampling rate, Hz
        'clf': the classifier you want to use
        'cv': the cross-validation method you use
    '''
    # define how many timewindows do we have
    n_timeWindows = len(timeWindows)
    print('You have {} timeWindows'.format(n_timeWindows))

    # define data and labels
    y = np.tile(np.array(labels), n_timeWindows)
    X = np.array([])
    # extract sub-data, smooth it, and average it within each timeWindow
    for i in range(n_timeWindows):
        curr_timeWindow = timeWindows[i]
        left_timepoint_index = int((curr_timeWindow[0] - all_time_points[0]) * sfreq)
        right_timepoint_index = int(len(all_time_points) - (all_time_points[-1] - curr_timeWindow[1]) * sfreq)
        sub_time_points = all_time_points[left_timepoint_index:right_timepoint_index]

        sub_data_i = data[:, :, left_timepoint_index:right_timepoint_index]

        if cat ==4: #with rest
            rest_data = data[:, :, int((-0.5 - all_time_points[0]) * sfreq):int(len(all_time_points) - (all_time_points[-1] - (0)) * sfreq)][y==20,:]
            sub_data_i = np.concatenate((sub_data_i,rest_data),0)
            y = np.concatenate((y,np.ones(np.sum(y==20))))



        # get information
        n_trial = sub_data_i.shape[0]
        n_channel = sub_data_i.shape[1]
        n_times = sub_data_i.shape[2]

        # average data within timewindow
        mean_data_i = np.expand_dims(np.mean(sub_data_i, axis=-1), axis=-1)

        # concatenate data within each timewindow
        if i == 0:
            X = mean_data_i
        else:
            X = np.concatenate((X, mean_data_i), axis=0)

    print(X.shape)

    # standardize features across channels
    X_normalized = Scaler(info).fit_transform(X)
    # vectorize features across different time windows
    X_vectorized = Vectorizer().fit_transform(X_normalized)
    print('shape before and after vectorize: ', X_normalized.shape, X_vectorized.shape)

    # performe classification
    y_true = np.array([], dtype=int)
    y_pred = np.array([], dtype=int)


    for train_ind, test_ind in cv.split(X_vectorized[:,:], y[:]): #[:len(data),:]
        # print("TRAIN:", train_ind, "TEST:", test_ind)
        # for i in range(n_timeWindows):
        X_train = X_vectorized[train_ind]
        y_train = y[train_ind]
        X_test = X_vectorized[test_ind]
        y_test = y[test_ind]

        if cat ==3:
            min_trial = np.min([np.sum(y_train==20),np.sum(y_train==21),np.sum(y_train==22)])
            X_train_bal = np.concatenate((X_train[np.where(y_train==20)[0][:min_trial],:],X_train[np.where(y_train==21)[0][:min_trial],:],X_train[np.where(y_train==22)[0][:min_trial],:]))
            y_train_bal = np.concatenate((20*np.ones(min_trial),21*np.ones(min_trial),22*np.ones(min_trial)))#20*np.ones(min_trial)
        elif cat ==2:
            min_trial = np.min([np.sum(y_train == 21), np.sum(y_train == 22)])
            X_train_bal = np.concatenate((X_train[np.where(y_train == 21)[0][:min_trial], :], X_train[np.where(y_train == 22)[0][:min_trial], :]))
            y_train_bal = np.concatenate(( 21 * np.ones(min_trial), 22 * np.ones(min_trial)))  # 20*np.ones(min_trial)
        if cat==4:
            min_trial = np.min([np.sum(y_train==20),np.sum(y_train==21),np.sum(y_train==22),np.sum(y_train==1)])
            X_train_bal = np.concatenate((X_train[np.where(y_train==20)[0][:min_trial],:],X_train[np.where(y_train==21)[0][:min_trial],:],X_train[np.where(y_train==22)[0][:min_trial],:],X_train[np.where(y_train==1)[0][:min_trial],:]))
            y_train_bal = np.concatenate((20*np.ones(min_trial),21*np.ones(min_trial),22*np.ones(min_trial),np.ones(min_trial)))#20*np.ones(min_trial)


        # X_train = X_vectorized[np.concatenate((train_ind,train_ind+len(data),train_ind+2*len(data),
        #                                        train_ind+3*len(data),train_ind+4*len(data),train_ind+5*len(data),
        #                                        train_ind+6*len(data),train_ind+7*len(data),train_ind+8*len(data))),:]
        # y_train = y[np.concatenate((train_ind,train_ind+len(data),train_ind+2*len(data),
        #                                        train_ind+3*len(data),train_ind+4*len(data),train_ind+5*len(data),
        #                                        train_ind+6*len(data),train_ind+7*len(data),train_ind+8*len(data)))]
        # X_test = X_vectorized[np.concatenate((test_ind,test_ind+len(data),test_ind+2*len(data),
        #                                        test_ind+3*len(data),test_ind+4*len(data),test_ind+5*len(data),
        #                                        test_ind+6*len(data),test_ind+7*len(data),test_ind+8*len(data))),:]
        # y_test = y[np.concatenate((test_ind,test_ind+len(data),test_ind+2*len(data),
        #                                        test_ind+3*len(data),test_ind+4*len(data),test_ind+5*len(data),
        #                                        test_ind+6*len(data),test_ind+7*len(data),test_ind+8*len(data)))]
        # print(X_train_bal.shape,y_train_bal.shape)
        y_true = np.hstack((y_true, y_test))
        print(X_train.shape,X_test.shape)
        # sfs = SequentialFeatureSelector(LogisticRegression(solver='lbfgs', max_iter=10e5, multi_class='ovr'), n_features_to_select=20,cv=cv)
        # sfs.fit(X_train_bal, y_train_bal)
        # #
        # X_train_sel =X_train_bal[:, sfs.get_support()] #X_train_transformed
        # X_test_sel = X_test[:, sfs.get_support()]

        # skbest = SelectKBest(k=20)
        # X_train_kbest = skbest.fit_transform(X_train_bal, y_train_bal)
        # X_test_kbest = X_test[:, skbest.get_support()]

        # clf.fit(X_train_kbest, y_train_bal)
        clf.fit(X_train_bal, y_train_bal)
        #best_C = clf.best_params_['C']
        best_model = clf.best_estimator_
        pred_clf =best_model.predict(X_test)
        y_pred = np.hstack((y_pred, pred_clf))



        # print('X train shape: ',X_train.shape)
        # print('X test shape: ', X_test.shape)

    # save results
    # results = {}
    # get the confusion matrix
    # con_matrix = confusion_matrix(y_true, y_pred, labels=category_labels,normalize='true')
    # transform it into probability
    # con_matrix = con_matrix / len(y_true) * 2

    # results['confusion_matrix'] = con_matrix
    # results['y_true'] = y_true
    # results['y_pred'] = y_pred

    # # save results to the disk
    # f = open(f_name, 'wb')
    # pickle.dump(results, f)
    # f.close

    return y_true, y_pred




def base_line(eeg_data,time,baseline_window):
    baselined_eeg = np.zeros(eeg_data.shape)
    base_line_aved = np.mean(eeg_data[:,:,(time>=baseline_window[0])&(time<=baseline_window[1])],2)
    for ti in np.arange(eeg_data.shape[2]):
        baselined_eeg[:,:,ti] = eeg_data[:,:,ti] - base_line_aved
    return baselined_eeg
def norm_tem(x, dim1):
    # x: ndarray trial by channel by time
    # dim1: dimension to average, time

    normed_x = np.ones(x.shape)
    normed_x[:] = np.NaN
    xmean = np.mean(x, dim1)
    xstd = np.std(x, dim1)
    for i in range(x.shape[dim1]):
        normed_x[:, :, i] = (x[:, :, i] - xmean)

    return normed_x

def norm_chan(x, dim1):
    # x: ndarray trial by channel by time
    # dim1: dimension to average, channel

    normed_x = np.ones(x.shape)
    normed_x[:] = np.NaN
    xmean = np.mean(x, dim1)
    xstd = np.std(x, dim1)
    for i in range(x.shape[dim1]):
        normed_x[:, i, :] = (x[:, i, :] - xmean)/xstd

    return normed_x
# -------------------------classifier training------------------------- #
# define function to train the classifier based on all data
def clf_training (data, labels, info, f_name, all_time_points, timeWindows, sfreq, clf,cat):
    '''
    parameters:
        'data': epochs, shape is (n_trial, n_channel, n_times)
        'labels': labels for all data, shape is n_trial
        'info': data information for scaler
        'f_name': the filename you want to save your results
        'all_time_points': array-like including all time points in the data
        'timeWindows': the time window you want to decode, it can be multiple windows
                        We treat different time windows as separate samples
        'sfreq': sampling rate, Hz
        'clf': the classifier you want to use
    '''

    # define how many timewindows do we have
    n_timeWindows = len(timeWindows)
    print('You have {} timeWindows'.format(n_timeWindows))


    # define data and labels
    y = np.tile(np.array(labels), n_timeWindows)

    X = np.array([])
    # extract sub-data, smooth it, and average it within each timeWindow
    for i in range(n_timeWindows):
        curr_timeWindow = timeWindows[i]
        left_timepoint_index = int((curr_timeWindow[0] - all_time_points[0]) * sfreq)
        right_timepoint_index = int(len(all_time_points) - (all_time_points[-1] - curr_timeWindow[1]) * sfreq)
        sub_time_points = all_time_points[left_timepoint_index:right_timepoint_index]

        sub_data_i = data[:, :, left_timepoint_index:right_timepoint_index]
        if cat ==4: #with rest
            rest_data = data[:, :, int((-0.5 - all_time_points[0]) * sfreq):int(len(all_time_points) - (all_time_points[-1] - (0)) * sfreq)][y==20,:]
            sub_data_i = np.concatenate((sub_data_i,rest_data),0)
            print(sub_data_i.shape)
            y = np.concatenate((y,np.ones(np.sum(y==20))))

        # if withrest ==1:
        #     rest_data = data[:, :, int((-0.5 - all_time_points[0]) * sfreq):right_timepoint_index]

        # average data within timewindow
        mean_data_i = np.expand_dims(np.mean(sub_data_i, axis=-1), axis=-1)

        # concatenate data within each timewindow
        if i == 0:
            X = mean_data_i
        else:
            X = np.concatenate((X, mean_data_i), axis=0)
    print(X.shape)
    # standardize features across channels
    X_normalized = Scaler(info).fit_transform(X)
    # vectorize features across different time windows
    X_vectorized = Vectorizer().fit_transform(X_normalized)
    X_train = X_vectorized
    y_train = y
    print([np.sum(y_train == 20), np.sum(y_train == 21), np.sum(y_train == 22)])
    if cat ==3:
        min_trial = np.min([np.sum(y_train == 20), np.sum(y_train == 21), np.sum(y_train == 22)])
        X_train_bal = np.concatenate((X_train[np.where(y_train == 20)[0][:min_trial], :],
                                  X_train[np.where(y_train == 21)[0][:min_trial], :],
                                  X_train[np.where(y_train == 22)[0][:min_trial], :])) #
        y_train_bal = np.concatenate((20 * np.ones(min_trial), 21 * np.ones(min_trial), 22 * np.ones(min_trial)))#20 * np.ones(min_trial),
    elif cat==2:
        min_trial = np.min([ np.sum(y_train == 21), np.sum(y_train == 22)])
        X_train_bal = np.concatenate((X_train[np.where(y_train == 21)[0][:min_trial], :],
                                  X_train[np.where(y_train == 22)[0][:min_trial], :])) #
        y_train_bal = np.concatenate(( 21 * np.ones(min_trial), 22 * np.ones(min_trial)))#20 * np.ones(min_trial),
    elif cat == 4:
        min_trial = np.min([np.sum(y_train == 20), np.sum(y_train == 21), np.sum(y_train == 22)])
        X_train_bal = np.concatenate((X_train[np.where(y_train == 20)[0][:min_trial], :],
                                      X_train[np.where(y_train == 21)[0][:min_trial], :],
                                      X_train[np.where(y_train == 22)[0][:min_trial], :],
                                      X_train[np.where(y_train == 1)[0][:min_trial], :]))
        y_train_bal = np.concatenate((20 * np.ones(min_trial), 21 * np.ones(min_trial), 22 * np.ones(min_trial),np.ones(min_trial)))  # 20*np.ones(min_trial)

    # train classifier
    clf.fit(X_train_bal, y_train_bal)
    best_model = clf.best_estimator_
    print(clf.best_params_['C'],clf.best_score_)

    clf_trained = best_model.fit(X_train_bal, y_train_bal)

    # # save trained classifier into disk
    # f = open(f_name, 'wb')
    # pickle.dump(clf_trained, f)
    # f.close


    return clf_trained
# define data path
datapath = os.path.join(os.path.expanduser('~'),'Desktop', 'timecomp','data_EEG')#os.getcwd() + '/preprocessed_training/'
# define output path
outpath = os.path.join(os.path.expanduser('~'),'Desktop', 'timecomp','MVPA_training_ovr_ZZ')
isExists = os.path.exists(outpath)
if not isExists:
    os.makedirs(outpath)


# define category labels
cat=3
if cat==3:
    categories = ['object', 'face', 'scene']
    category_labels = [20, 21, 22]
    fnames = '_encode_multi'
elif cat==2:
    categories = ['face', 'scene']
    category_labels = [21, 22]
    fnames = '_encode_2cat_fs'
elif cat==4:
    categories = ['object', 'face', 'scene','fixation']
    category_labels = [20, 21, 22,1]
    fnames = '_encode_withrest'



# define smoothing window time points
smooth_window = 5

sub_ids = [16,17]#[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,19,21,22,23,24,25,26] #1,2,3,4,5,6,7,8,9,10,11,12,


for i in sub_ids:
    sub = 'sub%.2d' % (i)
    subpath = os.path.join(datapath,sub,'preprocessed_training')
    # outfile = outpath + '/'+sub+'_MVPA_training_probe.pickle'
    mvpa_out =  outpath + '/'+sub+ '_MVPA_training'+fnames+'.pickle'
    # read preprocessed epoch data
    epochs_filename = subpath + '/'+sub+'_training_epochs.fif'
    epochs_raw = mne.read_epochs(epochs_filename, preload=True)


    # --------------------- select channels
    drops = [ 'M2', 'HR', 'VR', 'HL', 'M1', 'Status']
    epochs_raw.drop_channels(drops)


    # # visualize epochs
    # epochs_raw.plot(picks=['eeg', 'eog'], n_epochs=5)
    # # drop bad channels
    # picks = np.array(epochs_raw.ch_names)[mne.pick_types(epochs_raw.info, eeg=True, exclude='bads')]
    #
    # # record selected channels
    # np.savetxt(outpath+'training_selected_channels.txt', picks, fmt='%s')

####----------select eletrodes. Not used in the current analysis
    # selected channels recording
    # picks_manually = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'FT7', 'FC3', 'FC1', 'C1',
    #        'C3', 'C5', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7',
    #        'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'AFz',
    #        'Fz', 'F2', 'F4', 'F8', 'FT8', 'FC6', 'C2', 'C4', 'T8', 'P2', 'P4',
    #        'P6', 'P8', 'PO8', 'PO4', 'O2']

    '''
    # select all eeg channels
    # pick selected channels
    # occipital cortex
    picks = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
         'PO3', 'PO4', 'PO7', 'PO8', 'O1', 'O2', 'Pz', 'POz','Oz', 'lz']


    # temporal cortex
    picks = ['C1','C3','C5', 'Cz', 'C2', 'C4', 'C6',
        'CP5', 'CP3', 'CP1', 'CPz',  'CP6', 'CP4', 'CP2',
        'T7', 'TP7', 'T8','TP8']


    # pre-frontal cortex
    picks = ['Fp1','Fp2','Fpz',
        'AF7','AF3','AF8','AF4','AFz',
        'F1','F3','F5','F7','Fz','F2','F4','F6','F8',
        'FT7','FC5','FC3','FC1','FT8','FC6','FC4','FC2','FCz']
    '''
####-------------------------------------


    epochs = epochs_raw.copy()#.pick_channels(picks)



    # get labels from events
    labels = epochs.events[:, 2]



    # get sample frequency
    sfreq = epochs.info['sfreq']
    # down sample data into 500Hz
    epochs_downsampled = epochs.copy().resample(sfreq=500)
    sfreq_downsampled = epochs_downsampled.info['sfreq']
    epochs_downsampled.info['bads']=[]


    # get all time points
    all_time_points = epochs_downsampled.times

    # get epochs data
    raw_data = epochs_downsampled.get_data()


####---------tf decomposition not used-------
    # time-frequency
    #  define frequency range
    # freqs = np.arange(1., 32., 1)
    # n_cycles = freqs / 2.
    #
    # power = tfr_morlet(epochs_downsampled, freqs=freqs,
    #                n_cycles=n_cycles, picks=['eeg'],
    #                return_itc=False, average=False)
    #
    #
    # avgpower = power.average()
    # avgpower.plot(combine='mean',baseline=None, mode='mean',
    #             tmin=-0.5, tmax=5,
    #           title='Using Morlet wavelets', show=False)
    # plt.savefig(outpath+'Time_frequency_training.png', dpi=300)
    # plt.show()
####------------------
    # sfreq = 500
    # freqs = np.logspace(*np.log10([0.1, 40]), num=40)
    # n_cycles = freqs / 2.  # different number of cycle per frequency
    # eeg_powers = mne.time_frequency.tfr_array_morlet(epochs_downsampled.copy().get_data(),sfreq=sfreq,  freqs=freqs, n_cycles=n_cycles,
    #                                                  output='power')
    #
    # #get band power
    # delta_power =  norm_chan(norm_tem(np.mean(eeg_powers[:,:,(freqs<=4),:],2),2),1)
    # theta_power =  norm_chan(norm_tem(np.mean(eeg_powers[:,:,(freqs>4)&(freqs<=7),:],2),2),1)
    # alpha_power = norm_chan(norm_tem(np.mean(eeg_powers[:,:,(freqs>7)&(freqs<=13),:],2),2),1)
    # beta_power =  norm_chan(norm_tem(np.mean(eeg_powers[:,:,(freqs>13),:],2),2),1)
    # h_freq = 6
    # epochs_filter = epochs_downsampled.copy().filter(l_freq = None, h_freq=h_freq, phase='zero-double')

    baselined_eeg = base_line(raw_data,all_time_points,[-0.2,0])



    #get all power data
    power_all=  baselined_eeg#np.concatenate((delta_power,theta_power,alpha_power,beta_power),1)#baselined_eeg#baselined_eeg#baselined_eeg#baselined_eeg# np.concatenate((delta_power,theta_power,alpha_power,beta_power),1)#baselined_eeg#baselined_eeg#

    if cat==2:
        power_all=power_all[labels!=20,:,:]
        labels=labels[labels!=20]
    ##### takes too much time maybe later  -------------------------temporal decoding------------------------- #
    # define temporal decoding timeWindow
    timeWindow = [-0.8, 5.5]



    # -------------------------MVPA decoding------------------------- #
    # define classifier timeWindow
    timeWindow_encoding = [0, 1] #0.5, 1.0
    timeWindow_delay = [3.5, 4.0]
    timeWindow_probe = [4,5]
    # define classifier
    # params = {'C': np.logspace(-1, 2, 30)}

    # clf = make_pipeline(Scaler(epochs_downsampled.info),
    #                     Vectorizer(),
    #                     LinearDiscriminantAnalysis())

    # cv = StratifiedShuffleSplit(n_splits=10, test_size=1/3)
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    cv = StratifiedShuffleSplit(n_splits=10, test_size=1/3)
    clf = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=10e5, multi_class='ovr'), params, cv=cv, scoring='roc_auc_ovr')  # make_pipeline(SVC(kernel='linear'))#LogisticRegression(solver='lbfgs', max_iter=10e5, multi_class='ovr')



    decode_windows =  [[0.5,0.7],[0.6,0.8],[0.7,0.9],[0.8,1],[0.9,1.1],[1,1.2],[1.1,1.3],[1.2,1.4],[1.3,1.5],[1.4,1.6],[1.5,1.7],[1.6,1.8],[1.7,1.9],[1.8,2]]

    for i in np.arange(len(decode_windows)):
        print('process:', i+1)
        decode_window = [decode_windows[i]]
        y_true_delay_cur, y_pred_delay_cur = MVPA_decoding(power_all, labels, epochs_downsampled.info, mvpa_out, all_time_points, decode_window, sfreq_downsampled, clf, cv,cat)
        if i ==0:
            y_true_delay = y_true_delay_cur
            y_pred_delay = y_pred_delay_cur
        else:
            y_true_delay = np.concatenate((y_true_delay,y_true_delay_cur))
            y_pred_delay = np.concatenate((y_pred_delay,y_pred_delay_cur))

    results_delay = confusion_matrix(y_true_delay, y_pred_delay, labels=category_labels, normalize='true')


    # plot the confusion matrix
    fig_title = ''
    fig_name = 'confusion_matrix'+fnames+'.png'

    f, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title(fig_title, fontsize=20)
    sns.heatmap(results_delay, vmin=0, vmax=1, cmap='YlGnBu', linewidths=0.2,
            annot=True, annot_kws={"fontsize":20}, cbar=True)
    ax.set_xticklabels(labels=categories, fontsize=20)
    ax.set_yticklabels(labels=categories, fontsize=20)
    ax.set_xlabel('Predicted category', fontsize=20)
    ax.set_ylabel('Actual category', fontsize=20)
    plt.savefig(outpath+'/'+sub + '_'+fig_name)
    # plt.show()

    results={}
    results['confusion_matrix'] = results_delay
    results['y_true'] = y_true_delay
    results['y_pred'] = y_pred_delay

    # save results to the disk
    f = open(mvpa_out, 'wb')
    pickle.dump(results, f)
    f.close





# # -------------------------classifier training------------------------- #


    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    cv = StratifiedShuffleSplit(n_splits=10, test_size=1/3)
    clf = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=10e5, multi_class='ovr'), params, cv=cv, scoring='roc_auc_ovr')  # make_pipeline(SVC(kernel='linear'))#LogisticRegression(solver='lbfgs', max_iter=10e5, multi_class='ovr')
    clf_trained = {}

    for i in np.arange(len(decode_windows)):
        print('process:', i+1)
        decode_window = [decode_windows[i]]
        clf_encoding_raw = clf_training(power_all, labels, epochs_downsampled.info, outpath+'/'+sub+'_clf_trained'+fnames+'.pickle', all_time_points,  decode_window, sfreq_downsampled, clf,cat)
        clf_trained[str(i)] = clf_encoding_raw


    f = open(outpath+'/'+sub+'_clf_trained'+fnames+'.pickle', 'wb')
    pickle.dump(clf_trained, f)
    f.close



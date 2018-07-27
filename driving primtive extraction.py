from scipy import io
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
import copy, os
import Normalization as norm
import pyhsmm
from pyhsmm.util.text import progprint_xrange

rowData = io.loadmat('data.mat')
data = rowData['Data']
amount,_ = np.shape(data)


#####################################
# define personalized plot function #
#####################################
def Myplot(labels, raw, durations):
    # raw data with primitive colors
    length_labels = len(labels)
    colorbase=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gold', 'gray', 'pink']
    count = 0
    for i in range(length_labels):
        j = durations[i]
        colordefine=labels[i]
        if j >= 3 or i == 0:
            plt.figure(1)
            plt.subplot(211)
            plt.plot(raw[count:count+j,0], raw[count:count+j,1], '.', color= colorbase[colordefine], ms=3)
            plt.plot(raw[count:count+j,3], raw[count:count+j,4], '.', color=colorbase[colordefine], ms=3)

            plt.subplot(212)
            t = np.arange(count, count+len(raw[count:count+j,2]), 1)
            plt.plot(t,raw[count:count+j,2],'.', color =colorbase[colordefine], ms=1)
            plt.plot(t,raw[count:count+j,5],'.', color =colorbase[colordefine], ms=1)
            colorold = colordefine

        else:
            plt.figure(1)
            plt.subplot(211)
            plt.plot(raw[count:count+j,0], raw[count:count+j,1], '.', color= colorbase[colorold], ms=3)
            plt.plot(raw[count:count+j,3], raw[count:count+j,4], '.', color=colorbase[colorold], ms=3)

            plt.subplot(212)
            t = np.arange(count, count+len(raw[count:count+j,2]), 1)
            plt.plot(t,raw[count:count+j,2],'.', color =colorbase[colorold], ms=1)
            plt.plot(t,raw[count:count+j,5],'.', color =colorbase[colorold], ms=1)

        count = count + j

    return

##########################
# THe MAIN Funtion start #
##########################
Nmax = 10

for i in range(amount):
    countforwholecycle = i
    # and some hyperparameters
    dictforMatsave = {}
    obs_dim = 6
    obs_hypparams = {'mu_0': np.zeros(obs_dim),
                     'sigma_0': np.eye(obs_dim),
                     'kappa_0': 0.25,
                     'nu_0': obs_dim + 2}
    dur_hypparams = {'alpha_0': 2 * 30,
                     'beta_0': 2}

    obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
    dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

    posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        alpha=6., gamma=6.,  # these can matter; see concentration-resampling.py
        init_state_concentration=6.,  # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns)
    print("the number of the cycle: %d" %countforwholecycle)

    data1 = data[countforwholecycle][0]
    print('the shape of data is',np.shape(data1))

    # Set the weak limit truncation level

    posteriormodel.add_data(data1,trunc=60) # duration truncation speeds things up when it's possible

    for idx in progprint_xrange(200):
        posteriormodel.resample_model()

    #posteriormodel.plot()

    # personalized outputs
    durations1 = posteriormodel.durations
    labels = posteriormodel.stateseqs_norep# stateseqs
    states1 = posteriormodel.stateseqs

    labels_removesmallsize = []
    for i in range(len(labels[0])):
        j = durations1[0][i]
        if i == 0:
            oldlabel=labels[0][i]
            labels_removesmallsize.append(labels[0][i])
        if j >= 3:
            if oldlabel != labels[0][i]:
                labels_removesmallsize.append(labels[0][i])
                oldlabel = labels[0][i]

    #calculate the label without repeat
    labelsnoRepeat=list(OrderedDict.fromkeys(labels_removesmallsize))


    oldStates = -1
    countforStates = 0
    durations = []
    states2 = states1[0]
    for m in range(len(states2)):
        if m == 0:
            oldStates = states2[m]
        if oldStates == states2[m]:
            countforStates = countforStates + 1
        if countforStates == 1 and m == len(states2) - 1:
            durations.append(countforStates)
        if states2[m] != oldStates:
            durations.append(countforStates)
            countforStates = 1
            oldStates = states2[m]
        if m == len(states2) - 1:
            durations.append(countforStates)


    ##################################################
    # Calculate the durations without the small size #
    ##################################################
    _, length_labels = np.shape(labels)

    length_labels_remov = len(labels_removesmallsize)
    print(length_labels_remov)
    durations_removesmallsize = []
    tempdurations = 0
    countforRemovesmallszieofDuration = 0
    labelsShorten = labels
    for k in range(length_labels_remov):
        tempdurations_ss = 0
        length_durations_s = len(durations_removesmallsize)
        for n in range(length_durations_s):
            tempdurations_ss = tempdurations_ss + durations_removesmallsize[n]
        tempdurations_ss_m = 0
        for m in range(length_labels):
            tempdurations_ss_m = tempdurations_ss_m + durations[m]
            if tempdurations_ss_m <= tempdurations_ss:
                continue
            if labels[0][m] == labels_removesmallsize[k]:
                tempdurations = tempdurations + durations[m]
                countforRemovesmallszieofDuration = countforRemovesmallszieofDuration + 1
            if durations[m] < 3:
                tempdurations = tempdurations + durations[m]
                countforRemovesmallszieofDuration = countforRemovesmallszieofDuration + 1
            if labels[0][m] != labels_removesmallsize[k] and durations[m] >= 3:
                countforRemovesmallszieofDuration = countforRemovesmallszieofDuration + 1
                durations_removesmallsize.append(tempdurations)
                tempdurations = 0
                break
            if m == length_labels-1:
                durations_removesmallsize.append(tempdurations)
                tempdurations = 0
                break

    ##############################
    # Calculate the mean and std #
    ##############################
    countforDurations = 0
    durationswithSamelabel={}
    primitive_centers = []
    primitive_stds = []
    for i in range(len(labelsnoRepeat)):
        index = np.where(states1 == labelsnoRepeat[i])
        durationswithSamelabel[i] = data1[index[1], :]
        durationsforMeanandStd=durationswithSamelabel[i]
        
        centers = []
        stds = []
        
        for iii in range(6):
            kmeans = np.mean(durationsforMeanandStd[:,iii], axis=0)

            centers.append(kmeans)
            std = np.std(durationsforMeanandStd[:,iii], axis=0)
            stds.append(std)
  
        primitive_stds.append(stds)
        primitive_centers.append(centers)


    #######################
    #   Plot the result   #
    #######################
    Myplot(labels[0], data1, durations)
    plt.show()

    #Save the result data for Mat
    durations_s = 'duration_%0.0f' %countforwholecycle
    durations_rem_s = 'duration_rem%0.0f' %countforwholecycle
    labels_s = 'labels_%0.0f' %countforwholecycle
    lableswithoutsparse_s = 'labesnosparse_%0.0f' %countforwholecycle
    labelsnon_s = 'labelsnon_%0.0f' %countforwholecycle
    states_s = 'states_%0.0f' %countforwholecycle
    centers_s='center_%0.0f' %countforwholecycle
    stds_s='std_%0.0f' %countforwholecycle
    dictforMatsave[durations_s] = durations
    dictforMatsave[durations_rem_s] = durations_removesmallsize
    dictforMatsave[labels_s] = labels[0]
    dictforMatsave[lableswithoutsparse_s] = labels_removesmallsize
    dictforMatsave[labelsnon_s] = labelsnoRepeat
    dictforMatsave[states_s] = states1[0]
    dictforMatsave[centers_s] = primitive_centers
    dictforMatsave[stds_s] = primitive_stds



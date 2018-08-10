from scipy import io
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
from dtw import dtwforDistance
from dtw import dtwforSpeed

rowData = io.loadmat('data.mat')
data = rowData['Data']
amount,_ = np.shape(data)

interData_pos = io.loadmat('intersectionData_pos') #load the intersection position information
intersectionData_pos = interData_pos['intersectionData_pos_sum'][0]

######################################
# Scaling the duration's length to N #
######################################
def Scaling(dataofDuration, N):
    'Normalized data into 200 points for each variable using interploration algorithm'
    'dimension of data is nxTxd, where n is the number of driving encounter, d is the dimension, T is the length of data'
    amount, _ = np.shape(dataofDuration)
    dim = 6
    data_rescale = np.zeros((N, dim))
    data_new = dataofDuration
    for j in range(dim):
        var = data_new[:, j]
        raw_len = len(var)
        raw_seq = np.linspace(0, raw_len, raw_len)
        new_seq = np.linspace(0, raw_len, N)
        data_rescale[:,j]=np.interp(new_seq, raw_seq, var)

    return data_rescale

#####################
# The MAIN FUNCTION #
#####################
durationsforAll = []
labelsofdurationsforAll = []
dataforKmeans = []
dataforKmeans_norm = []
countforDurations = 0


for i in range(len(intersectionData_pos)):
    countforwholecycle = intersectionData_pos[i]
    print('the cycle is %d' %countforwholecycle)
    data1 = data[countforwholecycle][0]
    print(np.shape(data1))
    dictforData = io.loadmat('output%0.0f' %countforwholecycle)
    dict = dictforData['dictforMatsave']
    
    durations_s = 'duration_%0.0f' %countforwholecycle
    durations_rem_s = 'duration_rem%0.0f' %countforwholecycle
    labels_s = 'labels_%0.0f' %countforwholecycle
    lableswithoutsparse_s = 'labesnosparse_%0.0f' %countforwholecycle
    labelsnon_s = 'labelsnon_%0.0f' %countforwholecycle
    states_s = 'states_%0.0f' %countforwholecycle
    centers_s='center_%0.0f' %countforwholecycle
    stds_s='std_%0.0f' %countforwholecycle

    durations = dict[durations_s][0][0]
    durations_rem = dict[durations_rem_s][0][0]
    labels = dict[labels_s][0][0]
    labels_removesmallsize = dict[lableswithoutsparse_s][0][0]
    labelsnoRepeat = dict[labelsnon_s][0][0]
    states = dict[states_s][0][0]
    centers = dict[centers_s][0][0]
    stds = dict[stds_s][0][0]
    states1 = states[0]
 

    ###############################
    # Scaling the durations to 50 #
    ###############################

    _, length_labels = np.shape(labels)
    dataforMatafterScalingandCost = {}
    count = 0
    countforAccMatrix = 0
    for i in range(len(durations_rem[0])):
        dataforMatafterScaling_s = 'dataforMatafterScaling%0.0f%0.0f' % (countforwholecycle, countforAccMatrix)
        j = durations_rem[0][i]
        dataofDuration = data1[count:count + j, :]
        dataAfterscaling = Scaling(dataofDuration, 50)  # N=50 (5S)
        dataforMatafterScalingandCost[dataforMatafterScaling_s] = dataAfterscaling
        dataforMatafterScaling1 = dataAfterscaling[:, 0:3]
        dataforMatafterScaling2 = dataAfterscaling[:, 3:6]


        ####################################
        # Calculate the cost matrix by DTW #
        ####################################
        dist_D, cost_D, acc_D, path_D, costMatrix_D = dtwforDistance(dataforMatafterScaling1, dataforMatafterScaling2)
        dist_S, cost_S, acc_S, path_S, costMatrix_S = dtwforSpeed(dataforMatafterScaling1, dataforMatafterScaling2)
        costMatrix_D_S = np.concatenate((cost_D, cost_S), axis=0)
        cost_D_max = cost_D.max()
        cost_S_max = cost_S.max()
        if cost_D_max == 0:
            cost_D_norm = cost_D
        else:
            cost_D_norm = cost_D / cost_D_max

        if cost_S_max == 0:
            cost_S_norm = cost_S
        else:
            cost_S_norm = cost_S / cost_S_max

        costMatrix_D_S_norm = np.concatenate((cost_D_norm, cost_S_norm), axis=0)

        dataforMatcostMatix_s = 'dataforMatcostMatrix%0.0f%0.0f' % (countforwholecycle, countforAccMatrix)
        dataforMatafterScalingandCost[dataforMatcostMatix_s] = costMatrix_D_S
        countforAccMatrix = countforAccMatrix + 1
        count = count + j

        dataforKmeans.append(costMatrix_D_S.reshape(1, -1)[0])
        dataforKmeans_norm.append(costMatrix_D_S_norm.reshape(1, -1)[0])
        countforDurations = countforDurations + 1

    # Save the drivingEncounter result
    io.savemat('output%0.0f' %countforwholecycle, {'dataforMatafterScalingandCost':dataforMatafterScalingandCost})

    #####################################
    # Record the label of the durations #
    #####################################

    for m in range(len(durations_rem[0])):
        labelsofdurationsforAll.append(countforwholecycle)

    #########################################
    # Calculate the length of the durations #
    #########################################

    for n in range(len(durations_rem[0])):
        tempDuration = durations_rem[0][n]
        durationsforAll.append(tempDuration)

#########################
# Save the data to .Mat #
#########################
io.savemat('abelsofDurations', {'labelsofDurations': labelsofdurationsforAll})
io.savemat('dataforKmeans', {'dataforKmeans': dataforKmeans})
io.savemat('dataforKmeans_norm', {'dataforKmeans_norm': dataforKmeans_norm})
io.savemat('durations1', {'durations':durationsforAll})

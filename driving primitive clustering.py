from scipy import io
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances

rowData = io.loadmat('data.mat')
data = rowData['Data']
amount,_ = np.shape(data)


costMatrixforkmeans = io.loadmat('ataforKmeans')
costKmeans = costMatrixforkmeans['dataforKmeans']
costMatrixforkmeans_norm = io.loadmat('ataforKmeans_norm')
costKmeans_norm = costMatrixforkmeans_norm['dataforKmeans_norm']

labelsforKmeans = []
harabaz_score = []
sil_coe = []
BC_score = []
WC_score = []


def My_harabaz_score(X, labels, n_labels):

    extra_disp, intra_disp = 0., 0.
    mean = np.mean(X, axis=0)
    n_samples, _ = X.shape
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)

    return (extra_disp/ (n_labels - 1.)), (intra_disp/(n_samples - n_labels))


for i in range(49):
    cluster_num = i + 2
    print('the cluster num is: %d' %cluster_num)

    # Set the K-Means input vector here #
    Kmeans = KMeans(n_clusters=cluster_num, random_state=None).fit(costKmeans_norm)

    labels = np.array(Kmeans.labels_)

    BC, WC = My_harabaz_score(costKmeans_norm, Kmeans.labels_, cluster_num)
    temp_score = metrics.calinski_harabaz_score(costKmeans_norm, Kmeans.labels_)
    print('the harabaz score is: %f' %temp_score)
    harabaz_score.append(temp_score)
    temp_score1 = metrics.silhouette_score(costKmeans_norm, Kmeans.labels_, metric='euclidean')
    sil_coe.append(temp_score1)
    print('the sil_coe is: %f' %temp_score1)
    BC_score.append(BC)
    WC_score.append(WC)
    print(BC)
    print(WC)
    print(labels)
    labelsforKmeans.append(labels)


sil_coe_diff = np.diff(sil_coe)
harabaz_score_diff = np.diff(harabaz_score)


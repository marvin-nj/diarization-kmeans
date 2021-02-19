import numpy as np

def labels_to_segments(labels, window):
    """
    ARGUMENTS:
     - labels:     a sequence of class labels (per time window)
     - window:     window duration (in seconds)

    RETURNS:
     - segments:   a sequence of segment's limits: segs[i,0] is start and
                   segs[i,1] are start and end point of segment i
     - classes:    a sequence of class flags: class[i] is the class ID of
                   the i-th segment
    """

    if len(labels)==1:
        segs = [0, window]
        classes = labels
        return segs, classes


    num_segs = 0
    index = 0
    classes = []
    segment_list = []
    cur_label = labels[index]
    while index < len(labels) - 1:
        previous_value = cur_label
        while True:
            index += 1
            compare_flag = labels[index]
            if (compare_flag != cur_label) | (index == len(labels) - 1):
                num_segs += 1
                cur_label = labels[index]
                segment_list.append((index * window))
                classes.append(previous_value)
                break
    segments = np.zeros((len(segment_list), 2))

    for i in range(len(segment_list)):
        if i > 0:
            segments[i, 0] = segment_list[i-1]
        segments[i, 1] = segment_list[i]
    return segments, classes

def normalize_features(features):
    """
    This function normalizes a feature set to 0-mean and 1-std.
    Used in most classifier trainning cases.

    ARGUMENTS:
        - features:    list of feature matrices (each one of them is a np
                       matrix)
    RETURNS:
        - features_norm:    list of NORMALIZED feature matrices
        - mean:        mean vector
        - std:        std vector
    """
    temp_feats = np.array([])

    for count, f in enumerate(features):
        if f.shape[0] > 0:
            if count == 0:
                temp_feats = f
            else:
                temp_feats = np.vstack((temp_feats, f))
            count += 1

    mean = np.mean(temp_feats, axis=0) + 1e-14
    std = np.std(temp_feats, axis=0) + 1e-14

    features_norm = []
    for f in features:
        ft = f.copy()
        for n_samples in range(f.shape[0]):
            ft[n_samples, :] = (ft[n_samples, :] - mean) / std
        features_norm.append(ft)
    return features_norm, mean, std


def features_to_matrix(features):
    """
    features_to_matrix(features)

    This function takes a list of feature matrices as argument and returns
    a single concatenated feature matrix and the respective class labels.

    ARGUMENTS:
        - features:        a list of feature matrices

    RETURNS:
        - feature_matrix:    a concatenated matrix of features
        - labels:            a vector of class indices
    """

    labels = np.array([])
    feature_matrix = np.array([])
    for i, f in enumerate(features):
        if i == 0:
            feature_matrix = f
            labels = i * np.ones((len(f), 1))
        else:
            feature_matrix = np.vstack((feature_matrix, f))
            labels = np.append(labels, i * np.ones((len(f), 1)))
    return feature_matrix, labels



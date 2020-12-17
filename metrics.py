import numpy as np
from sklearn.metrics import jaccard_score

def jaccard(pred, real):
    jaccard_s = jaccard_score(real.astype(int).flatten(), pred.astype(int).flatten(), pos_label=0)
    return jaccard_s

def accuracy(pred, true):
    dist = np.count_nonzero(pred != true) / (true.shape[0] * true.shape[1])
    return 1-dist



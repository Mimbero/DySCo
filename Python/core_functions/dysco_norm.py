import numpy as np


def dysco_norm(eigenvalues, what_norm):
    if what_norm == 1:
        norm = np.sum(np.abs(eigenvalues),axis=0)
    elif what_norm == 2:
        norm = np.sqrt(np.sum(eigenvalues ** 2,axis=0))
        # norm = np.sum(eigenvalues ** 2)
    elif what_norm == np.inf:
        norm = np.max(eigenvalues,axis=0)

    return norm
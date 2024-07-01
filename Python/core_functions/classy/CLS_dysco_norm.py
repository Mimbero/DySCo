import numpy as np

class EIDA_Norm:
    def __init__(self, what_norm):
        self.what_norm = what_norm

        return

    def norm(self, eigenvalues):
        if self.what_norm == 1:
            norm = np.sum(np.abs(eigenvalues),axis=0)
        elif self.what_norm == 2:
            norm = np.sqrt(np.sum(eigenvalues ** 2,axis=0))
            # norm = np.sum(eigenvalues ** 2)
        elif self.what_norm == np.inf:
            norm = np.max(eigenvalues,axis=0)

        return norm
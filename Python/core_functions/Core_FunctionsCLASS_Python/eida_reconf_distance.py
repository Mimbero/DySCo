import numpy as np
from numpy.linalg import norm

# NOTE THIS NEEDS TO BE TESTED AGAINST MATLAB !!!!!



class Eida_reconf_distance:
    def __init__(self, normalize):
        self.normalize = normalize

        return

    def eida_reconf_distance(self, matrix_a, matrix_b):
        matrix_a = matrix_a.copy()
        matrix_b = matrix_b.copy()

        n_eigen = matrix_a.shape[1]

        if self.normalize == True:

            for i in range(n_eigen):
                matrix_a[:, i] = matrix_a[:, i]/norm(matrix_a[:, i])
                matrix_b[:, i] = matrix_b[:, i]/norm(matrix_b[:, i])

        # Define minimatrix
        minimatrix = np.dot(matrix_a.T,matrix_b)
        distance = norm(minimatrix,ord='fro')

        return distance
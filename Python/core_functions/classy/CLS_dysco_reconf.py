import numpy as np
from numpy.linalg import norm


class Dysco_reconf_distance:
    def __init__(self):
        return

    def dysco_mode_alignment(self, matrix_a, matrix_b):
        """
        This function returns the measure of reconfiguration (rotation of the eigenvectors in the multidimspace)

        :return: reconfiguration distance
        """
        matrix_a = matrix_a.copy()
        matrix_b = matrix_b.copy()

        n_eigen = matrix_a.shape[1]

        for i in range(n_eigen):
            matrix_a[:, i] = matrix_a[:, i] / norm(matrix_a[:, i])
            matrix_b[:, i] = matrix_b[:, i] / norm(matrix_b[:, i])

        # Define minimatrix
        minimatrix = np.dot(matrix_a.T, matrix_b)
        frob_product = norm(minimatrix, ord='fro')

        distance = 2 * (n_eigen - frob_product)

        return distance
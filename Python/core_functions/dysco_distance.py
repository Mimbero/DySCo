import numpy as np


def dysco_distance(matrix_a, matrix_b, what_distance):
    with np.errstate(invalid='ignore'):

        matrix_a = matrix_a.copy()
        matrix_b = matrix_b.copy()

        n_eigen = matrix_a.shape[1]

    # Define minimatrix
        minimatrix = np.zeros((2 * n_eigen, 2 * n_eigen))

    # Fill diagonal with the squared norms of eigenvectors
        for i in range(n_eigen):
            minimatrix[i, i] = np.dot(matrix_a[:, i].T, matrix_a[:, i])
            minimatrix[n_eigen + i, n_eigen + i] = -np.dot(matrix_b[:, i].T, matrix_b[:, i])

    # Fill the rest with scalar products
        minimatrix_up_right = np.dot(matrix_a.T, matrix_b)
        minimatrix[0:n_eigen, n_eigen:2 * n_eigen] = minimatrix_up_right
        minimatrix[n_eigen:2 * n_eigen, 0:n_eigen] = -minimatrix_up_right.T


    # Compute eigenvalues
        if what_distance != 2:
            lambdas = np.linalg.eigvals(minimatrix)
            lambdas = np.real(lambdas)

        if what_distance == 1:
            distance = np.sum(np.abs(lambdas))

        elif what_distance == 2:
            # distance = np.sqrt(np.sum(lambdas ** 2))
        # Modify the distance calculation
            distance = np.sqrt(np.sum(np.diag(minimatrix) ** 2) - 2 * np.sum(minimatrix_up_right ** 2))
        else:
            distance = np.max(lambdas)

    return distance
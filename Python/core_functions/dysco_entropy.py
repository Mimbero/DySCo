import numpy as np


def dysco_entropy(eigenvalues):
    """
    Calculates von Neumann entropy starting from the eigenvalue timeseries.
    Each row is eigenvalue n. 1, 2, 3..., each column is a time point.

    Note that if the matrix has a lower rank than expected, and there's a row
    of null eigenvalues, they should be discarded.
    """
    # Ensure eigenvalues are a numpy array
    eigenvalues = np.array(eigenvalues)

    # Number of eigenvalues
    n_eigenvalues = eigenvalues.shape[0]

    # Normalize the eigenvalues using np.tile
    von_neumann = eigenvalues / np.tile(np.sum(eigenvalues, axis=0), (n_eigenvalues, 1))

    # Avoid log(0) by setting zero values to a small positive number
    von_neumann[von_neumann == 0] = np.finfo(float).eps

    # Calculate von Neumann entropy
    von_neumann = -np.sum(np.log(von_neumann) * von_neumann, axis=0)

    return von_neumann


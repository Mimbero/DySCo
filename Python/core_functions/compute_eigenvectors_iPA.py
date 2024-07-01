import numpy as np
from scipy.signal import hilbert


def compute_eigenvectors_ipa(timeseries):
    """
    This function computes the timecourse of the 2 eigenvectors.
    They are scaled by sqrt of eigenvalues in order to recompose the full matrix.

    :param timeseries: Must be a 2D numpy array. Each column is a signal, each row is a time point.
    :return: eigenvectors, eigenvalues
    """
    n_channels = timeseries.shape[1]
    n = timeseries.shape[0]

    # Compute the phase of the Hilbert transform for each signal
    timeseries = np.angle(hilbert(timeseries, axis=0))

    eigenvectors = np.zeros((n_channels, 2, n))  # 2 because in iPA, always have 2 eigenvectors
    eigenvalues = np.zeros(n)

    for t in range(n):
        # Compute the iPL matrix, and the c and s vectors
        c = np.cos(timeseries[t, :])
        s = np.sin(timeseries[t, :])

        # Analytical method to compute eigenvalues and eigenvectors
        sigma = np.dot(s, s)
        gamma = np.dot(c, c)
        xi = np.dot(c, s)
        delta = (gamma - sigma) ** 2 + 4 * xi ** 2
        B1 = ((sigma - gamma) + np.sqrt(delta)) / (2 * xi)
        B2 = ((sigma - gamma) - np.sqrt(delta)) / (2 * xi)

        v1 = c + B1 * s
        v2 = c + B2 * s
        v1 /= np.linalg.norm(v1)  # Normalize eigenvectors
        v2 /= np.linalg.norm(v2)

        # Find eigenvalues
        lambda1 = gamma + B1 * xi
        lambda2 = gamma + B2 * xi
        eigenvalues[t] = lambda1  # Save the largest eigenvalue

        # Scale eigenvectors by eigenvalues
        v1 *= np.sqrt(lambda1)
        v2 *= np.sqrt(lambda2)

        # Switch eigenvectors such that v1 is the leading one
        if lambda2 > lambda1:
            v1, v2 = v2, v1
            eigenvalues[t] = lambda2

        # Ensure eigenvector time series are positively correlated
        if t > 0:
            if np.corrcoef(v1, eigenvectors[:, 0, t-1])[0, 1] < 0:
                v1 = -v1
            if np.corrcoef(v2, eigenvectors[:, 1, t-1])[0, 1] < 0:
                v2 = -v2

        # Stack eigenvectors
        eigenvectors[:, 0, t] = v1
        eigenvectors[:, 1, t] = v2

    # Discard first and last timepoint as they are not meaningful
    eigenvectors = eigenvectors[:, :, 1:-1]
    eigenvalues = eigenvalues[1:-1]

    return eigenvectors, eigenvalues

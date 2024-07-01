import numpy as np
from tqdm import tqdm
from scipy.linalg import eigh


def compute_eigs_cov(timeseries, n_eigen, half_window_size):

    """
    this function computes the pearson correlation matrices using a sliding window.
     then calculates eigenvectors using the formula presented in the paper

    :param timeseries should be TxN, so every row is a timepoint and every column is a dimension.
    :param half_window_size: half window size for sliding window
    :param n_eigen: number of eigenvectors
    :return: eigenvectors, eigenvalues
    """

    if n_eigen > 2 * half_window_size:
        raise ValueError('Number of requested eigenvectors is too large')
    t, n = timeseries.shape
    total_iterations = t - 2 * half_window_size
    progress_bar_eigs = tqdm(total=total_iterations, desc="Calculating eigenvectors and eigenvalues:")
    eigenvectors = np.zeros((t - 2 * half_window_size, n, n_eigen))

    eigenvalues = np.zeros((n_eigen, t - 2 * half_window_size))

    for i in range(t - 2 * half_window_size):
        truncated_timeseries = timeseries[i:i + 2 * half_window_size, :]

        z_scored_truncated = (truncated_timeseries - np.mean(truncated_timeseries, axis=0)) / np.std(
            truncated_timeseries, axis=0, ddof=1)

        normalizing_factor = z_scored_truncated.shape[0] - 1
        z_scored_truncated = (1 / np.sqrt(normalizing_factor)) * z_scored_truncated
        mini_matrix = z_scored_truncated @ z_scored_truncated.T
        ns = len(mini_matrix)

        eigenvalues_t, eigenvectors_t = eigh(mini_matrix, subset_by_index=[ns - n_eigen, ns - 1], overwrite_a=True,
                                             check_finite=False)
        eigenvectors_t = np.flip(eigenvectors_t, axis=1)
        eigenvalues_t = np.flip(eigenvalues_t, axis=0)
        eigenvalues[:, i] = eigenvalues_t
        eigenvectors[i, :, :] = np.dot(z_scored_truncated.T, eigenvectors_t)

        for j in range(n_eigen):
            eigenvectors[i, :, j] = eigenvectors[i, :, j] / np.linalg.norm(eigenvectors[i, :, j])
            eigenvectors[i, :, j] = eigenvectors[i, :, j] * np.sqrt(eigenvalues[j, i])

        progress_bar_eigs.update(1)

    progress_bar_eigs.close()
    return eigenvectors, eigenvalues
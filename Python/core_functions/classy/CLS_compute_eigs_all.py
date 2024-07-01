import numpy as np
from tqdm import tqdm
from scipy.linalg import eigh
from scipy.signal import hilbert


class Compute_Eigs:

    def __init__(self, timeseries, n_eigen, half_window_size):
        self.timeseries = timeseries
        self.n_eigen = n_eigen
        self.half_window_size = half_window_size

        return

    def compute_eigs_cov(self):

        if self.n_eigen > 2 * self.half_window_size:
            raise ValueError('Number of requested eigenvectors is too large')
        t, n = self.timeseries.shape
        total_iterations = t - 2 * self.half_window_size
        progress_bar_eigs = tqdm(total=total_iterations, desc="Calculating eigenvectors and eigenvalues:")
        eigenvectors = np.zeros((t - 2 * self.half_window_size, n, self.n_eigen))

        eigenvalues = np.zeros((self.n_eigen, t - 2 * self.half_window_size))

        for i in range(t - 2 * self.half_window_size):
            truncated_timeseries = self.timeseries[i:i + 2 * self.half_window_size, :]

            z_scored_truncated = (truncated_timeseries - np.mean(truncated_timeseries, axis=0)) / np.std(
                truncated_timeseries, axis=0, ddof=1)

            normalizing_factor = z_scored_truncated.shape[0] - 1
            z_scored_truncated = (1 / np.sqrt(normalizing_factor)) * z_scored_truncated
            mini_matrix = z_scored_truncated @ z_scored_truncated.T
            ns = len(mini_matrix)

            eigenvalues_t, eigenvectors_t = eigh(mini_matrix, subset_by_index=[ns - self.n_eigen, ns - 1], overwrite_a=True,
                                                 check_finite=False)
            eigenvectors_t = np.flip(eigenvectors_t, axis=1)
            eigenvalues_t = np.flip(eigenvalues_t, axis=0)
            eigenvalues[:, i] = eigenvalues_t
            eigenvectors[i, :, :] = np.dot(z_scored_truncated.T, eigenvectors_t)

            for j in range(self.n_eigen):
                eigenvectors[i, :, j] = eigenvectors[i, :, j] / np.linalg.norm(eigenvectors[i, :, j])
                eigenvectors[i, :, j] = eigenvectors[i, :, j] * np.sqrt(eigenvalues[j, i])

            progress_bar_eigs.update(1)

        progress_bar_eigs.close()
        return eigenvectors, eigenvalues

    def compute_eigs_corr(self):

        if self.n_eigen > 2 * self.half_window_size:  # maximum rank of corr matrix
            raise ValueError('Number of requested eigenvectors is too large')

        t, n = self.timeseries.shape
        total_iterations = t - 2 * self.half_window_size
        progress_bar_eigs = tqdm(total=total_iterations, desc="Calculating eigenvectors and eigenvalues:")

        eigenvectors = np.zeros((t - 2 * self.half_window_size, n, self.n_eigen))

        eigenvalues = np.zeros((self.n_eigen, t - 2 * self.half_window_size))

        for i in range(t - 2 * self.half_window_size):
            truncated_timeseries = self.timeseries[i:i + 2 * self.half_window_size + 1, :]
            zscored_truncated = (truncated_timeseries - np.mean(truncated_timeseries, axis=0)) / np.std(
                truncated_timeseries, axis=0)

            normalizing_factor = truncated_timeseries.shape[0] - 1
            zscored_truncated /= np.sqrt(normalizing_factor)

            # correlation matrix
            minimatrix = zscored_truncated @ zscored_truncated.T
            ns = len(minimatrix)

            # Gathering eigenvectors (columns of v) and eigenvalues (diagonal of a) of minimatrix.
            # Eigenvalues will be them. Eigenvectors will be the obtained coefficients x original vectors.
            eigenvalues_t, eigenvectors_t = eigh(minimatrix, subset_by_index=[ns - self.n_eigen, ns - 1],
                                                 overwrite_a=True,
                                                 check_finite=False)
            eigenvectors_t = np.flip(eigenvectors_t, axis=1)
            eigenvalues_t = np.flip(eigenvalues_t, axis=0)
            eigenvalues[:, i] = eigenvalues_t
            eigenvectors[i, :, :] = np.dot(zscored_truncated.T, eigenvectors_t)

            # normalizing the eigenvectors by sqrt of eigenvalues
            for j in range(self.n_eigen):
                eigenvectors[i, :, j] = eigenvectors[i, :, j] / np.linalg.norm(eigenvectors[i, :, j])
                eigenvectors[i, :, j] = eigenvectors[i, :, j] * np.sqrt(eigenvalues[j, i])

            progress_bar_eigs.update(1)

        return eigenvectors, eigenvalues

    def compute_eigenvectors_ipa(self):
        """
        This function computes the timecourse of the 2 eigenvectors.
        They are scaled by sqrt of eigenvalues in order to recompose the full matrix.

        :param timeseries: Must be a 2D numpy array. Each column is a signal, each row is a time point.
        :return: eigenvectors, eigenvalues
        """
        n_channels = self.timeseries.shape[1]
        n = self.timeseries.shape[0]

        # Compute the phase of the Hilbert transform for each signal
        timeseries = np.angle(hilbert(self.timeseries, axis=0))

        eigenvectors = np.zeros((n_channels, 2, n))  # 2 because in iPA I always have 2 eigenvectors
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
                if np.corrcoef(v1, eigenvectors[:, 0, t - 1])[0, 1] < 0:
                    v1 = -v1
                if np.corrcoef(v2, eigenvectors[:, 1, t - 1])[0, 1] < 0:
                    v2 = -v2

            # Stack eigenvectors
            eigenvectors[:, 0, t] = v1
            eigenvectors[:, 1, t] = v2

        # Discard first and last timepoint as they are not meaningful
        eigenvectors = eigenvectors[:, :, 1:-1]
        eigenvalues = eigenvalues[1:-1]

        return eigenvectors, eigenvalues

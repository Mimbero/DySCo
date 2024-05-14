import numpy as np
from tqdm import tqdm
from scipy.linalg import eigh
from scipy.signal import tukey
class Compute_Eigs:

    def __init__(self, timeseries, n_eigen, half_window_size):
        self.timeseries = timeseries
        self.n_eigen = n_eigen
        self.half_window_size = half_window_size

        return

    def compute_eigs_cov(self):

        weighted_res = False

        if self.n_eigen > 2 * self.half_window_size:
            raise ValueError('Number of requested eigenvectors is too large')
        t, n = self.timeseries.shape
        total_iterations = t - 2 * self.half_window_size
        progress_bar_eigs = tqdm(total=total_iterations, desc="Calculating eigenvectors and eigenvalues:")
        eigenvectors = np.zeros((t - 2 * self.half_window_size, n, self.n_eigen))

        eigenvalues = np.zeros((self.n_eigen, t - 2 * self.half_window_size))

        for i in range(t - 2 * self.half_window_size):
            truncated_timeseries = self.timeseries[i:i + 2 * self.half_window_size, :]

            # sigma = 3
            # weighted_result = self.apply_weighted_window(truncated_timeseries, sigma)

            if weighted_res:
                z_scored_truncated = (weighted_result - np.mean(weighted_result, axis=0)) / np.std(
                    weighted_result, axis=0, ddof=1)
            else:
                z_scored_truncated = (truncated_timeseries - np.mean(truncated_timeseries, axis=0)) / np.std(
                    truncated_timeseries, axis=0, ddof=1)

            normalizing_factor = z_scored_truncated.shape[0] - 1
            z_scored_truncated = (1 / np.sqrt(normalizing_factor)) * z_scored_truncated
            mini_matrix = z_scored_truncated @ z_scored_truncated.T
            ns = len(mini_matrix)

            # The Outputs from this are differnt to the matlab version of Eigs??
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

    def gaussian_weight(self, x, sigma):
        return np.exp(-0.5 * (x / sigma) ** 2)

    def tukey_window(self, num_points, alpha):
        return np.sqrt(tukey(num_points, alpha))

    def apply_weighted_window(self, data, sigma):
        tukey = False
        gauss = True
        alpha = 0.1
        num_time_points, num_measures = data.shape
        weighted_data = np.zeros_like(data)

        # Calculate weights for a symmetric window
        center_index = (num_time_points - 1) / 2
        gauss_weights = self.gaussian_weight(np.arange(num_time_points) - center_index, sigma)
        # tukey_weights = tukey(num_points, alpha)
        tukey_weights = self.tukey_window(num_time_points, alpha)

        if tukey:
            for i in range(num_time_points):
                weighted_data[i, :] = data[i, :] * tukey_weights[i]

        if gauss:
            for i in range(num_time_points):
                weighted_data[i, :] = data[i, :] * gauss_weights[i]

        return weighted_data
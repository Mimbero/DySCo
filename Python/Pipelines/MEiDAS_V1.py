import numpy as np
from scipy.linalg import eigh
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import cProfile
import tkinter as tk
from tkinter import ttk
from matplotlib.animation import FuncAnimation
import nibabel as nb
from multiprocessing import Pool
import threading
from scipy import stats


path_full = '/Users/oliversherwood/Documents/Academia/PhD/PhD_Main/AutoNeuro/Python_Leida_test/'
with h5py.File('/Users/oliversherwood/Documents/Academia/PhD/PhD_Main/AutoNeuro/Python_Leida_test/BOLD_AD_DC.h5', 'r') as file:
    dataset = file['numeric_data']
    numpy_array = np.array(dataset)
transposed_array = numpy_array.transpose(0, 2, 1)
data_full = transposed_array


def run_script(data_full, n_eigen, half_window_size, run_rebuild_and_plot, what_norm, what_distance, are_you_sure):
    brain_load = load_hcp_data()
    brain_load = brain_load.T
    zero_columns = np.all(brain_load == 0, axis=0)
    filtered_array = brain_load[:, ~zero_columns]
    brain = filtered_array
    #
    # # If reduce Dimensionality
    # # brain = reduce_dims(brain)
    #
    # singlesub = data_full[:, 1, :]
    # reshaped_data = singlesub.reshape((296, 90))

    inst = EIDA(brain, n_eigen, half_window_size, what_norm, what_distance)
    eig_vec, eig_val = inst.compute_eigs()
    if run_rebuild_and_plot:
        inst.rebuild_and_plot(eig_vec, eig_val, run_rebuild_and_plot, are_you_sure)

    print("Finished")
    # for i in range(data_full.shape[1]):
    #     singlesub = data_full[:, i, :]
    #     reshaped_data = singlesub.reshape((296, 90))
    #     # cProfile.run('inst.compute_eigs()')
    #     # cProfile.run('inst.rebuild_and_plot(eig_vec,eig_val)')
    #     inst = EIDA(reshaped_data, n_eigen, half_window_size, what_norm, what_distance)
    #     eig_vec, eig_val = inst.compute_eigs()
    #     animation = inst.rebuild_and_plot(eig_vec, eig_val, run_rebuild_and_plot)
    #     std_dev_norm = inst.calculate_metastablity()


def surf_data_from_cifti(data, axis, surf_name):
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:  # Just looking for a surface
            data = data.T[data_indices]  # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex  # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")


def load_hcp_data():
    cifti = nb.load(
        '/Users/oliversherwood/Documents/100307/MNINonLinear/Results/tfMRI_WM_LR/tfMRI_WM_LR_Atlas_MSMAll.dtseries.nii')
    cifti_data = cifti.get_fdata(dtype=np.float32)
    cifti_hdr = cifti.header
    nifti_hdr = cifti.nifti_header

    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
    left_brain = surf_data_from_cifti(cifti_data, axes[1], 'CIFTI_STRUCTURE_CORTEX_LEFT')
    right_brain = surf_data_from_cifti(cifti_data, axes[1], 'CIFTI_STRUCTURE_CORTEX_RIGHT')

    return left_brain

def reduce_dims(brain):
    num_columns = brain.shape[1]
    num_columns_to_sample = int(0.1*num_columns)  # Number of columns to sample

    # Generate random indices for sampling without replacement
    random_column_indices = np.random.choice(num_columns, size=num_columns_to_sample, replace=False)

    # Sample columns using the generated indices
    red_brain = brain[:, random_column_indices]

    return red_brain


def rebuilt_parallel(args):
    i, eig_vec = args
    rebuilt_matrix = eig_vec[i, :, :] @ eig_vec[i, :, :].T
    print("Built Matrix - " + str(i))


def pool_handler(eig_vec):

    # THIS DOESN'T WORK YET
    value_to_process = range(405)
    # p = Pool(2)
    # p.map(rebuilt_parallel, zip(value_to_process, eig_vec))

    threads = []
    values_to_process = zip(value_to_process, eig_vec)
    try:
        for i, data in enumerate(value_to_process):
            thread = threading.Thread(target=rebuilt_parallel, args=((i, data),))
            threads.append(thread)
            thread.start()
    finally:

    # Wait for all threads to complete
        for thread in threads:
            thread.join()


class EIDA:
    def __init__(self, timeseries, n_eigen, half_window_size, what_norm, what_distance): 
        self.timeseries = timeseries
        self.n_eigen = n_eigen
        self.half_window_size = half_window_size
        self.what_norm = what_norm
        self.norm_values = []
        self.what_distance = what_distance

        return

    def compute_eigs(self):

        if self.n_eigen > 2*self.half_window_size:
            raise ValueError('Number of requested eigenvectors is too large')
        t, n = self.timeseries.shape
        total_iterations = t-2*self.half_window_size
        progress_bar_eigs = tqdm(total=total_iterations, desc="Calculating eigenvectors and eigenvalues:")
        eigenvectors = np.zeros((t-2*self.half_window_size, n, self.n_eigen))

        eigenvalues = np.zeros((self.n_eigen, t-2*self.half_window_size))

        for i in range(t-2*self.half_window_size):

            truncated_timeseries = self.timeseries[i:i+2*self.half_window_size, :]

            # z_scored_truncated = (truncated_timeseries - np.mean(truncated_timeseries, axis=0)) / np.std(
            #     truncated_timeseries, axis=0, ddof=1)
            z_scored_truncated = stats.zscore(truncated_timeseries)

            normalizing_factor = z_scored_truncated.shape[0] - 1
            z_scored_truncated = (1 / np.sqrt(normalizing_factor)) * z_scored_truncated

            # This could be wrong perhaps it is multiplied not @
            mini_matrix = z_scored_truncated @ z_scored_truncated.T
            # sum_squared_elements = np.sum(mini_matrix ** 2)
            # print("Sum of squared elements - Miniatrix:", sum_squared_elements)
            ns = len(mini_matrix)

            # The Outputs from this are differnt to the matlab version of Eigs??
            eigenvalues_t, eigenvectors_t = eigh(mini_matrix, subset_by_index=[ns-self.n_eigen, ns-1], overwrite_a=True, check_finite=False)
            eigenvectors_t = np.flip(eigenvectors_t, axis=1)
            eigenvalues_t = np.flip(eigenvalues_t, axis=0)
            eigenvalues[:, i] = eigenvalues_t
            eigenvectors[i, :, :] = np.dot(z_scored_truncated.T, eigenvectors_t)

            for j in range(self.n_eigen):
                eigenvectors[i, :, j] = eigenvectors[i, :, j] / np.linalg.norm(eigenvectors[i, :, j])
                eigenvectors[i, :, j] = eigenvectors[i, :, j] * np.sqrt(eigenvalues[j, i])

            progress_bar_eigs.update(1)

        # NORM
        # eigval_norm = []
        # for norm_time in range(t-2*self.half_window_size):
        #     eig_val_time_norm = self.norm(eigenvalues[:, norm_time])
        #     eigval_norm.append(eig_val_time_norm)
        #
        #
        # # Calculating Distance - DISTANCE
        # distances_eida = []
        # tau = 15
        # for eig_time in range(t-2*self.half_window_size):
        #     if eig_time > tau:
        #         distance_eida = self.eida_distance(eigenvectors[eig_time, :, :], eigenvectors[eig_time-tau, :, :])
        #         distances_eida.append(distance_eida)

            # time_norm = self.norm(eigenvalues_t)
            # self.norm_values.append(time_norm)

        progress_bar_eigs.close()
        return eigenvectors, eigenvalues

    def rebuild_and_plot(self, eig_vec, eig_val, run_rebuild_and_plot, are_you_sure):
        # t, n = self.timeseries.shape
        t, n, e = eig_vec.shape
        total_iterations = t

        # Create a figure for the animation
        if are_you_sure:
            fig, axs = plt.subplots(2, 2)
            fig.suptitle("Comparison Plots")
            ims = []
        else:
            fig, axs = plt.subplots(1, 1)
            fig.suptitle("Rebuilt Plot")
            ims = []

        def update_plot(i):

            if are_you_sure == 1:
                axs[0, 0].clear()
                axs[0, 0].imshow(plots[i][0], interpolation='nearest')
                axs[0, 0].set_title('Empirical Data')

                axs[0, 1].clear()
                axs[0, 1].imshow(plots[i][1], interpolation='nearest')
                axs[0, 1].set_title('Rebuilt with MEiDAS')

                axs[1, 0].clear()
                axs[1, 0].scatter(plots[i][2][0], plots[i][2][1])
                axs[1, 0].set_title('MEiDAS eigenvalue versus empirical')

                axs[1, 1].clear()
                axs[1, 1].imshow(plots[i][3], interpolation='nearest')
                axs[1, 1].set_title('MEiDAS eigenvector versus empirical')

                plt.suptitle(f"Comparison Plots - Iteration {i + 1}/{total_iterations}")

            else:
                axs[0, 0].clear()
                axs[0, 0].imshow(plots[i][0], interpolation='nearest')
                axs[0, 0].set_title('Empirical Data')

            return axs

        progress_bar = tqdm(total=total_iterations, desc="Rebuilding and Plotting:")

        plots = []
        fro_dist = []
        for i in range(t):
            lower = max(i - self.half_window_size, 0)
            upper = min(i + self.half_window_size + 1, t)
            truncated_timeseries = self.timeseries[lower:upper, :]

            # if __name__ == "__main__":
            #     pool_handler(eig_vec)

            if are_you_sure:
                literature_corr = np.corrcoef(truncated_timeseries, rowvar=False)
                # print("Value T - " + str(i))
            # sum_squared_elements = np.sum(literature_corr ** 2)
            # print("Sum of squared elements - Empirical:", sum_squared_elements)
            # This takes excessive time with full matrix
                ns = len(literature_corr)
                l_lit, v_lit = eigh(literature_corr, subset_by_index=[ns - self.n_eigen, ns - 1],
                                overwrite_a=True, check_finite=False)
                v_lit = np.flip(v_lit, axis=1)
                l_lit = np.flip(l_lit, axis=0)

                rebuilt_matrix = eig_vec[i, :, :] @ eig_vec[i, :, :].T

            # sum_squared_elements = np.sum(rebuilt_matrix ** 2)
            # print("Sum of squared elements - Rebuilt:", sum_squared_elements)
                for r in range(self.n_eigen):
                    eig_vec[i, :, r] /= np.sqrt(eig_val[r, i])

                # Append the plots to the 'plots' list
                plots.append((literature_corr, rebuilt_matrix, (eig_val[:, i], l_lit), eig_vec[i, :, :].T @ v_lit))
            else:
                rebuilt_matrix = eig_vec[i, :, :] @ eig_vec[i, :, :].T
                plots.append(rebuilt_matrix)
            progress_bar.update(1)
            sub_i_eigvect = eig_vec[i, :, :]

            # Decide, when/how to reimplement
            # norm = self.norm(eig_val)
            # difference = self.frobenius_norm_dist(literature_corr, rebuilt_matrix)
            # fro_dist.append(difference)

            # distance_eida = self.eida_distance(sub_i_eigvect, v_lit, self.what_distance)
        progress_bar.close()

        if run_rebuild_and_plot:
            print("Playing Plot Animation")
            ani = FuncAnimation(fig, update_plot, frames=t, repeat=False, blit=False)
            plt.show()

        return

    def frobenius_norm_dist(self, matrix_a, matrix_b):
        difference = np.linalg.norm(matrix_a - matrix_b, 'fro')
        distance = np.sqrt(np.sum(np.diag(matrix_a) ** 2) - 2 * np.sum(matrix_b ** 2))
        return difference

    def eida_distance(self, matrix_a, matrix_b):

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
        lambdas = np.linalg.eigvals(minimatrix)
        lambdas = np.real(lambdas)

        if self.what_distance == 1:
            distance = np.sum(np.abs(lambdas))
        elif self.what_distance == 2:
            distance = np.sqrt(np.sum(lambdas ** 2))
        # Modify the distance calculation
            distance = np.sqrt(np.sum(np.diag(minimatrix) ** 2) - 2 * np.sum(minimatrix_up_right ** 2))
        else:
            distance = np.max(lambdas)

        return distance

    def norm(self, eigenvalues):

        if self.what_norm == 1:
            norm = np.sum(np.abs(eigenvalues))
        elif self.what_norm == 2:
            norm = np.sqrt(np.sum(eigenvalues ** 2))
            # norm = np.sum(eigenvalues ** 2)
        elif self.what_norm == np.inf:
            norm = np.max(eigenvalues)

        return norm

    def calculate_metastablity(self):

        std_dev = np.std(self.norm_values)

        print('Metastabilty' + str(std_dev))

        return std_dev



class MEIDAS_GUI:
    def __init__(self, root):

        self.root = root
        self.root.title("GUI for EIDA Class")

        ttk.Label(root, text="N - Eigenvectors:").grid(row=0, column=0)
        self.param1_entry = ttk.Entry(root)
        self.param1_entry.grid(row=0, column=1)

        ttk.Label(root, text="Half Window Size:").grid(row=1, column=0)
        self.param2_entry = ttk.Entry(root)
        self.param2_entry.grid(row=1, column=1)

        ttk.Label(root, text="What Norm:").grid(row=2, column=0)
        self.param3_entry = ttk.Entry(root)
        self.param3_entry.grid(row=2, column=1)

        ttk.Label(root, text="What Distance:").grid(row=3, column=0)
        self.param4_entry = ttk.Entry(root)
        self.param4_entry.grid(row=3, column=1)

        self.run_rebuild_and_plot = tk.BooleanVar()
        run_rebuild_and_plot_checkbutton = ttk.Checkbutton(root, text="Run Rebuild and Plot", variable=self.run_rebuild_and_plot)
        run_rebuild_and_plot_checkbutton.grid(row=4, columnspan=2, pady=5)

        self.are_you_sure = tk.BooleanVar()
        are_you_sure_checkbutton = ttk.Checkbutton(root, text="Computationally Intensive - *Caution* - Are you Sure?", variable=self.are_you_sure)
        are_you_sure_checkbutton.grid(row=5, columnspan=2, pady=10)

        run_button = ttk.Button(root, text="Run Script", command=self.run_script)
        run_button.grid(row=6, columnspan=5, pady=10)

    def run_script(self):
        n_eigen = int(self.param1_entry.get())
        half_window_size = int(self.param2_entry.get())
        run_rebuild_and_plot = self.run_rebuild_and_plot.get()
        are_you_sure = self.are_you_sure.get()
        what_norm = int(self.param3_entry.get())
        what_distance = int(self.param4_entry.get())

        try:
            run_script(data_full, n_eigen, half_window_size, run_rebuild_and_plot, what_norm, what_distance, are_you_sure)
        except ValueError:
            print("Invalid input. Please enter numeric values for parameters.")


root = tk.Tk()
gui = MEIDAS_GUI(root)
root.mainloop()

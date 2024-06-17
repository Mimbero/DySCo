import os
import numpy as np
import nibabel as nib
import nibabel as nb
from scipy.io import loadmat
import random

import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import threading
import warnings
import scipy

current_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
core_functions_path = os.path.join(project_dir, 'Python', 'core_functions', 'Core_FunctionsCLASS_Python')
sys.path.append(core_functions_path)

from compute_eigs import Compute_Eigs
from eida_distance import Eida_distance
from eida_reconf_distance import Eida_reconf_distance
from compute_norm import EIDA_Norm


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


def save_subject_data(subject_id, half_window_sz, n_eigs, eig_vect, eig_val):
    # subject_folder = f'/Users/oliversherwood/Documents/CODE/MEIDAS_MAIN/Data_HCPSubjects/{subject_id}/'
    # subject_folder = f'/Users/oliversherwood/Documents/CODE/Data_HCPSubjects_ALLEIGs_SLIDE/{subject_id}/'
    subject_folder = f'/Users/oliversherwood/Documents/BMIS_EPI_DC_05_06_rem10/Data_EPI_WS-{half_window_sz}__eigen-{n_eigs}/{subject_id}/'
    # subject_folder = f'/Users/oliversherwood/Documents/CODE/DySCo/Saved_out/SENSITIVTIY_09_06/Data_HCP_WS-{half_window_sz}__eigen-{n_eigs}/{subject_id}/'

    # Create the subject folder if it doesn't exist
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)

    # Save eigenvectors and eigenvalues in the subject folder
    np.save(os.path.join(subject_folder, f'eig_vect{subject_id}.npy'), eig_vect)
    np.save(os.path.join(subject_folder, f'eig_val{subject_id}.npy'), eig_val)


def extract_subj_id(subject_txt, subj_partition, n_part):
    with open(subject_txt, 'r') as file:
        subject_names = file.read().splitlines()

    subj_names = subject_names

    if subj_partition:
        subj_names = random.sample(subj_names, n_part)
        # subj_names = ['101410']
        # # subj_names = ['106521']
        # # subj_names = ['102614']


    return subj_names


def process_nifti(file_path, n_eigs, window_shape, window_sizes, subject_id):
    # Load NIfTI file
    cifti = nb.load(file_path)
    cifti_data = cifti.get_fdata(dtype=np.float32)
    cifti_hdr = cifti.header
    nifti_hdr = cifti.nifti_header

    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
    left_brain = surf_data_from_cifti(cifti_data, axes[1], 'CIFTI_STRUCTURE_CORTEX_LEFT')
    right_brain = surf_data_from_cifti(cifti_data, axes[1], 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    brain_load = left_brain

    brain_load = brain_load.T
    zero_columns = np.all(brain_load == 0, axis=0)
    filtered_array = brain_load[:, ~zero_columns]
    brain = filtered_array

    # scipy.io.savemat(f'//Brain_Mat/brain_{subject_id}.mat', {"array": brain})

    if window_shape == "Rect":
        eig_vec, eig_val = slide_window_rect(n_eigs, window_sizes, brain, subject_id)

    return eig_vec, eig_val


def process_mat(file_path, n_eigs, window_shape, window_sizes, subject_id):
    mat_data = loadmat(file_path)
    # Assuming 'brain' is the variable name inside the .mat file
    brain_struct = mat_data['data']
    # brain_struct = mat_data['currentRow']

    # reshaped_data = []
    #
    # for i in range(brain_struct.shape[1]):  # Assuming the struct is 1x90
    #     brain = brain_struct[0, i].flatten()  # Each cell is 1x296 array
    #     reshaped_data.append(brain[:296])  # Truncate each array to 1x275

    # brain_struct = brain_struct[:296, :]

    # brain_struct = brain_struct[:296, :]

    if "HC" in file_path:
        brain_struct = brain_struct[10:, :]
    else:
        brain_struct = brain_struct[6:, :]

    combined_brain = np.array(brain_struct)
    # Combine reshaped data into a 90x275 array
    # combined_brain = np.array(reshaped_data)
    # combined_brain = combined_brain.transpose()

    if window_shape == "Rect":
        eig_vec, eig_val = slide_window_rect(n_eigs, window_sizes, combined_brain)
    # Add other conditions and processing as needed
    return eig_vec, eig_val


def slide_window_rect(n_eigs, window_sizes, brain_load, subject_id):

    if not isinstance(window_sizes, int):
        for half_window_sz in window_sizes:
            inst = Compute_Eigs(brain_load, n_eigs, half_window_sz)
            eig_vec, eig_val = inst.compute_eigs_cov()
            save_subject_data(subject_id, half_window_sz, n_eigs, eig_vec, eig_val)
    else:
        inst = Compute_Eigs(brain_load, n_eigs, window_sizes)
        eig_vec, eig_val = inst.compute_eigs_cov()
        save_subject_data(subject_id, window_sizes, n_eigs, eig_vec, eig_val)

    return eig_vec, eig_val


def compute_fcd(i, j, eigvect, eigs_inst, eigs_inst_reconf, total_i):
    matrix_a = eigvect[i, :, :]
    matrix_b = eigvect[j, :, :]
    fcd_ij = eigs_inst.eida_distance(matrix_a, matrix_b)
    fcd_reconf_ij = eigs_inst_reconf.eida_reconf_distance(matrix_a, matrix_b)

    # global progress, prev_percentage
    # progress = i
    # percentage = (progress / total_i) * 100
    # if int(percentage) > prev_percentage:
    #     sys.stdout.write(f"\rProgress: {int(percentage)}%")
    #     sys.stdout.flush()
    #     # print(f"\rProgress: {int(percentage)}%")
    #     prev_percentage = int(percentage)

    return i, j, fcd_ij, fcd_reconf_ij


def fcd_calc_para(eigvect, T):
    fcd = np.zeros((T, T))
    fcd_reconf = np.zeros((T, T))
    eigs_inst = Eida_distance(2)
    eigs_inst_reconf = Eida_reconf_distance(True)
    total_i = 276  # Number of iterations
    # progress_counter = threading.Lock()  # Using a threading lock for thread-safe operations
    progress = 0
    prev_percentage = 0

    eigvect = eigvect[:, :, 0:8]


    results = Parallel(n_jobs=-1)(
        delayed(compute_fcd)(i, j, eigvect, eigs_inst, eigs_inst_reconf, total_i)
        for i in range(T) for j in range(i, T)
    )

    total_iterations = len(results)
    progress_bar_meas = tqdm(total=total_iterations, desc="Calculating Measures:")

    for i, j, fcd_ij, fcd_reconf_ij in results:
        fcd[i, j] = fcd_ij
        fcd[j, i] = fcd_ij
        fcd_reconf[i, j] = fcd_reconf_ij
        fcd_reconf[j, i] = fcd_reconf_ij
        progress_bar_meas.update(1)
    progress_bar_meas.close()

    return fcd, fcd_reconf


def load_saved_eigs_and_analyse(subj_names, data_folder, proxy_measure, fcd_calc, von_neum, norm_calc):
    calc_measures = True
    count = 0
    window_folders = []
    all_eig = 0
    matrix_count = 0
    for name in subj_names:
        for folder_name in os.listdir(data_folder):
            window_folder_path = os.path.join(data_folder, folder_name + '/')
            window_folders.append(window_folder_path)

        for window_dir in window_folders:
            # count += 1
            if os.path.exists(window_dir + name):
                count += 1
                subject_folder = window_dir + name

                analysis_flag = os.path.exists(os.path.join(subject_folder, "fcd.npy")) and \
                                os.path.exists(os.path.join(subject_folder, "fcd_reconf.npy")) and \
                                os.path.exists(os.path.join(subject_folder, "neumann.npy")) and \
                                os.path.exists(os.path.join(subject_folder, "norm1.npy")) and \
                                os.path.exists(os.path.join(subject_folder, "norm2.npy")) and \
                                os.path.exists(os.path.join(subject_folder, "norminf.npy"))

                eigvect = np.load(subject_folder + f'/eig_vect{name}.npy')

                # all_eig += eigvect[:, :, 0]
                # matrix_count += 1
                # print(f'All_eig + {count}')

                if calc_measures:
                    if not analysis_flag:
                        print("Calculating Measures for " + name)

                        eigval = np.load(subject_folder + f'/eig_val{name}.npy')
                        eigvect = np.load(subject_folder + f'/eig_vect{name}.npy')

                        # all_eig.append(eigvect)
                        n_eigen = eigval.shape[0]
                        T = eigvect.shape[0]

                        if proxy_measure:
                            von_neumann = eigval / np.tile(np.sum(eigval, axis=0), (n_eigen, 1))
                            von_neumann = -np.sum(np.log(von_neumann) * von_neumann, axis=0)
                            # Take off the first eigenval -
                            fig1, ax = plt.subplots(2, 1, sharex=True)
                            ax[0].plot(von_neumann)
                            plt.savefig(subject_folder + "/neumann_norm_figure")
                            np.save(subject_folder + "/neumann", von_neumann)

                            # data_folder = '/Users/oliversherwood/Documents/CODE/Data_HCPSubjects_ALLEIGs_SLIDE/'
                            # plt.savefig(data_folder + f'/neumann_norm_figure{name, count}')
                            # np.save(data_folder + f"/neumann{name, count}", von_neumann)
                        else:
                            print("No Proxy Measures")

                        if fcd_calc:
                            fcd, fcd_reconf = fcd_calc_para(eigvect, T)

                            # np.save(data_folder + name + "/fcd", fcd)
                            # np.save(data_folder + name + "/fcd_reconf", fcd_reconf)
                            np.save(subject_folder + "/fcd_reconf", fcd_reconf)
                            np.save(subject_folder + "/fcd", fcd)

                            fig2, ax = plt.subplots(1, 2)
                            ax[0].imshow(fcd)
                            ax[1].imshow(fcd_reconf)
                            # plt.savefig(data_folder + name + "/fcd_figure")
                            plt.savefig(subject_folder + "/fcd_figure")
                        else:
                            print("No FCD")

                        if von_neum:
                            von_neumann = eigval / np.tile(np.sum(eigval, axis=0), (n_eigen, 1))
                            von_neumann = -np.sum(np.log(von_neumann) * von_neumann, axis=0)
                            # np.save(data_folder + name + "/neumann", von_neumann)
                            np.save(subject_folder + "/neumann", von_neumann)
                        else:
                            print("No Von-Neum")

                        if norm_calc:
                            norm1_inst = EIDA_Norm(1)
                            norm2_inst = EIDA_Norm(2)
                            norminf_inst = EIDA_Norm(np.inf)

                            norm1 = norm1_inst.norm(eigval)
                            norm2 = norm2_inst.norm(eigval)
                            norminf = norminf_inst.norm(eigval)

                            # np.save(data_folder + name + "/norm1", norm1)
                            # np.save(data_folder + name + "/norm2", norm2)
                            # np.save(data_folder + name + "/norminf", norminf)

                            np.save(subject_folder + "/norm1", norm1)
                            np.save(subject_folder + "/norm2", norm2)
                            np.save(subject_folder + "/norminf", norminf)

                        else:
                            print("No Norm")

                        if von_neum and norm_calc:
                            fig1, ax = plt.subplots(2, 1, sharex=True)
                            ax[0].plot(von_neumann)
                            ax[1].plot(norm1)
                            ax[1].plot(norm2)
                            ax[1].plot(norminf)
                            # plt.savefig(data_folder + name + "/neumann_norm_figure")
                            plt.savefig(subject_folder + "/neumann_norm_figure")
                        else:
                            print("All FIG???")

                    else:
                        print(f"Analysis already performed for {name}. Skipping...")
                else:
                    print("No CALC")
        # else:
        #     print(f"Subject folder not found for {name}. Skipping...")

        # mean_across_all_matrices = all_eig / matrix_count
    return all_eig


def main():

    # n_eigs = 18
    n_eigs = 10
    # window_sizes = 10
    # window_sizes = [11, 15, 19, 25]
    window_sizes = [5, 7, 9, 11, 13]
    window_shape = "Rect"
    multi_eigs = False
    n_part = 50
    proxy_measure = True
    saveOUT = True
    subj_partition = False

    fcd_calc = True
    von_neum = False
    norm_calc = False
    # subj_partition = False

    load = False

    min_file_size_bytes = 140 * 1024 * 1024

    # data_folder = "/Users/oliversherwood/Documents/BMIS_DC/20191009/BOLD_StructRows/"
    # data_folder = "/Users/oliversherwood/Documents/BMIS_DC/20191009/SAVED/"
    data_folder = "/Users/oliversherwood/Documents/CODE/DySCo/Data/HCP/WMData_HCP_100"
    try:
        # subject_txt = "/Users/oliversherwood/Documents/BMIS_DC/20191009/DySCo_EPI_SubV1.txt"
        subject_txt = "/Users/oliversherwood/Documents/CODE/DySCo/Data/HCP/WMData_HCP_100/SubjectsToDownload_100_HCP.txt"
        subj_names = extract_subj_id(subject_txt, subj_partition, n_part)
    except:
        print('No Subject Text file defined')

    if load:
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            subject_id = os.path.splitext(filename)[0]

            if os.path.getsize(file_path) >= min_file_size_bytes:
                subject_id = filename.split('-')[0]

            if subject_id in subj_names:
                if filename.endswith('.nii'):
                    eig_vec, eig_val = process_nifti(file_path, n_eigs, window_shape, window_sizes, subject_id)
                    # save_subject_data(subject_id, window_sizes, n_eigs, eig_vec, eig_val)
                elif filename.endswith('.mat'):
                    eig_vec, eig_val = process_mat(file_path, n_eigs, window_shape, window_sizes, subject_id)
                    # save_subject_data(subject_id, window_sizes, n_eigs, eig_vec, eig_val)

            # Save results


    eig_data_folder = '/Users/oliversherwood/Documents/BMIS_EPI_DC_05_06_rem10/'
    # eig_data_folder = '/Users/oliversherwood/Documents/CODE/DySCo/Saved_out/SENSITIVTIY_09_06/'
    all_eig = load_saved_eigs_and_analyse(subj_names, eig_data_folder, proxy_measure, fcd_calc, von_neum, norm_calc)



a = main()
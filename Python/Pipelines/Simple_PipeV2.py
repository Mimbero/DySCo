import os
import numpy as np
import nibabel as nb
from scipy.io import loadmat
import random

import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from joblib import Parallel, delayed


current_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
core_functions_path = os.path.join(project_dir, 'Python', 'core_functions', 'classy')
sys.path.append(core_functions_path)

from CLS_compute_eigs_all import Compute_Eigs
from CLS_dysco_distance import Dysco_distance
from CLS_dysco_reconf import Dysco_reconf_distance
from CLS_dysco_norm import Dysco_Norm


def surf_data_from_cifti(data, axis, surf_name):

    """
    :param data: Cifti data (from process Nifti function)
    :param axis:
    :param surf_name:
    :return: Returns the surface data
    """

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

    """
    :param subject_id: Id of subject
    :param half_window_sz: The half window size (assigned in 'Main')
    :param n_eigs: Number of Eigenvectors (assigned in 'Main')
    :param eig_vect: The Eigenvector Array (returned from Class: ComputeEigs (in slide_window_rect))
    :param eig_val: The Eigenvalue Array (returned from Class: ComputeEigs (in slide_window_rect))
    :return: N/A, Saves .npy file to the specified path
    """

    subject_folder = os.path.join(
        project_dir, 'Data', 'Saved_out', f'Data_WS-{half_window_sz}__eigen-{n_eigs}', subject_id
    )

    # Create the subject folder if it doesn't exist
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)

    # Save eigenvectors and eigenvalues in the subject folder
    np.save(os.path.join(subject_folder, f'eig_vect{subject_id}.npy'), eig_vect)
    np.save(os.path.join(subject_folder, f'eig_val{subject_id}.npy'), eig_val)


def extract_subj_id(subject_txt, subj_partition, n_part):

    """
    :param subject_txt: Text file containing subject IDs
    :param subj_partition: Conditional, Partition if you don't want to use all the data (assigned in 'Main')
    :param n_part: The amount of data to use, IF partition == True
    :return:
    """

    with open(subject_txt, 'r') as file:
        subject_names = file.read().splitlines()

    subj_names = subject_names

    if subj_partition:
        subj_names = random.sample(subj_names, n_part)

    return subj_names


def process_nifti(file_path, n_eigs, window_shape, window_sizes, subject_id):
    # Load NIfTI file
    cifti = nb.load(file_path)
    cifti_data = cifti.get_fdata(dtype=np.float32)
    cifti_hdr = cifti.header
    nifti_hdr = cifti.nifti_header

    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
    left_brain = surf_data_from_cifti(cifti_data, axes[1], 'CIFTI_STRUCTURE_CORTEX_LEFT')
    # right_brain = surf_data_from_cifti(cifti_data, axes[1], 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    brain_load = left_brain

    brain_load = brain_load.T
    zero_columns = np.all(brain_load == 0, axis=0)
    filtered_array = brain_load[:, ~zero_columns]
    brain = filtered_array

    if window_shape == "Rect":
        eig_vec, eig_val = slide_window_rect(n_eigs, window_sizes, brain, subject_id)

    return eig_vec, eig_val


def slide_window_rect(n_eigs, window_sizes, brain_load, subject_id):

    """

    :param n_eigs: Number of Eigenvectors (assigned in 'Main')
    :param window_sizes: (assigned in 'Main')
    :param brain_load: Data, passed from process_nifti (or process_mat)
    :param subject_id:
    :return: Returns Eigenvector and Eigenvalue arrays, calculated using Class:Compute_Eigs, here (see line 132) can
    also be specified to use covariance, correlation, iPA etc.
    """

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


def compute_fcd(i, j, eigvect, eigs_inst, eigs_inst_reconf):
    """
    Called from function: fcd_calc_para.
    Computes fcd and mode alignment using core functions

    :return: fcd (at ij) mode alignment (fcd_reconf) (at ij)
    """

    matrix_a = eigvect[i, :, :]
    matrix_b = eigvect[j, :, :]
    fcd_ij = eigs_inst.dysco_distance(matrix_a, matrix_b)
    fcd_reconf_ij = eigs_inst_reconf.dysco_mode_alignment(matrix_a, matrix_b)

    return i, j, fcd_ij, fcd_reconf_ij


def fcd_calc_para(eigvect, T):
    """
    :param eigvect: eigenvector array (assigned in load_saved_eigs_and_analyse)
    :param T: time points (assigned in load_saved_eigs_and_analyse)
    :return: fcd, fcd_reconf
    """
    fcd = np.zeros((T, T))
    fcd_reconf = np.zeros((T, T))
    eigs_inst = Dysco_distance(2)
    eigs_inst_reconf = Dysco_reconf_distance(True)
    eigvect = eigvect[:, :, 0:8]

    results = Parallel(n_jobs=-1)(
        delayed(compute_fcd)(i, j, eigvect, eigs_inst, eigs_inst_reconf)
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

    """

    :param subj_names: subject names (from subject.txt)
    :param data_folder: data folders (containing saved eig_vect and eig_val arrays, for each subject, at different
    window sizes (if applicable)
    :param proxy_measure: Conditional, only calculates VN entropy
    :param fcd_calc: Conditional, perform fcd calculations
    :param von_neum: Conditional, Calculate VN entropy
    :param norm_calc: Conditional, Calculate Norms
    :return: Return all_eigs (a combination of all subject matrix of eigenvectors) (if required). Based on above
    conditionals will save result to appropriate subject folder.
    """
    calc_measures = True
    count = 0
    window_folders = []
    all_eig = []

    for folder_name in os.listdir(data_folder):
        window_folder_path = os.path.join(data_folder, folder_name + '/')
        window_folders.append(window_folder_path)

    for window_dir in window_folders:
        for name in subj_names:
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

                if calc_measures:
                    if not analysis_flag:
                        print("Calculating Measures for " + name)

                        eigval = np.load(subject_folder + f'/eig_val{name}.npy')
                        eigvect = np.load(subject_folder + f'/eig_vect{name}.npy')

                        all_eig.append(eigvect)

                        n_eigen = eigval.shape[0]
                        T = eigvect.shape[0]

                        if proxy_measure:
                            von_neumann = eigval / np.tile(np.sum(eigval, axis=0), (n_eigen, 1))
                            von_neumann = -np.sum(np.log(von_neumann) * von_neumann, axis=0)
                            fig1, ax = plt.subplots(2, 1, sharex=True)
                            ax[0].plot(von_neumann)
                            plt.savefig(subject_folder + "/neumann_norm_figure")
                            np.save(subject_folder + "/neumann", von_neumann)

                        else:
                            print("No Proxy Measures")

                        if fcd_calc:
                            fcd, fcd_reconf = fcd_calc_para(eigvect, T)
                            np.save(subject_folder + "/fcd_reconf", fcd_reconf)
                            np.save(subject_folder + "/fcd", fcd)

                            fig2, ax = plt.subplots(1, 2)
                            ax[0].imshow(fcd)
                            ax[1].imshow(fcd_reconf)
                            plt.savefig(data_folder + "/fcd_figure")
                        else:
                            print("No FCD")

                        if von_neum:
                            von_neumann = eigval / np.tile(np.sum(eigval, axis=0), (n_eigen, 1))
                            von_neumann = -np.sum(np.log(von_neumann) * von_neumann, axis=0)
                            np.save(subject_folder + "/neumann", von_neumann)
                        else:
                            print("No Von-Neum")

                        if norm_calc:
                            norm1_inst = Dysco_Norm(1)
                            norm2_inst = Dysco_Norm(2)
                            norminf_inst = Dysco_Norm(np.inf)

                            norm1 = norm1_inst.norm(eigval)
                            norm2 = norm2_inst.norm(eigval)
                            norminf = norminf_inst.norm(eigval)

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
                            plt.savefig(subject_folder + "/neumann_norm_figure")
                        else:
                            print("All FIG???")

                    else:
                        print(f"Analysis already performed for {name}. Skipping...")
                else:
                    print("No CALC")

    return all_eig


def main():
    """
    Main Function

    Set all variables and conditionals below.

    Data Should be in the 'Data_In Folder'
    :return:
    """

    n_eigs = 10
    window_sizes = [5, 7, 9, 11, 13]
    window_shape = "Rect"
    multi_eigs = False
    n_part = 50
    proxy_measure = True
    saveOUT = True
    subj_partition = False

    fcd_calc = False
    von_neum = False
    norm_calc = False

    load = True

    min_file_size_bytes = 140 * 1024 * 1024

    data_folder = os.path.join(project_dir, 'Data', 'Data_In')

    try:
        subject_txt = os.path.join(project_dir, 'Data', 'Subjects.txt')
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

    eig_data_folder = os.path.join(project_dir, 'Data', 'Saved_out')
    all_eig = load_saved_eigs_and_analyse(subj_names, eig_data_folder, proxy_measure, fcd_calc, von_neum, norm_calc)


if __name__ == "__main__":
    main()

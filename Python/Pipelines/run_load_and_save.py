import numpy as np
import os
import sys
import nibabel as nb

current_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
core_functions_path = os.path.join(project_dir, 'Python', 'core_functions', 'Core_FunctionsCLASS_Python')
sys.path.append(core_functions_path)

from compute_eigs import Compute_Eigs


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
    subject_folder = f'/Users/oliversherwood/Documents/CODE/Data_HCPSubjects_ALLEIGs_SLIDE/Data_HCPSubjects_WS-{half_window_sz}__eigen-{n_eigs}/{subject_id}/'

    # Create the subject folder if it doesn't exist
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)

    # Save eigenvectors and eigenvalues in the subject folder
    np.save(os.path.join(subject_folder, f'eig_vect{subject_id}.npy'), eig_vect)
    np.save(os.path.join(subject_folder, f'eig_val{subject_id}.npy'), eig_val)


def load_hcp_data(folder_path, subject_txt, n_eigs, window_sizes):

    min_file_size_bytes = 140 * 1024 * 1024

    with open(subject_txt, 'r') as file:
        subject_names = file.read().splitlines()

    # Iterate through NIfTI files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.nii'):
            file_path = os.path.join(folder_path, filename)

            if os.path.getsize(file_path) >= min_file_size_bytes:
                subject_id = filename.split('-')[0]  # Extract subject ID from the filename

            # Check if the subject ID is in subject_names
                if subject_id in subject_names:
                    # file_path = os.path.join(folder_path, filename)
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

                    eig_vec, eig_val = slide_window_and_save(n_eigs, window_sizes, brain, subject_id)

                    #
                    # inst = Compute_Eigs(brain, n_eigs, half_window_sz)
                    # eig_vec, eig_val = inst.compute_eigs_cov()
                    #
                    #
                    # save_subject_data(subject_id, eig_vec, eig_val)
            else:
                print(file_path)
                print("No Appropriate Files of Type .nii")

    return brain


def slide_window_and_save(n_eigs, window_sizes, brain_load, subject_id):

    for half_window_sz in window_sizes:

        # Perform eigenvalue calculations
        inst = Compute_Eigs(brain_load, n_eigs, half_window_sz)
        eig_vec, eig_val = inst.compute_eigs_cov()

        save_subject_data(subject_id, half_window_sz, n_eigs, eig_vec, eig_val)

        # Create a folder for the current window size
        # output_folder = f'/Users/oliversherwood/Documents/CODE/Data_HCPSubjects_{half_window_sz}eigen/'
        # if not os.path.exists(output_folder):
        #     os.makedirs(output_folder)
        #
        # # Save eigenvectors in the folder with the name of the window size
        # np.save(os.path.join(output_folder, f'eig_vect_{half_window_sz}.npy'), eig_vec)
        # np.save(os.path.join(output_folder, f'eig_val_{half_window_sz}.npy'), eig_val)

    return eig_vec, eig_val


def main_pipe():
    n_eigs = 20
    window_sizes = [10, 15, 20, 25, 30, 35]
    folder_path = '//WMDat/'
    subject_txt = "/Users/oliversherwood/Documents/CODE/MEIDAS_MAIN/WMDat/SubjectsToDownload_full.txt"
    brain_load = load_hcp_data(folder_path, subject_txt, n_eigs, window_sizes)
    print(brain_load)

    # inst = Compute_Eigs(brain_load, n_eigs, half_window_sz)
    # eig_vec, eig_val = inst.compute_eigs_cov()
    #
    # save_subject_data(subject_id, eig_vec, eig_val)
    # inst = Compute_Eigs(brain, n_eigs, half_window_sz)
    # eig_vec, eig_val = inst.compute_eigs_cov()
    # print(eig_vec)


a = main_pipe()
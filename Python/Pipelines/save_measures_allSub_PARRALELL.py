import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import threading
import warnings

# Define your paths and configurations here
current_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
core_functions_path = os.path.join(project_dir, 'Python', 'core_functions', 'Core_FunctionsCLASS_Python')
sys.path.append(core_functions_path)

from eida_distance import Eida_distance
from eida_reconf_distance import Eida_reconf_distance
from compute_norm import EIDA_Norm

# SPECIFY DATA FOLDER WITH SUBJECTS
# data_folder ='/Users/oliversherwood/Documents/CODE/MEIDAS_MAIN/Data_HCPSubjects/'
data_folder ='/Users/oliversherwood/Documents/CODE/Data_HCPSubjects_ALLEIGs_SLIDE/'

# DEFINE PARAMETERS
# half_window_size = 15
subject_txt = "/Users/oliversherwood/Documents/CODE/MEIDAS_MAIN/WMDat/SubjectsToDownload_full.txt"

proxy_measure = True

with open(subject_txt, 'r') as file:
    subject_names = file.read().splitlines()

subj_names = subject_names


def compute_fcd(i, j, eigvect, eigs_inst, eigs_inst_reconf):
    matrix_a = eigvect[i, :, :]
    matrix_b = eigvect[j, :, :]
    fcd_ij = eigs_inst.eida_distance(matrix_a, matrix_b)
    fcd_reconf_ij = eigs_inst_reconf.eida_reconf_distance(matrix_a, matrix_b)

    global progress, prev_percentage
    progress = i
    percentage = (progress / total_i) * 100
    if int(percentage) > prev_percentage:
        sys.stdout.write(f"\rProgress: {int(percentage)}%")
        sys.stdout.flush()
        # print(f"\rProgress: {int(percentage)}%")
        prev_percentage = int(percentage)

    return i, j, fcd_ij, fcd_reconf_ij


for name in subj_names:
    for folder_name in os.listdir(data_folder):
        window_folder_path = os.path.join(data_folder, folder_name + '/')

    if os.path.exists(window_folder_path + name):
        subject_folder = window_folder_path + name

        analysis_flag = os.path.exists(os.path.join(subject_folder, "fcd.npy")) and \
                        os.path.exists(os.path.join(subject_folder, "fcd_reconf.npy")) and \
                        os.path.exists(os.path.join(subject_folder, "neumann.npy")) and \
                        os.path.exists(os.path.join(subject_folder, "norm1.npy")) and \
                        os.path.exists(os.path.join(subject_folder, "norm2.npy")) and \
                        os.path.exists(os.path.join(subject_folder, "norminf.npy"))

        if not analysis_flag:
            print("Calculating Measures for " + name)

            eigval = np.load(subject_folder + f'/eig_val{name}.npy')
            eigvect = np.load(subject_folder + f'/eig_vect{name}.npy')

            n_eigen = eigval.shape[0]
            T = eigvect.shape[0]

            if proxy_measure:
                von_neumann = eigval / np.tile(np.sum(eigval, axis=0), (n_eigen, 1))
                von_neumann = -np.sum(np.log(von_neumann) * von_neumann, axis=0)

                fig1, ax = plt.subplots(2, 1, sharex=True)
                ax[0].plot(von_neumann)
                # ax[1].plot(norm1)
                # ax[1].plot(norm2)
                # ax[1].plot(norminf)
                plt.savefig(subject_folder + "/neumann_norm_figure")

                np.save(subject_folder + "/neumann", von_neumann)

            else:

                fcd = np.zeros((T, T))
                fcd_reconf = np.zeros((T, T))
                eigs_inst = Eida_distance(2)
                eigs_inst_reconf = Eida_reconf_distance(True)
                total_i = 375  # Number of iterations
                # progress_counter = threading.Lock()  # Using a threading lock for thread-safe operations
                progress = 0
                prev_percentage = 0

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

                print("Plotting")
                # Calculate von_neumann
                von_neumann = eigval / np.tile(np.sum(eigval, axis=0), (n_eigen, 1))
                von_neumann = -np.sum(np.log(von_neumann) * von_neumann, axis=0)


                norm1_inst = EIDA_Norm(1)
                norm2_inst = EIDA_Norm(2)
                norminf_inst = EIDA_Norm(np.inf)

                norm1 = norm1_inst.norm(eigval)
                norm2 = norm2_inst.norm(eigval)
                norminf = norminf_inst.norm(eigval)

                np.save(data_folder+name+"/fcd",fcd)
                np.save(data_folder+name+"/fcd_reconf",fcd_reconf)
                np.save(data_folder+name+"/neumann",von_neumann)
                np.save(data_folder+name+"/norm1",norm1)
                np.save(data_folder+name+"/norm2",norm2)
                np.save(data_folder+name+"/norminf",norminf)

                #plot quantities and save
                fig1,ax = plt.subplots(2,1,sharex=True)
                ax[0].plot(von_neumann)
                ax[1].plot(norm1)
                ax[1].plot(norm2)
                ax[1].plot(norminf)
                plt.savefig(data_folder+name+"/neumann_norm_figure")

                fig2,ax = plt.subplots(1,2)
                ax[0].imshow(fcd)
                ax[1].imshow(fcd_reconf)
                plt.savefig(data_folder+name+"/fcd_figure")

                plt.savefig(data_folder + name + "/fcd_figure")

        else:
            print(f"Analysis already performed for {name}. Skipping...")
    else:
        print(f"Subject folder not found for {name}. Skipping...")

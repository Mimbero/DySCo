# how do I clear vars in python to avoid conflicts?

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
core_functions_path = os.path.join(project_dir, 'Python', 'core_functions', 'Core_FunctionsCLASS_Python')
sys.path.append(core_functions_path)

from eida_distance import Eida_distance
from eida_reconf_distance import Eida_reconf_distance
from compute_norm import EIDA_Norm

# SPECIFY DATA FOLDER WITH SUBJECTS. EVERY SUBJECT IS A SUBFOLDER WITH A NAME
# This cant be changed for the moment / can be inputed
# data_folder = "/Users/giuseppe/OneDrive - King's College London/Londra/EiDA_2.0/data/"
# data_folder = "/Users/oliversherwood/Documents/CODE/data/"
data_folder = '/Users/oliversherwood/Documents/CODE/MEIDAS_MAIN/Data_HCPSubjects/'
# data_folder = '/Users/oliversherwood/Documents/CODE/Data_HCPSubjects_5eigen/'

# DEFINE PARAMETERS1
half_window_size = 15
# SPECIFY NAMES FOR THE ANALYSIS
subject_txt = "/Users/oliversherwood/Documents/CODE/MEIDAS_MAIN/WMDat/SubjectsToDownload_full.txt"


with open(subject_txt, 'r') as file:
    subject_names = file.read().splitlines()

subj_names = subject_names

#REPEAT THE COMPUTATION OF QUANTITIES OF INTEREST FOR EACH SUBJ
for name in subj_names:

    if os.path.exists(data_folder+name):
        subject_folder = data_folder+name

        analysis_flag = os.path.exists(os.path.join(subject_folder, "fcd.npy")) and \
                        os.path.exists(os.path.join(subject_folder, "fcd_reconf.npy")) and \
                        os.path.exists(os.path.join(subject_folder, "neumann.npy")) and \
                        os.path.exists(os.path.join(subject_folder, "norm1.npy")) and \
                        os.path.exists(os.path.join(subject_folder, "norm2.npy")) and \
                        os.path.exists(os.path.join(subject_folder, "norminf.npy"))

        if not analysis_flag:
            print("Calculating Measures for " + name)
            # Scale each subject

            eigval = np.load(data_folder+name+f'/eig_val{name}.npy')
            eigvect = np.load(data_folder+name+f'/eig_vect{name}.npy')

            n_eigen = eigval.shape[0] # this should be fixed for each subj, otherwise it won't work
            T = eigvect.shape[0]
            total_iterations = T
            progress_bar_meas = tqdm(total=total_iterations, desc="Calculating Measures:")

            # Initialize fcd matrix + with reconf speed
            fcd = np.zeros((T, T))
            fcd_reconf = np.zeros((T, T))
            eigs_inst = Eida_distance(2)
            eigs_inst_reconf = Eida_reconf_distance(True)

            # Calculate fcd values

            for i in range(T):
                for j in range(i, T):
                    # eig_vect_cp = eigvect.copy()
                    matrix_a = eigvect[i, :, :]
                    matrix_b = eigvect[j, :, :]
                    fcd[i, j] = eigs_inst.eida_distance(matrix_a, matrix_b)
                    fcd[j, i] = fcd[i, j]

                    fcd_reconf[i, j] = eigs_inst_reconf.eida_reconf_distance(matrix_a, matrix_b)
                    fcd_reconf[j, i] = fcd_reconf[i, j]
                progress_bar_meas.update(1)

            progress_bar_meas.close()

            print("Plotting")
            # Calculate von_neumann
            von_neumann = eigval / np.tile(np.sum(eigval, axis=0), (n_eigen, 1))
            von_neumann = -np.sum(np.log(von_neumann) * von_neumann, axis=0)

            # Calculate different possible norms

            norm1_inst = EIDA_Norm(1)
            norm2_inst = EIDA_Norm(2)
            norminf_inst = EIDA_Norm(np.inf)

            norm1 = norm1_inst.norm(eigval)
            norm2 = norm2_inst.norm(eigval)
            norminf = norminf_inst.norm(eigval)


            #SAVE DATA

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
            ax[0].imshow(fcd, vmin=-1, vmax=1)
            ax[1].imshow(fcd_reconf)
            plt.savefig(data_folder+name+"/fcd_figure")

        else:
            print(f"Analysis already performed for {name}. Skipping...")
    else:
        print(f"Subject folder not found for {name}. Skipping...")


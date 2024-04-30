import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.linalg import eigh
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm

# Set the main directory where your data is stored
# main_directory = '/Users/oliversherwood/Documents/CODE/Data_HCPSubjects_ALL_100/'
main_directory = '/Users/oliversherwood/Documents/CODE/Data_HCPSubjects_ALLEIGs_SLIDE'
# main_directory = '/Users/oliversherwood/Documents/CODE/Data_HCP_106521_ONESUB/'

task = np.loadtxt('//Python/DATA/G/TaskTC.txt')
task_reduced = np.mean(task, axis=1)
# task_reduced = task_reduced[15:390]

window_size_dirs = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]

# window_size_colors = {'10': 'red', '15': 'blue', '20': 'green', '25': 'purple', '30': 'orange'}
window_size_colors = {'5': 'cyan', '7': 'red', '9': 'blue', '12': 'green', '13': 'purple', '18': 'orange'}

plot = False
plot_all = False
plot_average = True
neum_only = True
fcd_only = False

# Plotting
# plt.figure(figsize=(10, 6))
# fig_1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

def plot_func(values, colourpallete, max_task_reduced_shift):
    # fig_directory = '/Users/oliversherwood/Documents/CODE/Figures_28.02/FIGS_New/'
    #
    # # FOR SPEED
    # # max_task_reduced_shift_1 = max_task_reduced_shift[0:367]
    #
    # # =============================
    # # CREATE FIG VAR
    # # =============================
    # x = pd.DataFrame(values)
    # # ent.to_csv('entropies.csv', index=False)
    # mean_x = x.mean(axis=0)
    # # savemat...
    # std_x = x.std(axis=0)
    # error = std_x / np.sqrt(np.shape(x)[0])
    # normalized_x = (mean_x - mean_x.min()) / (mean_x.max() - mean_x.min())
    #
    # relative_std_x = std_x / (mean_x.max() - mean_x.min())
    # relative_err_x = error / (mean_x.max() - mean_x.min())
    #
    # mean_colr = colourpallete[0]
    # shade_colr = colourpallete[1]
    # task_colr = colourpallete[2]
    #
    # # correlation_coefficient_reconf, p_value_reconf = pearsonr(mean_x, max_task_reduced_shift)
    #
    # # print("Pearson correlation coefficient:", correlation_coefficient_reconf)
    # # print("p-value:", p_value_reconf)
    #
    # # =============================
    # # PLOT FIG
    # # =============================
    # plt.figure(figsize=(12, 8))
    # # plt.figure(figsize=(10, 6))
    # sns.set(font="Arial")
    # sns.lineplot(x=x.columns, y=normalized_x, err_style="band", ci='sd', err_kws={'alpha': 0.3}, color=mean_colr)
    # plt.fill_between(x.columns, normalized_x - relative_err_x, normalized_x + relative_err_x, alpha=0.3, color=shade_colr)
    # sns.lineplot(x=range(len(max_task_reduced_shift)), y=max_task_reduced_shift, color=task_colr, linestyle='dashed')
    #
    # plt.xlabel('Time (TRs)', fontsize='16', fontfamily="Arial")
    # plt.ylabel('Speed', fontsize='16', fontfamily="Arial")
    # # plt.ylabel('Entropy', fontsize='12', fontfamily="Arial")
    # # plt.title('Reconfiguration Speed and Task Time-course for 1 Subjects', fontsize='12', fontfamily="Arial")
    # # plt.title('Entropy and Task Time-course for all Subjects', fontsize='12', fontfamily="Arial")
    #
    # mean_line = mlines.Line2D([], [], color=mean_colr, label='Mean Reconfiguration Speed')
    # # mean_line = mlines.Line2D([], [], color=mean_colr, label='Mean Entropy')
    # std_error_patch = mpatches.Patch(color=shade_colr, label='Standard Error (SE)')
    # task_line = mlines.Line2D([], [], color=task_colr, linestyle='dashed', label='Task Timecourse')
    #
    # # Add legend with custom handles
    # plt.legend(handles=[mean_line, std_error_patch, task_line], loc='upper right', frameon=False, bbox_to_anchor=(1.05, 1))
    # # plt.legend(handles=[mean_line, task_line], loc='upper right', frameon=False)
    # # plt.legend(title='Window Size')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(fig_directory + "Reconf_speed_05_03_ALLSUB.jpg", dpi=300)
    # # plt.savefig(fig_directory + "Entropies_28_02_1SUB.jpg")
    # plt.close()

    fig_directory = '/Users/oliversherwood/Documents/CODE/Figures_28.02/FIGS_New/'

    # =============================
    # CREATE FIG VAR
    # =============================
    x = pd.DataFrame(values)
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    error = std_x / np.sqrt(np.shape(x)[0])
    normalized_x = (mean_x - mean_x.min()) / (mean_x.max() - mean_x.min())
    relative_err_x = error / (mean_x.max() - mean_x.min())

    mean_colr = colourpallete[0]
    shade_colr = colourpallete[1]
    task_colr = colourpallete[2]

    # Create a subplot with defined spacing
    fig, ax = plt.subplots(figsize=(14, 4))  # Adjust figsize as needed
    fig.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)  # Adjust spacing as needed

    # =============================
    # PLOT FIG
    # =============================
    sns.set(font="Arial")
    sns.lineplot(x=x.columns, y=normalized_x, err_style="band", ci='sd', err_kws={'alpha': 0.3}, color=mean_colr, ax=ax)
    ax.fill_between(x.columns, normalized_x - relative_err_x, normalized_x + relative_err_x, alpha=0.3,
                    color=shade_colr)
    sns.lineplot(x=range(len(max_task_reduced_shift)), y=max_task_reduced_shift, color=task_colr, linestyle='dashed',
                 ax=ax)

    ax.set_xlabel('Time (TRs)', fontsize='16', fontfamily="Arial")
    ax.set_ylabel('Entropy', fontsize='16', fontfamily="Arial")
    # ax.set_ylabel('Reconfiguration Speed', fontsize='16', fontfamily="Arial")
    ax.tick_params(axis='both', which='major', labelsize=16)

    # mean_line = mlines.Line2D([], [], color=mean_colr, label='Mean Reconfiguration Speed')
    mean_line = mlines.Line2D([], [], color=mean_colr, label='Mean von-Neumann Entropy')
    std_error_patch = mpatches.Patch(color=shade_colr, label='Standard Error (SE)')
    task_line = mlines.Line2D([], [], color=task_colr, linestyle='dashed', label='Task Timecourse')

    # Add legend with custom handles, placing it outside the plot area
    # legend = ax.legend(handles=[mean_line, std_error_patch, task_line], loc='upper right', frameon=False,
    #           bbox_to_anchor=(1.32, 1))
    legend = ax.legend(handles=[mean_line, std_error_patch, task_line], loc='upper right', frameon=False,
                       bbox_to_anchor=(1.32, 1))
    for text in legend.get_texts():
        text.set_fontsize(14)

    ax.grid(True)

    plt.tight_layout()

    # plt.savefig(fig_directory + "Reconf_speed_06_03_1SUB.jpg", dpi=300)
    plt.savefig(fig_directory + "Entropies_05_03_1SUB.jpg", dpi=300)
    plt.close()

if plot:
    fig_1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # fig_2, (ax1_1, ax2_1, ax3_1) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig_2, (ax1_1, ax2_1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)



for window_size_dir in window_size_dirs:
    entropies = []
    fcd_le = []
    fcd_all = []
    window_size_path = os.path.join(main_directory, window_size_dir)

    # Get a list of all subject directories for the current window size
    subject_dirs = [d for d in os.listdir(window_size_path) if os.path.isdir(os.path.join(window_size_path, d))]

    match = re.search(r'WS-(\d+)', window_size_dir)
    match_eig = re.search(r'eigen-(\d+)', window_size_dir)
    if match:
        window_size = match.group(1)
    else:
        # If no match found, skip this directory
        continue

    if match_eig:
        eig_no = match_eig.group(1)
    else:
        continue

    # plt.figure(figsize=(10, 6))

    for subject_dir in subject_dirs:
        subject_path = os.path.join(window_size_path, subject_dir)
        npy_file_path_neum = os.path.join(subject_path, 'neumann.npy')
        npy_file_path_fcd = os.path.join(subject_path, 'fcd.npy')

        # SAVE ENTROPY MEASURES
        if os.path.exists(npy_file_path_neum):
            entropy_measures = np.load(npy_file_path_neum)
            entropies.append(entropy_measures)

            if plot_all:
                # sns.lineplot(x=range(len(entropy_measures)), y=entropy_measures, color='green')
                ax1.plot(entropy_measures, label=f'Window Size {window_size}',
                        color=window_size_colors.get(window_size, 'black'))

                corr = correlate(task_reduced, entropy_measures, mode='same')
                shift = np.argmax(corr) - len(task_reduced)
                aligned_entropy = np.roll(entropy_measures, -shift)
                cor_eigTtask = np.corrcoef(aligned_entropy, task_reduced)

                ax2.plot(aligned_entropy, label=f'Window Size {window_size}',
                         color=window_size_colors.get(window_size, 'black'))

        # SAVE FCD MEASURES
        if os.path.exists(npy_file_path_fcd):
            fcd_measures = np.load(npy_file_path_fcd)
            fcd_all.append(fcd_measures)
            for i in range(np.shape(fcd_measures)[0]):
                fcd_measures[i, i] = 0
            eigenvalues_t, eigenvectors_t = eigh(fcd_measures, overwrite_a=True, check_finite=False)
            le_fcd = eigenvectors_t[:, -1]
            fcd_le.append(le_fcd)

    if plot_average:

        if neum_only:
            mean_entropy = np.mean(entropies, axis=0)
            T = task_reduced.shape[0]

            if plot:
                ax1.plot(mean_entropy, color=window_size_colors.get(window_size, 'blue'))
                ax1.plot(mean_entropy, label=f'Window Size {window_size}',
                         color=window_size_colors.get(window_size, 'black'))

            scaled_task = (task_reduced - np.min(task_reduced)) / (np.max(task_reduced) - np.min(task_reduced))
            scaled_entropy = (mean_entropy - np.min(mean_entropy)) / (np.max(mean_entropy) - np.min(mean_entropy))


            max_corr_1 = []
            for i in range(0, int(window_size)):
                task_reduced_shift = scaled_task[int(window_size)-i:T - int(window_size) - i]
                corr_neum = np.corrcoef(scaled_entropy, task_reduced_shift)
                max_corr_1.append((i, corr_neum[0, 1]))
            # task_reduced = task_reduced[0:375-i]
            # neumann_red = neumann_indv[i:375]

            x_values = [item[0] for item in max_corr_1]
            y_values = [item[1] for item in max_corr_1]

            max_y_value = max(y_values)
            corresponding_x_value = x_values[y_values.index(max_y_value)]
            max_task_reduced_shift = scaled_task[int(window_size) - int(corresponding_x_value):T - int(window_size) - (corresponding_x_value)]
            corr_max_shift= np.corrcoef(scaled_entropy, max_task_reduced_shift)

            correlation_coefficient_Neum, p_value_neum = pearsonr(scaled_entropy, max_task_reduced_shift)

            print("Pearson correlation coefficient:", correlation_coefficient_Neum)
            print("p-value:", p_value_neum)

            # ax2.plot(max_task_reduced_shift, label=f'Window Size {window_size},Max Corr = {max_y_value:.2f}',
            #         color=window_size_colors.get(window_size, 'black'))
            if plot:
                ax2.plot(max_task_reduced_shift, label=f'Max Corr = {max_y_value:.2f}',
                         color=window_size_colors.get(window_size, 'black'))

        if fcd_only:

            mean_fcd_le = np.mean(fcd_le, axis=0)
            # ax1_1.plot(mean_fcd_le, label=f'Window Size {window_size}' + f'Eigs {eig_no}',
            #      color=window_size_colors.get(eig_no, 'black'))
            if plot:
                ax1_1.plot(mean_fcd_le, color=window_size_colors.get(eig_no, 'black'))

            T = task_reduced.shape[0]

            scaled_task = (task_reduced - np.min(task_reduced)) / (np.max(task_reduced) - np.min(task_reduced))
            scaled_fcd = (mean_fcd_le - np.min(mean_fcd_le)) / (np.max(mean_fcd_le) - np.min(mean_fcd_le))

            max_corr_2 = []
            for i in range(0, int(window_size)):
                task_reduced_shift = scaled_task[int(window_size)-i:T - int(window_size) - i]
                corr_fcd = np.corrcoef(scaled_fcd, task_reduced_shift)
                max_corr_2.append((i, corr_fcd[0, 1]))

            x_values = [item[0] for item in max_corr_2]
            y_values = [item[1] for item in max_corr_2]

            max_y_value = max(y_values)
            corresponding_x_value = x_values[y_values.index(max_y_value)]
            max_task_reduced_shift = scaled_task[int(window_size) - int(corresponding_x_value):T - int(window_size) - (corresponding_x_value)]
            corr_max_shift = np.corrcoef(scaled_fcd, max_task_reduced_shift)

            correlation_coefficient_, p_value_neum = pearsonr(scaled_entropy, max_task_reduced_shift)

            print("Pearson correlation coefficient:", correlation_coefficient_Neum)
            print("p-value:", p_value_neum)

            # ax2_1.plot(max_task_reduced_shift, label=f'Window Size {window_size}' + f'Eigs {eig_no}, Max Corr = {max_y_value:.2f}',
            #         color=window_size_colors.get(eig_no, 'black'))
            if plot:
                ax2_1.plot(max_task_reduced_shift,
                        label=f'Max Corr = {max_y_value:.2f}',
                        color=window_size_colors.get(eig_no, 'black'))

mean_matrix = np.mean(fcd_all, axis=0)
# mean_matrix = fcd_measures
delay = 18
T = mean_matrix.shape[0]
reconf_all = []
for i in range(len(fcd_all)):
    reconf_speed = np.empty((T-delay,))
    mat = np.asarray(fcd_all[i])
    for j in range(T-delay):
        reconf_speed[j] = mat[j, j+delay]
    reconf_all.append(reconf_speed)

# ====================================
# PLOTTING FOR RECONFIGURATION SPEED
# ====================================
# colourpallete_speed = ['mediumseagreen', 'mediumaquamarine', 'coral']
# plot_func(reconf_all, colourpallete_speed, max_task_reduced_shift)

# ====================================
# PLOTTING FOR ENTROPY
# ====================================
# colourpallete_ent = ['steelblue', 'skyblue', 'coral']
# plot_func(entropies, colourpallete_ent, max_task_reduced_shift)


# ====================================
# PLOTTING FOR MATRIX PLOT
# ====================================

mean_matrix = np.mean(fcd_all, axis=0)
# for i in range(np.shape(mean_matrix)[0]):
#     mean_matrix[i, i] = 0
# mean_matrix = fcd_measures
# threshold = 4000
# mean_matrix[mean_matrix < threshold] = np.nan
# np.fill_diagonal(mean_matrix, np.nan)


fig_directory = '/Users/oliversherwood/Documents/CODE/Figures_28.02/'
np.fill_diagonal(mean_matrix, np.nan)

band_width = 16
mask = np.zeros_like(mean_matrix, dtype=bool)
for i in range(len(mean_matrix)):
    for j in range(max(0, i - band_width), min(len(mean_matrix), i + band_width + 1)):
        mask[i, j] = True

# Plot the heatmap with the diagonal band masked
plt.figure(figsize=(10, 6), dpi=300)
sns.set(font="Arial")
sns.set_palette("spring")
s = sns.heatmap(mean_matrix, mask=mask, cmap='spring', square=True, vmin=6000, linecolor=None, linewidths=0, cbar_kws={"shrink": 0.7})

s.grid(False)

s.set_facecolor('white')

xtick_positions = [i for i in range(len(mean_matrix[0])) if i % 20 == 0]  # Show every other tick
ytick_positions = [i for i in range(len(mean_matrix)) if i % 20 == 0]

xtick_labels = [str(xtick) for xtick in xtick_positions]
ytick_labels = [str(ytick) for ytick in ytick_positions]

s.set_xticks(xtick_positions)
s.set_yticks(ytick_positions)

s.set_xticklabels(xtick_labels)
s.set_yticklabels(ytick_labels)

font_dict = {'fontsize': 12, 'fontfamily': 'Arial'}
s.set_xlabel('TR', fontdict=font_dict)
s.set_ylabel('TR', fontdict=font_dict)

plt.title('Mean FCD Matrix for One Subjects', fontsize='12', fontfamily="Arial")
plt.show()
plt.savefig(fig_directory + "FCD_28_02_SPRING_FINAL_1V3SUB.jpg", dpi=300)
plt.close()

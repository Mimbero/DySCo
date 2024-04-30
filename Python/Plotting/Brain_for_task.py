import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import nibabel as nib
import pandas as pd
from neuromaps import datasets
import matplotlib.colors as mcolors


# def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
#     new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
#         'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
#         cmap(np.linspace(minval, maxval, n)))
#     return new_cmap

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'truncated_' + cmap.name, cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def custom_red_blue_colormap():
    # Define the colors
    colors = [
        (1.0, 0.0, 0.0),   # Red
        (0.0, 0.0, 1.0)    # Blue
    ]

    # Create a colormap from the defined colors
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_red_blue', colors, N=256)

    # Adjust saturation and brightness for red
    red_multiplier = 1
    red_addition = 0.2
    cmap_colors = cmap(np.linspace(0, 1, 256))
    cmap_colors[:128, :3] = np.clip(cmap_colors[:128, :3] * red_multiplier + red_addition, 0, 1)

    # Adjust saturation and brightness for blue
    blue_multiplier = 1
    blue_addition = 0.2
    cmap_colors[128:, :3] = np.clip(cmap_colors[128:, :3] * blue_multiplier + blue_addition, 0, 1)

    # Create a colormap from the adjusted colors
    return mcolors.LinearSegmentedColormap.from_list('custom_pastel_blue_to_red', cmap_colors)


def write_plyRGB(filename, vertices, faces, colorsR, colorsG, colorsB,comment=None):
    print("writing ply format")
    # infer number of vertices and faces
    number_vertices = vertices.shape[0]
    number_faces = faces.shape[0]
    # make header dataframe
    header = ['ply',
            'format ascii 1.0',
            'comment %s' % comment,
            'element vertex %i' % number_vertices,
            'property float x',
            'property float y',
            'property float z',
            'property uchar red',
            'property uchar green',
            'property uchar blue',
            'element face %i' % number_faces,
            'property list uchar int vertex_indices',
            'end_header'
             ]
    header_df = pd.DataFrame(header)
    # make dataframe from vertices
    vertex_df = pd.DataFrame(vertices/50)
    #colors_df = pd.DataFrame(np.tile(np.round(colors/7*255), (3,1)).T)
    ColorsR_df=pd.DataFrame(colorsR)
    ColorsG_df=pd.DataFrame(colorsG)
    ColorsB_df=pd.DataFrame(colorsB)
    colorsConcat = pd.concat([ColorsR_df,ColorsG_df,ColorsB_df], axis=1)
    colors_df=pd.DataFrame(colorsConcat)
    colors_df=colorsConcat.astype(int)
    df_concat = pd.concat([vertex_df, colors_df], axis=1)
    # make dataframe from faces, adding first row of 3s (indicating triangles)
    triangles = np.reshape(3 * (np.ones(number_faces)), (number_faces, 1))
    triangles = triangles.astype(int)
    faces = faces.astype(int)
    faces_df = pd.DataFrame(np.concatenate((triangles, faces), axis=1))
    # write dfs to csv
    header_df.to_csv(filename, header=None, index=False)
    with open(filename, 'a') as f:
        df_concat.to_csv(f, header=False, index=False,
                         float_format='%.3f', sep=' ')
    with open(filename, 'a') as f:
        faces_df.to_csv(f, header=False, index=False,
                        float_format='%.0f', sep=' ')


def truncate_and_blue_to_red_colormap(minval, maxval):
    # cmap = blue_to_red_colormap()
    cmap = custom_red_blue_colormap()
    return truncate_colormap(cmap, minval=minval, maxval=maxval)


fslr = datasets.fetch_atlas(atlas='fslr', density='32k')
print(fslr.keys())

ltemp, rtemp = fslr['midthickness']
lvert, ltri = nib.load(ltemp).agg_data()

# EigVects = np.load('/Users/oliversherwood/Documents/CODE/MEIDAS_MAIN/EIDA_Old/DATA/G/Sub1_eigvect29_ws_15.npy')

EigVects = np.load('//Python/Scripts/Mean_EIGVECT.npy')
# EigVects = np.load('/Users/oliversherwood/Documents/CODE/MEIDAS_MAIN/Python/Scripts/Mean_EIGVECT_1_SUB.npy')

# EigVal = np.load('/Users/oliversherwood/Documents/CODE/MEIDAS_MAIN/EIDA_Old/DATA/G/Sub1_eigVAL29_ws_15.npy')

IndexZero = np.load('//EIDA_Old/DATA/IndexZero.npy')
task = np.loadtxt('//Python/DATA/G/TaskTC.txt')
task_reduced = np.mean(task, axis=1)
task_reduced = task_reduced[15:390]

task_block = np.zeros(len(task_reduced))
for i in range(len(task_reduced)):
    if task_reduced[i] > 0:
        task_block[i] = 1
    else:
        task_block[i] = 0

EigN = 2
# TR_list = [50, 90, 150, 190, 250, 290, 350]
# TR_list = [0, 11, 20, 40, 60, 90]
# TR_ranges = [(0, 11), (11, 20), (20, 40), (40, 60), (60, 90)]
TR_ranges = [(10, 70), (80, 100), (101, 170), (180, 200), (201, 270), (280, 300)]
# TR = 350
subject_folder = '/Users/oliversherwood/Documents/CODE/DYSCO_FIGS_09_04'

EigIm = np.zeros([IndexZero.shape[0], EigVects.shape[0]])

for i in range(EigVects.shape[0]):
    EigIm[IndexZero == False, i] = EigVects[i, :, EigN].flatten()
    # EigIm[IndexZero == False, i] = EigVects[i, :].flatten()

EigIm_flip = np.zeros([IndexZero.shape[0], EigVects.shape[0]])
for i in range(EigVects.shape[0]):
    for j in range(EigIm.shape[0]):
        if EigIm[j, i] > 0:
            EigIm_flip[j, i] = -EigIm[j, i]
        else:
            EigIm_flip[j, i] = EigIm[j, i]


overall_min = np.percentile(EigIm_flip, 30)
overall_max = np.percentile(EigIm_flip, 70)

for i, (start, end) in enumerate(TR_ranges):
    avg_TR_values = np.mean(EigIm_flip[:, start:end], axis=1)
    # TR = tr_num
    # Normalize values to [0, 1] based on overall_min and overall_max
    # normalized_values = (avg_TR_values - overall_min) / (overall_max - overall_min)
    # norm = matplotlib.colors.Normalize(vmin=overall_min, vmax=overall_max)
    # norm_colour_DAT = norm(avg_TR_values)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(lvert[:, 0], lvert[:, 1], lvert[:, 2], c=EigIm[:, TR], s=8, cmap='bwr_r')
    # ax.scatter(lvert[:, 0], lvert[:, 1], lvert[:, 2], c=EigIm_flip[:, TR], s=8, cmap='bwr_r')
    cmap = truncate_and_blue_to_red_colormap(minval=0, maxval=1)
    # cmap = truncate_colormap(cm.get_cmap('plasma_r'), minval=0.25, maxval=0.8)
    ax.scatter(lvert[:, 0], lvert[:, 1], lvert[:, 2], vmin=overall_min, vmax=overall_max, c=avg_TR_values, s=8, cmap=cmap)
    ax.view_init(elev=0, azim=180)

    norm = matplotlib.colors.Normalize(vmin=overall_min, vmax=overall_max)
    norm_colour_DAT = cmap(norm(avg_TR_values))*255
    filename = subject_folder + f'/DYSCO_Brain_blob_EIG3_TR{i}.ply'
    write_plyRGB(filename, lvert, ltri, norm_colour_DAT[:, 0], norm_colour_DAT[:, 1], norm_colour_DAT[:, 2])

    x, y, z = lvert[:, 0], lvert[:, 1], lvert[:, 2]

    xmin, xmax = np.min(lvert[:, 0]), np.max(lvert[:, 0])
    ymin, ymax = np.min(lvert[:, 1]), np.max(lvert[:, 1])
    zmin, zmax = np.min(lvert[:, 2]), np.max(lvert[:, 2])

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
    ax.autoscale(enable=True, axis='both', tight=True)
    plt.axis('off')
    ax.xaxis.set_tick_params(pad=0)
    ax.yaxis.set_tick_params(pad=0)
    ax.zaxis.set_tick_params(pad=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tight_layout()

    plt.savefig(subject_folder + f'/Brain_blob_EIG3_TR{i}')
    plt.show()
    # dict_keys(['midthickness', 'inflated', 'veryinflated', 'sphere', 'medial', 'sulc', 'vaavg'])
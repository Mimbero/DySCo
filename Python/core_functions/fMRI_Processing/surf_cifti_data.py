import numpy as np
import nibabel as nb


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
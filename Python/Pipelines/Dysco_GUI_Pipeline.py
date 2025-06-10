import os
import numpy as np
import nibabel as nb
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import gc
from PyQt6.QtWidgets import QApplication

# --- Import DySCo core functions ---
from core_functions.compute_eigenvectors_sliding_corr import compute_eigs_corr
from core_functions.compute_eigenvectors_sliding_cov import compute_eigs_cov
from core_functions.compute_eigenvectors_iPA import compute_eigenvectors_ipa
from core_functions.dysco_entropy import dysco_entropy
from core_functions.dysco_norm import dysco_norm
from core_functions.dysco_distance import dysco_distance
from core_functions.dysco_mode_alignment import dysco_mode_alignment


class DyscoPipeline:
    def __init__(self, methods, params, data_folder, save_path, save_preferences, log_function=None, visualization_preferences=None):
        self.methods = methods
        self.params = params
        self.data_folder = data_folder
        self.save_path = save_path
        self.save_preferences = save_preferences
        self.log_function = log_function or self.default_log_function
        self.visualization_preferences = visualization_preferences or {}
        self.log_message(f"Visualization preferences: {self.visualization_preferences}")
        
        # Set matplotlib backend once at initialization
        plt.switch_backend('Agg')

    def map_params(self, method, params):
        mapping = {
            "Sliding Window Correlation": {
                "Window Size": "window_size",
                "Step Size": "step_size",
                "# Eigenvectors": "n_eigs"
            },
            "Sliding Window Covariance": {
                "Window Size": "window_size",
                "Step Size": "step_size",
                "# Eigenvectors": "n_eigs"
            },
            "Co-Fluctuation Matrix": {
                "Z-Score Type": "z_score_type"
            },
            "Instantaneous Phase Alignment": {
                "Phase Extraction Method": "phase_extraction_method"
            },
            "Instantaneous Phase Locking": {
                "Phase Extraction Method": "phase_extraction_method"
            },
            "Wavelet Coherence": {
                "Wavelet Type": "wavelet_type"
            }
        }
        mapped = {}
        for k, v in params.items():
            mapped_key = mapping.get(method, {}).get(k, k)
            mapped[mapped_key] = v
        return mapped

    def run(self):
        self.log_message(f"Starting pipeline with methods: {list(self.methods.keys())}")
        self.log_message(f"Data folder: {self.data_folder}")
        
        # Get list of files, excluding hidden files
        files = [f for f in os.listdir(self.data_folder) if not f.startswith('._')]
        self.log_message(f"Files to process: {files}")

        for method, method_params in self.methods.items():
            self.log_message(f"\n{'='*50}")
            self.log_message(f"Starting processing for method: {method}")
            self.log_message(f"{'='*50}")
            mapped_params = self.map_params(method, method_params)
            self.log_message(f"Parameters for {method}: {mapped_params}")

            for file_idx, file in enumerate(files, 1):
                try:
                    self.log_message(f"\n{'-'*30}")
                    self.log_message(f"Processing file {file_idx}/{len(files)}: {file}")
                    self.log_message(f"{'-'*30}")
                    
                    file_path = os.path.join(self.data_folder, file)
                    subject_id, ext = os.path.splitext(file)
                    
                    # Handle .dtseries.nii files
                    is_dtseries = False
                    if ext.lower() == '.nii':
                        subject_id, ext2 = os.path.splitext(subject_id)
                        if ext2.lower() == '.dtseries':
                            ext = '.dtseries.nii'
                            is_dtseries = True
                    
                    self.log_message(f"File type: {ext}")

                    # Handle supported file types
                    if ext.lower() in [".nii", ".mat", ".dtseries", ".dtseries.nii"]:
                        self.log_message(f"Processing {ext} file: {file}")
                        try:
                            # Process file
                            if is_dtseries:
                                self.log_message("Processing as HCP dtseries file")
                                data = self.process_dtseries(file_path)
                            elif ext.lower() == ".nii":
                                self.log_message("Processing as regular NIfTI file")
                                data = self.process_nifti(file_path)
                            elif ext.lower() == ".mat":
                                self.log_message("Processing as MATLAB file")
                                data = self.process_mat(file_path)

                            self.log_message(f"Data shape: {data.shape}")
                            results = {}
                            
                            # Process with selected method
                            if method == 'Sliding Window Covariance':
                                self.log_message("Running Sliding Window Covariance")
                                n_eigs = mapped_params.get('n_eigs', 5)
                                window_size = mapped_params.get('window_size', 30)
                                half_window = window_size // 2
                                eig_vec, eig_val = compute_eigs_cov(data, n_eigs, half_window)
                                results['Eigenvectors'] = eig_vec
                                results['Eigenvalues'] = eig_val
                            elif method == 'Sliding Window Correlation':
                                self.log_message("Running Sliding Window Correlation")
                                n_eigs = mapped_params.get('n_eigs', 5)
                                window_size = mapped_params.get('window_size', 30)
                                half_window = window_size // 2
                                eig_vec, eig_val = compute_eigs_corr(data, n_eigs, half_window)
                                results['Eigenvectors'] = eig_vec
                                results['Eigenvalues'] = eig_val
                            elif method == 'Instantaneous Phase Alignment':
                                self.log_message("Running Instantaneous Phase Alignment")
                                eig_vec, eig_val = compute_eigenvectors_ipa(data)
                                results['Eigenvectors'] = eig_vec
                                results['Eigenvalues'] = eig_val
                            elif method == 'Co-Fluctuation Matrix':
                                self.co_fluctuation_matrix(data, mapped_params, file)
                                continue
                            elif method == 'Instantaneous Phase Locking':
                                self.instantaneous_phase_locking(data, mapped_params, file)
                                continue
                            elif method == 'Wavelet Coherence':
                                self.wavelet_coherence(data, mapped_params, file)
                                continue
                            else:
                                self.log_message(f"Method {method} is not implemented.")
                                continue

                            # Clear large data structure
                            del data

                            # Compute derived metrics
                            if 'Eigenvalues' in results:
                                self.log_message("Computing derived metrics from eigenvalues")
                                eig_val = results['Eigenvalues']
                                results['Entropy'] = dysco_entropy(eig_val)
                                results['Norm_L1'] = dysco_norm(eig_val, 1)
                                results['Norm_L2'] = dysco_norm(eig_val, 2)
                                results['Norm_Inf'] = dysco_norm(eig_val, np.inf)
                            if 'Eigenvectors' in results:
                                self.log_message("Computing reconfiguration metrics")
                                eig_vec = results['Eigenvectors']
                                distances = []
                                mode_alignments = []
                                for t in range(eig_vec.shape[0] - 1):
                                    a = eig_vec[t]
                                    b = eig_vec[t + 1]
                                    distances.append(dysco_distance(a, b, 1))
                                    mode_alignments.append(dysco_mode_alignment(a, b))
                                results['Reconfiguration_Speed'] = np.array(distances)
                                results['Mode_Alignment'] = np.array(mode_alignments)

                            # Save results
                            self.log_message(f"Saving results for {subject_id}")
                            self.save_results(
                                method=method,
                                subject_id=subject_id,
                                **results
                            )

                            # Generate plots
                            try:
                                plot_types = self.visualization_preferences.get('plot_types', [])
                                save_options = self.visualization_preferences.get('save_options', {})
                                plot_format = save_options.get('format', 'png')
                                plot_dpi = save_options.get('dpi', 300)
                                method_dir = os.path.join(self.save_path, method)
                                subject_dir = os.path.join(method_dir, subject_id)
                                
                                if 'Entropy Time Series' in plot_types and 'Entropy' in results:
                                    self.log_message("Generating Entropy Time Series plot")
                                    self.plot_entropy(results['Entropy'], subject_dir, plot_format, plot_dpi)
                                if 'Reconfiguration Speed' in plot_types and 'Reconfiguration_Speed' in results:
                                    self.log_message("Generating Reconfiguration Speed plot")
                                    self.plot_reconfiguration_speed(results['Reconfiguration_Speed'], subject_dir, plot_format, plot_dpi)
                                if 'FCD Matrix' in plot_types and 'Eigenvectors' in results:
                                    self.log_message("Generating FCD Matrix plot")
                                    self.plot_fcd_matrix(results['Eigenvectors'], subject_dir, plot_format, plot_dpi)
                            except Exception as e:
                                self.log_message(f"Error during plotting: {str(e)}")
                                self.log_message(traceback.format_exc())
                            finally:
                                plt.close('all')
                                
                            # Clear results
                            del results
                            
                            self.log_message(f"Completed processing {file} for method {method}")
                        except Exception as e:
                            self.log_message(f"Error processing file {file}: {str(e)}")
                            self.log_message(traceback.format_exc())
                            plt.close('all')
                            continue
                    else:
                        self.log_message(f"Skipping unsupported file: {file}")
                        continue
                except Exception as e:
                    self.log_message(f"Error in main processing loop: {str(e)}")
                    self.log_message(traceback.format_exc())
                    plt.close('all')
                    continue

            self.log_message(f"Completed processing all files for method {method}")
            plt.close('all')
            
        self.log_message(f"Pipeline completed. Results saved to {self.save_path}")

    def default_log_function(self, message):
        print(message)

    def log_message(self, message):
        if self.log_function:
            self.log_function(message)

    def save_results(self, method, subject_id, **kwargs):
        method_dir = os.path.join(self.save_path, method)
        subject_dir = os.path.join(method_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        plot_types = self.visualization_preferences.get('plot_types', [])
        save_options = self.visualization_preferences.get('save_options', {})
        plot_format = save_options.get('format', 'png')
        plot_dpi = save_options.get('dpi', 300)
        self.log_message(f"Saving with plot_types: {plot_types}, format: {plot_format}, dpi: {plot_dpi}")
        for key, value in kwargs.items():
            if key not in self.save_preferences or not self.save_preferences[key]:
                self.log_message(f"Skipping save for {key} as per user preferences.")
                print(f"Skipping save for {key} as per user preferences.")
                continue
            if isinstance(value, np.ndarray):
                file_name = f"{key}.npy"
                file_path = os.path.join(subject_dir, file_name)
                np.save(file_path, value)
                self.log_message(f"Saved {key} for {subject_id} in {file_path}")
                print(f"Saved {key} for {subject_id} in {file_path}")
            elif isinstance(value, dict) and all(hasattr(v, 'savefig') for v in value.values()):
                for name, figure in value.items():
                    file_name = f"{name}.{plot_format}"
                    file_path = os.path.join(subject_dir, file_name)
                    figure.savefig(file_path, format=plot_format, dpi=plot_dpi)
                    self.log_message(f"Saved figure {name} for {subject_id} in {file_path}")
                    print(f"Saved figure {name} for {subject_id} in {file_path}")
            else:
                self.log_message(f"Unsupported data type for {key}. Skipping.")
                print(f"Unsupported data type for {key}. Skipping.")
        self.log_message(f"All requested results saved for {subject_id} in {subject_dir}")
        print(f"All requested results saved for {subject_id} in {subject_dir}")

    @staticmethod
    def process_nifti(file_path):
        """Process regular .nii files."""
        nifti = nb.load(file_path)
        data = nifti.get_fdata(dtype=np.float32)
        
        # Handle 4D data (time series)
        if len(data.shape) == 4:
            # Reshape to 2D: time x voxels
            data = data.reshape(data.shape[0], -1)
        elif len(data.shape) == 3:
            # Single volume, reshape to 2D: 1 x voxels
            data = data.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported NIfTI data shape: {data.shape}")
            
        # Remove any zero-variance voxels
        return data[:, ~np.all(data == 0, axis=0)]

    @staticmethod
    def process_dtseries(file_path):
        """Process HCP .dtseries.nii files."""
        cifti = nb.load(file_path)
        cifti_data = cifti.get_fdata(dtype=np.float32)
        axes = [cifti.header.get_axis(i) for i in range(cifti.ndim)]
        left_brain = DyscoPipeline.surf_data_from_cifti(cifti_data, axes[1], 'CIFTI_STRUCTURE_CORTEX_LEFT')
        brain_load = left_brain.T
        return brain_load[:, ~np.all(brain_load == 0, axis=0)]

    @staticmethod
    def surf_data_from_cifti(data, axis, surf_name):
        assert isinstance(axis, nb.cifti2.BrainModelAxis)
        for name, data_indices, model in axis.iter_structures():
            if name == surf_name:
                data = data.T[data_indices]
                vtx_indices = model.vertex
                surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
                surf_data[vtx_indices] = data
                return surf_data
        raise ValueError(f"No structure named {surf_name}")

    @staticmethod
    def process_mat(file_path):
        mat_data = loadmat(file_path)
        return mat_data['data']

    # --- Stubs for missing methods ---
    def co_fluctuation_matrix(self, data, params, file):
        print(f"[STUB] Co-Fluctuation Matrix not yet implemented. Params: {params}, File: {file}")
        self.log_message(f"[STUB] Co-Fluctuation Matrix not yet implemented. Params: {params}, File: {file}")

    def instantaneous_phase_alignment(self, data, params, file):
        print(f"[STUB] Instantaneous Phase Alignment not yet implemented. Params: {params}, File: {file}")
        self.log_message(f"[STUB] Instantaneous Phase Alignment not yet implemented. Params: {params}, File: {file}")

    def instantaneous_phase_locking(self, data, params, file):
        print(f"[STUB] Instantaneous Phase Locking not yet implemented. Params: {params}, File: {file}")
        self.log_message(f"[STUB] Instantaneous Phase Locking not yet implemented. Params: {params}, File: {file}")

    def wavelet_coherence(self, data, params, file):
        print(f"[STUB] Wavelet Coherence not yet implemented. Params: {params}, File: {file}")
        self.log_message(f"[STUB] Wavelet Coherence not yet implemented. Params: {params}, File: {file}")

    def plot_entropy(self, entropy, subject_dir, plot_format, plot_dpi):
        try:
            plt.figure(figsize=(8, 4))
            plt.plot(entropy)
            plt.title('Entropy Time Series')
            plt.xlabel('Time')
            plt.ylabel('Entropy')
            plt.tight_layout()
            file_path = os.path.join(subject_dir, f"Entropy.{plot_format}")
            plt.savefig(file_path, format=plot_format, dpi=plot_dpi)
            plt.close()
            self.log_message(f"Saved entropy plot: {file_path}")
        except Exception as e:
            self.log_message(f"Error generating entropy plot: {str(e)}")
            self.log_message(traceback.format_exc())

    def plot_reconfiguration_speed(self, reconfig_speed, subject_dir, plot_format, plot_dpi):
        try:
            plt.figure(figsize=(8, 4))
            plt.plot(reconfig_speed)
            plt.title('Reconfiguration Speed')
            plt.xlabel('Time')
            plt.ylabel('Speed')
            plt.tight_layout()
            file_path = os.path.join(subject_dir, f"Reconfiguration_Speed.{plot_format}")
            plt.savefig(file_path, format=plot_format, dpi=plot_dpi)
            plt.close()
            self.log_message(f"Saved reconfiguration speed plot: {file_path}")
        except Exception as e:
            self.log_message(f"Error generating reconfiguration speed plot: {str(e)}")
            self.log_message(traceback.format_exc())

    def plot_fcd_matrix(self, eig_vec, subject_dir, plot_format, plot_dpi):
        try:
            n_time = eig_vec.shape[0]
            fcd_matrix = np.zeros((n_time, n_time))
            for i in range(n_time):
                for j in range(n_time):
                    fcd_matrix[i, j] = dysco_distance(eig_vec[i], eig_vec[j], 1)
            plt.figure(figsize=(6, 5))
            sns.heatmap(fcd_matrix, cmap='viridis')
            plt.title('FCD Matrix')
            plt.xlabel('Time')
            plt.ylabel('Time')
            plt.tight_layout()
            file_path = os.path.join(subject_dir, f"FCD_Matrix.{plot_format}")
            plt.savefig(file_path, format=plot_format, dpi=plot_dpi)
            plt.close()
            self.log_message(f"Saved FCD matrix plot: {file_path}")
        except Exception as e:
            self.log_message(f"Error generating FCD matrix plot: {str(e)}")
            self.log_message(traceback.format_exc())

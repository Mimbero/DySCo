import sys
import webbrowser
import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import logging
import time
import traceback
import gc

current_dir = os.path.dirname(os.path.realpath(__file__))  # Path to GUI_V3.py
python_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # Path to 'Python'
sys.path.append(python_dir)

from Pipelines.Dysco_GUI_Pipeline import DyscoPipeline
from utils import logger, ErrorHandler, LoadingOverlay

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QComboBox, QFileDialog, QTextEdit, QProgressBar,
    QListWidget, QListWidgetItem, QCheckBox, QMessageBox, QGroupBox, QFormLayout, QDoubleSpinBox
)

from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl, Qt, QSize, QThread, QMetaObject, Q_ARG
from PyQt6.QtGui import QPixmap, QIcon, QColor, QPalette, QTextCursor





class DySCoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DySCo GUI")
        self.setMinimumSize(800, 600)
        
        # Set up logging to file
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"dysco_gui_{time.strftime('%Y%m%d_%H%M%S')}.log")
        self.file_logger = logging.getLogger('dysco_gui')
        self.file_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.file_logger.addHandler(file_handler)
        self.file_logger.info("DySCo GUI started")
        
        # Initialize progress bar and log output
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        
        # Initialize loading overlay
        self.loading_overlay = LoadingOverlay(self)
        
        # Initialize multimedia features
        self.setup_multimedia()
        
        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # Setup welcome tab
        self.setup_welcome_tab()
        
        # Setup other tabs
        self.setup_data_tab()
        self.setup_parameters_tab()
        self.setup_visualization_tab()
        self.setup_run_tab()
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Show welcome tab by default
        self.tabs.setCurrentIndex(0)
        
        logger.info("DySCo GUI initialized successfully")
        self.file_logger.info("DySCo GUI initialized successfully")

    def setup_multimedia(self):
        """Initialize multimedia features with error handling."""
        try:
            # Video player setup
            self.media_player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.media_player.setAudioOutput(self.audio_output)
            
            # Music player setup
            self.music_player = QMediaPlayer()
            self.music_audio_output = QAudioOutput()
            self.music_player.setAudioOutput(self.music_audio_output)
            
            self.is_music_playing = False
            logger.info("Multimedia features initialized successfully")
            
        except Exception as e:
            error_msg = (
                "Failed to initialize multimedia features. "
                "Please ensure you have installed PyQt6 and its multimedia dependencies:\n"
                "pip install PyQt6 PyQt6-Qt6 PyQt6-sip"
            )
            ErrorHandler.handle_error(self, e, "Multimedia Error")
            self.media_player = None
            self.music_player = None
            self.is_music_playing = False

    def setup_welcome_tab(self):
        welcome_tab = QWidget()
        layout = QVBoxLayout(welcome_tab)
        
        # Create video player
        self.video_widget = QVideoWidget()
        if self.media_player:
            self.media_player.setVideoOutput(self.video_widget)
        
        # Set video source
        video_path = os.path.join(os.path.dirname(__file__), "resources", "DySCO_opener.mp4")
        logger.info(f"Loading video from: {video_path}")
        
        if os.path.exists(video_path) and self.media_player:
            self.media_player.setSource(QUrl.fromLocalFile(video_path))
            # Set volume and start playing
            # self.media_player.setVolume(50)
            self.media_player.play()
            logger.info("Video loaded and started playing")
        else:
            logger.warning(f"Video file not found at: {video_path}")
            ErrorHandler.handle_warning("Video file not found", "The welcome video could not be loaded.", self)
        
        # Add video widget to layout
        layout.addWidget(self.video_widget)
        
        # Create button container
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        
        # Create disco ball button
        disco_ball_path = os.path.join(os.path.dirname(__file__), "resources", "Disco_ball_blackBG.jpeg")
        try:
            if os.path.exists(disco_ball_path):
                self.disco_ball_button = QPushButton()
                self.disco_ball_button.setIcon(QIcon(disco_ball_path))
                self.disco_ball_button.setIconSize(QSize(50, 50))
                self.disco_ball_button.clicked.connect(self.toggle_music)
                button_layout.addWidget(self.disco_ball_button)
            else:
                logger.warning(f"Disco ball image not found at: {disco_ball_path}")
                ErrorHandler.handle_warning("Image not found", "The disco ball image could not be loaded.", self)
        except Exception as e:
            logger.warning(f"Error loading disco ball image: {str(e)}")
            ErrorHandler.handle_warning("Image loading error", "There was an error loading the disco ball image.", self)
        
        # Add button container to layout
        layout.addWidget(button_container)
        
        # Add welcome tab
        self.tabs.addTab(welcome_tab, "Welcome")
        
        # Connect video player signals
        if self.media_player:
            self.media_player.errorOccurred.connect(self.handle_video_error)
            self.media_player.mediaStatusChanged.connect(self.handle_media_status)

    def handle_video_error(self, error, error_string):
        logger.error(f"Video player error: {error_string}")
        ErrorHandler.handle_error("Video playback error", f"An error occurred during video playback: {error_string}", self)

    def handle_media_status(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            logger.info("Video playback ended, attempting to loop")
            self.loop_video()

    def loop_video(self):
        try:
            logger.info("Attempting to loop video")
            self.media_player.setPosition(0)
            self.media_player.play()
            logger.info("Video looped successfully")
        except Exception as e:
            logger.warning(f"Error looping video: {str(e)}")
            ErrorHandler.handle_warning("Video loop error", "There was an error looping the video.", self)

    def toggle_music(self):
        """Toggle the music on and off when the disco ball button is clicked."""
        if self.music_player is None:
            ErrorHandler.handle_warning(
                "Music Player",
                "Music playback is not available on this system.",
                self
            )
            return

        if self.is_music_playing:
            # Stop music
            self.music_player.stop()
            self.is_music_playing = False
            logger.info("Music playback stopped")
        else:
            try:
                # Play music
                music_path = os.path.join(os.path.dirname(__file__), "resources", "The_Tramps_Disco_Inferno.mp3")
                logger.info(f"Looking for music at: {music_path}")
                
                if os.path.exists(music_path):
                    logger.info(f"Music file found at: {music_path}")
                    self.music_player.setSource(QUrl.fromLocalFile(music_path))
                    self.music_player.play()
                    self.is_music_playing = True
                    logger.info("Music playback started")
                else:
                    ErrorHandler.handle_warning(
                        "Missing Music File",
                        f"Music file not found at {music_path}",
                        self
                    )
            except Exception as e:
                ErrorHandler.handle_error(
                    "Music Player Error",
                    f"Failed to play music: {str(e)}",
                    self
                )
                self.is_music_playing = False

    def get_started(self):
        """Handle the action to guide the user to get started."""
        QMessageBox.information(self, "Get Started", "Welcome! Let's guide you through getting started with DySCo!")

    def open_settings(self):
        """Handle the action to open the settings page."""
        QMessageBox.information(self, "Settings", "Navigating to settings...")

    def open_tutorial(self):
        """Open a Jupyter Notebook tutorial for DySCo."""
        notebook_path = "/path/to/your/tutorial_notebook.ipynb"  # Update to your notebook path
        subprocess.Popen(["jupyter", "notebook", notebook_path])

    def open_link(self, url):
        """Open a URL in the default web browser."""
        webbrowser.open(url)

    # ===========================
    # PARAMETER TAB
    # ===========================

    def setup_data_tab(self):
        """Set up the data input tab."""
        data_tab = QWidget()
        layout = QVBoxLayout(data_tab)
        
        # Create file selection group
        file_group = QGroupBox("Data Files")
        file_layout = QVBoxLayout()
        
        # Add input data folder selection
        input_group = QGroupBox("Input Data")
        input_layout = QVBoxLayout()
        
        self.data_folder_label = QLabel("No input folder selected")
        self.data_folder_label.setStyleSheet("color: gray;")
        
        select_folder_btn = QPushButton("Select Input Data Folder")
        select_folder_btn.clicked.connect(self.select_data_folder)
        
        input_layout.addWidget(self.data_folder_label)
        input_layout.addWidget(select_folder_btn)
        input_group.setLayout(input_layout)
        file_layout.addWidget(input_group)
        
        # Add output folder selection
        output_group = QGroupBox("Output Location")
        output_layout = QVBoxLayout()
        
        self.output_folder_label = QLabel("No output folder selected")
        self.output_folder_label.setStyleSheet("color: gray;")
        
        select_output_btn = QPushButton("Select Output Folder")
        select_output_btn.clicked.connect(self.select_output_folder)
        
        output_layout.addWidget(self.output_folder_label)
        output_layout.addWidget(select_output_btn)
        output_group.setLayout(output_layout)
        file_layout.addWidget(output_group)
        
        # Add task time series section
        task_group = QGroupBox("Task Time Series (Optional)")
        task_layout = QVBoxLayout()
        
        self.use_task_ts = QCheckBox("Include Task Time Series")
        self.use_task_ts.stateChanged.connect(self.toggle_task_ts)
        
        self.task_ts_label = QLabel("No task time series selected")
        self.task_ts_label.setStyleSheet("color: gray;")
        self.task_ts_label.setEnabled(False)
        
        self.select_task_ts_btn = QPushButton("Select Task Time Series")
        self.select_task_ts_btn.clicked.connect(self.select_task_ts)
        self.select_task_ts_btn.setEnabled(False)
        
        task_layout.addWidget(self.use_task_ts)
        task_layout.addWidget(self.task_ts_label)
        task_layout.addWidget(self.select_task_ts_btn)
        task_group.setLayout(task_layout)
        file_layout.addWidget(task_group)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Add data preview section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        
        self.data_preview = QTextEdit()
        self.data_preview.setReadOnly(True)
        self.data_preview.setPlaceholderText("Select a data folder to preview its contents...")
        
        preview_layout.addWidget(self.data_preview)
        preview_group.setLayout(preview_layout)
        
        layout.addWidget(preview_group)
        
        # Add stretch to push widgets to the top
        layout.addStretch()
        
        # Add the tab
        self.tabs.addTab(data_tab, "Data")
        
        # Initialize task time series variables
        self.task_ts_path = None

    def select_output_folder(self):
        """Handle output folder selection."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.output_folder_label.setText(folder)
            logger.info(f"Selected output folder: {folder}")

    def toggle_task_ts(self, state):
        """Enable/disable task time series selection."""
        enabled = state == Qt.CheckState.Checked
        self.task_ts_label.setEnabled(enabled)
        self.select_task_ts_btn.setEnabled(enabled)
        if not enabled:
            self.task_ts_path = None
            self.task_ts_label.setText("No task time series selected")

    def select_task_ts(self):
        """Handle task time series file selection."""
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Task Time Series File",
            "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)"
        )
        
        if file:
            self.task_ts_path = file
            self.task_ts_label.setText(os.path.basename(file))
            logger.info(f"Selected task time series file: {file}")

    def select_data_folder(self):
        """Handle data folder selection."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.data_folder_label.setText(folder)
            self.update_data_preview(folder)
            logger.info(f"Selected data folder: {folder}")

    def update_data_preview(self, folder):
        """Update the data preview with folder contents."""
        try:
            files = os.listdir(folder)
            preview_text = "Folder contents:\n\n"
            
            for file in files:
                if file.endswith(('.nii', '.mat')):
                    preview_text += f"✓ {file}\n"
                else:
                    preview_text += f"✗ {file} (unsupported format)\n"
            
            self.data_preview.setText(preview_text)
            logger.info(f"Updated data preview for folder: {folder}")
            
        except Exception as e:
            logger.error(f"Error updating data preview: {str(e)}")
            ErrorHandler.handle_error(
                "Data Preview Error",
                f"Could not read folder contents: {str(e)}",
                self
            )

    def setup_parameters_tab(self):
        """Set up the parameters tab for method selection and parameter input."""
        param_tab = QWidget()
        main_layout = QVBoxLayout(param_tab)

        # Create horizontal split layout for methods and parameters
        split_layout = QHBoxLayout()
        
        # Left side: Method selection
        method_group = QGroupBox("Select DySCo Methods")
        method_layout = QVBoxLayout()
        self.method_checkboxes = {}
        self.method_param_widgets = {}

        # Define methods and their parameters
        methods = {
            "Sliding Window Correlation": [
                ("Window Size", QSpinBox, {"min": 1, "max": 1000, "value": 30, "suffix": " TR"}),
                ("Step Size", QSpinBox, {"min": 1, "max": 100, "value": 1, "suffix": " TR"}),
                ("# Eigenvectors", QSpinBox, {"min": 1, "max": 50, "value": 5})
            ],
            "Sliding Window Covariance": [
                ("Window Size", QSpinBox, {"min": 1, "max": 1000, "value": 30, "suffix": " TR"}),
                ("Step Size", QSpinBox, {"min": 1, "max": 100, "value": 1, "suffix": " TR"}),
                ("# Eigenvectors", QSpinBox, {"min": 1, "max": 50, "value": 5})
            ],
            "Co-Fluctuation Matrix": [
                ("Z-Score Type", QComboBox, {"items": ["Whole Recording", "Within Window"]})
            ],
            "Instantaneous Phase Alignment": [
                ("Phase Extraction Method", QComboBox, {"items": ["Hilbert Transform", "Wavelet Transform"]})
            ],
            "Instantaneous Phase Locking": [
                ("Phase Extraction Method", QComboBox, {"items": ["Hilbert Transform", "Wavelet Transform"]})
            ],
            "Wavelet Coherence": [
                ("Wavelet Type", QComboBox, {"items": ["Morlet", "Mexican Hat", "Paul"]})
            ]
        }

        # Create checkboxes and view buttons for each method
        for method, params in methods.items():
            # Create horizontal layout for method row
            method_row = QHBoxLayout()
            
            # Add checkbox
            checkbox = QCheckBox(method)
            self.method_checkboxes[method] = checkbox
            method_row.addWidget(checkbox)

            # Add view parameters button
            view_button = QPushButton("View Parameters")
            view_button.clicked.connect(lambda checked, m=method: self.show_method_parameters(m))
            method_row.addWidget(view_button)

            # Add the row to the method layout
            method_layout.addLayout(method_row)

            # Create parameter widget for this method
            param_widget = QWidget()
            param_layout = QFormLayout(param_widget)
            param_fields = {}
            for label, widget_type, opts in params:
                if widget_type == QSpinBox:
                    widget = QSpinBox()
                    widget.setRange(opts["min"], opts["max"])
                    widget.setValue(opts["value"])
                    if "suffix" in opts:
                        widget.setSuffix(opts["suffix"])
                elif widget_type == QComboBox:
                    widget = QComboBox()
                    widget.addItems(opts["items"])
                else:
                    continue
                param_layout.addRow(QLabel(label), widget)
                param_fields[label] = widget
            self.method_param_widgets[method] = (param_widget, param_fields)
            param_widget.setVisible(False)  # Initially hidden

        method_group.setLayout(method_layout)
        split_layout.addWidget(method_group)

        # Right side: Parameter display
        param_group = QGroupBox("Method Parameters")
        param_layout = QVBoxLayout()
        self.param_display = QWidget()
        param_layout.addWidget(self.param_display)
        param_group.setLayout(param_layout)
        split_layout.addWidget(param_group)

        # Add the split layout to the main layout
        main_layout.addLayout(split_layout)

        # Save output preferences group
        save_outputs_group = QGroupBox("Save Output Preferences")
        save_outputs_layout = QVBoxLayout()
        self.save_output_checkboxes = {}
        output_types = [
            "Eigenvectors", "Eigenvalues", "Entropy", "Norm_L1", "Norm_L2", "Norm_Inf",
            "Reconfiguration_Speed", "Mode_Alignment"
        ]
        for output in output_types:
            checkbox = QCheckBox(output)
            checkbox.setChecked(True)
            self.save_output_checkboxes[output] = checkbox
            save_outputs_layout.addWidget(checkbox)
        save_outputs_group.setLayout(save_outputs_layout)
        main_layout.addWidget(save_outputs_group)

        # Add the tab
        self.tabs.addTab(param_tab, "Parameters")

    def show_method_parameters(self, method):
        """Show parameters for the selected method."""
        # Clear the parameter display area
        for i in reversed(range(self.param_display.layout().count() if self.param_display.layout() else 0)):
            self.param_display.layout().itemAt(i).widget().setParent(None)

        # Add the method's parameters to the display
        param_widget, _ = self.method_param_widgets[method]
        if not self.param_display.layout():
            self.param_display.setLayout(QVBoxLayout())
        self.param_display.layout().addWidget(param_widget)
        param_widget.setVisible(True)

    def get_selected_methods_and_params(self):
        """Return selected methods and their parameters as a dict."""
        selected = {}
        self.log_output.append("Checking selected methods:")
        for method, checkbox in self.method_checkboxes.items():
            is_checked = checkbox.isChecked()
            self.log_output.append(f"Method {method}: {'Selected' if is_checked else 'Not selected'}")
            if is_checked:
                param_widget, param_fields = self.method_param_widgets[method]
                params = {}
                for label, widget in param_fields.items():
                    if isinstance(widget, QSpinBox):
                        params[label] = widget.value()
                    elif isinstance(widget, QComboBox):
                        params[label] = widget.currentText()
                selected[method] = params
                self.log_output.append(f"Parameters for {method}: {params}")
        return selected

    def setup_run_tab(self):
        """Set up run and monitor tab."""
        self.run_tab = QWidget()
        layout = QVBoxLayout()

        # Run button
        self.run_button = QPushButton("Run DySCo")
        self.run_button.clicked.connect(self.run_dysco_pipeline)
        layout.addWidget(self.run_button)

        # Progress bar
        layout.addWidget(self.progress_bar)

        # Log output
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log_output)

        self.run_tab.setLayout(layout)
        self.tabs.addTab(self.run_tab, "Run & Monitor")

    def collect_save_preferences(self):
        """Collect and return save preferences from the GUI."""
        return {key: cb.isChecked() for key, cb in self.save_output_checkboxes.items()}

    def log_message(self, message):
        """Log message to both GUI and file."""
        # Use QMetaObject.invokeMethod to ensure thread safety
        QMetaObject.invokeMethod(self.log_output, "append", Qt.ConnectionType.QueuedConnection, Q_ARG(str, message))
        self.file_logger.info(message)

    def run_dysco_pipeline(self):
        """Run the DySCo pipeline with the set parameters."""
        try:
            # Disable run button while processing
            self.run_button.setEnabled(False)
            
            # Stop video playback before starting pipeline
            if hasattr(self, 'media_player') and self.media_player:
                self.media_player.stop()
                self.media_player.setPosition(0)

            # Retrieve selected dFC methods and their parameters
            selected_methods_dict = self.get_selected_methods_and_params()

            # Check if any method is selected
            if not selected_methods_dict:
                self.log_message("Error: No dFC method selected.")
                self.run_button.setEnabled(True)
                return

            # Retrieve the data folder
            data_folder = self.data_folder_label.text()
            if not data_folder or data_folder == "No input folder selected":
                self.log_message("Error: Data folder not selected.")
                self.run_button.setEnabled(True)
                return

            # Retrieve save path
            save_path = self.output_folder_label.text()
            if not save_path or save_path == "No output folder selected":
                self.log_message("Error: Save path not specified.")
                self.run_button.setEnabled(True)
                return

            # Visualization preferences
            visualization_preferences = self.get_visualization_preferences()

            # Log the configuration
            self.log_message("\nStarting DySCo Pipeline...")
            self.log_message(f"Data Folder: {data_folder}")
            self.log_message(f"Save Path: {save_path}")
            self.log_message(f"Selected Methods: {', '.join(selected_methods_dict.keys())}")
            for method, params in selected_methods_dict.items():
                self.log_message(f"Parameters for {method}: {params}")
            self.log_message(f"Visualization Preferences: {visualization_preferences}")

            save_preferences = self.collect_save_preferences()

            # Log save preferences
            self.log_message(
                f"Save Preferences: {', '.join([key for key, value in save_preferences.items() if value])}")

            def pipeline_task():
                try:
                    self.log_message("Running DySCo Pipeline...")
                    pipeline = DyscoPipeline(
                        methods=selected_methods_dict,
                        params={},
                        data_folder=data_folder,
                        save_path=save_path,
                        save_preferences=save_preferences,
                        log_function=self.log_message,
                        visualization_preferences=visualization_preferences
                    )
                    pipeline.run()
                    self.log_message("Pipeline completed successfully.")
                    QMetaObject.invokeMethod(self.progress_bar, "setValue", Qt.ConnectionType.QueuedConnection, Q_ARG(int, 100))
                except Exception as e:
                    error_msg = f"Error in pipeline: {str(e)}\n{traceback.format_exc()}"
                    self.log_message(error_msg)
                    QMetaObject.invokeMethod(self.progress_bar, "setValue", Qt.ConnectionType.QueuedConnection, Q_ARG(int, 0))
                finally:
                    # Ensure we clean up any remaining resources
                    plt.close('all')
                    gc.collect()
                    # Re-enable run button
                    QMetaObject.invokeMethod(self.run_button, "setEnabled", Qt.ConnectionType.QueuedConnection, Q_ARG(bool, True))

            # Run the pipeline in a separate thread
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(pipeline_task)
            
            # Add a callback to handle completion
            def on_completion(future):
                try:
                    future.result()  # This will raise any exceptions that occurred
                except Exception as e:
                    self.log_message(f"Pipeline thread error: {str(e)}")
                finally:
                    executor.shutdown(wait=False)
                    # Restart video playback after pipeline completes
                    if hasattr(self, 'media_player') and self.media_player:
                        QMetaObject.invokeMethod(self.media_player, "play", Qt.ConnectionType.QueuedConnection)

            future.add_done_callback(on_completion)

        except Exception as e:
            error_msg = f"Error in GUI: {str(e)}\n{traceback.format_exc()}"
            self.log_message(error_msg)
            self.progress_bar.setValue(0)
            self.run_button.setEnabled(True)

    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        # Update loading overlay size
        self.loading_overlay.setGeometry(0, 0, self.width(), self.height())

    def apply_dark_theme(self):
        """Apply a dark theme to the GUI."""
        # Set the application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 8px 12px;
                border: 1px solid #4c4c4c;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4c4c4c;
                border-bottom: 1px solid #4c4c4c;
            }
            QTabBar::tab:hover {
                background-color: #454545;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #4c4c4c;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #454545;
            }
            QPushButton:pressed {
                background-color: #505050;
            }
            QLineEdit, QTextEdit, QSpinBox, QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #4c4c4c;
                padding: 3px;
                border-radius: 3px;
            }
            QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {
                border: 1px solid #5c5c5c;
            }
            QGroupBox {
                border: 1px solid #4c4c4c;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QProgressBar {
                border: 1px solid #4c4c4c;
                border-radius: 3px;
                text-align: center;
                background-color: #3c3c3c;
            }
            QProgressBar::chunk {
                background-color: #4c4c4c;
                width: 10px;
                margin: 0.5px;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #3c3c3c;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #5c5c5c;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background-color: #3c3c3c;
                height: 10px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #5c5c5c;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        
        logger.info("Dark theme applied successfully")

    def setup_visualization_tab(self):
        """Set up the visualization tab for collecting plot preferences."""
        vis_tab = QWidget()
        layout = QVBoxLayout(vis_tab)
        
        # Create plot selection group
        selection_group = QGroupBox("Plot Selection")
        selection_layout = QVBoxLayout()
        
        # Add plot type checkboxes
        self.plot_checkboxes = {}
        plot_types = [
            "Eigenvector Time Series",
            "Eigenvalue Distribution",
            "Entropy Time Series",
            "Reconfiguration Speed",
            "FCD Matrix"
        ]
        
        for plot_type in plot_types:
            checkbox = QCheckBox(plot_type)
            self.plot_checkboxes[plot_type] = checkbox
            selection_layout.addWidget(checkbox)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Create save options group
        save_group = QGroupBox("Save Options")
        save_layout = QVBoxLayout()
        
        # Add format selection
        format_layout = QHBoxLayout()
        self.save_format = QComboBox()
        self.save_format.addItems(["PNG", "PDF", "SVG"])
        format_layout.addWidget(QLabel("Save Format:"))
        format_layout.addWidget(self.save_format)
        
        # Add DPI selection
        self.save_dpi = QSpinBox()
        self.save_dpi.setRange(72, 600)
        self.save_dpi.setValue(300)
        self.save_dpi.setSuffix(" DPI")
        format_layout.addWidget(QLabel("Resolution:"))
        format_layout.addWidget(self.save_dpi)
        
        save_layout.addLayout(format_layout)
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        # Add the tab
        self.tabs.addTab(vis_tab, "Visualization")

    def get_visualization_preferences(self):
        """Get the current visualization preferences from the UI."""
        return {
            'plot_types': [plot_type for plot_type, checkbox in self.plot_checkboxes.items() 
                          if checkbox.isChecked()],
            'save_options': {
                'format': self.save_format.currentText().lower(),
                'dpi': self.save_dpi.value()
            }
        }


def main():
    app = QApplication(sys.argv)
    window = DySCoGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

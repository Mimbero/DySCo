import sys
import webbrowser
import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

current_dir = os.path.dirname(os.path.realpath(__file__))  # Path to GUI_V3.py
python_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # Path to 'Python'
sys.path.append(python_dir)

from Pipelines.Dysco_GUI_Pipeline import DyscoPipeline
# from Python.Pipelines.Dysco_GUI_Pipeline import DyscoPipeline

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QComboBox, QFileDialog, QTextEdit, QProgressBar,
    QListWidget, QListWidgetItem, QCheckBox, QMessageBox
)

from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl, Qt, QSize  # Import QSize here
from PyQt6.QtGui import QPixmap, QIcon, QColor, QPalette
from PyQt6.QtWidgets import QMessageBox, QApplication





class DySCoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DySCo Framework GUI")
        self.setGeometry(100, 100, 900, 700)

        # Initialize progress bar and log output attributes
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.log_output = QTextEdit()

        # Main tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Initialize multimedia features with error handling
        self.setup_multimedia()

        # Initialize tabs
        self.setup_welcome_tab()
        self.setup_data_and_parameters_tab()
        # self.setup_data_tab()
        self.setup_run_tab()

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
            
        except Exception as e:
            print(f"Error setting up multimedia: {str(e)}")
            print("Please ensure you have installed PyQt6 and its multimedia dependencies:")
            print("pip install PyQt6 PyQt6-Qt6 PyQt6-sip")
            self.media_player = None
            self.music_player = None
            self.is_music_playing = False

    def setup_welcome_tab(self):
        """Set up the welcome tab with a disco ball video and interactive elements."""
        self.welcome_tab = QWidget()
        layout = QVBoxLayout()
        
        # Set dark theme
        self.welcome_tab.setStyleSheet("""
            QWidget {
                background-color: black;
                color: white;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #2c2c2c;
                color: white;
                border: 1px solid #3c3c3c;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3c3c3c;
            }
            QPushButton:pressed {
                background-color: #4c4c4c;
            }
        """)

        # Video player setup with error handling
        if self.media_player is not None:
            try:
                self.video_widget = QVideoWidget()
                self.media_player.setVideoOutput(self.video_widget)
                self.video_widget.setMinimumSize(800, 400)
                self.video_widget.setStyleSheet("background-color: black;")
                
                # Set video source with error handling
                video_path = Path(current_dir) / "DySCO_opener.mp4"
                print(f"Looking for video at: {video_path}")  # Debug: Print video path
                if video_path.exists():
                    print(f"Video file found at: {video_path}")  # Debug: Confirm file exists
                    media_content = QUrl.fromLocalFile(str(video_path))
                    self.media_player.setSource(media_content)
                    
                    # Debug: Print media player state
                    print(f"Media player state: {self.media_player.playbackState()}")
                    print(f"Media player error: {self.media_player.error()}")
                    
                    self.media_player.play()
                    self.media_player.mediaStatusChanged.connect(self.loop_video)
                    layout.addWidget(self.video_widget, alignment=Qt.AlignmentFlag.AlignCenter)
                else:
                    print(f"Warning: Video file not found at {video_path}")
            except Exception as e:
                print(f"Warning: Video player initialization failed: {str(e)}")

        # Title layout with disco ball button
        title_layout = QHBoxLayout()

        # Disco ball button with error handling
        disco_ball_button = QPushButton()
        disco_ball_button.setFixedSize(80, 80)
        try:
            disco_ball_path = Path(current_dir) / "Disco_ball_blackBG.jpeg"
            if disco_ball_path.exists():
                pixmap = QPixmap(str(disco_ball_path))
                icon = QIcon(pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                disco_ball_button.setIcon(icon)
                disco_ball_button.setIconSize(disco_ball_button.size())
        except Exception as e:
            print(f"Warning: Could not load disco ball image: {str(e)}")

        disco_ball_button.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
            }
            QPushButton:hover {
                background-color: rgba(173, 216, 230, 100);
            }
            QPushButton:pressed {
                background-color: rgba(0, 0, 255, 100);
            }
        """)
        disco_ball_button.clicked.connect(self.toggle_music)

        # Disco-style title with modern styling
        title_label = QLabel("<h1>Welcome to the DySCo Framework!</h1>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            color: white;
            font-family: 'Arial', sans-serif;
            font-size: 32px;
            font-weight: bold;
            padding: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF00FF, stop:1 #00FFFF);
            color: white;
        """)

        # Add button and title to title layout
        title_layout.addWidget(disco_ball_button)
        title_layout.addWidget(title_label)

        # Center title layout
        title_widget = QWidget()
        title_widget.setLayout(title_layout)
        layout.addWidget(title_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        # Buttons for navigation or actions
        button_layout = QVBoxLayout()

        # Create buttons with consistent styling
        buttons = [
            ("Get Started", "Click to get started with project", self.get_started),
            ("Documentation", "View the full documentation for DySCo", lambda: self.open_link("https://github.com/Mimbero/DySCo")),
            ("Tutorial", "Watch the tutorial to understand the framework", self.open_tutorial),
            ("Settings", "Open settings", self.open_settings)
        ]

        for text, tooltip, callback in buttons:
            button = QPushButton(text)
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            button_layout.addWidget(button)

        layout.addLayout(button_layout)

        # Add a footer message
        footer_label = QLabel("<p>DySCo Framework - Your solution for dynamic connectivity analysis.</p>")
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(footer_label)

        # Set the layout
        self.welcome_tab.setLayout(layout)
        self.tabs.addTab(self.welcome_tab, "Welcome")

    def loop_video(self, status):
        """Loop the video when it ends."""
        if self.media_player is None:
            return
            
        print(f"Video status changed to: {status}")  # Debug: Print status changes
        
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            try:
                print("Attempting to loop video")  # Debug: Print loop attempt
                self.media_player.setPosition(0)
                self.media_player.play()
                print("Video looped successfully")  # Debug: Print success
            except Exception as e:
                print(f"Warning: Could not loop video: {str(e)}")

    def toggle_music(self):
        """Toggle the music on and off when the disco ball button is clicked."""
        if self.music_player is None:
            QMessageBox.warning(self, "Music Player", "Music playback is not available on this system.")
            return

        if self.is_music_playing:
            # Stop music
            self.music_player.stop()
            self.is_music_playing = False
        else:
            try:
                # Play music
                music_path = Path(current_dir) / "The_Tramps_Disco_Inferno.mp3"
                print(f"Looking for music at: {music_path}")  # Debug: Print music path
                if music_path.exists():
                    print(f"Music file found at: {music_path}")  # Debug: Confirm file exists
                    self.music_player.setSource(QUrl.fromLocalFile(str(music_path)))
                    
                    # Debug: Print music player state
                    print(f"Music player state: {self.music_player.playbackState()}")
                    print(f"Music player error: {self.music_player.error()}")
                    
                    self.music_player.play()
                    self.is_music_playing = True
                else:
                    print(f"Warning: Music file not found at {music_path}")
                    QMessageBox.warning(self, "Music Player", "Music file not found.")
            except Exception as e:
                print(f"Error playing music: {str(e)}")  # Debug: Print detailed error
                QMessageBox.warning(self, "Music Player", f"Could not play music: {str(e)}")
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

    def setup_data_and_parameters_tab(self):
        """Set up the combined data loading, parameter setting, visualization, and save options tab."""
        self.data_and_parameters_tab = QWidget()
        layout = QVBoxLayout()

        # Top Section: Data and Parameters
        top_section = QHBoxLayout()

        # Left Section: List of dFC methods with checkboxes
        self.method_list = QListWidget()
        dfc_methods = ["Sliding Window Covariance", "Sliding Window Correlation", "Instantaneous Phase Alignment",
                       "Instantaneous Phase Locking", "Co-Activation"]
        for method in dfc_methods:
            item = QListWidgetItem(method)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.method_list.addItem(item)
        self.method_list.currentItemChanged.connect(self.update_parameters_section)
        top_section.addWidget(self.method_list, 1)

        # Center Section: Parameter fields for the selected method
        self.parameter_area = QVBoxLayout()
        self.parameter_placeholder = QLabel("Select a dFC method to set parameters.")
        self.parameter_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.parameter_area.addWidget(self.parameter_placeholder)

        self.parameter_widget = QWidget()
        self.parameter_widget.setLayout(self.parameter_area)

        top_section.addWidget(self.parameter_widget, 3)

        # Right Section: Data loading button
        self.data_button_layout = QVBoxLayout()
        data_label = QLabel("Load Data Folder:")
        self.data_folder_path = QLineEdit()
        self.data_folder_path.setPlaceholderText("Path to data folder")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.select_data_folder)
        self.data_button_layout.addWidget(data_label)
        self.data_button_layout.addWidget(self.data_folder_path)
        self.data_button_layout.addWidget(browse_button)
        top_section.addLayout(self.data_button_layout, 1)

        layout.addLayout(top_section)

        # Middle Section: Visualization Options
        visualization_section = QVBoxLayout()
        visualization_section.addWidget(QLabel("<b>Visualization Options</b>"))

        self.visualize_checkbox = QCheckBox("Visualize")
        self.visualize_checkbox.stateChanged.connect(self.update_visualization_options)
        visualization_section.addWidget(self.visualize_checkbox)

        self.plot_list = QListWidget()
        plots = ["Signal Distribution", "Entropy Plot", "Reconfiguration Speed Plot", "FCD Matrix"]
        for plot in plots:
            plot_item = QListWidgetItem(plot)
            plot_item.setCheckState(Qt.CheckState.Unchecked)
            self.plot_list.addItem(plot_item)
        self.plot_list.setEnabled(False)
        visualization_section.addWidget(self.plot_list)

        # Task time series
        self.task_series_checkbox = QCheckBox("Include Task Time Series")
        self.task_series_checkbox.stateChanged.connect(self.toggle_task_time_series)
        visualization_section.addWidget(self.task_series_checkbox)

        self.task_series_path = QLineEdit()
        self.task_series_path.setPlaceholderText("Path to task time series")
        self.task_series_path.setEnabled(False)

        task_browse_button = QPushButton("Browse")
        task_browse_button.clicked.connect(self.select_task_series_file)
        task_browse_button.setEnabled(False)

        self.task_time_layout = QHBoxLayout()
        self.task_time_layout.addWidget(self.task_series_path)
        self.task_time_layout.addWidget(task_browse_button)

        visualization_section.addLayout(self.task_time_layout)

        layout.addLayout(visualization_section)

        # Bottom Section: Save Options
        save_section = QVBoxLayout()
        save_section.addWidget(QLabel("<b>Save Options</b>"))

        # Save Path with Browse Button
        save_path_label = QLabel("Save Path:")
        self.save_path_input = QLineEdit()
        self.save_path_input.setPlaceholderText("Path to save outputs")
        save_browse_button = QPushButton("Browse")
        save_browse_button.clicked.connect(self.select_save_folder)

        save_path_layout = QHBoxLayout()
        save_path_layout.addWidget(self.save_path_input)
        save_path_layout.addWidget(save_browse_button)

        save_section.addWidget(save_path_label)
        save_section.addLayout(save_path_layout)

        # Outputs to Save
        save_outputs_label = QLabel("Outputs to Save:")
        self.save_outputs_list = QListWidget()
        outputs = ["Eigenvalues", "Eigenvectors", "Plots", "Logs"]
        for output in outputs:
            output_item = QListWidgetItem(output)
            output_item.setCheckState(Qt.CheckState.Unchecked)
            self.save_outputs_list.addItem(output_item)
        save_section.addWidget(save_outputs_label)
        save_section.addWidget(self.save_outputs_list)

        layout.addLayout(save_section)

        self.data_and_parameters_tab.setLayout(layout)
        self.tabs.addTab(self.data_and_parameters_tab, "Data & Parameters")

        # Parameter configurations for each method
        self.method_parameters = {
            "Sliding Window Covariance": self.create_sliding_window_parameters,
            "Sliding Window Correlation": self.create_sliding_window_parameters,
            "IPA": self.create_ipa_parameters,
            "IPL": self.create_ipl_parameters,
            "Co-fluctuation": self.create_coactivation_parameters,
        }

        # Store parameter widgets for each method
        self.parameters_storage = {method: {} for method in dfc_methods}

    def toggle_task_time_series(self):
        """Enable or disable the task time series fields based on the checkbox state."""
        enabled = self.task_series_checkbox.isChecked()
        self.task_series_path.setEnabled(enabled)
        self.task_time_layout.itemAt(1).widget().setEnabled(enabled)

    def select_task_series_file(self):
        """Open a file dialog to select the task time series file."""
        file, _ = QFileDialog.getOpenFileName(self, "Select Task Time Series File", "", "All Files (*)")
        if file:
            self.task_series_path.setText(file)

    def update_visualization_options(self):
        """Enable or disable plot options based on the 'Visualize' checkbox state."""
        self.plot_list.setEnabled(self.visualize_checkbox.isChecked())

    def update_parameters_section(self, current_item, previous_item):
        """Update the parameter section based on the selected dFC method."""
        # Clear the current parameter layout
        for i in reversed(range(self.parameter_area.count())):
            widget = self.parameter_area.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        if current_item:
            method = current_item.text()

            # Retrieve stored parameters if they exist, otherwise create new ones
            if method in self.method_parameters:
                self.method_parameters[method](self.parameters_storage[method])
        else:
            # Show placeholder if no method is selected
            self.parameter_area.addWidget(self.parameter_placeholder)

    def create_sliding_window_parameters(self, storage):
        """Create parameters for sliding window methods."""
        self.parameter_area.addWidget(QLabel("Number of Eigenvectors:"))
        eig_spinbox = QSpinBox()
        eig_spinbox.setRange(1, 100)
        eig_spinbox.setValue(storage.get("n_eigs", 10))
        eig_spinbox.valueChanged.connect(lambda val: storage.update({"n_eigs": val}))
        self.parameter_area.addWidget(eig_spinbox)

        self.parameter_area.addWidget(QLabel("Window Size:"))
        window_spinbox = QSpinBox()
        window_spinbox.setRange(1, 50)
        window_spinbox.setValue(storage.get("window_size", 5))
        window_spinbox.valueChanged.connect(lambda val: storage.update({"window_size": val}))
        self.parameter_area.addWidget(window_spinbox)

        self.parameter_area.addWidget(QLabel("Window Shape:"))
        window_shape_combobox = QComboBox()
        window_shape_combobox.addItems(["Rectangular", "Hann", "Hamming"])
        window_shape_combobox.setCurrentText(storage.get("window_shape", "Rectangular"))
        window_shape_combobox.currentTextChanged.connect(lambda text: storage.update({"window_shape": text}))
        self.parameter_area.addWidget(window_shape_combobox)

    def create_ipa_parameters(self, storage):
        """Create parameters for IPA method."""
        self.parameter_area.addWidget(QLabel("Number of Components:"))
        components_spinbox = QSpinBox()
        components_spinbox.setRange(1, 50)
        components_spinbox.setValue(storage.get("n_components", 5))
        components_spinbox.valueChanged.connect(lambda val: storage.update({"n_components": val}))
        self.parameter_area.addWidget(components_spinbox)

    def create_ipl_parameters(self, storage):
        """Create parameters for IPL method."""
        self.parameter_area.addWidget(QLabel("Threshold:"))
        threshold_spinbox = QSpinBox()
        threshold_spinbox.setRange(0, 100)
        threshold_spinbox.setValue(storage.get("threshold", 10))
        threshold_spinbox.valueChanged.connect(lambda val: storage.update({"threshold": val}))
        self.parameter_area.addWidget(threshold_spinbox)

    def create_coactivation_parameters(self, storage):
        """Create parameters for Co-Activation method."""
        self.parameter_area.addWidget(QLabel("Activation Threshold:"))
        activation_spinbox = QSpinBox()
        activation_spinbox.setRange(0, 100)
        activation_spinbox.setValue(storage.get("activation_threshold", 20))
        activation_spinbox.valueChanged.connect(lambda val: storage.update({"activation_threshold": val}))
        self.parameter_area.addWidget(activation_spinbox)

    def select_data_folder(self):
        """Open a file dialog to select the data folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            self.data_folder_path.setText(folder)
            QMessageBox.information(self, "Data Folder", f"Data folder selected: {folder}")

    def select_save_folder(self):
        """Open a file dialog to select the save folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self.save_path_input.setText(folder)

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
        save_preferences = {}
        for index in range(self.save_outputs_list.count()):
            item = self.save_outputs_list.item(index)
            save_preferences[item.text()] = (item.checkState() == Qt.CheckState.Checked)
        return save_preferences

    def log_message(self, message):
        self.log_output.append(message)
        self.log_output.ensureCursorVisible()  # Scroll to the latest log message
        QApplication.processEvents()

    def run_dysco_pipeline(self):
        multi_run = False
        """Run the DySCo pipeline with the set parameters."""
        # Retrieve selected dFC methods and their parameters
        selected_methods = []
        for index in range(self.method_list.count()):
            item = self.method_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                method = item.text()
                selected_methods.append((method, self.parameters_storage.get(method, {})))

        # Check if any method is selected
        if not selected_methods:
            self.log_output.append("Error: No dFC method selected.")
            return

        # Retrieve the data folder
        data_folder = self.data_folder_path.text()
        if not data_folder:
            self.log_output.append("Error: Data folder not selected.")
            return

        # Retrieve save path
        save_path = self.save_path_input.text()
        if not save_path:
            self.log_output.append("Error: Save path not specified.")
            return

        # Visualization options
        visualize = self.visualize_checkbox.isChecked()
        selected_plots = [
            self.plot_list.item(i).text()
            for i in range(self.plot_list.count())
            if self.plot_list.item(i).checkState() == Qt.CheckState.Checked
        ]

        # Task time series
        use_task_series = self.task_series_checkbox.isChecked()
        task_series_path = self.task_series_path.text() if use_task_series else None

        # Log the configuration
        self.log_output.append("Starting DySCo Pipeline...")
        self.log_output.append(f"Data Folder: {data_folder}")
        self.log_output.append(f"Save Path: {save_path}")
        self.log_output.append(f"Selected Methods: {', '.join([method[0] for method in selected_methods])}")
        for method, params in selected_methods:
            self.log_output.append(f"Parameters for {method}: {params}")
        self.log_output.append(f"Visualization: {'Enabled' if visualize else 'Disabled'}")
        if visualize:
            self.log_output.append(f"Selected Plots: {', '.join(selected_plots)}")
        if use_task_series:
            self.log_output.append(f"Task Time Series: {task_series_path}")

        save_preferences = self.collect_save_preferences()

        # Log save preferences
        self.log_output.append(
            f"Save Preferences: {', '.join([key for key, value in save_preferences.items() if value])}")

        selected_methods_dict = dict(selected_methods)

        def pipeline_task():
            try:
                self.log_message("Running DySCo Pipeline...")
                pipeline = DyscoPipeline(
                    methods=selected_methods_dict,
                    params={},
                    data_folder=data_folder,
                    save_path=save_path,
                    save_preferences=save_preferences,
                    log_function=self.log_message
                )
                pipeline.run()
                self.log_message("Pipeline completed successfully.")
                self.progress_bar.setValue(100)
            except Exception as e:
                self.log_message(f"Error: {str(e)}")
                self.progress_bar.setValue(0)

        # Run the pipeline in a separate thread
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(pipeline_task)

        if multi_run:
            try:
                self.log_output.append("Starting DySCo Pipeline...")
                with ThreadPoolExecutor() as executor:
                    futures = {
                        executor.submit(DyscoPipeline, {method: params}, data_folder, save_path): method
                        for method, params in selected_methods.items()
                    }
                    for future in futures:
                        method = futures[future]
                        try:
                            future.result()
                            self.log_output.append(f"{method} completed successfully.")
                        except Exception as e:
                            self.log_output.append(f"Error in {method}: {str(e)}")
                self.log_output.append("All pipelines completed.")
                self.progress_bar.setValue(100)
            except Exception as e:
                self.log_output.append(f"Error: {str(e)}")
                self.progress_bar.setValue(0)


def main():
    app = QApplication(sys.argv)
    window = DySCoGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

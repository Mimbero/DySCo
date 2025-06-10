from PyQt6.QtWidgets import QMessageBox
from .logger import logger

class ErrorHandler:
    """Handles errors in the DySCo GUI with proper logging and user feedback."""
    
    @staticmethod
    def handle_error(title, message, parent=None):
        """Handle error messages with optional parent widget."""
        logger.error(f"{title}: {message}")
        QMessageBox.critical(parent, title, message)

    @staticmethod
    def handle_warning(title, message, parent=None):
        """Handle warning messages with optional parent widget."""
        logger.warning(f"{title}: {message}")
        QMessageBox.warning(parent, title, message)

    @staticmethod
    def handle_info(title, message, parent=None):
        """Handle info messages with optional parent widget."""
        logger.info(f"{title}: {message}")
        QMessageBox.information(parent, title, message)

    @staticmethod
    def handle_file_error(operation, file_path, error, parent=None):
        """Handle file-related errors with optional parent widget."""
        message = f"Error during {operation} of file {file_path}: {str(error)}"
        logger.error(message)
        QMessageBox.critical(parent, "File Operation Error", message)

    @staticmethod
    def handle_permission_error(operation, path, error, parent=None):
        """Handle permission-related errors with optional parent widget."""
        message = f"Permission error during {operation} of {path}: {str(error)}"
        logger.error(message)
        QMessageBox.critical(parent, "Permission Error", message) 
import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
import sys

class DySCoLogger:
    """Custom logger for DySCo GUI with file and console handlers."""
    
    def __init__(self, name="DySCo", log_dir="logs"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler (with rotation)
        file_handler = RotatingFileHandler(
            self.log_dir / "dysco.log",
            maxBytes=1024*1024,  # 1MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical message."""
        self.logger.critical(message)
    
    def exception(self, message):
        """Log exception with traceback."""
        self.logger.exception(message)

# Create a singleton instance
logger = DySCoLogger() 
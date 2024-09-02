import os
import logging

#Setuping logging to store logs at its perfect place
def setup_logging(log_path):
    """
    Sets up logging configuration.
    Parameters:
    - log_path: Path to the log file
    """
    log_directory = os.path.dirname(log_path)
    # Ensure the directory exists
    os.makedirs(log_directory, exist_ok=True)
    # Clear previous handlers if any
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Set up the logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger
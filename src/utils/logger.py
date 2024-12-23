import logging
import os

LOG_FOLDER = "data/logs"  # Folder to store log files

def setup_logger(name: str, log_file: str, level: int = logging.INFO):
    """
    Sets up a logger instance that outputs to both console and a file.

    Args:
        name (str): The name of the logger (typically `__name__`).
        log_file (str): The file to which logs should be written.
        level (int): The logging level (default is `logging.INFO`).

    Returns:
        logging.Logger: Configured logger instance.
    """

    os.makedirs(LOG_FOLDER, exist_ok=True)
    log_path = os.path.join(LOG_FOLDER, log_file)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

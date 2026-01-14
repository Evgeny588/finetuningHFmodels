import logging 

from pathlib import Path


def setup_logging(
        log_path = 'logs/', 
        filename = 'app_logs.log',
        root_level = logging.DEBUG,
        file_level = logging.DEBUG,
        console_level = logging.INFO,
        file_mode = 'w'):
    
    log_path = Path(log_path)
    log_path.mkdir(parents = True, exist_ok = True)
    path_to_file = Path(log_path) / filename

    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    formatter = logging.Formatter(
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

    # File log
    file_logger = logging.FileHandler(
        path_to_file,
        encoding = 'utf-8',
        mode = file_mode
    )
    file_logger.setLevel(file_level)
    file_logger.setFormatter(formatter)

    # Console log 
    console_logger = logging.StreamHandler()
    console_logger.setLevel(console_level)
    console_logger.setFormatter(formatter)

    if not root_logger.handlers:
        root_logger.addHandler(file_logger)
        root_logger.addHandler(console_logger)
import logging
import sys

from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
filename = Path(__file__).stem
sys.path.append(str(root_path))

from set_logging import setup_logging
setup_logging(console_level = logging.WARNING, file_mode = 'w')

logging = logging.getLogger(filename)

logging.debug('debug message')
logging.info('info message')
logging.warning('warning message')
logging.error('error message')
logging.critical('critical message')


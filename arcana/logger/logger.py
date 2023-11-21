'''This module defines the logger for our application.
The file is a adopted copy of the file from MADAP project by Fuzhan Rahmanian. cite: https://github.com/fuzhanrahmanian/MADAP'''

import os
import sys
import logging
import time

APP_LOGGER_NAME = 'ARCANA'

def setup_applevel_logger(logger_name = APP_LOGGER_NAME, file_name=None):
    """Sets up the app level logger
    Args:
        logger_name (str, optional): Logger name. Defaults to APP_LOGGER_NAME.
        file_name (str, optional): file name. Defaults to None.
    Returns:
        logger.Logger: app level logger
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    if file_name:
        if not os.path.exists("logs"):
            os.makedirs("logs")
        # create subfolder for the logger with the timestamp
        timestr = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(f"logs/{timestr}")
        file_handler = logging.FileHandler(f"logs/{timestr}/{file_name}", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_logger(module_name):
    """Returns a logger with the name of the module calling this function
    Args:
        module_name (str): name of the module calling this function
    Returns:
        logging.Logger: logger with the name of the module calling this function
    """
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)

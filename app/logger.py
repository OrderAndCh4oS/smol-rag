import logging
import os.path
import os

from app.definitions import LOG_DIR

logger = logging.getLogger("mini-rag")


def set_logger(log_file: str):
    configured_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, configured_level, logging.INFO)
    logger.setLevel(log_level)
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_file))
    file_handler.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

import sys

from loguru import logger


def setup_logger(
    log_file: str = "logs/rag_app.log",
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Setup loguru logger with console and rotating file handlers.

    Args:
        log_file (str): Path to the log file.
        log_level (str): Logging level, e.g. "INFO", "DEBUG".
        rotation (str): Rotation size or interval (e.g. "10 MB", "1 day").
        retention (str): Log retention period (e.g. "7 days").
    """
    logger.remove()  # Remove default logger

    # Console logging
    logger.add(sys.stdout, level=log_level)

    # Rotating file logging
    logger.add(
        log_file,
        rotation=rotation,
        retention=retention,
        compression="zip",
        level=log_level,
        enqueue=True,  # safe multi-thread/process logging
    )

    logger.info("Logger initialised")

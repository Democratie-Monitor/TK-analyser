"""
Logging Configuration Utility for TK-Analyser

Provides consistent logging setup across all modules with:
- Configurable log levels (DEBUG/INFO/WARNING/ERROR)
- File and console output
- Rotation for log files
- Performance metrics logging
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    console: bool = True,
    format_style: str = "detailed"
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional specific log file path
        log_dir: Directory for log files
        console: Enable console output
        format_style: 'simple' or 'detailed'
    """
    # Create log directory
    if log_file or log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Set format
    if format_style == "simple":
        log_format = "%(levelname)s - %(message)s"
    else:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    handlers = []

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)

    # File handler
    if log_file:
        file_path = log_file
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f"{log_dir}/tk_analyser_{timestamp}.log"

    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG to file
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    ))
    handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Capture everything, handlers filter
        handlers=handlers,
        force=True
    )

    # Set specific module levels
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={level}, file={file_path}")


def get_performance_logger(name: str = "performance") -> logging.Logger:
    """Get a logger specifically for performance metrics."""
    perf_logger = logging.getLogger(f"tk_analyser.{name}")
    return perf_logger


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        self.operation = operation
        self.logger = logger or logging.getLogger("tk_analyser.performance")
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time

        if exc_type:
            self.logger.error(f"Failed: {self.operation} after {elapsed:.3f}s - {exc_val}")
        else:
            self.logger.info(f"Completed: {self.operation} in {elapsed:.3f}s")

        return False  # Don't suppress exceptions

    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def log_system_info() -> None:
    """Log system information for debugging."""
    import platform
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("System Information:")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Processor: {platform.processor()}")

    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info(f"  Memory: {mem.total / 1024 / 1024 / 1024:.1f} GB total, {mem.available / 1024 / 1024 / 1024:.1f} GB available")
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"  CUDA: Available ({torch.cuda.get_device_name(0)})")
        else:
            logger.info("  CUDA: Not available")
    except ImportError:
        pass

    logger.info("=" * 50)


def log_package_versions() -> None:
    """Log versions of key packages."""
    logger = logging.getLogger(__name__)
    packages = [
        'anthropic', 'pandas', 'transformers', 'torch',
        'pattern', 'textblob', 'vaderSentiment', 'pyyaml', 'tqdm'
    ]

    logger.info("Package versions:")
    for pkg in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"  {pkg}: {version}")
        except ImportError:
            logger.debug(f"  {pkg}: not installed")

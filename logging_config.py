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
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Any


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        return json.dumps(log_data, default=str)


class StructuredLogger(logging.LoggerAdapter):
    """Logger adapter that supports structured extra fields."""

    def process(self, msg, kwargs):
        """Add extra fields to log record."""
        extra = kwargs.get('extra', {})
        if 'extra_fields' not in extra:
            extra['extra_fields'] = {}

        # Move any additional kwargs to extra_fields
        for key in list(kwargs.keys()):
            if key not in ('exc_info', 'stack_info', 'stacklevel', 'extra'):
                extra['extra_fields'][key] = kwargs.pop(key)

        kwargs['extra'] = extra
        return msg, kwargs

    def with_fields(self, **fields) -> 'StructuredLogger':
        """Create a new logger with additional default fields."""
        new_extra = dict(self.extra)
        new_extra.update(fields)
        return StructuredLogger(self.logger, new_extra)


def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    base_logger = logging.getLogger(name)
    return StructuredLogger(base_logger, {})


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    console: bool = True,
    format_style: str = "detailed",
    json_output: bool = False
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional specific log file path
        log_dir: Directory for log files
        console: Enable console output
        format_style: 'simple' or 'detailed'
        json_output: Use JSON structured logging
    """
    # Create log directory
    if log_file or log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Set format
    if json_output:
        formatter = JSONFormatter()
    elif format_style == "simple":
        formatter = logging.Formatter("%(levelname)s - %(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    handlers = []

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # File handler
    if log_file:
        file_path = log_file
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f"{log_dir}/tk_analyser_{timestamp}.log"

    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG to file

    # Use JSON for file if enabled, otherwise detailed format
    if json_output:
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        ))
    handlers.append(file_handler)

    # Optional: Add JSON log file for structured log aggregation
    if json_output:
        json_file_path = file_path.replace('.log', '.jsonl')
        json_handler = logging.FileHandler(json_file_path, encoding='utf-8')
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        handlers.append(json_handler)

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
    logger.info(f"Logging configured: level={level}, file={file_path}, json={json_output}")


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


class MetricsLogger:
    """Logger for tracking analysis metrics."""

    def __init__(self, name: str = "metrics"):
        self.logger = get_structured_logger(f"tk_analyser.{name}")
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_processing_time": 0.0
        }

    def log_request(
        self,
        success: bool,
        tokens: int,
        cost: float,
        processing_time: float,
        **extra
    ) -> None:
        """Log a single request with metrics."""
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        self.metrics["total_tokens"] += tokens
        self.metrics["total_cost"] += cost
        self.metrics["total_processing_time"] += processing_time

        self.logger.info(
            "Request completed",
            extra={
                "extra_fields": {
                    "event": "request_complete",
                    "success": success,
                    "tokens": tokens,
                    "cost": cost,
                    "processing_time": processing_time,
                    **extra
                }
            }
        )

    def log_batch_complete(self, batch_size: int, **extra) -> None:
        """Log batch completion with cumulative metrics."""
        self.logger.info(
            f"Batch of {batch_size} completed",
            extra={
                "extra_fields": {
                    "event": "batch_complete",
                    "batch_size": batch_size,
                    "cumulative_metrics": self.metrics.copy(),
                    **extra
                }
            }
        )

    def get_metrics(self) -> dict:
        """Get current metrics."""
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        for key in self.metrics:
            if isinstance(self.metrics[key], int):
                self.metrics[key] = 0
            else:
                self.metrics[key] = 0.0

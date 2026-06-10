"""
Một module dùng để cấu hình và quản lý loggers trong dự án.

Author: Khang P. Nguyen
Date: 2026-05-22

Example:
    from exact.core.logger import setup_logging, get_request_logger

    setup_logging(level="INFO", log_file="outputs/logs/app.log")

    logger = get_request_logger(
        __name__,
        request_id="q001",
        task_type="type2_physics",
    )

    logger.info("Start pipeline")
    logger.warning("Parser failed, retrying with repair prompt")
    logger.error("Pipeline failed", exc_info=True)
"""
from __future__ import annotations

import logging
import sys
import json

from typing import Any
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "request_id=%(request_id)s | task_type=%(task_type)s | %(message)s"
)

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class SafeExtraFormatter(logging.Formatter):
    '''
    A class used to format logs in a safe way. We can avoid errors when formatting
    log having fields 'extra' like request_id or task_type but some log line
    don't have these fields.

    Usage:
        logger = get_logger(__name__)
        logger.info("Hello World", extra={"request_id": "123", "task_type": "task1"})
        logger.info("Hello World")
    '''

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        if not hasattr(record, "task_type"):
            record.task_type = "-"
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for production-freiendly structure logs"""

    def format(self, record: logging.LogRecord) -> str:
        '''
        A function is used to format log records into JSON format.
        
        Args:
            record (logging.LogRecord): The log record to format.
            
        Returns:
            str: The formatted log record in JSON format.
        '''
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "task_type": getattr(record, "task_type", "-"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)

def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    json_logs: bool = False,
) -> None:
    """
    Configure logging for the whole application

    Args:
        - level (str): The logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        - log_file (str | Path | None): Optional path to a log file.
        - json_logs (bool): If True, logs will be in JSON format.
    """

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if json_logs:
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = SafeExtraFormatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
        )

    # Create handlers based on configuration
    # Why: Handle logging output to different sinks (console, files, etc.)
    #       For example, console logger and file logger can have different formatters   
    #       And handlers can be enabled/disabled independently
    
    handlers: list[logging.Handler] = []

    # Console handler - always present
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # Kiểm tra xem có log ra file hay không
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Lấy logger toàn cục, xóa toàn bộ các logger cục bộ của ứng dụng, reset lại cấu hình, đặt lại level với
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)

    for handler in handlers:
        root_logger.addHandler(handler)

    _quiet_noisy_loggers()

def _quiet_noisy_loggers() -> None:
    """
    Dùng để giảm bớt các log rác từ các thư viện bên ngoài như httpx, aiohttp...

    Trong project này, ta dùng rất nhiều thư viện như FastAPI / Uvicorn, httpx / request các thư viện này có thể tự in ra rất nhiều log
    nên ta cần dùng để có thể xóa bớt và chặn đi các log đó, tránh việc terminal hoặc file bị quá tải.
    """

    noisy_loggers = [
        "httpx",
        "urllib3",
        "transformers",
        "asyncio",
        "uvicorn.access",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    Args:
        name (str): The name of the logger (usually __name__)
    Returns:
        logging.Logger: The logger instance
    """
    return logging.getLogger(name)

def get_request_logger(
    name: str,
    request_id: str | None = None,
    task_type: str | None = None,
    **extra: Any
) -> logging.LoggerAdapter:
    """
    Trả về một logger với bối cảnh của request-level .

    Ví dụ:
        logger = get_request_logger(
        "__name__, 
        request_id="q001", 
        task_type="type2_physics")
    """

    context : dict[str, Any] = {
        "request_id": request_id or "-",
        "task_type": task_type or "-",
    }

    return logging.LoggerAdapter(logging.getLogger(name), context)

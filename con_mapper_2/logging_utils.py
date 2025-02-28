"""
Logging utilities for Philosophical Concept Map Generator.
"""
import logging
import os
import sys
import time
from functools import wraps
from queue import Queue
from threading import Lock


class LogManager:
    """
    Centralized logging manager for the application
    """
    _instance = None
    _initialized = False
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LogManager, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._loggers = {}
        self._log_directory = 'logs'
        self._log_level = logging.INFO
        self._ui_queue = Queue()
        
        # Ensure log directory exists
        os.makedirs(self._log_directory, exist_ok=True)
        
        # Initialize default logger
        self._default_logger = self._create_logger('concept_mapper')
        
        self._initialized = True
    
    def _create_logger(self, name):
        """Create a new logger with file and console handlers"""
        logger = logging.getLogger(name)
        logger.setLevel(self._log_level)
        
        # Remove existing handlers if any
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create log file with timestamp
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        log_file = os.path.join(self._log_directory, f'{name}_{timestamp}.log')
        
        # Create handlers
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def get_logger(self, name=None):
        """Get or create a logger by name"""
        if name is None:
            return self._default_logger
            
        if name not in self._loggers:
            self._loggers[name] = self._create_logger(name)
            
        return self._loggers[name]
    
    def set_log_level(self, level):
        """Set the logging level for all loggers"""
        self._log_level = level
        self._default_logger.setLevel(level)
        
        for logger in self._loggers.values():
            logger.setLevel(level)
    
    def log_to_ui(self, message, level=logging.INFO):
        """Add a log message to the UI queue"""
        self._ui_queue.put((message, level))
    
    def get_ui_logs(self):
        """Get all pending UI log messages"""
        logs = []
        while not self._ui_queue.empty():
            logs.append(self._ui_queue.get())
        return logs


# Global log manager instance
log_manager = LogManager()


def get_logger(name=None):
    """Get a logger instance"""
    return log_manager.get_logger(name)


def log_execution(logger=None):
    """
    Decorator to log function execution time and status
    
    Args:
        logger: Optional logger instance (uses default if None)
    """
    if logger is None:
        logger = log_manager.get_logger()
        
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.info(f"Starting {func_name}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"Completed {func_name} in {elapsed:.2f} seconds")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Failed {func_name} after {elapsed:.2f} seconds: {str(e)}")
                raise
                
        return wrapper
    return decorator


def log_to_ui(message, level=logging.INFO):
    """Add a log message to the UI queue"""
    # Log to normal log as well
    logger = log_manager.get_logger()
    logger.log(level, message)
    
    # Add to UI queue
    log_manager.log_to_ui(message, level)
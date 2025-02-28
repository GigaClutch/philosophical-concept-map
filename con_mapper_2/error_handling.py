"""
Error handling utilities for Philosophical Concept Map Generator.
"""
import logging
import traceback
import sys
from enum import Enum, auto


class ErrorLevel(Enum):
    """Error severity levels"""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class ConceptMapError(Exception):
    """Base exception class for all application-specific errors"""
    def __init__(self, message, level=ErrorLevel.ERROR, original_exception=None):
        self.message = message
        self.level = level
        self.original_exception = original_exception
        super().__init__(self.message)


class WikipediaError(ConceptMapError):
    """Errors related to Wikipedia API"""
    pass


class NLPProcessingError(ConceptMapError):
    """Errors related to NLP processing"""
    pass


class VisualizationError(ConceptMapError):
    """Errors related to visualization generation"""
    pass


class DataStorageError(ConceptMapError):
    """Errors related to data storage/retrieval"""
    pass


class UIError(ConceptMapError):
    """Errors related to UI interactions"""
    pass


def setup_logger(name='concept_mapper', log_file='concept_mapper.log', level=logging.INFO):
    """Configure and return a logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Global logger instance
logger = setup_logger()


def handle_error(error, raise_exception=True, log_traceback=True, ui_callback=None):
    """
    Centralized error handling function
    
    Args:
        error: Exception object or string message
        raise_exception: Whether to re-raise the exception
        log_traceback: Whether to log the full traceback
        ui_callback: Optional callback function to display error in UI
    
    Returns:
        None
    """
    if isinstance(error, str):
        error_message = error
        error_level = ErrorLevel.ERROR
        exc_info = sys.exc_info()
    else:
        error_message = str(error)
        error_level = getattr(error, 'level', ErrorLevel.ERROR)
        exc_info = (type(error), error, error.__traceback__)
    
    # Log the error
    if error_level == ErrorLevel.INFO:
        logger.info(error_message)
    elif error_level == ErrorLevel.WARNING:
        logger.warning(error_message)
    elif error_level == ErrorLevel.ERROR:
        logger.error(error_message)
    elif error_level == ErrorLevel.CRITICAL:
        logger.critical(error_message)
    
    # Log traceback if requested
    if log_traceback and exc_info[0] is not None:
        logger.error("Exception traceback:", exc_info=exc_info)
    
    # Call UI callback if provided
    if ui_callback is not None:
        ui_callback(error_message, error_level)
    
    # Re-raise if requested
    if raise_exception and isinstance(error, Exception):
        raise error


def safe_execute(func, error_message="An error occurred", *args, **kwargs):
    """
    Execute a function safely with error handling
    
    Args:
        func: Function to execute
        error_message: Message to log if an error occurs
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        Result of the function or None if an error occurred
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_error(f"{error_message}: {str(e)}", raise_exception=False)
        return None
"""
Logging utilities for the nephrotoxicity analysis pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
import pandas as pd


def setup_logging(verbose: bool = True, 
                 log_file: Optional[Union[str, Path]] = None,
                 log_level: Optional[str] = None) -> logging.Logger:
    """
    Setup centralized logging for the pipeline.
    
    Args:
        verbose: Enable verbose output to console
        log_file: Optional path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    # Determine log level
    if log_level is None:
        log_level = logging.INFO if verbose else logging.WARNING
    else:
        log_level = getattr(logging, log_level.upper())
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always capture debug in file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Suppress third-party library noise
    _suppress_third_party_logs()
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def _suppress_third_party_logs():
    """Suppress verbose output from third-party libraries."""
    # Suppress common noisy loggers
    noisy_loggers = [
        'matplotlib',
        'PIL',
        'urllib3',
        'requests',
        'scanpy',
        'anndata',
        'numba',
        'h5py'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


class LoggingContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(self, logger: logging.Logger, level: int):
        """
        Initialize logging context.
        
        Args:
            logger: Logger to modify
            level: New logging level
        """
        self.logger = logger
        self.new_level = level
        self.old_level = None
    
    def __enter__(self):
        """Enter context and set new logging level."""
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original logging level."""
        self.logger.setLevel(self.old_level)


class ComponentLogger:
    """Logger wrapper for pipeline components."""
    
    def __init__(self, component_name: str, verbose: bool = True):
        """
        Initialize component logger.
        
        Args:
            component_name: Name of the component
            verbose: Enable verbose logging
        """
        self.component_name = component_name
        self.logger = get_logger(f"nephrotoxicity.{component_name}")
        self.verbose = verbose
        
        # Track component execution
        self.start_time = None
        self.steps_completed = 0
        self.total_steps = 0
    
    def start_component(self, total_steps: int = 0):
        """Mark the start of component execution."""
        self.start_time = pd.Timestamp.now()
        self.total_steps = total_steps
        self.steps_completed = 0
        
        self.logger.info(f"Starting {self.component_name} analysis")
        if total_steps > 0:
            self.logger.info(f"Total steps: {total_steps}")
    
    def step_completed(self, step_name: str):
        """Mark completion of a processing step."""
        self.steps_completed += 1
        
        if self.total_steps > 0:
            progress = (self.steps_completed / self.total_steps) * 100
            self.logger.info(f"Step {self.steps_completed}/{self.total_steps} completed: {step_name} ({progress:.1f}%)")
        else:
            self.logger.info(f"Completed: {step_name}")
    
    def finish_component(self, success: bool = True):
        """Mark the end of component execution."""
        if self.start_time:
            duration = pd.Timestamp.now() - self.start_time
            duration_str = self._format_duration(duration)
            
            if success:
                self.logger.info(f"{self.component_name} completed successfully in {duration_str}")
            else:
                self.logger.error(f"{self.component_name} failed after {duration_str}")
        else:
            status = "completed successfully" if success else "failed"
            self.logger.info(f"{self.component_name} {status}")
    
    def _format_duration(self, duration: pd.Timedelta) -> str:
        """Format duration for display."""
        total_seconds = duration.total_seconds()
        
        if total_seconds < 60:
            return f"{total_seconds:.1f} seconds"
        elif total_seconds < 3600:
            minutes = total_seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = total_seconds / 3600
            return f"{hours:.1f} hours"
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        if self.verbose:
            self.logger.debug(message)


def log_dataframe_info(logger: logging.Logger, df: pd.DataFrame, 
                      name: str = "DataFrame") -> None:
    """
    Log information about a DataFrame.
    
    Args:
        logger: Logger instance
        df: DataFrame to log information about
        name: Name to use in log messages
    """
    logger.info(f"{name} info:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"  Missing values: {missing}")
    
    # Check data types
    numeric_cols = df.select_dtypes(include=['number']).shape[1]
    categorical_cols = df.select_dtypes(include=['object']).shape[1]
    logger.info(f"  Numeric columns: {numeric_cols}")
    logger.info(f"  Categorical columns: {categorical_cols}")


def log_analysis_summary(logger: logging.Logger, component_name: str,
                        input_data: dict, results: dict) -> None:
    """
    Log a standardized analysis summary.
    
    Args:
        logger: Logger instance
        component_name: Name of the analysis component
        input_data: Dictionary describing input data
        results: Dictionary of analysis results
    """
    logger.info(f"=== {component_name.upper()} ANALYSIS SUMMARY ===")
    
    # Log input data summary
    if input_data:
        logger.info("Input data:")
        for key, value in input_data.items():
            logger.info(f"  {key}: {value}")
    
    # Log results summary
    if results:
        logger.info("Results:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value}")
            elif isinstance(value, (list, tuple)):
                logger.info(f"  {key}: {len(value)} items")
            elif isinstance(value, dict):
                logger.info(f"  {key}: {len(value)} entries")
            elif hasattr(value, 'shape'):
                logger.info(f"  {key}: shape {value.shape}")
            else:
                logger.info(f"  {key}: {type(value).__name__}")
    
    logger.info("=" * 50)


class ProgressTracker:
    """Track and log progress for long-running operations."""
    
    def __init__(self, total_items: int, description: str = "Processing",
                 logger: logging.Logger = None, log_interval: int = 10):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            description: Description of the operation
            logger: Logger instance (creates default if None)
            log_interval: Log progress every N percent
        """
        self.total_items = total_items
        self.description = description
        self.logger = logger or get_logger(__name__)
        self.log_interval = log_interval
        
        self.processed_items = 0
        self.last_logged_percent = 0
        self.start_time = pd.Timestamp.now()
    
    def update(self, increment: int = 1):
        """
        Update progress counter.
        
        Args:
            increment: Number of items processed
        """
        self.processed_items += increment
        current_percent = (self.processed_items / self.total_items) * 100
        
        # Log progress at intervals
        if current_percent - self.last_logged_percent >= self.log_interval:
            elapsed = pd.Timestamp.now() - self.start_time
            items_per_second = self.processed_items / elapsed.total_seconds()
            
            remaining_items = self.total_items - self.processed_items
            eta_seconds = remaining_items / items_per_second if items_per_second > 0 else 0
            eta = pd.Timedelta(seconds=eta_seconds)
            
            self.logger.info(
                f"{self.description}: {self.processed_items}/{self.total_items} "
                f"({current_percent:.1f}%) - ETA: {self._format_timedelta(eta)}"
            )
            
            self.last_logged_percent = current_percent
    
    def finish(self):
        """Log completion of operation."""
        total_time = pd.Timestamp.now() - self.start_time
        items_per_second = self.total_items / total_time.total_seconds()
        
        self.logger.info(
            f"{self.description} completed: {self.total_items} items in "
            f"{self._format_timedelta(total_time)} "
            f"({items_per_second:.1f} items/sec)"
        )
    
    @staticmethod
    def _format_timedelta(td: pd.Timedelta) -> str:
        """Format timedelta for display."""
        total_seconds = td.total_seconds()
        
        if total_seconds < 60:
            return f"{total_seconds:.1f}s"
        elif total_seconds < 3600:
            minutes = total_seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = total_seconds / 3600
            return f"{hours:.1f}h"

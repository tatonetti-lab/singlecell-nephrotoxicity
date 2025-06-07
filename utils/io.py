"""
Input/Output utilities for the nephrotoxicity analysis pipeline.
"""

import pickle
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import pandas as pd
import numpy as np
import logging

from .logging import get_logger


class DataIO:
    """Centralized data input/output operations."""
    
    def __init__(self, base_path: Union[str, Path] = None):
        """
        Initialize DataIO.
        
        Args:
            base_path: Base path for relative file operations
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.logger = get_logger(__name__)
    
    def load_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load CSV file with robust error handling.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded DataFrame
        """
        file_path = self._resolve_path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # Default CSV reading parameters
        default_kwargs = {
            'encoding': 'utf-8',
            'low_memory': False
        }
        default_kwargs.update(kwargs)
        
        try:
            self.logger.debug(f"Loading CSV: {file_path}")
            df = pd.read_csv(file_path, **default_kwargs)
            self.logger.info(f"Loaded CSV {file_path.name}: {df.shape}")
            return df
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    default_kwargs['encoding'] = encoding
                    df = pd.read_csv(file_path, **default_kwargs)
                    self.logger.warning(f"Loaded CSV with {encoding} encoding: {file_path}")
                    return df
                except UnicodeDecodeError:
                    continue
            raise
        
        except Exception as e:
            # Try with different separators
            for sep in ['\t', ';', '|']:
                try:
                    default_kwargs['sep'] = sep
                    df = pd.read_csv(file_path, **default_kwargs)
                    self.logger.warning(f"Loaded CSV with separator '{sep}': {file_path}")
                    return df
                except:
                    continue
            
            self.logger.error(f"Failed to load CSV {file_path}: {e}")
            raise
    
    def save_csv(self, df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> Path:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            **kwargs: Additional arguments for df.to_csv
            
        Returns:
            Path to saved file
        """
        file_path = self._resolve_path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Default CSV saving parameters
        default_kwargs = {
            'index': False,
            'encoding': 'utf-8'
        }
        default_kwargs.update(kwargs)
        
        try:
            df.to_csv(file_path, **default_kwargs)
            self.logger.info(f"Saved CSV {file_path.name}: {df.shape}")
            return file_path
        except Exception as e:
            self.logger.error(f"Failed to save CSV {file_path}: {e}")
            raise
    
    def load_excel(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load Excel file.
        
        Args:
            file_path: Path to Excel file
            **kwargs: Additional arguments for pd.read_excel
            
        Returns:
            Loaded DataFrame
        """
        file_path = self._resolve_path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        
        try:
            self.logger.debug(f"Loading Excel: {file_path}")
            df = pd.read_excel(file_path, **kwargs)
            self.logger.info(f"Loaded Excel {file_path.name}: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load Excel {file_path}: {e}")
            raise
    
    def save_excel(self, df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> Path:
        """
        Save DataFrame to Excel file.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            **kwargs: Additional arguments for df.to_excel
            
        Returns:
            Path to saved file
        """
        file_path = self._resolve_path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Default Excel saving parameters
        default_kwargs = {
            'index': False,
            'engine': 'openpyxl'
        }
        default_kwargs.update(kwargs)
        
        try:
            df.to_excel(file_path, **default_kwargs)
            self.logger.info(f"Saved Excel {file_path.name}: {df.shape}")
            return file_path
        except Exception as e:
            self.logger.error(f"Failed to save Excel {file_path}: {e}")
            raise
    
    def load_h5ad(self, file_path: Union[str, Path]) -> 'AnnData':
        """
        Load h5ad file (single-cell data).
        
        Args:
            file_path: Path to h5ad file
            
        Returns:
            AnnData object
        """
        try:
            import scanpy as sc
        except ImportError:
            raise ImportError("scanpy is required to load h5ad files")
        
        file_path = self._resolve_path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"h5ad file not found: {file_path}")
        
        try:
            self.logger.debug(f"Loading h5ad: {file_path}")
            adata = sc.read_h5ad(file_path)
            self.logger.info(f"Loaded h5ad {file_path.name}: {adata.n_obs} cells, {adata.n_vars} genes")
            return adata
        except Exception as e:
            self.logger.error(f"Failed to load h5ad {file_path}: {e}")
            raise
    
    def save_h5ad(self, adata: 'AnnData', file_path: Union[str, Path]) -> Path:
        """
        Save AnnData object to h5ad file.
        
        Args:
            adata: AnnData object to save
            file_path: Output file path
            
        Returns:
            Path to saved file
        """
        file_path = self._resolve_path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            adata.write_h5ad(file_path)
            self.logger.info(f"Saved h5ad {file_path.name}: {adata.n_obs} cells, {adata.n_vars} genes")
            return file_path
        except Exception as e:
            self.logger.error(f"Failed to save h5ad {file_path}: {e}")
            raise
    
    def load_pickle(self, file_path: Union[str, Path]) -> Any:
        """
        Load object from pickle file.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Loaded object
        """
        file_path = self._resolve_path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
            self.logger.info(f"Loaded pickle: {file_path.name}")
            return obj
        except Exception as e:
            self.logger.error(f"Failed to load pickle {file_path}: {e}")
            raise
    
    def save_pickle(self, obj: Any, file_path: Union[str, Path]) -> Path:
        """
        Save object to pickle file.
        
        Args:
            obj: Object to save
            file_path: Output file path
            
        Returns:
            Path to saved file
        """
        file_path = self._resolve_path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
            self.logger.info(f"Saved pickle: {file_path.name}")
            return file_path
        except Exception as e:
            self.logger.error(f"Failed to save pickle {file_path}: {e}")
            raise
    
    def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded dictionary
        """
        file_path = self._resolve_path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded JSON: {file_path.name}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load JSON {file_path}: {e}")
            raise
    
    def save_json(self, data: Dict[str, Any], file_path: Union[str, Path], **kwargs) -> Path:
        """
        Save dictionary to JSON file.
        
        Args:
            data: Dictionary to save
            file_path: Output file path
            **kwargs: Additional arguments for json.dump
            
        Returns:
            Path to saved file
        """
        file_path = self._resolve_path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Default JSON saving parameters
        default_kwargs = {
            'indent': 2,
            'ensure_ascii': False,
            'default': self._json_serializer
        }
        default_kwargs.update(kwargs)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, **default_kwargs)
            self.logger.info(f"Saved JSON: {file_path.name}")
            return file_path
        except Exception as e:
            self.logger.error(f"Failed to save JSON {file_path}: {e}")
            raise
    
    def load_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Loaded dictionary
        """
        file_path = self._resolve_path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"YAML file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self.logger.info(f"Loaded YAML: {file_path.name}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load YAML {file_path}: {e}")
            raise
    
    def save_yaml(self, data: Dict[str, Any], file_path: Union[str, Path], **kwargs) -> Path:
        """
        Save dictionary to YAML file.
        
        Args:
            data: Dictionary to save
            file_path: Output file path
            **kwargs: Additional arguments for yaml.dump
            
        Returns:
            Path to saved file
        """
        file_path = self._resolve_path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Default YAML saving parameters
        default_kwargs = {
            'default_flow_style': False,
            'sort_keys': False,
            'allow_unicode': True
        }
        default_kwargs.update(kwargs)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, **default_kwargs)
            self.logger.info(f"Saved YAML: {file_path.name}")
            return file_path
        except Exception as e:
            self.logger.error(f"Failed to save YAML {file_path}: {e}")
            raise
    
    def _resolve_path(self, file_path: Union[str, Path]) -> Path:
        """Resolve file path relative to base path."""
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        return file_path.resolve()
    
    @staticmethod
    def _json_serializer(obj: Any) -> str:
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return str(obj)


class FileValidator:
    """Validate file existence and formats."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_files(self, file_paths: Dict[str, Union[str, Path]]) -> Dict[str, bool]:
        """
        Validate existence of multiple files.
        
        Args:
            file_paths: Dictionary mapping names to file paths
            
        Returns:
            Dictionary mapping names to validation results
        """
        results = {}
        
        for name, path in file_paths.items():
            try:
                path = Path(path)
                exists = path.exists()
                results[name] = exists
                
                if exists:
                    self.logger.debug(f"✓ {name}: {path}")
                else:
                    self.logger.warning(f"✗ {name}: {path} (not found)")
                    
            except Exception as e:
                self.logger.error(f"Error validating {name}: {e}")
                results[name] = False
        
        return results
    
    def validate_file_format(self, file_path: Union[str, Path], 
                           expected_formats: List[str]) -> bool:
        """
        Validate file format based on extension.
        
        Args:
            file_path: Path to file
            expected_formats: List of expected file extensions (e.g., ['.csv', '.txt'])
            
        Returns:
            True if format is valid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return False
        
        file_ext = file_path.suffix.lower()
        expected_formats = [fmt.lower() for fmt in expected_formats]
        
        if file_ext in expected_formats:
            self.logger.debug(f"✓ Valid format {file_ext}: {file_path.name}")
            return True
        else:
            self.logger.warning(f"✗ Invalid format {file_ext} (expected {expected_formats}): {file_path.name}")
            return False
    
    def check_file_size(self, file_path: Union[str, Path], 
                       max_size_mb: Optional[float] = None) -> Dict[str, Any]:
        """
        Check file size and return information.
        
        Args:
            file_path: Path to file
            max_size_mb: Optional maximum size in MB
            
        Returns:
            Dictionary with size information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'exists': False, 'size_mb': 0, 'valid_size': False}
        
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        result = {
            'exists': True,
            'size_bytes': size_bytes,
            'size_mb': round(size_mb, 2),
            'valid_size': True
        }
        
        if max_size_mb and size_mb > max_size_mb:
            result['valid_size'] = False
            self.logger.warning(f"File size {size_mb:.2f}MB exceeds limit {max_size_mb}MB: {file_path}")
        else:
            self.logger.debug(f"File size: {size_mb:.2f}MB - {file_path.name}")
        
        return result


def safe_file_operation(operation_func, *args, retries: int = 3, **kwargs):
    """
    Perform file operation with retry logic.
    
    Args:
        operation_func: Function to execute
        *args: Arguments for the function
        retries: Number of retry attempts
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the operation
    """
    logger = get_logger(__name__)
    
    for attempt in range(retries):
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"File operation failed (attempt {attempt + 1}/{retries}): {e}")
                import time
                time.sleep(1)  # Brief delay before retry
            else:
                logger.error(f"File operation failed after {retries} attempts: {e}")
                raise


def create_directory_structure(base_path: Union[str, Path], 
                             subdirs: List[str]) -> List[Path]:
    """
    Create directory structure.
    
    Args:
        base_path: Base directory path
        subdirs: List of subdirectory names to create
        
    Returns:
        List of created directory paths
    """
    logger = get_logger(__name__)
    base_path = Path(base_path)
    created_dirs = []
    
    # Create base directory
    base_path.mkdir(parents=True, exist_ok=True)
    created_dirs.append(base_path)
    
    # Create subdirectories
    for subdir in subdirs:
        subdir_path = base_path / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        created_dirs.append(subdir_path)
        logger.debug(f"Created directory: {subdir_path}")
    
    logger.info(f"Created directory structure at: {base_path}")
    return created_dirs


def backup_file(file_path: Union[str, Path], backup_suffix: str = ".bak") -> Optional[Path]:
    """
    Create a backup copy of a file.
    
    Args:
        file_path: Path to file to backup
        backup_suffix: Suffix for backup file
        
    Returns:
        Path to backup file or None if backup failed
    """
    logger = get_logger(__name__)
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"Cannot backup non-existent file: {file_path}")
        return None
    
    backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
    
    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return None

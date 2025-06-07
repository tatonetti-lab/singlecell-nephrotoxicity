"""
Configuration management for the nephrotoxicity analysis pipeline.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union
import logging


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")
    
    # Validate configuration
    _validate_config(config)
    
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and values.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = [
        'output_base_dir',
        'analysis_steps'
    ]
    
    # Check required top-level keys
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate analysis steps
    analysis_steps = config['analysis_steps']
    valid_steps = {
        'drug2cell', 'statistical', 'ml_drug',
        'lincs_qc', 'differential', 'ml_lincs', 'simulation'
    }
    
    for step in analysis_steps:
        if step not in valid_steps:
            raise ValueError(f"Invalid analysis step: {step}")
    
    # Validate data paths if provided
    if 'data_paths' in config:
        data_paths = config['data_paths']
        for path_name, path_value in data_paths.items():
            if not isinstance(path_value, str):
                raise ValueError(f"Data path '{path_name}' must be a string")
    
    # Validate machine learning configuration
    if 'ml_config' in config:
        ml_config = config['ml_config']
        
        # Check test_size is valid
        if 'test_size' in ml_config:
            test_size = ml_config['test_size']
            if not 0 < test_size < 1:
                raise ValueError("test_size must be between 0 and 1")
        
        # Check cv_folds is valid
        if 'cv_folds' in ml_config:
            cv_folds = ml_config['cv_folds']
            if not isinstance(cv_folds, int) or cv_folds < 2:
                raise ValueError("cv_folds must be an integer >= 2")
    
    # Validate statistical configuration
    if 'statistical_config' in config:
        stat_config = config['statistical_config']
        
        if 'alpha' in stat_config:
            alpha = stat_config['alpha']
            if not 0 < alpha < 1:
                raise ValueError("alpha must be between 0 and 1")
        
        if 'correction_method' in stat_config:
            valid_methods = ['bonferroni', 'sidak', 'holm-sidak', 'holm',
                           'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by']
            if stat_config['correction_method'] not in valid_methods:
                raise ValueError(f"Invalid correction method: {stat_config['correction_method']}")


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    
    try:
        with open(output_path, 'w') as f:
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            elif output_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {output_path.suffix}")
    except Exception as e:
        raise ValueError(f"Error saving config: {e}")


def get_output_path(config: Dict[str, Any], component: str, filename: str = None) -> Path:
    """
    Get standardized output path for a component.
    
    Args:
        config: Configuration dictionary
        component: Component name (e.g., 'drug2cell', 'statistical')
        filename: Optional filename to append
        
    Returns:
        Path object for output location
    """
    base_dir = Path(config['output_base_dir'])
    
    if component in config.get('output_subdirs', {}):
        output_dir = base_dir / config['output_subdirs'][component]
    else:
        output_dir = base_dir / component
    
    if filename:
        return output_dir / filename
    else:
        return output_dir


def create_output_directories(config: Dict[str, Any]) -> None:
    """
    Create all necessary output directories.
    
    Args:
        config: Configuration dictionary
    """
    base_dir = Path(config['output_base_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each component
    for subdir in config.get('output_subdirs', {}).values():
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)


def update_config_paths(config: Dict[str, Any], base_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Update relative paths in configuration to be relative to base_path.
    
    Args:
        config: Configuration dictionary
        base_path: Base path for resolving relative paths
        
    Returns:
        Updated configuration dictionary
    """
    config = config.copy()
    base_path = Path(base_path).resolve()
    
    # Update data paths if they're relative
    if 'data_paths' in config:
        for key, path in config['data_paths'].items():
            path_obj = Path(path)
            if not path_obj.is_absolute():
                config['data_paths'][key] = str(base_path / path)
    
    # Update merge data source paths if they're relative
    if 'merge_drug_data_sources' in config and 'data_paths' in config['merge_drug_data_sources']:
        for key, path in config['merge_drug_data_sources']['data_paths'].items():
            path_obj = Path(path)
            if not path_obj.is_absolute():
                config['merge_drug_data_sources']['data_paths'][key] = str(base_path / path)
    
    return config


class ConfigManager:
    """Configuration manager for pipeline components."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize configuration manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._logger = logging.getLogger(__name__)
    
    def get_component_config(self, component: str) -> Dict[str, Any]:
        """Get configuration specific to a component."""
        component_configs = {
            'drug2cell': 'drug2cell_config',
            'statistical': 'statistical_config', 
            'ml_drug': 'ml_config',
            'ml_lincs': 'ml_config',
            'lincs_qc': 'lincs_config',
            'differential': 'lincs_config',
            'simulation': 'simulation_config'
        }
        
        config_key = component_configs.get(component)
        if config_key and config_key in self.config:
            return self.config[config_key]
        else:
            return {}
    
    def is_component_enabled(self, component: str) -> bool:
        """Check if a component is enabled for execution."""
        return self.config.get('analysis_steps', {}).get(component, False)
    
    def get_output_path(self, component: str, filename: str = None) -> Path:
        """Get output path for a component."""
        return get_output_path(self.config, component, filename)
    
    def should_cleanup_intermediate(self) -> bool:
        """Check if intermediate files should be cleaned up."""
        return self.config.get('cleanup_intermediate', False)
    
    def should_generate_report(self) -> bool:
        """Check if reports should be generated."""
        return self.config.get('generate_report', True)

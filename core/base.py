"""
Base classes for nephrotoxicity analysis pipeline components.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd

from config.settings import ConfigManager, get_output_path, create_output_directories
from utils.logging import get_logger
from .types import ComponentResult


class BaseAnalysisComponent(ABC):
    """Base class for all analysis components in the pipeline."""
    
    def __init__(self, config: Dict[str, Any], component_name: str):
        """
        Initialize analysis component.
        
        Args:
            config: Global configuration dictionary
            component_name: Name of this component
        """
        self.config = config
        self.component_name = component_name
        self.config_manager = ConfigManager(config)
        self.logger = get_logger(f"{__name__}.{component_name}")
        
        # Get component-specific configuration
        self.component_config = self.config_manager.get_component_config(component_name)
        
        # Setup output directory
        self.output_dir = self.config_manager.get_output_path(component_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized {component_name} component")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run the analysis component.
        
        Returns:
            Dictionary containing analysis results and metadata
        """
        pass
    
    def validate_inputs(self) -> bool:
        """
        Validate that required input files exist.
        
        Returns:
            True if all inputs are valid, False otherwise
        """
        return True
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> Path:
        """
        Save analysis results to file.
        
        Args:
            results: Results dictionary to save
            filename: Optional filename (defaults to component_results.json)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"{self.component_name}_results.json"
        
        output_path = self.output_dir / filename
        
        # Convert non-serializable objects to strings
        serializable_results = self._make_serializable(results)
        
        import json
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return f"DataFrame/Series with shape {obj.shape}"
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return f"{type(obj).__name__} object"
        else:
            try:
                import json
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    def cleanup_intermediate_files(self, file_patterns: list = None) -> None:
        """
        Clean up intermediate files if configured to do so.
        
        Args:
            file_patterns: List of file patterns to clean up
        """
        if not self.config_manager.should_cleanup_intermediate():
            return
        
        if file_patterns is None:
            file_patterns = ["*.tmp", "*.temp", "*_intermediate.*"]
        
        for pattern in file_patterns:
            for file_path in self.output_dir.glob(pattern):
                try:
                    file_path.unlink()
                    self.logger.debug(f"Cleaned up intermediate file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {file_path}: {e}")


class BaseDataProcessor(BaseAnalysisComponent):
    """Base class for data processing components."""
    
    def __init__(self, config: Dict[str, Any], component_name: str):
        super().__init__(config, component_name)
    
    def load_data(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from file with error handling.
        
        Args:
            file_path: Path to data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.logger.info(f"Loading data from: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
            elif file_path.suffix.lower() == '.h5ad':
                import scanpy as sc
                adata = sc.read_h5ad(file_path)
                return adata  # Return AnnData object directly
            else:
                # Try CSV as fallback
                df = pd.read_csv(file_path, **kwargs)
            
            self.logger.info(f"Loaded data shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            raise
    
    def save_data(self, data: pd.DataFrame, filename: str, **kwargs) -> Path:
        """
        Save DataFrame to file.
        
        Args:
            data: DataFrame to save
            filename: Output filename
            **kwargs: Additional arguments for pandas save functions
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        try:
            if filename.endswith('.csv'):
                data.to_csv(output_path, **kwargs)
            elif filename.endswith(('.xlsx', '.xls')):
                data.to_excel(output_path, **kwargs)
            elif filename.endswith('.parquet'):
                data.to_parquet(output_path, **kwargs)
            else:
                # Default to CSV
                data.to_csv(output_path, **kwargs)
            
            self.logger.info(f"Data saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save data to {output_path}: {e}")
            raise


class BaseMLComponent(BaseAnalysisComponent):
    """Base class for machine learning components."""
    
    def __init__(self, config: Dict[str, Any], component_name: str):
        super().__init__(config, component_name)
        
        # ML-specific configuration
        self.test_size = self.component_config.get('test_size', 0.2)
        self.cv_folds = self.component_config.get('cv_folds', 5)
        self.random_state = self.component_config.get('random_state', 42)
        self.verbose = self.component_config.get('verbose', True)
    
    def prepare_ml_data(self, data: pd.DataFrame, target_column: str) -> tuple:
        """
        Prepare data for machine learning.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple of (features, target)
        """
        self.logger.info("Preparing data for machine learning...")
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            self.logger.warning("Found missing values, filling with median/mode")
            # Fill numeric columns with median
            numeric_columns = X.select_dtypes(include=['number']).columns
            X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
            
            # Fill categorical columns with mode
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'unknown')
        
        self.logger.info(f"ML data prepared - Features: {X.shape}, Target: {y.shape}")
        return X, y
    
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, classification_report
        )
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC if probabilities available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
            except ValueError:
                pass  # Skip if only one class present
        
        return metrics


class BaseVisualizationComponent(BaseAnalysisComponent):
    """Base class for components that generate visualizations."""
    
    def __init__(self, config: Dict[str, Any], component_name: str):
        super().__init__(config, component_name)
        
        # Plotting configuration
        self.plot_config = self.config.get('plot_config', {})
        self.dpi = self.plot_config.get('dpi', 300)
        self.bbox_inches = self.plot_config.get('bbox_inches', 'tight')
        self.figure_formats = self.plot_config.get('figure_formats', ['png'])
        
        # Set publication style if configured
        if 'publication_style' in self.plot_config:
            import matplotlib.pyplot as plt
            plt.rcParams.update(self.plot_config['publication_style'])
    
    def save_figure(self, fig, filename: str, **kwargs) -> Path:
        """
        Save figure to file(s) in specified format(s).
        
        Args:
            fig: Matplotlib figure object
            filename: Base filename (without extension)
            **kwargs: Additional arguments for savefig
            
        Returns:
            Path to the first saved file
        """
        import matplotlib.pyplot as plt
        
        saved_path = None
        for fmt in self.figure_formats:
            output_path = self.output_dir / f"{filename}.{fmt}"
            
            # Set default kwargs
            save_kwargs = {
                'dpi': self.dpi,
                'bbox_inches': self.bbox_inches,
                **kwargs
            }
            
            fig.savefig(output_path, **save_kwargs)
            
            if saved_path is None:
                saved_path = output_path
            
            self.logger.debug(f"Figure saved to: {output_path}")
        
        return saved_path
    
    def close_figure(self, fig) -> None:
        """
        Close matplotlib figure to free memory.
        
        Args:
            fig: Matplotlib figure object
        """
        import matplotlib.pyplot as plt
        plt.close(fig)
    

class BaseVisualizationAnalyzer(BaseVisualizationComponent, BaseDataProcessor):
    """Combined base class for components that do both analysis and visualization."""
    
    def __init__(self, config: Dict[str, Any], component_name: str):
        """Initialize combined analyzer."""
        # Since both parent classes call BaseAnalysisComponent.__init__, 
        # we only need to call one of them to avoid duplicate initialization
        BaseVisualizationComponent.__init__(self, config, component_name)
        
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        pass



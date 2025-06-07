"""
Main pipeline orchestrator for nephrotoxicity analysis.
"""

import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from .types import ComponentResult
from config.settings import ConfigManager, create_output_directories
from utils.logging import get_logger
from utils.helpers import format_execution_time


class NephrotoxicityPipeline:
    """Main pipeline orchestrator for nephrotoxicity analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the analysis pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.config_manager = ConfigManager(config)
        self.logger = get_logger(__name__)
        
        # Create output directories
        create_output_directories(config)
        
        # Initialize component registry
        self._components = {}
        self._register_components()
        
        # Track execution results
        self.results = {}
        
        self.logger.info("Pipeline initialized")
        self.logger.info(f"Output directory: {config['output_base_dir']}")
    
    def _register_components(self) -> None:
        """Register all available analysis components."""
        from data.merger import DrugDataMerger
        from data.loader import LincsDataLoader
        from analysis.drug2cell import Drug2CellAnalyzer
        from analysis.statistical import StatisticalAnalyzer
        from analysis.differential import DifferentialExpressionAnalyzer
        from analysis.simulation import PowerAnalysisSimulator
        from ml.drug_classification import DrugToxicityClassifier
        from ml.lincs_classification import LincsMLClassifier
        
        self._components = {
            'merge_data': DrugDataMerger,
            'drug2cell': Drug2CellAnalyzer,
            'statistical': StatisticalAnalyzer,
            'ml_drug': DrugToxicityClassifier,
            'load_qc': LincsDataLoader,
            'differential': DifferentialExpressionAnalyzer,
            'ml_lincs': LincsMLClassifier,
            'simulation': PowerAnalysisSimulator
        }
    
    def run_all(self) -> Dict[str, ComponentResult]:
        """
        Run all enabled analysis pipelines.
        
        Returns:
            Dictionary mapping component names to their results
        """
        self.logger.info("Starting complete analysis pipeline")
        
        results = {}
        
        # Run Drug2Cell track
        if self._should_run_track('drug2cell'):
            self.logger.info("Running Drug2Cell analysis track")
            drug2cell_results = self.run_drug2cell_track()
            results.update(drug2cell_results)
        
        # Run LINCS track
        if self._should_run_track('lincs'):
            self.logger.info("Running LINCS analysis track")
            lincs_results = self.run_lincs_track()
            results.update(lincs_results)
        
        # Run simulation
        if self.config_manager.is_component_enabled('simulation'):
            self.logger.info("Running power analysis simulation")
            sim_results = self.run_simulation()
            results.update(sim_results)
        
        self.results.update(results)
        self._generate_summary_report()
        
        self.logger.info("Complete pipeline execution finished")
        return results
    
    def run_drug2cell_track(self) -> Dict[str, ComponentResult]:
        """
        Run Drug2Cell analysis track.
        
        Returns:
            Dictionary of component results
        """
        self.logger.info("Starting Drug2Cell analysis track")
        
        track_components = ['merge_data', 'drug2cell', 'statistical', 'ml_drug']
        return self._run_component_sequence(track_components)
    
    def run_lincs_track(self) -> Dict[str, ComponentResult]:
        """
        Run LINCS analysis track.
        
        Returns:
            Dictionary of component results
        """
        self.logger.info("Starting LINCS analysis track")
        
        track_components = ['load_qc', 'differential', 'ml_lincs']
        return self._run_component_sequence(track_components)
    
    def run_simulation(self) -> Dict[str, ComponentResult]:
        """
        Run power analysis simulation.
        
        Returns:
            Dictionary of component results
        """
        self.logger.info("Starting power analysis simulation")
        
        return self._run_component_sequence(['simulation'])
    
    def run_components(self, component_names: List[str]) -> Dict[str, ComponentResult]:
        """
        Run specific components.
        
        Args:
            component_names: List of component names to run
            
        Returns:
            Dictionary of component results
        """
        self.logger.info(f"Running components: {component_names}")
        
        return self._run_component_sequence(component_names)
    
    def _run_component_sequence(self, component_names: List[str]) -> Dict[str, ComponentResult]:
        """
        Run a sequence of components.
        
        Args:
            component_names: List of component names to run in order
            
        Returns:
            Dictionary of component results
        """
        results = {}
        
        for component_name in component_names:
            if not self.config_manager.is_component_enabled(component_name):
                self.logger.info(f"Skipping disabled component: {component_name}")
                continue
            
            result = self._run_single_component(component_name)
            results[component_name] = result
            
            # Stop execution if component failed and it's critical
            if not result.success and self._is_critical_component(component_name):
                self.logger.error(f"Critical component {component_name} failed, stopping execution")
                break
        
        return results
    
    def _run_single_component(self, component_name: str) -> ComponentResult:
        """
        Run a single analysis component.
        
        Args:
            component_name: Name of component to run
            
        Returns:
            Component execution result
        """
        if component_name not in self._components:
            error_msg = f"Unknown component: {component_name}"
            self.logger.error(error_msg)
            return ComponentResult(component_name, success=False, error=error_msg)
        
        self.logger.info(f"Running component: {component_name}")
        
        try:
            # Get component class and instantiate
            component_class = self._components[component_name]
            component = component_class(self.config, component_name)
            
            # Validate inputs
            if not component.validate_inputs():
                error_msg = f"Input validation failed for {component_name}"
                self.logger.error(error_msg)
                return ComponentResult(component_name, success=False, error=error_msg)
            
            # Run component
            start_time = pd.Timestamp.now()
            component_data = component.run()
            end_time = pd.Timestamp.now()
            
            execution_time = format_execution_time(start_time, end_time)
            self.logger.info(f"Component {component_name} completed in {execution_time}")
            
            # Create successful result
            result = ComponentResult(
                component_name=component_name,
                success=True,
                data=component_data
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Component {component_name} failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            
            return ComponentResult(
                component_name=component_name,
                success=False,
                error=error_msg
            )
    
    def _should_run_track(self, track_name: str) -> bool:
        """Check if any components in a track are enabled."""
        track_components = {
            'drug2cell': ['merge_data', 'drug2cell', 'statistical', 'ml_drug'],
            'lincs': ['load_qc', 'differential', 'ml_lincs']
        }
        
        if track_name not in track_components:
            return False
        
        return any(
            self.config_manager.is_component_enabled(comp)
            for comp in track_components[track_name]
        )
    
    def _is_critical_component(self, component_name: str) -> bool:
        """Check if a component is critical for the pipeline."""
        # Components that are critical for their respective tracks
        critical_components = {
            'merge_data', 'drug2cell', 'load_qc'
        }
        return component_name in critical_components
    
    def get_component_result(self, component_name: str) -> Optional[ComponentResult]:
        """
        Get result for a specific component.
        
        Args:
            component_name: Name of component
            
        Returns:
            Component result or None if not found
        """
        return self.results.get(component_name)
    
    def get_successful_components(self) -> List[str]:
        """Get list of successfully executed components."""
        return [
            name for name, result in self.results.items()
            if result.success
        ]
    
    def get_failed_components(self) -> List[str]:
        """Get list of failed components."""
        return [
            name for name, result in self.results.items()
            if not result.success
        ]
    
    def _generate_summary_report(self) -> None:
        """Generate a summary report of pipeline execution."""
        if not self.config_manager.should_generate_report():
            return
        
        self.logger.info("Generating pipeline summary report")
        
        output_dir = Path(self.config['output_base_dir'])
        report_path = output_dir / "pipeline_summary_report.txt"
        
        successful = self.get_successful_components()
        failed = self.get_failed_components()
        
        with open(report_path, 'w') as f:
            f.write("NEPHROTOXICITY ANALYSIS PIPELINE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total components executed: {len(self.results)}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(failed)}\n\n")
            
            if successful:
                f.write("SUCCESSFUL COMPONENTS:\n")
                f.write("-" * 25 + "\n")
                for component in successful:
                    result = self.results[component]
                    f.write(f"✓ {component} (completed at {result.timestamp})\n")
                f.write("\n")
            
            if failed:
                f.write("FAILED COMPONENTS:\n")
                f.write("-" * 20 + "\n")
                for component in failed:
                    result = self.results[component]
                    f.write(f"✗ {component}: {result.error}\n")
                f.write("\n")
            
            # Add component-specific summaries
            f.write("COMPONENT DETAILS:\n")
            f.write("-" * 20 + "\n")
            for component_name, result in self.results.items():
                f.write(f"\n{component_name.upper()}:\n")
                f.write(f"  Status: {'SUCCESS' if result.success else 'FAILED'}\n")
                f.write(f"  Timestamp: {result.timestamp}\n")
                
                if result.success and result.data:
                    f.write(f"  Output data keys: {list(result.data.keys())}\n")
                elif not result.success:
                    f.write(f"  Error: {result.error}\n")
            
            # Add configuration summary
            f.write(f"\nCONFIGURATION:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Output directory: {self.config['output_base_dir']}\n")
            f.write(f"Cell type annotation: {self.config.get('cell_type_annotation', 'N/A')}\n")
            f.write(f"Cleanup intermediate: {self.config.get('cleanup_intermediate', False)}\n")
            
            enabled_steps = [
                step for step, enabled in self.config.get('analysis_steps', {}).items()
                if enabled
            ]
            f.write(f"Enabled analysis steps: {', '.join(enabled_steps)}\n")
        
        self.logger.info(f"Summary report saved to: {report_path}")
    
    def export_results(self, output_path: Union[str, Path] = None) -> Path:
        """
        Export all pipeline results to a single file.
        
        Args:
            output_path: Optional path for export file
            
        Returns:
            Path to exported results file
        """
        if output_path is None:
            output_path = Path(self.config['output_base_dir']) / "pipeline_results.json"
        else:
            output_path = Path(output_path)
        
        # Prepare results for export
        export_data = {
            'pipeline_config': self.config,
            'execution_summary': {
                'total_components': len(self.results),
                'successful_components': len(self.get_successful_components()),
                'failed_components': len(self.get_failed_components()),
                'execution_timestamp': pd.Timestamp.now().isoformat()
            },
            'component_results': {
                name: result.to_dict() for name, result in self.results.items()
            }
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Pipeline results exported to: {output_path}")
        return output_path
    
    def validate_pipeline_config(self) -> List[str]:
        """
        Validate pipeline configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check if at least one analysis step is enabled
        analysis_steps = self.config.get('analysis_steps', {})
        if not any(analysis_steps.values()):
            errors.append("No analysis steps are enabled")
        
        # Check for required data paths based on enabled components
        if self.config_manager.is_component_enabled('drug2cell'):
            data_paths = self.config.get('data_paths', {})
            required_files = ['merged_drug_dataset', 'single_cell_data']
            for file_key in required_files:
                if file_key not in data_paths:
                    errors.append(f"Missing required data path: {file_key}")
                elif not Path(data_paths[file_key]).exists():
                    errors.append(f"Data file not found: {data_paths[file_key]}")
        
        # Check LINCS data paths
        if any(self.config_manager.is_component_enabled(comp) 
               for comp in ['load_qc', 'differential', 'ml_lincs']):
            merge_config = self.config.get('merge_drug_data_sources', {})
            if 'lincs_data_paths' in merge_config:
                lincs_paths = merge_config['lincs_data_paths']
                for path_key, path_value in lincs_paths.items():
                    if not Path(path_value).exists():
                        errors.append(f"LINCS data file not found: {path_value}")
        
        return errors

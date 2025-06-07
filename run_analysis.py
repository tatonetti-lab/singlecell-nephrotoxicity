#!/usr/bin/env python3
"""
Nephrotoxicity Analysis Pipeline Runner

This script properly sets up the import path and runs the analysis pipeline.
Use this script instead of main.py to avoid import issues.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Optional

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Set environment to allow relative imports
sys.path.insert(0, str(current_dir.parent))
os.environ['PYTHONPATH'] = str(current_dir) + ':' + os.environ.get('PYTHONPATH', '')

# Fix scipy array API compatibility with scikit-learn
os.environ['SCIPY_ARRAY_API'] = '1'

from core.pipeline import NephrotoxicityPipeline
from core.base import ComponentResult
from config.settings import load_config
from utils.logging import setup_logging


def setup_package_imports():
    """Setup package structure for imports to work."""
    import types
    
    # Create the main package module
    package = types.ModuleType('nephrotoxicity_analysis')
    package.__path__ = [str(current_dir)]
    package.__file__ = str(current_dir / '__init__.py')
    sys.modules['nephrotoxicity_analysis'] = package
    
    # Create subpackage modules
    for submodule in ['core', 'config', 'utils', 'analysis', 'ml', 'data', 'visualization']:
        subpackage = types.ModuleType(f'nephrotoxicity_analysis.{submodule}')
        subpackage.__path__ = [str(current_dir / submodule)]
        subpackage.__file__ = str(current_dir / submodule / '__init__.py')
        sys.modules[f'nephrotoxicity_analysis.{submodule}'] = subpackage

# Setup imports
setup_package_imports()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Nephrotoxicity Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available pipeline components:
  drug2cell_track    - Run complete Drug2Cell analysis track
  lincs_track       - Run complete LINCS analysis track  
  simulation        - Run power analysis simulation
  
  merge_data        - Merge drug data sources
  drug2cell         - Drug2cell analysis
  statistical       - Statistical analysis
  ml_drug           - Machine learning on drug scores
  load_qc           - Data loading and QC
  differential      - Differential expression analysis
  ml_lincs          - Machine learning on expression data

Examples:
  # Run all pipelines
  python run_analysis.py --all
  
  # Run specific track
  python run_analysis.py --pipeline drug2cell_track
  
  # Run individual components
  python run_analysis.py --components merge_data drug2cell
  
  # Use custom config
  python run_analysis.py --config my_config.yaml --all
        """
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Override output directory from config"
    )
    
    # Execution options
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all analysis pipelines"
    )
    
    group.add_argument(
        "--pipeline",
        choices=["drug2cell_track", "lincs_track", "simulation"],
        help="Run specific analysis track"
    )
    
    group.add_argument(
        "--components",
        nargs="+",
        choices=[
            "merge_data", "drug2cell", "statistical", "ml_drug",
            "load_qc", "differential", "ml_lincs", "simulation"
        ],
        help="Run specific components"
    )
    
    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: output_dir/pipeline.log)"
    )
    
    return parser.parse_args()


def validate_config_paths(config: dict) -> bool:
    """Validate that required input files exist."""
    required_files = []
    
    # Drug2Cell track files
    if 'data_paths' in config:
        data_paths = config['data_paths']
        drug2cell_files = [
            'merged_drug_dataset', 'single_cell_data'
        ]
        for key in drug2cell_files:
            if key in data_paths:
                required_files.append(data_paths[key])
    
    # LINCS track files (from merge_drug_data_sources data_paths)
    merge_config = config.get('merge_drug_data_sources', {})
    if 'data_paths' in merge_config:
        for file_path in merge_config['data_paths'].values():
            required_files.append(file_path)
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("ERROR: Missing required input files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    return True


def print_execution_plan(pipeline, args: argparse.Namespace):
    """Print what will be executed."""
    print("=" * 60)
    print("EXECUTION PLAN")
    print("=" * 60)
    
    print(f"Configuration: {args.config}")
    print(f"Output directory: {pipeline.config['output_base_dir']}")
    print(f"Verbose mode: {args.verbose}")
    
    if args.all:
        print("\nWill execute all analysis pipelines:")
        print("  1. Drug2Cell Track:")
        print("     - Merge drug data sources")
        print("     - Drug2cell analysis") 
        print("     - Statistical analysis")
        print("     - Machine learning on drug scores")
        print("  2. LINCS Track:")
        print("     - Data loading and QC")
        print("     - Differential expression analysis")
        print("     - Machine learning on expression data")
        print("  3. Simulation:")
        print("     - Power analysis simulation")
        
    elif args.pipeline:
        print(f"\nWill execute pipeline: {args.pipeline}")
        
    elif args.components:
        print(f"\nWill execute components: {', '.join(args.components)}")
    
    print("=" * 60)


def main():
    """Main entry point."""
    # Setup basic logging first
    logger = setup_logging()
    
    try:
        args = parse_arguments()
        
        # Load configuration
        config = load_config(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            config['output_base_dir'] = args.output_dir
        
        # Setup detailed logging with file
        log_file = args.log_file
        if not log_file:
            log_file = Path(config['output_base_dir']) / "pipeline.log"
        
        logger = setup_logging(
            verbose=args.verbose,
            log_file=str(log_file)
        )
        
        # Validate configuration
        if not validate_config_paths(config):
            sys.exit(1)
        
        # Initialize pipeline
        pipeline = NephrotoxicityPipeline(config)
        
        # Print execution plan
        print_execution_plan(pipeline, args)
        
        if args.dry_run:
            print("\nDRY RUN - No analysis will be performed")
            return
        
        print("\nStarted analysis... Please check the pipeline.log file for details.")
        
        # Execute based on arguments
        results = None
        if args.all:
            logger.info("Running complete analysis pipeline")
            results = pipeline.run_all()
            
        elif args.pipeline:
            logger.info(f"Running pipeline: {args.pipeline}")
            if args.pipeline == "drug2cell_track":
                results = pipeline.run_drug2cell_track()
            elif args.pipeline == "lincs_track":
                results = pipeline.run_lincs_track()
            elif args.pipeline == "simulation":
                results = pipeline.run_simulation()
            
        elif args.components:
            logger.info(f"Running components: {args.components}")
            results = pipeline.run_components(args.components)
        
        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETED")
        print("=" * 60)
        print(f"Results saved to: {config['output_base_dir']}")
        
        if isinstance(results, dict):
            print(f"Completed components: {len(results)}")
            for component, result in results.items():
                if isinstance(result, ComponentResult):
                    status = "✓" if result.success else "✗"
                elif isinstance(result, dict):
                    status = "✓" if result.get('success', True) else "✗"
                else:
                    status = "✓"  # Default to success for other types
                print(f"  {status} {component}")
        
        logger.info("Pipeline execution completed successfully")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        logger.warning("Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}")
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
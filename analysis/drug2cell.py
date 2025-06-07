"""
Drug2Cell analysis component for calculating drug scores on single-cell data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from collections import Counter

from core.base import BaseVisualizationAnalyzer
from utils.helpers import create_toxicity_dictionary
from utils.logging import ComponentLogger


class Drug2CellAnalyzer(BaseVisualizationAnalyzer):
    """
    Component for Drug2Cell analysis.
    
    This component:
    1. Loads drug dataset and single-cell data
    2. Creates toxicity dictionary for drug2cell analysis
    3. Runs drug2cell scoring
    4. Processes and averages drug scores by cell type
    5. Creates visualizations
    6. Saves results for downstream analysis
    """
    
    def __init__(self, config: Dict[str, Any], component_name: str = 'drug2cell'):
        super().__init__(config, component_name)
        
        # Get data paths
        self.data_paths = config.get('data_paths', {})
        self.merged_drug_dataset = self.data_paths.get('merged_drug_dataset')
        self.single_cell_data = self.data_paths.get('single_cell_data')
        self.cell_type_annotation = config.get('cell_type_annotation', 'kca')
        
        self.component_logger = ComponentLogger(component_name, verbose=self.component_config.get('verbose', True))
    
    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        if not self.merged_drug_dataset or not Path(self.merged_drug_dataset).exists():
            self.logger.error(f"Merged drug dataset not found: {self.merged_drug_dataset}")
            return False
        
        if not self.single_cell_data or not Path(self.single_cell_data).exists():
            self.logger.error(f"Single-cell data not found: {self.single_cell_data}")
            return False
        
        return True
    
    def run(self) -> Dict[str, Any]:
        """Run Drug2Cell analysis pipeline."""
        self.component_logger.start_component(total_steps=7)
        
        try:
            # Step 1: Load datasets
            self.component_logger.step_completed("Loading datasets")
            drug_df, adata = self._load_datasets()
            
            # Step 2: Create toxicity dictionary
            self.component_logger.step_completed("Creating toxicity dictionary")
            toxicity_dict = self._create_toxicity_dictionary(drug_df)
            
            # Step 3: Run drug2cell scoring
            self.component_logger.step_completed("Running drug2cell analysis")
            adata = self._run_drug2cell_scoring(adata, toxicity_dict)
            
            # Step 4: Extract and process drug scores
            self.component_logger.step_completed("Processing drug scores")
            cell_drug_matrix = self._extract_drug_scores(adata)
            toxic_matrix, non_toxic_matrix = self._process_drug_categories(toxicity_dict, cell_drug_matrix)
            
            # Step 5: Average by cell type
            self.component_logger.step_completed("Averaging scores by cell type")
            nephrotoxic_avg, non_nephrotoxic_avg = self._average_drug_scores_by_cell_type(
                toxic_matrix, non_toxic_matrix
            )
            
            # Step 6: Create combined matrix
            self.component_logger.step_completed("Creating combined analysis matrix")
            combined_matrix = self._create_combined_drug_matrix(nephrotoxic_avg, non_nephrotoxic_avg)
            
            # Step 7: Create visualizations and save results
            self.component_logger.step_completed("Creating visualizations and saving results")
            self._create_visualizations(adata)
            saved_files = self._save_results(nephrotoxic_avg, non_nephrotoxic_avg, combined_matrix)
            
            self.component_logger.finish_component(success=True)
            
            return {
                'adata': adata,
                'toxicity_dict': toxicity_dict,
                'cell_drug_matrix': cell_drug_matrix,
                'nephrotoxic_avg': nephrotoxic_avg,
                'non_nephrotoxic_avg': non_nephrotoxic_avg,
                'combined_matrix': combined_matrix,
                'saved_files': saved_files,
                'summary_stats': self._generate_summary_stats(toxicity_dict, nephrotoxic_avg, non_nephrotoxic_avg)
            }
            
        except Exception as e:
            self.component_logger.finish_component(success=False)
            self.logger.error(f"Drug2Cell analysis failed: {e}")
            raise
    
    def _load_datasets(self) -> Tuple[pd.DataFrame, Any]:
        """Load drug dataset and single-cell data."""
        self.logger.info("Loading datasets...")
        
        # Load drug dataset
        drug_df = pd.read_csv(self.merged_drug_dataset)
        self.logger.info(f"Loaded drug dataset: {drug_df.shape}")
        
        # Remove drugs not found in any toxicity dataset
        initial_count = len(drug_df)
        drug_df = drug_df[drug_df['sources'] != 'NotFound']
        self.logger.info(f"Removed {initial_count - len(drug_df)} drugs not found in toxicity datasets")
        
        # Load single-cell data
        try:
            import scanpy as sc
            adata = sc.read_h5ad(self.single_cell_data)
            # Ensure gene names are strings
            adata.var.index = adata.var.index.astype(str)
            self.logger.info(f"Loaded single-cell data: {adata.n_obs} cells, {adata.n_vars} genes")
        except ImportError:
            raise ImportError("scanpy is required for loading single-cell data")
        
        return drug_df, adata
    
    def _create_toxicity_dictionary(self, drug_df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
        """Create toxicity dictionary for drug2cell analysis."""
        self.logger.info("Creating toxicity dictionary...")
        
        toxicity_dict = create_toxicity_dictionary(
            drug_df,
            drug_col='drug_name',
            toxicity_col='is_toxic',
            genes_col='combined_genes'
        )
        
        return toxicity_dict
    
    def _run_drug2cell_scoring(self, adata: Any, toxicity_dict: Dict) -> Any:
        """Run drug2cell scoring analysis."""
        self.logger.info("Running drug2cell scoring...")
        
        try:
            import drug2cell as d2c
            
            # Run drug2cell scoring
            scoring = d2c.score(
                adata, 
                targets=toxicity_dict, 
                nested=self.component_config.get('nested', True),
                use_raw=self.component_config.get('use_raw', True)
            )
            
            self.logger.info("Drug2cell scoring completed")
            return adata
            
        except ImportError:
            raise ImportError("drug2cell package is required for drug2cell analysis")
        except Exception as e:
            self.logger.error(f"Drug2cell scoring failed: {e}")
            raise
    
    def _extract_drug_scores(self, adata: Any) -> pd.DataFrame:
        """Extract drug scores and create a combined matrix."""
        self.logger.info("Extracting drug scores...")
        
        # Extract drug scores
        dr2c = adata.uns['drug2cell']
        cell_drug_matrix = dr2c.X
        drug_names = list(dr2c.var.index)
        cell_types = adata.obs[self.cell_type_annotation]
        
        # Create named matrix
        cell_drug_named_matrix = pd.DataFrame(
            cell_drug_matrix, 
            columns=drug_names, 
            index=cell_types
        )
        
        self.logger.info(f"Created drug score matrix: {cell_drug_named_matrix.shape}")
        return cell_drug_named_matrix
    
    def _process_drug_categories(self, toxicity_dict: Dict, 
                               cell_drug_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process drug scores by toxicity category."""
        self.logger.info("Processing drug categories...")
        
        def process_category(category: str) -> pd.DataFrame:
            drugs = list(toxicity_dict[category].keys())
            available_drugs = list(set(drugs) & set(cell_drug_matrix.columns))
            
            if available_drugs:
                drug_matrix = cell_drug_matrix[available_drugs]
                self.logger.info(f"Created {category} drug matrix with {len(available_drugs)} drugs")
            else:
                self.logger.warning(f"No {category} drugs found in the cell_drug_matrix")
                drug_matrix = pd.DataFrame()
            
            missing_drugs = set(drugs) - set(cell_drug_matrix.columns)
            if missing_drugs:
                self.logger.info(f"Missing {category} drugs: {len(missing_drugs)}")
            
            return drug_matrix
        
        toxic_matrix = process_category('toxic')
        non_toxic_matrix = process_category('non_toxic')
        
        return toxic_matrix, non_toxic_matrix
    
    def _average_drug_scores_by_cell_type(self, toxic_matrix: pd.DataFrame, 
                                        non_toxic_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Average drug scores for each cell type."""
        self.logger.info("Averaging drug scores by cell type...")
        
        # Group by cell type annotation and calculate mean
        nephrotoxic_avg = toxic_matrix.groupby(self.cell_type_annotation).mean()
        non_nephrotoxic_avg = non_toxic_matrix.groupby(self.cell_type_annotation).mean()
        
        # Remove duplicate columns
        nephrotoxic_avg = nephrotoxic_avg.loc[:, ~nephrotoxic_avg.columns.duplicated()]
        non_nephrotoxic_avg = non_nephrotoxic_avg.loc[:, ~non_nephrotoxic_avg.columns.duplicated()]
        
        self.logger.info(f"Averaged drug scores by cell type:")
        self.logger.info(f"  Nephrotoxic matrix: {nephrotoxic_avg.shape}")
        self.logger.info(f"  Non-nephrotoxic matrix: {non_nephrotoxic_avg.shape}")
        
        return nephrotoxic_avg, non_nephrotoxic_avg
    
    def _create_combined_drug_matrix(self, nephrotoxic_avg: pd.DataFrame, 
                                   non_nephrotoxic_avg: pd.DataFrame) -> pd.DataFrame:
        """Create combined matrix for analysis and visualization."""
        self.logger.info("Creating combined drug matrix...")
        
        # Transpose and add group labels
        t_nephrotoxic = nephrotoxic_avg.T
        t_non_nephrotoxic = non_nephrotoxic_avg.T
        
        t_nephrotoxic['group'] = 'nephrotoxic'
        t_non_nephrotoxic['group'] = 'non_nephrotoxic'
        
        # Combine matrices
        combined = pd.concat([t_nephrotoxic, t_non_nephrotoxic])
        combined = combined.reset_index().rename(columns={'index': 'drug_id'})
        
        # Convert to long format
        combined_long = combined.melt(
            id_vars=['drug_id', 'group'], 
            var_name='cell_type', 
            value_name='drug_score'
        )
        
        # Add log drug scores
        combined_long['log_drug_score'] = np.log(combined_long['drug_score'])
        
        # Handle infinite and NaN values
        combined_long = combined_long.replace([np.inf, -np.inf], np.nan)
        combined_long = combined_long.fillna(0)
        
        self.logger.info(f"Created combined drug matrix: {combined_long.shape}")
        return combined_long
    
    def _create_visualizations(self, adata: Any) -> None:
        """Create sample UMAP plots for selected drugs."""
        self.logger.info("Creating visualizations...")
        
        try:
            import scanpy as sc
            
            dr2c = adata.uns['drug2cell']
            
            # Sample drugs for visualization
            sample_drugs = {
                'nephrotoxic': ['ibuprofen', 'acyclovir', 'cyclosporine'],
                'non_nephrotoxic': ['ferrous gluconate', 'amoxicillin', 'chlorpheniramine']
            }
            
            fig, axs = plt.subplots(2, 3, figsize=(17, 10))
            
            # Plot sample drugs
            for i, (category, drugs) in enumerate(sample_drugs.items()):
                for j, drug in enumerate(drugs):
                    if drug in dr2c.var.index:
                        sc.pl.umap(dr2c, color=drug, color_map="twilight", 
                                  ax=axs[i, j], show=False, vmin=0, vmax=3)
                        axs[i, j].set_title(drug.title())
                    else:
                        axs[i, j].text(0.5, 0.5, f'{drug}\n(not available)', 
                                     ha='center', va='center', transform=axs[i, j].transAxes)
                        axs[i, j].set_title(drug.title())
            
            # Add category labels
            fig.text(0.5, 0.95, 'Nephrotoxic', ha='center', va='center', 
                     fontsize=16, fontweight='bold')
            fig.text(0.5, 0.48, 'Non-nephrotoxic', ha='center', va='center', 
                     fontsize=16, fontweight='bold')
            
            plt.subplots_adjust(hspace=0.3)
            self.save_figure(fig, 'sample_drug_umaps')
            self.close_figure(fig)
            
            self.logger.info("Sample drug UMAP plots created")
            
        except Exception as e:
            self.logger.warning(f"Failed to create UMAP visualizations: {e}")
    
    def _save_results(self, nephrotoxic_avg: pd.DataFrame, non_nephrotoxic_avg: pd.DataFrame,
                     combined_matrix: pd.DataFrame) -> Dict[str, Path]:
        """Save analysis results."""
        self.logger.info("Saving results...")
        
        saved_files = {}
        
        # Save key matrices
        saved_files['combined_matrix'] = self.save_data(combined_matrix, 'combined_drug_matrix.csv', index=False)
        saved_files['nephrotoxic_avg'] = self.save_data(nephrotoxic_avg, 'nephrotoxic_avg_matrix.csv', index=True)
        saved_files['non_nephrotoxic_avg'] = self.save_data(non_nephrotoxic_avg, 'non_nephrotoxic_avg_matrix.csv', index=True)
        
        return saved_files
    
    def _generate_summary_stats(self, toxicity_dict: Dict, 
                              nephrotoxic_avg: pd.DataFrame,
                              non_nephrotoxic_avg: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary_stats = {
            'total_drugs_analyzed': len(toxicity_dict['toxic']) + len(toxicity_dict['non_toxic']),
            'nephrotoxic_drugs': len(toxicity_dict['toxic']),
            'non_nephrotoxic_drugs': len(toxicity_dict['non_toxic']),
            'cell_types': list(nephrotoxic_avg.index),
            'n_cell_types': len(nephrotoxic_avg.index),
            'nephrotoxic_matrix_shape': nephrotoxic_avg.shape,
            'non_nephrotoxic_matrix_shape': non_nephrotoxic_avg.shape
        }
        
        # Save summary statistics
        summary_path = self.output_dir / 'analysis_summary.json'
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        self.logger.info(f"Analysis summary saved to: {summary_path}")
        return summary_stats
    
    def save_figure(self, fig, filename: str) -> Path:
        """Save matplotlib figure to output directory."""
        output_path = self.output_dir / filename
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    def save_data(self, data: pd.DataFrame, filename: str, **kwargs) -> Path:
        """Save pandas DataFrame to output directory."""
        output_path = self.output_dir / filename
        data.to_csv(output_path, **kwargs)
        return output_path

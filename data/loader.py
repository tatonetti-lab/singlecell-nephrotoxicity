"""
LINCS data loading and quality control component.

This module handles:
1. Loading and merging LINCS datasets with toxicity data
2. Filtering for kidney cell lines
3. Parsing gene expression data from GCTX files
4. Basic quality control checks
5. Gene ID to symbol mapping
6. Saving cleaned datasets for downstream analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from core.base import BaseDataProcessor, BaseVisualizationComponent
from config.constants import KIDNEY_CELL_IDS
from utils.logging import ComponentLogger


class LincsDataLoader(BaseDataProcessor, BaseVisualizationComponent):
    """Component for loading and processing LINCS data."""
    
    def __init__(self, config: Dict[str, Any], component_name: str = 'load_qc'):
        super().__init__(config, component_name)
        
        # Get LINCS-specific configuration
        self.lincs_config = config.get('merge_drug_data_sources', {})
        self.data_paths = self.lincs_config.get('lincs_data_paths', {})
        self.kidney_cell_ids = config.get('lincs_config', {}).get('kidney_cell_ids', KIDNEY_CELL_IDS)
        
        self.component_logger = ComponentLogger(component_name, verbose=self.component_config.get('verbose', True))
    
    def validate_inputs(self) -> bool:
        """Validate that required LINCS input files exist."""
        required_files = ['merged_drug', 'lincs_pert', 'lincs_gene', 'lincs_cell', 'lincs_sig', 'gctx']
        missing_files = []
        
        for file_key in required_files:
            if file_key not in self.data_paths:
                missing_files.append(f"Missing LINCS data path: {file_key}")
            elif not Path(self.data_paths[file_key]).exists():
                missing_files.append(f"LINCS file not found: {self.data_paths[file_key]}")
        
        if missing_files:
            for error in missing_files:
                self.logger.error(error)
            return False
        
        return True
    
    def run(self) -> Dict[str, Any]:
        """Run the LINCS data loading and QC pipeline."""
        self.component_logger.start_component(total_steps=8)
        
        try:
            # Step 1: Load and merge metadata
            self.component_logger.step_completed("Loading and merging metadata")
            toxicity_data, lincs_gene = self._load_and_merge_metadata()
            
            # Step 2: Filter for kidney cells and expand distil_ids
            self.component_logger.step_completed("Filtering for kidney cells")
            expanded_data = self._filter_kidney_cells_and_expand(toxicity_data)
            
            # Step 3: Validate sample IDs against GCTX metadata
            self.component_logger.step_completed("Validating sample IDs")
            expanded_data_filtered, valid_ids = self._validate_sample_ids(expanded_data)
            
            # Step 4: Load gene expression data
            self.component_logger.step_completed("Loading gene expression data")
            gctx_data = self._load_gene_expression_data(valid_ids)
            
            if gctx_data is None:
                raise ValueError("Failed to load gene expression data")
            
            # Step 5: Map gene IDs to symbols
            self.component_logger.step_completed("Mapping gene IDs to symbols")
            expression_df = self._map_gene_ids_to_symbols(gctx_data, lincs_gene)
            
            # Step 6: Prepare metadata for expression data
            self.component_logger.step_completed("Preparing metadata")
            expression_df_final, metadata_df_final = self._prepare_metadata_for_expression(
                expanded_data_filtered, expression_df
            )
            
            # Step 7: Perform quality control
            self.component_logger.step_completed("Performing quality control")
            qc_results = self._perform_basic_qc(expression_df_final, metadata_df_final)
            
            # Step 8: Save processed data
            self.component_logger.step_completed("Saving processed data")
            saved_files = self._save_processed_data(expression_df_final, metadata_df_final, expanded_data_filtered)
            
            self.component_logger.finish_component(success=True)
            
            return {
                'gene_expression_data': expression_df_final,
                'sample_metadata': metadata_df_final,
                'expanded_toxicity_metadata': expanded_data_filtered,
                'qc_results': qc_results,
                'saved_files': saved_files,
                'n_genes': len(expression_df_final),
                'n_samples': len(expression_df_final.columns),
                'n_toxic_samples': sum(metadata_df_final['toxicity_label'] == 'Toxic'),
                'n_nontoxic_samples': sum(metadata_df_final['toxicity_label'] == 'Non-toxic')
            }
            
        except Exception as e:
            self.component_logger.finish_component(success=False)
            self.logger.error(f"LINCS data loading failed: {e}")
            raise
    
    def _load_and_merge_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and merge all LINCS metadata files with toxicity data."""
        self.logger.info("Loading LINCS metadata files...")
        
        # Load datasets
        merged_db = pd.read_csv(self.data_paths['merged_drug'])
        lincs_pert = pd.read_csv(self.data_paths['lincs_pert'], on_bad_lines="warn", sep="\t")
        lincs_gene = pd.read_csv(self.data_paths['lincs_gene'], sep="\t")
        lincs_cell = pd.read_csv(self.data_paths['lincs_cell'], sep="\t")
        lincs_sig = pd.read_csv(self.data_paths['lincs_sig'], sep="\t")
        
        self.logger.info(f"Loaded datasets:")
        self.logger.info(f"  - Drug toxicity data: {merged_db.shape}")
        self.logger.info(f"  - LINCS perturbation info: {lincs_pert.shape}")
        self.logger.info(f"  - LINCS gene info: {lincs_gene.shape}")
        self.logger.info(f"  - LINCS cell info: {lincs_cell.shape}")
        self.logger.info(f"  - LINCS signature info: {lincs_sig.shape}")
        
        # Filter for compound treatments only
        filt_lincs_pert = lincs_pert[lincs_pert['pert_type'] == "trt_cp"]
        self.logger.info(f"  - Filtered perturbations (compounds only): {filt_lincs_pert.shape}")
        
        # Merge perturbation with signature info
        merged_pert_sig = pd.merge(filt_lincs_pert, lincs_sig, on="pert_id", how="inner")
        self.logger.info(f"  - Merged perturbation-signature data: {merged_pert_sig.shape}")
        
        # Merge with toxicity data
        toxicity_pert_sig = pd.merge(merged_pert_sig, merged_db, 
                                    left_on="pert_iname_x", right_on="drug_name", how="inner")
        self.logger.info(f"  - Merged with toxicity data: {toxicity_pert_sig.shape}")
        
        # Merge with cell info
        toxicity_pert_sig_cell = pd.merge(toxicity_pert_sig, lincs_cell, on="cell_id", how="inner")
        self.logger.info(f"  - Final merged dataset: {toxicity_pert_sig_cell.shape}")
        
        return toxicity_pert_sig_cell, lincs_gene
    
    def _filter_kidney_cells_and_expand(self, toxicity_data: pd.DataFrame) -> pd.DataFrame:
        """Filter for kidney cell lines and expand distil_ids."""
        self.logger.info(f"Filtering for kidney cell lines: {self.kidney_cell_ids}")
        
        # Convert toxicity labels
        toxicity_data['toxicity_label'] = toxicity_data['is_toxic'].map({
            'False': 'Non-toxic', 
            'True': 'Toxic',
            'Conflict': 'Toxic'  # Treating conflicts as toxic
        })
        
        # Filter for kidney cell lines
        filtered_data = toxicity_data[toxicity_data['cell_id'].isin(self.kidney_cell_ids)]
        self.logger.info(f"After kidney cell filtering: {filtered_data.shape}")
        
        # Expand distil_ids (some entries have multiple IDs separated by |)
        expanded_rows = []
        for _, row in filtered_data.iterrows():
            distil_ids = str(row['distil_id']).split('|')
            for did in distil_ids:
                new_row = row.copy()
                new_row['distil_id'] = did.strip()
                expanded_rows.append(new_row)
        
        expanded_data = pd.DataFrame(expanded_rows).reset_index(drop=True)
        self.logger.info(f"After expanding distil_ids: {expanded_data.shape}")
        
        return expanded_data
    
    def _validate_sample_ids(self, expanded_data: pd.DataFrame) -> Tuple[pd.DataFrame, set]:
        """Validate which distil_ids exist in the GCTX metadata."""
        self.logger.info("Validating sample IDs against GCTX metadata...")
        
        try:
            # Import here to avoid dependency issues if not available
            from cmapPy.pandasGEXpress.parse import parse
            
            # Get metadata to check which IDs exist
            col_metadata = parse(self.data_paths['gctx'], col_meta_only=True)
            valid_ids = set(expanded_data['distil_id']).intersection(set(col_metadata.index))
            
            self.logger.info(f"Total IDs in expanded data: {len(expanded_data['distil_id'].unique())}")
            self.logger.info(f"Valid IDs found in GCTX: {len(valid_ids)}")
            
            # Filter expanded data to only include valid IDs
            expanded_data_filtered = expanded_data[expanded_data['distil_id'].isin(valid_ids)]
            self.logger.info(f"After filtering: {len(expanded_data_filtered['distil_id'].unique())} unique samples")
            
            return expanded_data_filtered, valid_ids
            
        except Exception as e:
            self.logger.error(f"Error validating sample IDs: {e}")
            return expanded_data, set(expanded_data['distil_id'].unique())
    
    def _load_gene_expression_data(self, valid_ids: set) -> Optional[Any]:
        """Load gene expression data from GCTX file."""
        self.logger.info(f"Loading gene expression data for {len(valid_ids)} samples...")
        
        try:
            from cmapPy.pandasGEXpress.parse import parse
            
            # Parse the GCTX file with valid IDs
            gctx_data = parse(self.data_paths['gctx'], cid=list(valid_ids))
            self.logger.info(f"Successfully loaded expression data: {gctx_data.data_df.shape}")
            return gctx_data
            
        except Exception as e:
            self.logger.error(f"Error loading GCTX data: {e}")
            return None
    
    def _map_gene_ids_to_symbols(self, gctx_data: Any, lincs_gene: pd.DataFrame) -> pd.DataFrame:
        """Map gene IDs to gene symbols."""
        self.logger.info("Mapping gene IDs to symbols...")
        
        # Create mapping dictionary
        gene_id_to_symbol = pd.Series(
            lincs_gene.pr_gene_symbol.values, 
            index=lincs_gene.pr_gene_id.astype(str).values
        ).to_dict()
        
        # Map gene IDs to symbols
        expression_df = gctx_data.data_df.copy()
        new_index = [gene_id_to_symbol.get(str(idx), str(idx)) for idx in expression_df.index]
        expression_df.index = new_index
        
        # Check mapping success
        unmapped_genes = [idx for idx in new_index if idx.isdigit()]
        if unmapped_genes:
            self.logger.warning(f"{len(unmapped_genes)} genes could not be mapped to symbols")
        else:
            self.logger.info("All genes successfully mapped to symbols")
        
        return expression_df
    
    def _prepare_metadata_for_expression(self, expanded_data: pd.DataFrame, 
                                       expression_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare metadata that matches expression data samples."""
        self.logger.info("Preparing metadata for expression data...")
        
        # Create toxicity mapping
        toxicity_map = dict(zip(expanded_data['distil_id'], expanded_data['toxicity_label']))
        
        # Create metadata dataframe matching expression columns
        metadata_df = pd.DataFrame(index=expression_df.columns)
        metadata_df['toxicity_label'] = metadata_df.index.map(toxicity_map)
        
        # Add other relevant metadata
        for col in ['cell_id', 'pert_iname_x', 'pert_dose', 'pert_time']:
            if col in expanded_data.columns:
                col_map = dict(zip(expanded_data['distil_id'], expanded_data[col]))
                metadata_df[col] = metadata_df.index.map(col_map)
        
        # Remove samples without toxicity labels
        valid_samples = metadata_df['toxicity_label'].notna()
        expression_df_filtered = expression_df.loc[:, valid_samples]
        metadata_df_filtered = metadata_df.loc[valid_samples]
        
        self.logger.info(f"Final dataset after metadata matching:")
        self.logger.info(f"  Expression data: {expression_df_filtered.shape}")
        self.logger.info(f"  Metadata: {metadata_df_filtered.shape}")
        self.logger.info(f"  Toxicity distribution: {metadata_df_filtered['toxicity_label'].value_counts()}")
        
        return expression_df_filtered, metadata_df_filtered
    
    def _perform_basic_qc(self, expression_df: pd.DataFrame, metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic quality control checks."""
        self.logger.info("Performing Quality Control checks...")
        
        # Basic statistics
        qc_results = {
            'expression_shape': expression_df.shape,
            'value_range': [float(expression_df.values.min()), float(expression_df.values.max())],
            'missing_values': int(expression_df.isnull().sum().sum()),
            'toxicity_distribution': metadata_df['toxicity_label'].value_counts().to_dict(),
            'cell_distribution': metadata_df['cell_id'].value_counts().to_dict() if 'cell_id' in metadata_df.columns else {}
        }
        
        self.logger.info(f"Expression data shape: {qc_results['expression_shape']}")
        self.logger.info(f"Value range: [{qc_results['value_range'][0]:.3f}, {qc_results['value_range'][1]:.3f}]")
        self.logger.info(f"Missing values: {qc_results['missing_values']}")
        
        self.logger.info("Sample distribution:")
        for label, count in qc_results['toxicity_distribution'].items():
            self.logger.info(f"  {label}: {count}")
        
        if qc_results['cell_distribution']:
            self.logger.info("Cell line distribution:")
            for cell, count in qc_results['cell_distribution'].items():
                self.logger.info(f"  {cell}: {count}")
        
        # Create QC plots
        self._create_qc_plots(expression_df, metadata_df)
        
        return qc_results
    
    def _create_qc_plots(self, expression_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
        """Create quality control plots."""
        self.logger.info("Generating QC plots...")
        
        # Plot 1: Sample distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Toxicity distribution
        toxicity_counts = metadata_df['toxicity_label'].value_counts()
        toxicity_counts.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Sample Distribution by Toxicity')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Cell line distribution (if available)
        if 'cell_id' in metadata_df.columns:
            cell_counts = metadata_df['cell_id'].value_counts()
            cell_counts.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Sample Distribution by Cell Line')
            axes[1].set_ylabel('Count')
            axes[1].tick_params(axis='x', rotation=45)
        else:
            axes[1].text(0.5, 0.5, 'Cell ID data not available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Cell Line Distribution')
        
        plt.tight_layout()
        self.save_figure(fig, 'qc_sample_distribution')
        self.close_figure(fig)
        
        # Plot 2: Expression value distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(expression_df.values.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Expression Values')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Gene Expression Values')
        
        plt.tight_layout()
        self.save_figure(fig, 'qc_expression_distribution')
        self.close_figure(fig)
        
        # Plot 3: Sample correlation heatmap (subset for visualization)
        if expression_df.shape[1] > 50:
            subset_cols = np.random.choice(expression_df.columns, 50, replace=False)
            corr_matrix = expression_df[subset_cols].corr()
        else:
            corr_matrix = expression_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, 
                    linewidths=0.1, cbar_kws={"shrink": .5}, ax=ax)
        ax.set_title('Sample Correlation Heatmap')
        
        plt.tight_layout()
        self.save_figure(fig, 'qc_sample_correlation')
        self.close_figure(fig)
        
        self.logger.info(f"QC plots saved to {self.output_dir}")
    
    def _save_processed_data(self, expression_df: pd.DataFrame, metadata_df: pd.DataFrame, 
                           expanded_data: pd.DataFrame) -> Dict[str, Path]:
        """Save all processed datasets."""
        self.logger.info(f"Saving processed data to {self.output_dir}...")
        
        saved_files = {}
        
        # Save main datasets
        saved_files['gene_expression'] = self.save_data(expression_df, 'gene_expression_data.csv')
        saved_files['sample_metadata'] = self.save_data(metadata_df, 'sample_metadata.csv')
        saved_files['expanded_metadata'] = self.save_data(expanded_data, 'expanded_toxicity_metadata.csv', index=False)
        
        # Save summary statistics
        summary_stats = {
            'total_genes': len(expression_df),
            'total_samples': len(expression_df.columns),
            'toxic_samples': sum(metadata_df['toxicity_label'] == 'Toxic'),
            'nontoxic_samples': sum(metadata_df['toxicity_label'] == 'Non-toxic'),
            'kidney_cell_lines': self.kidney_cell_ids,
            'expression_range': [float(expression_df.values.min()), float(expression_df.values.max())]
        }
        
        import json
        summary_path = self.output_dir / 'data_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        saved_files['summary'] = summary_path
        
        self.logger.info("Files saved:")
        self.logger.info(f"  - gene_expression_data.csv: {expression_df.shape}")
        self.logger.info(f"  - sample_metadata.csv: {metadata_df.shape}")
        self.logger.info(f"  - expanded_toxicity_metadata.csv: {expanded_data.shape}")
        self.logger.info(f"  - data_summary.json: Summary statistics")
        
        return saved_files

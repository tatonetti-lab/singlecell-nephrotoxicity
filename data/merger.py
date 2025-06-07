"""
Drug data merging component for combining multiple drug databases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

from core.base import BaseDataProcessor
from utils.helpers import clean_drug_names, combine_gene_information
from utils.logging import ComponentLogger


class DrugDataMerger(BaseDataProcessor):
    """
    Component for merging drug data from multiple sources:
    - ChEMBL (drug targets)
    - DrugBank (drug targets) 
    - Ryan Reference (nephrotoxicity)
    - ACS FDA (nephrotoxicity)
    - DIRIL (nephrotoxicity)
    """
    
    def __init__(self, config: Dict[str, Any], component_name: str = 'merge_data'):
        super().__init__(config, component_name)
        
        # Get merge-specific configuration
        self.merge_config = config.get('merge_drug_data_sources', {})
        self.data_paths = self.merge_config.get('data_paths', {})
        self.output_file = self.merge_config.get('output_file', 'V2_merged_drug_dataset.csv')
        self.toxicity_columns = self.merge_config.get('toxicity_columns', [])
        self.target_condition = self.merge_config.get('target_condition', 'acute kidney injury')
        
        self.component_logger = ComponentLogger(component_name, verbose=self.component_config.get('verbose', True))
    
    def validate_inputs(self) -> bool:
        """Validate that all required input files exist."""
        required_files = ['chembl', 'drugbank', 'ryan_ref', 'acs_fda', 'diril']
        missing_files = []
        
        for file_key in required_files:
            if file_key not in self.data_paths:
                missing_files.append(f"Missing data path: {file_key}")
            elif not Path(self.data_paths[file_key]).exists():
                missing_files.append(f"File not found: {self.data_paths[file_key]}")
        
        if missing_files:
            for error in missing_files:
                self.logger.error(error)
            return False
        
        return True
    
    def run(self) -> Dict[str, Any]:
        """Run the drug data merging pipeline."""
        self.component_logger.start_component(total_steps=8)
        
        try:
            # Step 1: Load all datasets
            self.component_logger.step_completed("Loading datasets")
            datasets = self._load_all_datasets()
            
            # Step 2: Preprocess individual datasets
            self.component_logger.step_completed("Preprocessing datasets")
            processed_datasets = self._preprocess_datasets(datasets)
            
            # Step 3: Merge target databases
            self.component_logger.step_completed("Merging target databases")
            merged_targets = self._merge_target_databases(
                processed_datasets['chembl'], 
                processed_datasets['drugbank']
            )
            
            # Step 4: Merge toxicity databases
            self.component_logger.step_completed("Merging toxicity databases")
            merged_toxicity = self._merge_toxicity_databases(
                processed_datasets['ryan_ref'],
                processed_datasets['acs_fda'],
                processed_datasets['diril']
            )
            
            # Step 5: Create final dataset
            self.component_logger.step_completed("Creating final dataset")
            final_dataset = self._create_final_dataset(merged_targets, merged_toxicity)
            
            # Step 6: Generate summary statistics
            self.component_logger.step_completed("Generating summary statistics")
            summary_stats = self._generate_summary_statistics(final_dataset)
            
            # Step 7: Save results
            self.component_logger.step_completed("Saving results")
            output_path = self._save_results(final_dataset, summary_stats)
            
            # Step 8: Generate report
            self.component_logger.step_completed("Generating report")
            self._generate_report(summary_stats)
            
            self.component_logger.finish_component(success=True)
            
            return {
                'final_dataset': final_dataset,
                'summary_stats': summary_stats,
                'output_path': output_path,
                'n_drugs_total': len(final_dataset),
                'n_drugs_toxic': summary_stats['toxic_drugs'],
                'n_drugs_nontoxic': summary_stats['non_toxic_drugs']
            }
            
        except Exception as e:
            self.component_logger.finish_component(success=False)
            self.logger.error(f"Drug data merging failed: {e}")
            raise
    
    def _load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all drug databases."""
        datasets = {}
        
        for name, path in self.data_paths.items():
            self.logger.info(f"Loading {name} dataset from {path}")
            try:
                df = self._load_csv_robust(path)
                datasets[name] = df
                self.logger.info(f"Loaded {name}: {len(df)} records")
            except Exception as e:
                self.logger.error(f"Failed to load {name}: {e}")
                raise
        
        return datasets
    
    def _load_csv_robust(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with robust error handling for different formats."""
        file_path = Path(file_path)
        
        # Try different loading strategies
        strategies = [
            {'engine': 'c'},
            {'engine': 'python'},
            {'sep': '\t', 'on_bad_lines': 'skip'},
            {'sep': ',', 'on_bad_lines': 'skip', 'low_memory': False}
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                df = pd.read_csv(file_path, **strategy)
                if i > 0:
                    self.logger.info(f"Loaded using strategy {i+1}")
                return df
            except Exception as e:
                if i == len(strategies) - 1:
                    raise e
                continue
    
    def _preprocess_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Preprocess individual datasets."""
        processed = {}
        
        # Preprocess ChEMBL
        processed['chembl'] = self._preprocess_chembl(datasets['chembl'])
        
        # Preprocess DrugBank
        processed['drugbank'] = self._preprocess_drugbank(datasets['drugbank'])
        
        # Preprocess Ryan Reference
        processed['ryan_ref'] = self._preprocess_ryan_ref(datasets['ryan_ref'])
        
        # Preprocess ACS FDA
        processed['acs_fda'] = self._preprocess_acs_fda(datasets['acs_fda'])
        
        # Preprocess DIRIL
        processed['diril'] = self._preprocess_diril(datasets['diril'])
        
        return processed
    
    def _preprocess_chembl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess ChEMBL dataset."""
        df = df.copy()
        
        # Split combined column if it exists
        if 'chembl|drugname' in df.columns:
            df[['chembl', 'drugname']] = df['chembl|drugname'].str.split("|", expand=True)
        
        # Clean drug names
        if 'drugname' in df.columns:
            df = clean_drug_names(df, 'drugname')
            df = df.rename(columns={'drugname': 'drug_name'})
        
        # Rename columns for consistency
        rename_map = {
            'genes': 'genes_chembl',
            'chembl': 'chembl_id'
        }
        df = df.rename(columns=rename_map)
        
        # Select relevant columns
        relevant_cols = ['drug_name', 'genes_chembl', 'chembl_id']
        available_cols = [col for col in relevant_cols if col in df.columns]
        
        return df[available_cols]
    
    def _preprocess_drugbank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DrugBank dataset."""
        df = df.copy()
        
        # Clean drug names
        if 'Drug Name' in df.columns:
            df = clean_drug_names(df, 'Drug Name')
            df = df.rename(columns={'Drug Name': 'drug_name'})
        
        # Rename columns for consistency
        rename_map = {
            'Unique Target IDs': 'target_ids_drugbank',
            'Unique Target Names': 'target_names_drugbank',
            'Unique Gene Names': 'genes_drugbank'
        }
        df = df.rename(columns=rename_map)
        
        # Select relevant columns
        relevant_cols = ['drug_name', 'target_ids_drugbank', 'target_names_drugbank', 'genes_drugbank']
        available_cols = [col for col in relevant_cols if col in df.columns]
        
        return df[available_cols]
    
    def _preprocess_ryan_ref(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Ryan reference dataset."""
        df = df.copy()
        
        # Clean drug names
        if 'drug_name' in df.columns:
            df = clean_drug_names(df, 'drug_name')
        
        # Filter for specified condition
        if 'condition_name' in df.columns:
            df = df[df['condition_name'].str.lower() == self.target_condition.lower()]
        
        # Create toxicity indicator
        if 'affect' in df.columns:
            df['is_toxic_ryan'] = df['affect'] == 1
        
        return df[['drug_name', 'is_toxic_ryan']]
    
    def _preprocess_acs_fda(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess ACS FDA dataset."""
        df = df.copy()
        
        # Clean drug names
        if 'salt_prefname' in df.columns:
            df = clean_drug_names(df, 'salt_prefname')
            df = df.rename(columns={'salt_prefname': 'drug_name'})
        
        # Create toxicity indicator
        if 'manual_nephrotoxicity_category' in df.columns:
            df['is_toxic_acs'] = df['manual_nephrotoxicity_category'] == 'nephrotoxicity'
        
        return df[['drug_name', 'is_toxic_acs']]
    
    def _preprocess_diril(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DIRIL dataset."""
        df = df.copy()
        
        # Clean drug names
        if 'name' in df.columns:
            df = clean_drug_names(df, 'name')
            df = df.rename(columns={'name': 'drug_name'})
        
        # Process toxicity information
        df = self._process_diril_toxicity(df)
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset='drug_name')
        if len(df) < initial_count:
            self.logger.info(f"Removed {initial_count - len(df)} duplicate drug names from DIRIL")
        
        # Select relevant columns
        columns = ['drug_name', 'is_toxic_diril']
        if 'drugbank_id' in df.columns:
            columns.append('drugbank_id')
        
        return df[columns]
    
    def _process_diril_toxicity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DIRIL toxicity columns to create binary nephrotoxicity indicator."""
        df = df.copy()
        
        # Find existing toxicity columns
        existing_columns = [col for col in self.toxicity_columns if col in df.columns]
        self.logger.info(f"Found toxicity columns in DIRIL: {existing_columns}")
        
        if not existing_columns:
            self.logger.warning("No expected nephrotoxicity columns found in DIRIL dataset!")
            df['is_toxic_diril'] = False
            return df
        
        # Create binary indicators for each toxicity column
        binary_columns = []
        for col in existing_columns:
            binary_col = f"{col}_binary"
            try:
                df[binary_col] = df[col].fillna('').astype(str).str.lower().apply(
                    lambda x: 'nephrotoxic' in x and 'non-nephrotoxic' not in x
                )
                positive_count = df[binary_col].sum()
                self.logger.info(f"Nephrotoxic count in {col}: {positive_count}")
                binary_columns.append(binary_col)
            except Exception as e:
                self.logger.error(f"Error processing column {col}: {e}")
        
        # Combine using logical OR
        if binary_columns:
            df['is_toxic_diril'] = df[binary_columns].any(axis=1)
        else:
            df['is_toxic_diril'] = False
            self.logger.warning("No binary columns created, setting all DIRIL drugs to non-nephrotoxic")
        
        return df
    
    def _merge_target_databases(self, chembl_df: pd.DataFrame, drugbank_df: pd.DataFrame) -> pd.DataFrame:
        """Merge ChEMBL and DrugBank target information."""
        self.logger.info(f"Merging targets - ChEMBL: {len(chembl_df)}, DrugBank: {len(drugbank_df)}")
        
        merged = pd.merge(drugbank_df, chembl_df, on='drug_name', how='outer')
        
        # Combine gene information
        gene_columns = []
        if 'genes_drugbank' in merged.columns:
            gene_columns.append(merged['genes_drugbank'])
        if 'genes_chembl' in merged.columns:
            gene_columns.append(merged['genes_chembl'])
        
        if gene_columns:
            merged['combined_genes'] = combine_gene_information(*gene_columns)
        
        self.logger.info(f"Merged target database: {len(merged)} drugs")
        return merged
    
    def _merge_toxicity_databases(self, ryan_df: pd.DataFrame, acs_df: pd.DataFrame, 
                                diril_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all toxicity databases."""
        self.logger.info(f"Merging toxicity databases - Ryan: {len(ryan_df)}, ACS: {len(acs_df)}, DIRIL: {len(diril_df)}")
        
        # Start with DIRIL as base (typically largest)
        combined = diril_df.copy()
        
        # Merge with other datasets
        combined = pd.merge(combined, ryan_df, on='drug_name', how='outer')
        combined = pd.merge(combined, acs_df, on='drug_name', how='outer')
        
        # Track data sources
        combined['sources'] = combined.apply(self._create_source_string, axis=1)
        
        # Fill NaN values with False for toxicity columns
        toxicity_cols = ['is_toxic_diril', 'is_toxic_ryan', 'is_toxic_acs']
        for col in toxicity_cols:
            if col in combined.columns:
                combined[col] = combined[col].fillna(False)
        
        # Create consensus toxicity column
        available_toxicity_cols = [col for col in toxicity_cols if col in combined.columns]
        combined['is_toxic'] = combined[available_toxicity_cols].any(axis=1)
        
        self.logger.info(f"Combined toxicity database: {len(combined)} drugs, {combined['is_toxic'].sum()} toxic")
        
        return combined
    
    def _create_source_string(self, row: pd.Series) -> str:
        """Create a string indicating which datasets contributed data for this drug."""
        sources = []
        
        if pd.notna(row.get('is_toxic_diril')):
            sources.append('DIRIL')
        if pd.notna(row.get('is_toxic_ryan')):
            sources.append('Ryan')
        if pd.notna(row.get('is_toxic_acs')):
            sources.append('ACS')
            
        return ','.join(sources) if sources else 'NotFound'
    
    def _create_final_dataset(self, merged_targets: pd.DataFrame, 
                            merged_toxicity: pd.DataFrame) -> pd.DataFrame:
        """Create final merged dataset."""
        self.logger.info("Creating final dataset...")
        
        # Merge target and toxicity data
        final_dataset = pd.merge(merged_targets, merged_toxicity, on='drug_name', how='outer')
        
        # Remove duplicates
        initial_count = len(final_dataset)
        final_dataset = final_dataset.drop_duplicates(subset='drug_name', keep='first')
        if len(final_dataset) < initial_count:
            self.logger.info(f"Removed {initial_count - len(final_dataset)} duplicate drugs")
        
        # Handle missing toxicity data
        toxicity_drugs = set(merged_toxicity['drug_name'].dropna())
        not_in_toxicity = ~final_dataset['drug_name'].isin(toxicity_drugs)
        final_dataset.loc[not_in_toxicity, 'is_toxic'] = False
        final_dataset.loc[not_in_toxicity, 'sources'] = 'NotFound'
        
        self.logger.info(f"Final dataset: {len(final_dataset)} drugs")
        
        return final_dataset
    
    def _generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the final dataset."""
        stats = {
            'total_drugs': len(df),
            'toxic_drugs': int(df['is_toxic'].sum()),
            'non_toxic_drugs': int((~df['is_toxic']).sum()),
            'not_found_count': int(df['sources'].str.contains('NotFound', na=False).sum())
        }
        
        # Calculate toxicity percentage
        total_with_data = stats['toxic_drugs'] + stats['non_toxic_drugs']
        if total_with_data > 0:
            stats['toxic_percentage'] = (stats['toxic_drugs'] / total_with_data) * 100
        else:
            stats['toxic_percentage'] = 0
        
        # Source-specific statistics
        for source in ['diril', 'ryan', 'acs']:
            col = f'is_toxic_{source}'
            if col in df.columns:
                toxic_count = int(df[col].sum())
                total_count = int(df[col].notna().sum())
                stats[f'{source}_toxic'] = toxic_count
                stats[f'{source}_total'] = total_count
                stats[f'{source}_percentage'] = (toxic_count / total_count * 100) if total_count > 0 else 0
        
        return stats
    
    def _save_results(self, final_dataset: pd.DataFrame, summary_stats: Dict[str, Any]) -> Path:
        """Save final dataset and summary statistics."""
        # Save final dataset
        output_path = self.output_dir / self.output_file
        final_dataset.to_csv(output_path, index=False)
        self.logger.info(f"Final dataset saved to: {output_path}")
        
        # Save summary statistics
        summary_path = self.output_dir / "merge_summary_statistics.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        return output_path
    
    def _generate_report(self, summary_stats: Dict[str, Any]) -> None:
        """Generate a summary report."""
        if not self.config_manager.should_generate_report():
            return
        
        report_path = self.output_dir / "drug_data_merge_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("DRUG DATABASE MERGING REPORT\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"  Total drugs: {summary_stats['total_drugs']}\n")
            f.write(f"  Toxic drugs: {summary_stats['toxic_drugs']}\n")
            f.write(f"  Non-toxic drugs: {summary_stats['non_toxic_drugs']}\n")
            f.write(f"  Not found in toxicity datasets: {summary_stats['not_found_count']}\n")
            f.write(f"  Toxicity percentage: {summary_stats['toxic_percentage']:.2f}%\n\n")
            
            f.write("SOURCE-SPECIFIC BREAKDOWN:\n")
            for source in ['diril', 'ryan', 'acs']:
                if f'{source}_total' in summary_stats:
                    f.write(f"  {source.upper()}: {summary_stats[f'{source}_toxic']}/{summary_stats[f'{source}_total']} ")
                    f.write(f"({summary_stats[f'{source}_percentage']:.2f}% toxic)\n")
            
            f.write(f"\nConfiguration used:\n")
            f.write(f"  Target condition: {self.target_condition}\n")
            f.write(f"  Toxicity columns: {self.toxicity_columns}\n")
            f.write(f"  Output file: {self.output_file}\n")
        
        self.logger.info(f"Report saved to: {report_path}")

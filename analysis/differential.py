import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import re
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import warnings

from core.base import BaseVisualizationAnalyzer
from utils.logging import ComponentLogger

warnings.filterwarnings('ignore')


class DifferentialExpressionAnalyzer(BaseVisualizationAnalyzer):
    """Differential expression analysis component using the working logic from diff.py. Always runs both direct and drug-averaged approaches."""

    def __init__(self, config: Dict[str, Any], component_name: str = 'differential_expression'):
        super().__init__(config, component_name)

        # Initialize data paths
        self.input_dir = Path('.')

        # Initialize data storage (following diff.py pattern)
        self.gene_expression_df = None
        self.metadata_df = None
        self.toxicity_labels = None
        self.pattern_metadata = None
        self.results = None

        # Analysis parameters
        self.significance_thresholds = self.component_config.get('significance_thresholds', {
            'fdr': 0.05,
            'log2fc': 1.0
        })

        # Always run both analysis approaches

        self.component_logger = ComponentLogger(component_name, verbose=self.component_config.get('verbose', True))
        
        # Override figure formats to exclude SVG
        self.figure_formats = ['png']

    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        required_files = [
            'input_data/recheck-kidney_lincs_matched_df.csv',
            'input_data/kidney_filtered_expanded_toxicity_df.csv'
        ]

        missing_files = []
        for filename in required_files:
            file_path = Path(filename)
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            self.logger.error("Missing required input files:")
            for file_path in missing_files:
                self.logger.error(f"  - {file_path}")
            return False

        return True

    def load_full_dataset(self, expression_file="input_data/recheck-kidney_lincs_matched_df.csv",
                         metadata_file="input_data/kidney_filtered_expanded_toxicity_df.csv"):
        """Load the full dataset for analysis approach 1 (using diff.py logic)"""
        self.logger.info("Loading full dataset...")

        # Load gene expression data
        self.gene_expression_df = pd.read_csv(expression_file, index_col=0)
        self.logger.info(f"Gene expression data shape: {self.gene_expression_df.shape}")

        # Load metadata
        self.metadata_df = pd.read_csv(metadata_file)
        self.logger.info(f"Metadata shape: {self.metadata_df.shape}")

        # Create toxicity mapping (exact logic from diff.py)
        toxicity_map = dict(zip(self.metadata_df['distil_id'], self.metadata_df['toxicity_label']))

        # Map toxicity to expression data columns
        self.toxicity_labels = pd.Series([toxicity_map.get(col, 'Unknown') for col in self.gene_expression_df.columns],
                                       index=self.gene_expression_df.columns)

        self.logger.info(f"Toxicity distribution:\n{self.toxicity_labels.value_counts()}")
        return True

    def create_averaged_dataset(self, expression_file="input_data/recheck-kidney_lincs_matched_df.csv",
                              metadata_file="input_data/kidney_filtered_expanded_toxicity_df.csv"):
        """Create averaged dataset for analysis approach 2 (using diff.py logic)"""
        self.logger.info("Creating averaged dataset...")

        # Load data
        expression_df = pd.read_csv(expression_file, index_col=0)
        metadata_df = pd.read_csv(metadata_file)

        self.logger.info(f"Original metadata shape: {metadata_df.shape}")
        self.logger.info(f"Toxicity distribution in metadata:\n{metadata_df['toxicity_label'].value_counts()}")

        def extract_base_pattern(text):
            """Extract base pattern from sig_id (exact logic from diff.py)"""
            if pd.isna(text):
                return None
            pattern = re.match(r'(.*?)(?=:B\d+)', str(text))
            return pattern.group(1) if pattern else text

        # Add base pattern to metadata
        metadata_df['base_pattern'] = metadata_df['sig_id'].apply(extract_base_pattern)

        # Create a mapping from distil_id to toxicity (more reliable)
        distil_toxicity_map = dict(zip(metadata_df['distil_id'], metadata_df['toxicity_label']))

        # Calculate averages for each pattern and preserve toxicity info
        averaged_expressions = {}
        pattern_metadata = {}

        # Group metadata by base pattern and get all associated distil_ids
        for base_pattern in metadata_df['base_pattern'].dropna().unique():
            # Create regex pattern to match columns
            regex_pattern = f"{re.escape(base_pattern)}(?:_X1)?(?:_B\\d+)?:B\\d+"

            # Find matching columns in expression data
            matching_columns = [col for col in expression_df.columns
                              if re.match(regex_pattern, col)]

            if matching_columns:
                # Calculate mean expression
                averaged_expressions[base_pattern] = expression_df[matching_columns].mean(axis=1)

                # Get all metadata entries for this pattern
                pattern_rows = metadata_df[metadata_df['base_pattern'] == base_pattern]

                # Find toxicity labels for matching distil_ids
                matching_distil_ids = [col for col in matching_columns if col in distil_toxicity_map]

                if matching_distil_ids:
                    # Use the toxicity from matching distil_ids
                    toxicities = [distil_toxicity_map[did] for did in matching_distil_ids]
                    # Use the most common toxicity label (in case of mixed)
                    toxicity_counts = pd.Series(toxicities).value_counts()
                    most_common_toxicity = toxicity_counts.index[0]
                else:
                    # Fallback to first entry's toxicity
                    most_common_toxicity = pattern_rows.iloc[0]['toxicity_label']

                pattern_metadata[base_pattern] = {
                    'toxicity_label': most_common_toxicity,
                    'drug_name': pattern_rows.iloc[0]['drug_name'],
                    'cell_id': pattern_rows.iloc[0]['cell_id'],
                    'num_replicates': len(matching_columns),
                    'matched_distil_ids': matching_distil_ids[:5]  # Store first 5 for reference
                }

        # Alternative approach: Use drug-based grouping for more reliable toxicity mapping
        self.logger.info("Trying drug-based grouping...")
        drug_based_expressions = {}
        drug_metadata = {}

        # Group by drug name and calculate averages
        for drug_name in metadata_df['drug_name'].dropna().unique():
            drug_rows = metadata_df[metadata_df['drug_name'] == drug_name]

            # Get all distil_ids for this drug
            drug_distil_ids = drug_rows['distil_id'].unique()

            # Find matching columns in expression data
            matching_columns = [col for col in expression_df.columns if col in drug_distil_ids]

            if len(matching_columns) >= 3:  # Only include drugs with at least 3 samples
                # Calculate mean expression
                drug_based_expressions[drug_name] = expression_df[matching_columns].mean(axis=1)

                # Get toxicity (should be consistent for same drug)
                toxicity_counts = drug_rows['toxicity_label'].value_counts()
                most_common_toxicity = toxicity_counts.index[0]

                drug_metadata[drug_name] = {
                    'toxicity_label': most_common_toxicity,
                    'cell_id': drug_rows.iloc[0]['cell_id'],
                    'num_replicates': len(matching_columns)
                }

        self.logger.info(f"Drug-based grouping: {len(drug_based_expressions)} drugs with ≥3 samples")

        # Use drug-based grouping if it gives better toxicity distribution
        if len(drug_based_expressions) > 0:
            drug_toxicity_dist = pd.Series([drug_metadata[drug]['toxicity_label']
                                          for drug in drug_based_expressions.keys()]).value_counts()
            self.logger.info(f"Drug-based toxicity distribution:\n{drug_toxicity_dist}")

            # Use drug-based if we have both toxic and non-toxic
            if len(drug_toxicity_dist) > 1:
                self.gene_expression_df = pd.DataFrame(drug_based_expressions)
                self.toxicity_labels = pd.Series([drug_metadata[col]['toxicity_label']
                                                for col in self.gene_expression_df.columns],
                                               index=self.gene_expression_df.columns)
                self.pattern_metadata = pd.DataFrame(drug_metadata).T
                self.logger.info("Using drug-based grouping")
            else:
                # Fallback to pattern-based
                self.gene_expression_df = pd.DataFrame(averaged_expressions)
                self.toxicity_labels = pd.Series([pattern_metadata[col]['toxicity_label']
                                                for col in self.gene_expression_df.columns],
                                               index=self.gene_expression_df.columns)
                self.pattern_metadata = pd.DataFrame(pattern_metadata).T
                self.logger.info("Using pattern-based grouping")
        else:
            # Use pattern-based as fallback
            self.gene_expression_df = pd.DataFrame(averaged_expressions)
            self.toxicity_labels = pd.Series([pattern_metadata[col]['toxicity_label']
                                            for col in self.gene_expression_df.columns],
                                           index=self.gene_expression_df.columns)
            self.pattern_metadata = pd.DataFrame(pattern_metadata).T
            self.logger.info("Using pattern-based grouping")

        self.logger.info(f"Final averaged gene expression data shape: {self.gene_expression_df.shape}")
        self.logger.info(f"Final toxicity distribution:\n{self.toxicity_labels.value_counts()}")

        # Debug: Show some examples of the mapping
        self.logger.info(f"First 5 samples and their toxicity labels:")
        for i, (sample, toxicity) in enumerate(self.toxicity_labels.head().items()):
            self.logger.info(f"  {sample}: {toxicity}")
            if i >= 4:
                break

        return True

    def perform_differential_expression(self, fdr_threshold=None, log2fc_threshold=None):
        """Perform differential expression analysis using exact diff.py logic"""
        self.logger.info("Performing differential expression analysis...")

        # Use configured thresholds or defaults
        if fdr_threshold is None:
            fdr_threshold = self.significance_thresholds['fdr']
        if log2fc_threshold is None:
            log2fc_threshold = self.significance_thresholds['log2fc']

        if self.gene_expression_df is None:
            raise ValueError("No data loaded. Call load_full_dataset() or create_averaged_dataset() first.")

        # Get sample indices for each group (exact logic from diff.py)
        toxic_samples = self.toxicity_labels[self.toxicity_labels == 'Toxic'].index
        nontoxic_samples = self.toxicity_labels[self.toxicity_labels == 'Non-toxic'].index

        self.logger.info(f"Number of toxic samples: {len(toxic_samples)}")
        self.logger.info(f"Number of non-toxic samples: {len(nontoxic_samples)}")

        if len(toxic_samples) == 0 or len(nontoxic_samples) == 0:
            raise ValueError("Not enough samples in one or both groups for comparison")

        # Initialize results (exact logic from diff.py)
        results = []

        for gene in self.gene_expression_df.index:
            toxic_expr = self.gene_expression_df.loc[gene, toxic_samples]
            nontoxic_expr = self.gene_expression_df.loc[gene, nontoxic_samples]

            # Calculate statistics
            t_stat, p_val = stats.ttest_ind(toxic_expr, nontoxic_expr)
            mean_toxic = np.mean(toxic_expr)
            mean_nontoxic = np.mean(nontoxic_expr)

            # Calculate log2 fold change (exact logic from diff.py)
            # Handle negative values by adding offset
            min_val = min(mean_toxic, mean_nontoxic)
            if min_val <= 0:
                offset = abs(min_val) + 1
                log2_fc = np.log2(mean_toxic + offset) - np.log2(mean_nontoxic + offset)
            else:
                log2_fc = np.log2(mean_toxic + 0.1) - np.log2(mean_nontoxic + 0.1)

            results.append({
                'gene': gene,
                'pvalue': p_val,
                'log2FoldChange': log2_fc,
                'mean_toxic': mean_toxic,
                'mean_nontoxic': mean_nontoxic,
                't_statistic': t_stat
            })

        # Create results dataframe (exact logic from diff.py)
        self.results = pd.DataFrame(results).set_index('gene')

        # Apply FDR correction
        rejected, fdr_pvals = fdrcorrection(self.results['pvalue'], alpha=fdr_threshold)
        self.results['FDR'] = fdr_pvals

        # Add significance flags
        self.results['significant'] = ((self.results['FDR'] < fdr_threshold) &
                                     (abs(self.results['log2FoldChange']) > log2fc_threshold))

        # Sort by FDR
        self.results = self.results.sort_values('FDR')

        self.logger.info(f"Differential Expression Results:")
        self.logger.info(f"Total genes analyzed: {len(self.results)}")
        self.logger.info(f"Significantly DE genes (FDR < {fdr_threshold}, |log2FC| > {log2fc_threshold}): {sum(self.results['significant'])}")
        self.logger.info(f"Upregulated genes: {sum((self.results['significant']) & (self.results['log2FoldChange'] > 0))}")
        self.logger.info(f"Downregulated genes: {sum((self.results['significant']) & (self.results['log2FoldChange'] < 0))}")

        return self.results

    def run(self) -> Dict[str, Any]:
        """Run differential expression analysis pipeline using diff.py logic - always runs both approaches."""
        total_steps = 10
        self.component_logger.start_component(total_steps=total_steps)

        try:
            results = {}

            # Method 1: Direct analysis on full dataset (diff.py approach 1)
            self.component_logger.step_completed("Loading full dataset for direct analysis")
            self.load_full_dataset()

            self.component_logger.step_completed("Performing direct differential expression analysis")
            direct_results = self.perform_differential_expression()

            results['direct_analysis'] = {
                'results': direct_results.copy(),
                'expression_data': self.gene_expression_df.copy(),
                'toxicity_labels': self.toxicity_labels.copy(),
                'analysis_type': 'direct'
            }

            # Method 2: Averaged analysis (diff.py approach 2)
            self.component_logger.step_completed("Creating averaged dataset for drug-based analysis")
            self.create_averaged_dataset()

            self.component_logger.step_completed("Performing drug-averaged differential expression analysis")
            drug_results = self.perform_differential_expression()

            results['drug_averaged_analysis'] = {
                'results': drug_results.copy(),
                'expression_data': self.gene_expression_df.copy(),
                'toxicity_labels': self.toxicity_labels.copy(),
                'pattern_metadata': getattr(self, 'pattern_metadata', None),
                'analysis_type': 'drug_averaged'
            }

            # Save all results
            self.component_logger.step_completed("Saving results")
            saved_files = self._save_all_results(results)

            # Generate final comprehensive report
            self.component_logger.step_completed("Generating comprehensive report")
            self._generate_comprehensive_report(results)

            # Create ML ready datasets
            self.component_logger.step_completed("Creating ML-ready datasets")
            self._create_ml_ready_datasets(results)

            self.component_logger.finish_component(success=True)

            # Add summary statistics to results
            for analysis_type, analysis_data in results.items():
                if analysis_type != 'comparison' and 'results' in analysis_data:
                    res_df = analysis_data['results']
                    results[analysis_type].update({
                        'n_significant_genes': int(res_df['significant'].sum()),
                        'n_upregulated': int((res_df['significant'] & (res_df['log2FoldChange'] > 0)).sum()),
                        'n_downregulated': int((res_df['significant'] & (res_df['log2FoldChange'] < 0)).sum())
                    })

            results['saved_files'] = saved_files
            return results

        except Exception as e:
            self.component_logger.finish_component(success=False)
            self.logger.error(f"Differential expression analysis failed: {e}")
            raise

    def _create_ml_ready_datasets(self, results: dict[str, pd.DataFrame]) -> dict[str, Path]:
        """Create ML-ready datasets for machine learning classification."""
        self.logger.info("Creating ML-ready datasets...")

        ml_files = {}

        # Process both analysis types
        for analysis_type in ['direct_analysis', 'drug_averaged_analysis']:
            if analysis_type not in results:
                continue

            self.logger.info(f"Creating ML datasets for {analysis_type}...")

            # Get data for this analysis type
            analysis_results = results[analysis_type]
            expression_data = analysis_results['expression_data']
            toxicity_labels = analysis_results['toxicity_labels']
            de_results = analysis_results['results']

            # Ensure expression data is in correct format (samples x genes)
            if expression_data.shape[0] > expression_data.shape[1]:
                # Likely already samples x genes
                ml_expression = expression_data.copy()
            else:
                # Transpose to get samples x genes
                ml_expression = expression_data.T

            # toxicity_labels is a pandas Series, not DataFrame
            # Align samples between expression and toxicity data
            common_samples = ml_expression.index.intersection(toxicity_labels.index)
            if len(common_samples) == 0:
                continue

            # Filter to common samples
            ml_expression_filtered = ml_expression.loc[common_samples]
            toxicity_filtered = toxicity_labels.loc[common_samples]

            self.logger.info(f"{analysis_type}: {len(common_samples)} samples, {ml_expression_filtered.shape[1]} genes")
            self.logger.info(f"Toxicity distribution: {toxicity_filtered.value_counts().to_dict()}")

            # 1. All genes dataset
            ml_data_all = ml_expression_filtered.copy()
            ml_data_all['toxicity_label'] = toxicity_filtered

            # Save with analysis-specific names
            if analysis_type == 'drug_averaged_analysis':
                # Main files for lincs_classification.py
                ml_files['all_genes'] = self.save_data(ml_data_all, 'ml_ready_all_genes.csv')

            ml_files[f'{analysis_type}_all_genes'] = self.save_data(
                ml_data_all, f'{analysis_type}_ml_ready_all_genes.csv'
            )

            # 2. Significant DE genes only
            if 'significant' in de_results.columns and de_results['significant'].sum() > 0:
                significant_genes = de_results[de_results['significant']].index
                available_sig_genes = ml_expression_filtered.columns.intersection(significant_genes)

                if len(available_sig_genes) > 0:
                    ml_data_de = ml_expression_filtered[available_sig_genes].copy()
                    ml_data_de['toxicity_label'] = toxicity_filtered

                    # Save with analysis-specific names
                    if analysis_type == 'drug_averaged_analysis':
                        # Main files for lincs_classification.py
                        ml_files['de_genes'] = self.save_data(ml_data_de, 'ml_ready_de_genes_only.csv')

                    ml_files[f'{analysis_type}_de_genes'] = self.save_data(
                        ml_data_de, f'{analysis_type}_ml_ready_de_genes_only.csv'
                    )

                    self.logger.info(f"{analysis_type}: Created DE genes dataset with {len(available_sig_genes)} genes")

        self.logger.info(f"Created {len(ml_files)} ML-ready datasets")
        return ml_files

    def _save_all_results(self, results: Dict[str, Any]) -> Dict[str, Path]:
        """Save all results from different analyses."""
        self.logger.info(f"Saving all results to {self.output_dir}...")

        saved_files = {}

        # Save results for each analysis type
        for analysis_type, analysis_data in results.items():
            if analysis_type == 'comparison':
                continue  # Handle comparison separately

            prefix = analysis_type

            # Main results
            if 'results' in analysis_data:
                saved_files[f'{prefix}_results'] = self.save_data(
                    analysis_data['results'], f'{prefix}_differential_expression_results.csv'
                )

                # Significant genes only
                significant_genes = analysis_data['results'][analysis_data['results']['significant']]
                if len(significant_genes) > 0:
                    saved_files[f'{prefix}_significant'] = self.save_data(
                        significant_genes, f'{prefix}_significant_de_genes.csv'
                    )

            # Expression data and metadata
            if 'expression_data' in analysis_data:
                saved_files[f'{prefix}_expression'] = self.save_data(
                    analysis_data['expression_data'], f'{prefix}_expression_data.csv'
                )

            if 'toxicity_labels' in analysis_data:
                saved_files[f'{prefix}_toxicity_labels'] = self.save_data(
                    analysis_data['toxicity_labels'].to_frame('toxicity_label'), f'{prefix}_toxicity_labels.csv'
                )

            if 'pattern_metadata' in analysis_data and analysis_data['pattern_metadata'] is not None:
                saved_files[f'{prefix}_pattern_metadata'] = self.save_data(
                    analysis_data['pattern_metadata'], f'{prefix}_pattern_metadata.csv'
                )

        # Save comparison results if available
        if 'comparison' in results:
            comp_data = results['comparison']
            saved_files['comparison_stats'] = self.save_data(
                pd.DataFrame([comp_data['statistics']]), 'comparison_statistics.csv', index=False
            )
            saved_files['comparison_detailed'] = self.save_data(
                comp_data['detailed_comparison'], 'detailed_gene_comparison.csv'
            )

        return saved_files

    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive report for all analyses."""
        self.logger.info("Generating comprehensive differential expression report...")

        report_lines = [
            "=" * 80,
            "COMPREHENSIVE DIFFERENTIAL EXPRESSION ANALYSIS REPORT",
            "=" * 80,
            "",
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Analysis Approach: Both Direct and Drug-Averaged",
            "",
            "METHODOLOGY:",
            "1. Direct Analysis: Full dataset differential expression (diff.py approach 1)",
            "2. Drug-Averaged Analysis: Replicate-collapsed analysis (diff.py approach 2)",
            "",
        ]

        # Add results for each analysis type
        for analysis_type, analysis_data in results.items():
            if analysis_type == 'comparison':
                continue  # Handle comparison separately

            if 'results' in analysis_data:
                res_df = analysis_data['results']
                n_sig = int(res_df['significant'].sum())
                n_up = int((res_df['significant'] & (res_df['log2FoldChange'] > 0)).sum())
                n_down = int((res_df['significant'] & (res_df['log2FoldChange'] < 0)).sum())

                report_lines.extend([
                    f"{analysis_type.upper().replace('_', ' ')} ANALYSIS:",
                    "-" * 40,
                    f"Total genes analyzed: {len(res_df):,}",
                    f"Significant genes: {n_sig:,}",
                    f"  - Upregulated: {n_up:,}",
                    f"  - Downregulated: {n_down:,}",
                    "",
                ])

                # Add top genes
                if n_sig > 0:
                    top_genes = res_df[res_df['significant']].head(5)
                    report_lines.append(f"Top 5 DE genes ({analysis_type}):")
                    for gene, row in top_genes.iterrows():
                        direction = "↑" if row['log2FoldChange'] > 0 else "↓"
                        report_lines.append(
                            f"  {direction} {gene}: log2FC={row['log2FoldChange']:.3f}, "
                            f"FDR={row['FDR']:.2e}"
                        )
                    report_lines.append("")

        # Add comparison results if available
        if 'comparison' in results:
            comp_stats = results['comparison']['statistics']
            report_lines.extend([
                "COMPARISON OF ANALYSIS APPROACHES:",
                "-" * 40,
                f"Direct analysis significant genes: {comp_stats['direct_significant']:,}",
                f"Drug-averaged analysis significant genes: {comp_stats['drug_averaged_significant']:,}",
                f"Overlapping genes: {comp_stats['overlap']:,}",
                f"Jaccard similarity index: {comp_stats['jaccard_index']:.3f}",
                f"Direct-only genes: {comp_stats['direct_only']:,}",
                f"Drug-averaged-only genes: {comp_stats['drug_averaged_only']:,}",
                "",
            ])

        report_lines.extend([
            "ANALYSIS PARAMETERS:",
            "-" * 40,
            f"FDR threshold: {self.significance_thresholds['fdr']}",
            f"Log2 fold change threshold: {self.significance_thresholds['log2fc']}",
            "",
            "INPUT FILES:",
            "- recheck-kidney_lincs_matched_df.csv (expression data)",
            "- kidney_filtered_expanded_toxicity_df.csv (metadata)",
            "",
            "=" * 80
        ])

        # Save report
        report_content = "\n".join(report_lines)
        report_path = self.output_dir / "comprehensive_differential_expression_report.txt"

        try:
            with open(report_path, 'w') as f:
                f.write(report_content)
            self.logger.info(f"Comprehensive report saved to {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save comprehensive report: {e}")
            raise
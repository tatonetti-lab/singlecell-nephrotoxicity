"""
Power analysis simulation component.

This component performs power analysis simulation using single-cell and pseudo-bulk methods.
The simulation code is preserved exactly as in the original with minimal changes for class integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
import os
import datetime
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import multiprocessing as mp
from itertools import product
from functools import partial
from pathlib import Path

from core.base import BaseDataProcessor
from utils.logging import ComponentLogger

warnings.filterwarnings("ignore", category=DeprecationWarning)


class PowerAnalysisSimulator(BaseDataProcessor):
    """
    Component for power analysis simulation.
    
    This component performs power analysis comparing single-cell and pseudo-bulk methods
    across different effect sizes, response rates, and cell types.
    
    Note: The simulation code is preserved exactly as in the original implementation
    to ensure consistency with previous results.
    """
    
    def __init__(self, config: Dict[str, Any], component_name: str = 'simulation'):
        super().__init__(config, component_name)
        
        # Get simulation-specific configuration
        self.sim_config = config.get('simulation_config', {})
        
        # Simulation parameters
        self.effect_sizes = self.sim_config.get('effect_sizes', [0.01, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0, 7.0, 10.0])
        self.response_rates = self.sim_config.get('response_rates', [0.3, 0.8, 0.95])
        self.methods = self.sim_config.get('methods', ['single_cell', 'pseudo_bulk'])
        self.n_iterations = self.sim_config.get('n_iterations', 100)
        self.n_samples = self.sim_config.get('n_samples', 10)
        self.p_value_threshold = self.sim_config.get('p_value_threshold', 0.0000025)
        self.log_fc_threshold = self.sim_config.get('log_fc_threshold', 2.0)
        self.n_processes = self.sim_config.get('default_processes', 48)
        
        # Input data file
        self.input_data_file = self.sim_config.get('input_data_file', 'input_data/base_sampled_data_1_percent.csv')
        
        self.component_logger = ComponentLogger(component_name, verbose=self.component_config.get('verbose', True))
    
    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        if not Path(self.input_data_file).exists():
            self.logger.error(f"Input data file not found: {self.input_data_file}")
            return False
        
        return True
    
    def run(self) -> Dict[str, Any]:
        """Run power analysis simulation."""
        self.component_logger.start_component(total_steps=5)
        
        try:
            # Step 1: Load and prepare data
            self.component_logger.step_completed("Loading and preparing data")
            input_data = self._load_and_prepare_data()
            
            # Step 2: Save simulation parameters
            self.component_logger.step_completed("Saving simulation parameters")
            self._save_parameters()
            
            # Step 3: Run power analysis
            self.component_logger.step_completed("Running power analysis simulation")
            results = self._run_power_analysis_parallel(input_data)
            
            # Step 4: Create visualizations
            self.component_logger.step_completed("Creating power analysis plots")
            self._create_power_plot(results)
            
            # Step 5: Save results
            self.component_logger.step_completed("Saving simulation results")
            saved_files = self._save_results(results)
            
            self.component_logger.finish_component(success=True)
            
            return {
                'simulation_results': results,
                'saved_files': saved_files,
                'n_combinations': len(results),
                'parameters': self._get_parameter_summary()
            }
            
        except Exception as e:
            self.component_logger.finish_component(success=False)
            self.logger.error(f"Power analysis simulation failed: {e}")
            raise
    
    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare simulation data."""
        self.logger.info("Loading simulation data...")
        
        input_data = pd.read_csv(self.input_data_file)
        self.logger.info(f"Data loaded: {len(input_data)} rows")
        
        cell_types = input_data['Cell_Type'].unique()
        self.logger.info(f"Cell types detected: {len(cell_types)} types")
        
        # Convert to numeric (preserving original function)
        input_data = self._convert_to_numeric(input_data)
        
        return input_data
    
    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert gene columns to numeric (preserved from original)."""
        gene_columns = df.columns[2:]  # All columns except index and Cell_Type
        df[gene_columns] = df[gene_columns].astype(np.float64)
        return df
    
    def _save_parameters(self) -> None:
        """Save simulation parameters."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(self.output_dir / 'parameters.txt', 'w') as f:
            f.write(f"Effect Sizes: {self.effect_sizes}\n")
            f.write(f"Response Rates: {self.response_rates}\n")
            f.write(f"Methods: {self.methods}\n")
            f.write(f"Number of Iterations: {self.n_iterations}\n")
            f.write(f"Number of Samples (Pseudo-bulk): {self.n_samples}\n")
            f.write(f"P-value Threshold: {self.p_value_threshold}\n")
            f.write(f"Log Fold Change Threshold: {self.log_fc_threshold}\n")
            f.write(f"Number of Processes: {self.n_processes}\n")
            f.write(f"Timestamp: {timestamp}\n")
    
    # ========================================================================
    # CORE SIMULATION FUNCTIONS (PRESERVED FROM ORIGINAL)
    # ========================================================================
    
    def _get_responsive_cells(self, input_data: pd.DataFrame, cell_type: str, response_rate: float) -> np.ndarray:
        """Identify responsive cells from the target cell type (preserved from original)."""
        # Get indices of cells of the target cell type
        cell_type_indices = np.where(input_data['Cell_Type'] == cell_type)[0]
        n_cells_of_type = len(cell_type_indices)
        
        # Select responsive cells (only within the target cell type)
        n_responsive = int(n_cells_of_type * response_rate)
        responsive_indices = np.random.choice(cell_type_indices, n_responsive, replace=False)
        
        return responsive_indices
    
    def _apply_effect(self, input_data: pd.DataFrame, gene_columns: List[str],
                     responsive_indices: np.ndarray, effect_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply effect to responsive cells and return baseline and simulated datasets (preserved from original)."""
        baseline_data = input_data[gene_columns].values
        simulated_data = baseline_data.copy()
        
        # Apply effect size to responsive cells
        simulated_data[responsive_indices] *= (1 + effect_size)
        
        return baseline_data, simulated_data
    
    def _calculate_log_fold_change(self, baseline: np.ndarray, simulated: np.ndarray) -> np.ndarray:
        """Calculate log2 fold change between baseline and simulated data (preserved from original)."""
        # Add small epsilon to avoid division by zero or log of zero
        epsilon = 1e-10
        baseline_mean = np.mean(baseline, axis=0) + epsilon
        simulated_mean = np.mean(simulated, axis=0) + epsilon
        
        # Calculate log2 fold change
        log2_fold_change = np.log2(simulated_mean / baseline_mean)
        
        return log2_fold_change
    
    def _run_single_cell_analysis(self, baseline_data: np.ndarray, simulated_data: np.ndarray,
                                 gene_columns: List[str], p_value_threshold: float = 0.0000025,
                                 log_fc_threshold: float = 2.0) -> Dict:
        """Run single-cell analysis using one-to-one t-test with log fold change filtering (preserved from original)."""
        # Calculate p-values using one-to-one mapping
        p_values = np.ones(len(gene_columns))
        for i in range(len(gene_columns)):
            _, p_val = stats.ttest_ind(baseline_data[:, i], simulated_data[:, i])
            if not np.isnan(p_val):
                p_values[i] = p_val
        
        # Calculate log fold changes
        log_fold_changes = self._calculate_log_fold_change(baseline_data, simulated_data)
        
        # Identify significant genes based on both p-value and log fold change
        is_significant = (p_values < p_value_threshold) & (np.abs(log_fold_changes) >= log_fc_threshold)
        significant_genes = np.sum(is_significant)
        total_genes = len(gene_columns)
        power = significant_genes / total_genes
        
        return {
            'significant_genes': significant_genes,
            'total_genes': total_genes,
            'power': power,
            'mean_log_fc': np.mean(np.abs(log_fold_changes[is_significant])) if significant_genes > 0 else 0
        }
    
    def _run_pseudo_bulk_analysis(self, baseline_data: np.ndarray, simulated_data: np.ndarray,
                                 gene_columns: List[str], n_samples: int = 10,
                                 p_value_threshold: float = 0.0000025,
                                 log_fc_threshold: float = 2.0) -> Dict:
        """Run pseudo-bulk analysis using mean-based t-test with log fold change filtering (preserved from original)."""
        # Split baseline and simulated data into n_samples
        baseline_split = np.array_split(baseline_data, n_samples)
        simulated_split = np.array_split(simulated_data, n_samples)
        
        # Calculate mean for each sample
        baseline_means = np.vstack([np.mean(split, axis=0) for split in baseline_split])
        simulated_means = np.vstack([np.mean(split, axis=0) for split in simulated_split])
        
        # Calculate p-values on the means
        p_values = np.ones(len(gene_columns))
        for i in range(len(gene_columns)):
            _, p_val = stats.ttest_ind(baseline_means[:, i], simulated_means[:, i])
            if not np.isnan(p_val):
                p_values[i] = p_val
        
        # Calculate log fold changes using the aggregated means
        log_fold_changes = self._calculate_log_fold_change(baseline_means, simulated_means)
        
        # Identify significant genes based on both p-value and log fold change
        is_significant = (p_values < p_value_threshold) & (np.abs(log_fold_changes) >= log_fc_threshold)
        significant_genes = np.sum(is_significant)
        total_genes = len(gene_columns)
        power = significant_genes / total_genes
        
        return {
            'significant_genes': significant_genes,
            'total_genes': total_genes,
            'power': power,
            'mean_log_fc': np.mean(np.abs(log_fold_changes[is_significant])) if significant_genes > 0 else 0
        }
    
    def _process_single_combination(self, params, input_data, gene_columns, methods,
                                   n_samples, p_value_threshold, log_fc_threshold):
        """Process a single combination of parameters (preserved from original)."""
        cell_type, effect_size, response_rate, iteration = params
        results = []
        
        # Generate a seed based on parameters to ensure different random selections across processes
        seed = hash((cell_type, effect_size, response_rate, iteration)) % (2**32)
        np.random.seed(seed)
        
        # Get only the data for the specific cell type
        cell_type_mask = input_data['Cell_Type'] == cell_type
        cell_type_data = input_data[cell_type_mask].copy()
        
        # Get responsive cells only within this cell type's data
        n_cells_of_type = len(cell_type_data)
        n_responsive = int(n_cells_of_type * response_rate)
        responsive_indices = np.random.choice(range(n_cells_of_type), n_responsive, replace=False)
        
        # Apply effect to create baseline and simulated datasets
        baseline_data = cell_type_data[gene_columns].values
        simulated_data = baseline_data.copy()
        
        # Apply effect size to responsive cells
        simulated_data[responsive_indices] *= (1 + effect_size)
        
        # Run both analysis methods on the same data
        for method in methods:
            if method == 'single_cell':
                analysis_result = self._run_single_cell_analysis(
                    baseline_data, simulated_data, gene_columns,
                    p_value_threshold, log_fc_threshold
                )
            else:  # method == 'pseudo_bulk'
                analysis_result = self._run_pseudo_bulk_analysis(
                    baseline_data, simulated_data, gene_columns,
                    n_samples, p_value_threshold, log_fc_threshold
                )
            
            # Create full result dictionary
            result = {
                'cell_type': cell_type,
                'effect_size': effect_size,
                'response_rate': response_rate,
                'iteration': iteration,
                'method': method,
                **analysis_result
            }
            
            results.append(result)
        
        return results
    
    def _run_power_analysis_parallel(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Run power analysis in parallel using multiprocessing (preserved from original)."""
        gene_columns = input_data.columns[2:]
        cell_types = input_data['Cell_Type'].unique()
        
        # Create all parameter combinations
        all_combinations = list(product(
            cell_types,
            self.effect_sizes,
            self.response_rates,
            range(self.n_iterations)
        ))
        
        self.logger.info(f"Total combinations to process: {len(all_combinations)}")
        
        # Create a partial function with fixed parameters
        process_func = partial(
            self._process_single_combination,
            input_data=input_data,
            gene_columns=gene_columns,
            methods=self.methods,
            n_samples=self.n_samples,
            p_value_threshold=self.p_value_threshold,
            log_fc_threshold=self.log_fc_threshold
        )
        
        # Configure multiprocessing for the current platform
        try:
            mp.set_start_method('fork')
        except:
            pass  # Use default method if fork is not available
        
        # Set up multiprocessing pool with the specified number of processes
        with mp.Pool(processes=self.n_processes) as pool:
            # Use imap_unordered to get results as they are completed
            results_nested = list(tqdm(
                pool.imap_unordered(process_func, all_combinations),
                total=len(all_combinations),
                desc="Processing parameter combinations"
            ))
        
        # Flatten results
        results = [item for sublist in results_nested for item in sublist]
        
        return pd.DataFrame(results)
    
    def _create_power_plot(self, results_df: pd.DataFrame) -> None:
        """Create high-quality power plot comparing single-cell and pseudo-bulk methods."""
        self.logger.info("Creating power analysis plots...")
        
        # Calculate average values across iterations
        summary = results_df.groupby(['cell_type', 'effect_size', 'response_rate', 'method']).agg({
            'power': 'mean',
            'significant_genes': 'mean',
            'total_genes': 'first',
            'mean_log_fc': 'mean'
        }).reset_index()
        
        # Get unique response rates and cell types
        response_rates = sorted(summary['response_rate'].unique())
        cell_types = sorted(summary['cell_type'].unique())
        methods = summary['method'].unique()
        
        # Create a simple subplot grid with response rates
        n_cols = min(len(response_rates), 3)  # Maximum 3 columns
        n_rows = (len(response_rates) + n_cols - 1) // n_cols
        
        # Enhanced subplot titles with better formatting
        subplot_titles = [f"<b>Response Rate: {rate*100:.0f}%</b>" for rate in response_rates]
        
        # Create figure for power analysis with enhanced styling
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=subplot_titles,
            shared_yaxes=True, 
            shared_xaxes=True,
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )
        
        # Create a separate figure for log fold change with enhanced styling
        fig_logfc = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=subplot_titles,
            shared_yaxes=True, 
            shared_xaxes=True,
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )
        
        # Enhanced color mapping for cell types with better contrast and professional colors
        cell_type_colors = {
            'Endothelium': '#2E8B57',  # Sea Green
            'Immune': '#8A2BE2',      # Blue Violet
            'Nephron': '#DC143C',     # Crimson
            'Stroma': '#4169E1'       # Royal Blue
        }
        
        # Enhanced line style and width mapping for methods
        method_styles = {
            'single_cell': {'dash': 'solid', 'width': 3},
            'pseudo_bulk': {'dash': 'dash', 'width': 3}
        }
        
        # For each response rate, create a subplot
        for i, rate in enumerate(response_rates):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            # Get data for this response rate
            rate_data = summary[summary['response_rate'] == rate]
            
            # For each cell type and method combination
            for cell_type in cell_types:
                cell_data = rate_data[rate_data['cell_type'] == cell_type]
                
                for method in methods:
                    method_data = cell_data[cell_data['method'] == method]
                    
                    if not method_data.empty:
                        # Sort by effect size for proper line plotting
                        method_data = method_data.sort_values('effect_size')
                        
                        # Enhanced marker styling
                        marker_style = dict(
                            size=8,
                            line=dict(width=2, color='white'),
                            opacity=0.8
                        )
                        
                        # Enhanced line styling
                        line_style = dict(
                            color=cell_type_colors.get(cell_type, '#808080'),
                            width=method_styles.get(method, {'width': 3})['width'],
                            dash=method_styles.get(method, {'dash': 'solid'})['dash']
                        )
                        
                        # Add power trace to subplot with enhanced styling
                        fig.add_trace(
                            go.Scatter(
                                x=method_data['effect_size'],
                                y=method_data['power'],
                                name=f"{cell_type} ({method.replace('_', ' ').title()})",
                                mode='lines+markers',
                                line=line_style,
                                marker=marker_style,
                                hovertemplate=(
                                    f"<b>{cell_type} ({method.replace('_', ' ').title()})</b><br>"
                                    "Effect Size: %{x}<br>"
                                    "Power: %{y:.3f}<br>"
                                    "<extra></extra>"
                                ),
                                showlegend=(i == 0)  # Only show legend for first subplot
                            ),
                            row=row, col=col
                        )
                        
                        # Add log fold change trace to separate subplot with enhanced styling
                        fig_logfc.add_trace(
                            go.Scatter(
                                x=method_data['effect_size'],
                                y=method_data['mean_log_fc'],
                                name=f"{cell_type} ({method.replace('_', ' ').title()})",
                                mode='lines+markers',
                                line=line_style,
                                marker=marker_style,
                                hovertemplate=(
                                    f"<b>{cell_type} ({method.replace('_', ' ').title()})</b><br>"
                                    "Effect Size: %{x}<br>"
                                    "Mean |Log2 FC|: %{y:.3f}<br>"
                                    "<extra></extra>"
                                ),
                                showlegend=(i == 0)  # Only show legend for first subplot
                            ),
                            row=row, col=col
                        )
        
        # Enhanced layout for power plot
        fig.update_layout(
            title={
                'text': "<b>Power Analysis by Response Rate</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2F4F4F'}
            },
            height=400 * n_rows,
            width=500 * n_cols,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2F4F4F"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#E6E6FA",
                borderwidth=1,
                font=dict(size=11)
            ),
            margin=dict(l=80, r=80, t=120, b=100)
        )
        
        # Enhanced layout for log fold change plot
        fig_logfc.update_layout(
            title={
                'text': "<b>Mean Log₂ Fold Change of Significant Genes by Response Rate</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2F4F4F'}
            },
            height=400 * n_rows,
            width=500 * n_cols,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2F4F4F"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#E6E6FA",
                borderwidth=1,
                font=dict(size=11)
            ),
            margin=dict(l=80, r=80, t=120, b=100)
        )
        
        # Enhanced axes styling for power plot
        fig.update_xaxes(
            title_text="<b>Effect Size</b>",
            type="log",
            showgrid=True,
            gridwidth=1,
            gridcolor='#E6E6FA',
            showline=True,
            linewidth=2,
            linecolor='#2F4F4F',
            mirror=True,
            title_font=dict(size=14, color='#2F4F4F'),
            tickfont=dict(size=11, color='#2F4F4F')
        )
        fig.update_yaxes(
            title_text="<b>Statistical Power</b>",
            showgrid=True,
            gridwidth=1,
            gridcolor='#E6E6FA',
            showline=True,
            linewidth=2,
            linecolor='#2F4F4F',
            mirror=True,
            title_font=dict(size=14, color='#2F4F4F'),
            tickfont=dict(size=11, color='#2F4F4F'),
            range=[0, 1.05]
        )
        
        # Enhanced axes styling for log fold change plot
        fig_logfc.update_xaxes(
            title_text="<b>Effect Size</b>",
            type="log",
            showgrid=True,
            gridwidth=1,
            gridcolor='#E6E6FA',
            showline=True,
            linewidth=2,
            linecolor='#2F4F4F',
            mirror=True,
            title_font=dict(size=14, color='#2F4F4F'),
            tickfont=dict(size=11, color='#2F4F4F')
        )
        fig_logfc.update_yaxes(
            title_text="<b>Mean |Log₂ Fold Change|</b>",
            showgrid=True,
            gridwidth=1,
            gridcolor='#E6E6FA',
            showline=True,
            linewidth=2,
            linecolor='#2F4F4F',
            mirror=True,
            title_font=dict(size=14, color='#2F4F4F'),
            tickfont=dict(size=11, color='#2F4F4F')
        )
        
        # Save response rate comparison plots with high quality settings
        fig.write_html(
            self.output_dir / 'power_analysis_by_response_rate.html',
            config={'displayModeBar': True, 'displaylogo': False}
        )
        fig.write_image(
            self.output_dir / 'power_analysis_by_response_rate.png',
            width=500 * n_cols,
            height=400 * n_rows,
            scale=2
        )
        
        fig_logfc.write_html(
            self.output_dir / 'logfc_analysis_by_response_rate.html',
            config={'displayModeBar': True, 'displaylogo': False}
        )
        fig_logfc.write_image(
            self.output_dir / 'logfc_analysis_by_response_rate.png',
            width=500 * n_cols,
            height=400 * n_rows,
            scale=2
        )
        
        # Also create individual cell type plots for deeper analysis
        for cell_type in cell_types:
            self._create_cell_type_plot(summary, cell_type, response_rates, methods, method_styles, n_rows, n_cols)
        
        self.logger.info("Power analysis plots created")
    
    def _create_cell_type_plot(self, summary: pd.DataFrame, cell_type: str, response_rates: List[float],
                              methods: List[str], method_styles: Dict[str, Dict], n_rows: int, n_cols: int) -> None:
        """Create high-quality individual cell type plots."""
        cell_data = summary[summary['cell_type'] == cell_type]
        
        # Enhanced subplot titles
        subplot_titles = [f"<b>Response Rate: {rate*100:.0f}%</b>" for rate in response_rates]
        
        # Create a subplot grid for power analysis with enhanced styling
        fig_cell = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=subplot_titles,
            shared_yaxes=True, 
            shared_xaxes=True,
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )
        
        # Create a subplot grid for log fold change with enhanced styling
        fig_cell_logfc = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=subplot_titles,
            shared_yaxes=True, 
            shared_xaxes=True,
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )
        
        # For each response rate
        for i, rate in enumerate(response_rates):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            # Get data for this response rate and cell type
            rate_cell_data = cell_data[cell_data['response_rate'] == rate]
            
            # For each method
            for method in methods:
                method_data = rate_cell_data[rate_cell_data['method'] == method]
                
                if not method_data.empty:
                    # Sort by effect size for proper line plotting
                    method_data = method_data.sort_values('effect_size')
                    
                    # Enhanced color scheme for different methods
                    method_colors = {
                        'single_cell': '#FF6B6B',  # Coral Red
                        'pseudo_bulk': '#4ECDC4'   # Teal
                    }
                    
                    # Enhanced marker and line styling
                    marker_style = dict(
                        size=10,
                        line=dict(width=2, color='white'),
                        opacity=0.8
                    )
                    
                    line_style = dict(
                        color=method_colors.get(method, '#808080'),
                        width=method_styles.get(method, {'width': 4})['width'],
                        dash=method_styles.get(method, {'dash': 'solid'})['dash']
                    )
                    
                    # Add power trace to subplot with enhanced styling
                    fig_cell.add_trace(
                        go.Scatter(
                            x=method_data['effect_size'],
                            y=method_data['power'],
                            name=method.replace('_', ' ').title(),
                            mode='lines+markers',
                            line=line_style,
                            marker=marker_style,
                            hovertemplate=(
                                f"<b>{method.replace('_', ' ').title()}</b><br>"
                                "Effect Size: %{x}<br>"
                                "Power: %{y:.3f}<br>"
                                "<extra></extra>"
                            ),
                            showlegend=(i == 0)  # Only show legend for first subplot
                        ),
                        row=row, col=col
                    )
                    
                    # Add log fold change trace to subplot with enhanced styling
                    fig_cell_logfc.add_trace(
                        go.Scatter(
                            x=method_data['effect_size'],
                            y=method_data['mean_log_fc'],
                            name=method.replace('_', ' ').title(),
                            mode='lines+markers',
                            line=line_style,
                            marker=marker_style,
                            hovertemplate=(
                                f"<b>{method.replace('_', ' ').title()}</b><br>"
                                "Effect Size: %{x}<br>"
                                "Mean |Log2 FC|: %{y:.3f}<br>"
                                "<extra></extra>"
                            ),
                            showlegend=(i == 0)  # Only show legend for first subplot
                        ),
                        row=row, col=col
                    )
        
        # Enhanced layout for cell type power plot
        fig_cell.update_layout(
            title={
                'text': f"<b>Power Analysis for {cell_type} Cells by Response Rate</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2F4F4F'}
            },
            height=400 * n_rows,
            width=500 * n_cols,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2F4F4F"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#E6E6FA",
                borderwidth=1,
                font=dict(size=12),
                title_text="<b>Analysis Method</b>"
            ),
            margin=dict(l=80, r=80, t=120, b=100)
        )
        
        # Enhanced layout for cell type log fold change plot
        fig_cell_logfc.update_layout(
            title={
                'text': f"<b>Mean Log₂ Fold Change for {cell_type} Cells by Response Rate</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2F4F4F'}
            },
            height=400 * n_rows,
            width=500 * n_cols,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2F4F4F"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#E6E6FA",
                borderwidth=1,
                font=dict(size=12),
                title_text="<b>Analysis Method</b>"
            ),
            margin=dict(l=80, r=80, t=120, b=100)
        )
        
        # Enhanced axes styling for cell type power plot
        fig_cell.update_xaxes(
            title_text="<b>Effect Size</b>",
            type="log",
            showgrid=True,
            gridwidth=1,
            gridcolor='#E6E6FA',
            showline=True,
            linewidth=2,
            linecolor='#2F4F4F',
            mirror=True,
            title_font=dict(size=14, color='#2F4F4F'),
            tickfont=dict(size=11, color='#2F4F4F')
        )
        fig_cell.update_yaxes(
            title_text="<b>Statistical Power</b>",
            showgrid=True,
            gridwidth=1,
            gridcolor='#E6E6FA',
            showline=True,
            linewidth=2,
            linecolor='#2F4F4F',
            mirror=True,
            title_font=dict(size=14, color='#2F4F4F'),
            tickfont=dict(size=11, color='#2F4F4F'),
            range=[0, 1.05]
        )
        
        # Enhanced axes styling for cell type log fold change plot
        fig_cell_logfc.update_xaxes(
            title_text="<b>Effect Size</b>",
            type="log",
            showgrid=True,
            gridwidth=1,
            gridcolor='#E6E6FA',
            showline=True,
            linewidth=2,
            linecolor='#2F4F4F',
            mirror=True,
            title_font=dict(size=14, color='#2F4F4F'),
            tickfont=dict(size=11, color='#2F4F4F')
        )
        fig_cell_logfc.update_yaxes(
            title_text="<b>Mean |Log₂ Fold Change|</b>",
            showgrid=True,
            gridwidth=1,
            gridcolor='#E6E6FA',
            showline=True,
            linewidth=2,
            linecolor='#2F4F4F',
            mirror=True,
            title_font=dict(size=14, color='#2F4F4F'),
            tickfont=dict(size=11, color='#2F4F4F')
        )
        
        # Save cell type-specific plots with high quality settings
        fig_cell.write_html(
            self.output_dir / f'power_analysis_{cell_type}_by_response_rate.html',
            config={'displayModeBar': True, 'displaylogo': False}
        )
        fig_cell.write_image(
            self.output_dir / f'power_analysis_{cell_type}_by_response_rate.png',
            width=500 * n_cols,
            height=400 * n_rows,
            scale=2
        )
        
        fig_cell_logfc.write_html(
            self.output_dir / f'logfc_analysis_{cell_type}_by_response_rate.html',
            config={'displayModeBar': True, 'displaylogo': False}
        )
        fig_cell_logfc.write_image(
            self.output_dir / f'logfc_analysis_{cell_type}_by_response_rate.png',
            width=500 * n_cols,
            height=400 * n_rows,
            scale=2
        )
    
    def _save_results(self, results_df: pd.DataFrame) -> Dict[str, Path]:
        """Save simulation results."""
        saved_files = {}
        
        # Save main results
        saved_files['results'] = self.save_data(results_df, 'power_analysis_results.csv', index=False)
        
        return saved_files
    
    def _get_parameter_summary(self) -> Dict[str, Any]:
        """Get summary of simulation parameters."""
        return {
            'effect_sizes': self.effect_sizes,
            'response_rates': self.response_rates,
            'methods': self.methods,
            'n_iterations': self.n_iterations,
            'n_samples': self.n_samples,
            'p_value_threshold': self.p_value_threshold,
            'log_fc_threshold': self.log_fc_threshold,
            'n_processes': self.n_processes
        }

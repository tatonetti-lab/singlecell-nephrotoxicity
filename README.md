# Nephrotoxicity Analysis Pipeline

A comprehensive pipeline for analyzing drug-induced nephrotoxicity using single-cell RNA sequencing data and machine learning approaches.
Zenodo link to data: https://doi.org/10.5281/zenodo.15724290
Pre-print: https://doi.org/10.1101/2025.06.17.660070

## Overview

This pipeline integrates multiple analysis approaches to study nephrotoxicity:
- **Differential Expression Analysis**: Compare gene expression between toxic and non-toxic drug treatments
- **Drug2Cell Analysis**: Perform drug-cell interaction analysis using LINCS L1000 data
- **Statistical Analysis**: Conduct comprehensive statistical comparisons and visualizations
- **Machine Learning**: Build predictive models for nephrotoxicity classification
- **Simulation Analysis**: Perform power analysis and effect size simulations

## Prerequisites
### Before diving into this analysis, please make sure to run the R workflows provided in /kidney_single_cell_data_processing. 
There is an additional readme within the directory for more explanations on that pipeline and how to run it.

### Python Dependencies
Install the required packages:

```bash
pip install -r requirements.txt
```

### Required Data Files
**Download the input data folder from Zenodo and place it in the project root directory.**
Link: https://doi.org/10.5281/zenodo.15724290

The pipeline expects the following files in the `input_data/` directory:

**Core Dataset Files:**
- `kidney_data.h5ad` - Single-cell annotated RNA-seq data in AnnData format
- `V2_merged_drug_dataset.csv` - Merged drug information dataset
- `kidney_filtered_expanded_toxicity_df.csv` - Toxicity labels and metadata

**Reference Data Files:**
- `DIRIL_508.csv` - DIRIL reference dataset
- `pryan_reference_set_ades.csv` - Reference adverse events dataset
- `toxic_drugList_ACS.csv` - ACS toxic drug list
- `refined_drugbank_targets.csv` - DrugBank target information
- `recheck-kidney_lincs_matched_df.csv` - LINCS matched dataset
- `dr2c_var_genes.csv` - Variable genes for Drug2Cell analysis
- `base_sampled_data_1_percent.csv` - Sampled data for simulation analysis

## Quick Start

### Option 1: Run Complete Pipeline
Execute the full analysis pipeline with default settings:

```bash
python run_analysis.py --all
```

### Option 2: Custom Configuration
1. Copy and modify the configuration file:
```bash
cp config/default_config.yaml config/my_config.yaml
# Edit my_config.yaml with your specific settings
```

2. Run with custom configuration:
```bash
python run_analysis.py --config config/my_config.yaml
```

### Option 3: Run Specific Components
Run individual analysis components:

```bash
# Differential expression analysis only
python run_analysis.py --components differential

# Statistical analysis and visualization
python run_analysis.py --components statistical

# Machine learning analysis
python run_analysis.py --components ml

# Multiple specific components
python run_analysis.py --components differential,statistical,ml
```

## Output Structure

All results are saved in the `pipeline_results/` directory:

```
pipeline_results/
├── differential_expression_results/     # DE analysis outputs
│   ├── direct_analysis_*.csv           # Direct comparison results
│   ├── drug_averaged_analysis_*.csv    # Drug-averaged results
│   └── *.png/*.svg                     # Visualization plots
├── statistical_results/                # Statistical analysis outputs
│   ├── correlation_matrix.csv          # Gene correlation data
│   ├── mannwhitney_results.csv         # Statistical test results
│   └── *.png/*.svg                     # Statistical plots
├── drug2cell_results/                  # Drug2Cell analysis outputs
│   ├── combined_drug_matrix.csv        # Drug signature matrix
│   └── analysis_summary.json          # Summary statistics
├── drug_score_ml_results/              # Drug score-based ML outputs
│   ├── best_model_predictions.csv      # Model predictions from drug scores
│   ├── feature_importance_*.csv        # Feature importance scores
│   ├── cross_validation_metrics.csv    # CV performance
│   └── *.png/*.svg                     # ML visualization plots
├── expression_ml_results/              # Expression-based ML outputs
│   ├── best_model_predictions.csv      # Model predictions from gene expression
│   ├── feature_importance_*.csv        # Feature importance scores
│   ├── cross_validation_metrics.csv    # CV performance
│   └── *.png/*.svg                     # ML visualization plots
├── simulation_results/                 # Power analysis outputs
│   ├── power_analysis_*.html/.png      # Interactive/static plots
│   └── power_analysis_results.csv     # Simulation results
└── pipeline_summary_report.txt         # Overall summary
```

## Configuration

### Key Configuration Options

Edit `config/default_config.yaml` to customize:

**Input Data Paths:**
```yaml
input_data:
  merged_drug_dataset: "input_data/V2_merged_drug_dataset.csv"
  single_cell_data: "input_data/kidney_data.h5ad"
  # ... other data paths
```

**Output Settings:**
```yaml
output:
  base_dir: "pipeline_results"
  formats: ["png", "svg"]  # Figure formats
  cleanup_intermediate: false
```

## Analysis Components

### 1. Differential Expression Analysis
- **Direct Method**: Direct comparison between toxic vs non-toxic samples
- **Drug-Averaged Method**: Compare drug-averaged expression profiles
- **Output**: DE gene lists, fold changes, statistical significance, visualizations

### 2. Statistical Analysis
- **Correlation Analysis**: Gene-gene and drug-drug correlations
- **Statistical Tests**: Mann-Whitney U tests, t-tests
- **Visualizations**: Correlation heatmaps, boxplots, distribution plots

### 3. Drug2Cell Analysis
- **LINCS Integration**: Connect drug perturbations to cellular responses
- **Signature Analysis**: Analyze drug signature patterns
- **Visualization**: UMAP projections, drug clustering

### 4. Machine Learning
Two complementary ML approaches:

**Drug Score-Based ML** (`drug_score_ml_results/`):
- **Input**: Aggregated drug scores from Drug2Cell analysis
- **Models**: Random Forest, XGBoost, SVM, Logistic Regression
- **Purpose**: Predict nephrotoxicity using drug signature patterns

**Expression-Based ML** (`expression_ml_results/`):
- **Input**: Differential gene expression data
- **Models**: Random Forest, XGBoost, Logistic Regression
- **Purpose**: Predict nephrotoxicity using gene expression profiles
- **Features**: Differentially expressed genes, top variable genes

**Common Outputs**: Cross-validation metrics, ROC curves, feature importance, model predictions

### 5. Simulation Analysis
- **Power Analysis**: Determine optimal sample sizes
- **Effect Size Analysis**: Simulate various biological effect sizes

## Example Workflows

### Research Workflow
```bash
# 1. Explore data with statistical analysis
python run_analysis.py --components statistical

# 2. Identify differentially expressed genes
python run_analysis.py --components differential

# 3. Build predictive models
python run_analysis.py --components ml

# 4. Validate with simulation
python run_analysis.py --components simulation
```

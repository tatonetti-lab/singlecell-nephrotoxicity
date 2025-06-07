# Kidney Single-Cell RNA-seq Analysis Pipeline

This repository contains a comprehensive workflow for processing and annotating kidney single-cell RNA-seq data. The pipeline integrates multiple datasets, performs quality control, normalization, integration, clustering, and detailed cell type annotation using the Kidney Cell Atlas.

## Overview

The pipeline consists of two main scripts:

1. **`main_kidney_scrnaseq_workflow.R`** - Main processing workflow
2. **`kidney_annotation_workflow.R`** - Advanced annotation using Kidney Cell Atlas

## Workflow Summary

### Main Processing Workflow
1. **Data Loading & Merging** - Loads multiple h5/10X format datasets and merges them
2. **Quality Control** - Applies filtering based on cell and gene quality metrics
3. **Normalization & Scaling** - Normalizes expression data and identifies variable features
4. **Integration** - Integrates multiple samples using Seurat's integration methods
5. **Clustering** - Performs dimensionality reduction, clustering, and visualization
6. **Marker Identification** - Finds cluster-specific marker genes
7. **Data Export** - Saves processed data for downstream analysis

### Advanced Annotation Workflow
1. **Reference Loading** - Loads Kidney Cell Atlas as reference
2. **SingleR Annotation** - Performs detailed cell type annotation
3. **Cell Type Grouping** - Creates hierarchical cell type categories
4. **Visualization** - Generates comprehensive plots and analysis
5. **Quality Assessment** - Evaluates annotation quality
6. **Data Export** - Saves results in multiple formats

## Requirements

### R Packages
```r
# Core Seurat packages
install.packages(c("Seurat", "SeuratObject", "SeuratDisk"))

# Data manipulation and visualization
install.packages(c("dplyr", "ggplot2", "Matrix", "cowplot", "purrr"))

# Single-cell analysis
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("SingleR", "SingleCellExperiment", "celldex"))

# Additional packages
install.packages(c("pheatmap", "data.table", "magrittr", "reticulate"))

# For format conversion (optional)
remotes::install_github("cellgeni/sceasy")
```

### Python Dependencies (for h5ad file handling)
```bash
pip install scanpy pandas
```

### Data Requirements

1. **Single-cell RNA-seq data** in one of these formats:
   - HDF5 format (`.h5` files from 10X Genomics)
   - 10X format (directory with `barcodes.tsv`, `features.tsv`, `matrix.mtx`)

2. **Kidney Cell Atlas reference** (for advanced annotation):
   - Download `Mature_Full_v3.h5ad` from [kidneycellatlas.org](https://www.kidneycellatlas.org/)

## Setup and Configuration

### 1. Clone/Download Scripts
Save both R scripts in your working directory.

### 2. Update Data Paths
Edit the configuration sections in both scripts:

**In `main_kidney_scrnaseq_workflow.R`:**
```r
DATA_PATHS <- list(
  h5_base_path = "/path/to/your/h5/files/",
  h5_samples = paste0("kN", 1:21),  # Adjust sample names
  tenx_base_path = "/path/to/your/10x/files/",
  tenx_samples = c("sample1", "sample2", "sample3")  # Adjust sample names
)
```

**In `kidney_annotation_workflow.R`:**
```r
CONFIG <- list(
  kidney_atlas_path = "path/to/Mature_Full_v3.h5ad",
  seurat_object_path = "kidney_integrated_processed.rds",
  output_dir = "annotation_results"
)
```

### 3. Adjust Parameters (Optional)
You can modify filtering and processing parameters in the configuration sections:

```r
# Quality control parameters
QC_PARAMS <- list(
  min_count_rna = 500,        # Minimum UMI count per cell
  min_feature_rna = 250,      # Minimum genes per cell
  min_log10_genes_per_umi = 0.85,  # Gene complexity threshold
  min_cells_per_gene = 10     # Minimum cells expressing each gene
)

# Integration parameters
INTEGRATION_PARAMS <- list(
  n_features = 2000,          # Number of variable features
  n_dims_pca = 50,           # PCA dimensions
  n_dims_integration = 50,    # Integration dimensions
  cluster_resolution = 0.1    # Clustering resolution
)
```

## Running the Pipeline

### Method 1: Interactive Execution

1. **Run Main Workflow:**
```r
source("main_kidney_scrnaseq_workflow.R")
result_main <- main_workflow()
```

2. **Run Annotation Workflow:**
```r
source("kidney_annotation_workflow.R")
result_annotated <- annotation_workflow()
```

### Method 2: Command Line Execution

```bash
# Run main workflow
Rscript main_kidney_scrnaseq_workflow.R

# Run annotation workflow
Rscript kidney_annotation_workflow.R
```

### Method 3: Step-by-Step Execution

You can also run individual functions if you need more control:

```r
source("main_kidney_scrnaseq_workflow.R")

# Load and merge data
seurat_objects <- load_sample_data(...)
merged_seurat <- merge(...)

# Apply QC filtering
filtered_seurat <- perform_qc_filtering(merged_seurat, QC_PARAMS)

# Continue with other steps...
```

## Expected Outputs

### Main Workflow Outputs
- `kidney_integrated_processed.rds` - Processed and integrated Seurat object
- `all_cluster_markers.csv` - Complete list of cluster markers
- `top_cluster_markers.csv` - Top 10 markers per cluster
- `processing_summary.csv` - Processing statistics
- QC plots: `pre_filtering_*.png`, `post_filtering_*.png`
- Integration plots: `integration_clusters.png`, `integration_samples.png`

### Annotation Workflow Outputs
- `kidney_annotated_final.rds` - Final annotated Seurat object
- `kidney_annotated_final.h5ad` - H5AD format for Python analysis
- `annotation_detailed_celltypes.png` - Detailed cell type visualization
- `annotation_broad_celltypes.png` - Broad cell type categories
- `cell_counts_*.png` - Cell count analysis plots
- `annotation_qc_*.png` - Annotation quality assessment
- Various CSV tables with cell counts and statistics

## Cell Type Annotations

The pipeline provides two levels of cell type annotation:

### Detailed Cell Types (from Kidney Cell Atlas)
32 specific kidney cell types including:
- Proximal tubule variants
- Intercalated cell types
- Endothelial cell subtypes
- Immune cell populations
- And more...

### Broad Cell Type Categories
4 major categories:
- **Nephron** 
- **Endothelium** 
- **Stroma** 
- **Immune** 

## License

This pipeline is provided as-is for research purposes. Please ensure you have appropriate permissions for any datasets used.

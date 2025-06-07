# =============================================================================
# KIDNEY CELL ATLAS ANNOTATION WORKFLOW
# =============================================================================
# 
# This script performs advanced cell type annotation using the Kidney Cell Atlas
# as a reference dataset. It converts h5ad format data, performs SingleR 
# annotation, and creates hierarchical cell type groupings.
#
# Input: Processed Seurat object from main workflow + Kidney Cell Atlas h5ad file
# Output: Annotated Seurat object with detailed and broad cell type categories
#
# Author: Based on singleR_workflow.Rmd and h5ad_to_h5seurat.Rmd by Aditi Kuchi
# =============================================================================

# Load required libraries (matching the original Rmd versions)
library(Seurat)
library(SeuratDisk)
library(SeuratObject)
library(dplyr)
library(Matrix)
library(pheatmap)
library(ggplot2)
library(SingleR)
library(SingleCellExperiment)
library(reticulate)
library(data.table)
library(magrittr)
use_condaenv("seurat4")
# Try to load sceasy for h5ad conversion (optional)
tryCatch({
  library(sceasy)
}, error = function(e) {
  cat("Warning: sceasy not available. h5ad export will be skipped.\n")
})

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data paths (modify as needed)
CONFIG <- list(
  # Path to kidney cell atlas reference (download from kidneycellatlas.org)
  kidney_atlas_path = "Mature_Full_v3.h5ad",
  
  # Path to processed Seurat object from main workflow
  seurat_object_path = "kidney_integrated_processed.rds",
  
  # Output directory
  output_dir = "annotation_results"
)

# Cell type grouping definitions based on Kidney Cell Atlas (exactly from original)
CELL_TYPE_GROUPS <- list(
  stroma_cells = c("Myofibroblast", "Fibroblast"),
  
  endothelium_cells = c(
    "Ascending vasa recta endothelium", 
    "Descending vasa recta endothelium", 
    "Glomerular endothelium", 
    "Peritubular capillary endothelium 1", 
    "Peritubular capillary endothelium 2"
  ),
  
  nephron_cells = c(
    "Connecting tubule", 
    "Distinct proximal tubule 1", 
    "Distinct proximal tubule 2", 
    "Epithelial progenitor cell", 
    "Indistinct intercalated cell", 
    "Pelvic epithelium", 
    "Podocyte", 
    "Principal cell", 
    "Proliferating Proximal Tubule", 
    "Proximal tubule", 
    "Thick ascending limb of Loop of Henle", 
    "Type A intercalated cell", 
    "Type B intercalated cell", 
    "Transitional Urothelium"
  ),
  
  immune_cells = c(
    "B cell", 
    "CD4 T cell", 
    "CD8 T cell", 
    "Mast cell", 
    "MNP-a/classical monocyte derived", 
    "MNP-d/Tissue macrophage", 
    "MNP-c/dendritic cell", 
    "MNP-b/non-classical monocyte derived", 
    "Neutrophil", 
    "NK cell", 
    "NKT cell", 
    "Plasmacytoid dendritic cell"
  )
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

#' Load and convert Kidney Cell Atlas reference data (exactly from original)
#' @param atlas_path Path to h5ad file
#' @return Seurat object containing reference data
load_kidney_atlas_reference <- function(atlas_path) {
  cat("Loading Kidney Cell Atlas reference data...\n")
  
  # Import scanpy (exactly as in original)
  sc <- import("scanpy")
  
  # Read h5ad file (exactly as in original)
  atlas_data <- sc$read_h5ad(atlas_path)
  
  # Extract count matrix and transpose (exactly as in original)
  counts <- t(atlas_data$layers["counts"])
  colnames(counts) <- atlas_data$obs_names$to_list()
  rownames(counts) <- atlas_data$var_names$to_list()
  counts <- Matrix::Matrix(as.matrix(counts), sparse = TRUE)
  
  # Create Seurat object (exactly as in original)
  kidney_ref <- CreateSeuratObject(counts)
  kidney_ref <- AddMetaData(kidney_ref, atlas_data$obs)
  
  cat(sprintf("Reference loaded: %d cells, %d genes\n", 
              ncol(kidney_ref), nrow(kidney_ref)))
  
  return(kidney_ref)
}

#' Perform SingleR annotation using kidney atlas (exactly from original)
#' @param query_object Seurat object to annotate
#' @param reference_object Reference Seurat object
#' @param label_column Column name in reference containing cell type labels
#' @return SingleR results object
perform_singler_annotation <- function(query_object, reference_object, 
                                     label_column = "celltype") {
  cat("Performing SingleR annotation...\n")
  
  # Convert to SingleCellExperiment objects (exactly as in original)
  kidney_integrated_sce <- as.SingleCellExperiment(query_object)
  kidney_ref_sce <- as.SingleCellExperiment(reference_object)
  
  # Run SingleR (exactly as in original)
  kca <- SingleR(
    test = kidney_integrated_sce, 
    assay.type.test = "logcounts", 
    ref = kidney_ref_sce, 
    labels = kidney_ref_sce[[label_column]], 
    aggr.ref = TRUE
  )
  
  cat(sprintf("Annotation complete. %d cells annotated.\n", length(kca$labels)))
  
  return(kca)
}

#' Create cell type groupings (exactly from original function)
#' @param seurat_obj Seurat object with detailed annotations
#' @param annotation_column Column containing detailed annotations
#' @param new_column_name Name for new grouped annotation column
#' @return Modified Seurat object
create_cell_type_groups <- function(seurat_obj, annotation_column = "kca", 
                                   new_column_name = "kca_large") {
  cat("Creating cell type groupings...\n")
  
  # Copy original annotations (exactly as in original)
  seurat_obj@meta.data[[new_column_name]] <- seurat_obj@meta.data[[annotation_column]]
  
  # Apply groupings (exactly as in original function)
  seurat_obj@meta.data[[new_column_name]][seurat_obj@meta.data[[new_column_name]] %in% CELL_TYPE_GROUPS$stroma_cells] <- "Stroma"
  seurat_obj@meta.data[[new_column_name]][seurat_obj@meta.data[[new_column_name]] %in% CELL_TYPE_GROUPS$endothelium_cells] <- "Endothelium"
  seurat_obj@meta.data[[new_column_name]][seurat_obj@meta.data[[new_column_name]] %in% CELL_TYPE_GROUPS$nephron_cells] <- "Nephron"
  seurat_obj@meta.data[[new_column_name]][seurat_obj@meta.data[[new_column_name]] %in% CELL_TYPE_GROUPS$immune_cells] <- "Immune"
  
  cat(sprintf("Created broad cell type groups:\n"))
  print(unique(seurat_obj@meta.data[[new_column_name]]))
  
  return(seurat_obj)
}

#' Create comprehensive visualization plots (following original patterns)
#' @param seurat_obj Annotated Seurat object
#' @param output_prefix Prefix for output files
create_annotation_plots <- function(seurat_obj, output_prefix = "annotation") {
  cat("Creating annotation visualizations...\n")
  
  # Detailed cell type plot (exactly as in original with small legend)
  seurat_obj <- SetIdent(seurat_obj, value = "kca")
  
  p1 <- DimPlot(seurat_obj, label = FALSE, label.size = 3, raster = FALSE) +
    guides(
      shape = guide_legend(override.aes = list(size = 2.5)),
      color = guide_legend(override.aes = list(size = 2.5))
    ) +
    theme(
      legend.title = element_text(size = 6), 
      legend.text = element_text(size = 6),
      legend.key.size = unit(0.6, "lines"),
      axis.title = element_text(size = 6),
      axis.text.x = element_text(size = 6)
    ) +
    ggtitle("Kidney Cell Atlas Detailed Annotation")
  
  ggsave(paste0(output_prefix, "_detailed_celltypes.png"), p1, 
         dpi = 300, width = 13, height = 7, bg = "white")
  
  # Detailed with labels (exactly as in original)
  p1_labeled <- DimPlot(seurat_obj, label = TRUE, label.size = 3, raster = FALSE) + NoLegend()
  ggsave(paste0(output_prefix, "_detailed_celltypes_labeled.png"), p1_labeled, 
         width = 10, height = 8, bg = "white")
  
  # Broad cell type plot (exactly as in original)
  seurat_obj <- SetIdent(seurat_obj, value = "kca_large")
  
  p2 <- DimPlot(seurat_obj, label = TRUE, label.size = 4, raster = FALSE) + 
    NoLegend() +
    ggtitle("Broad Cell Type Categories")
  
  ggsave(paste0(output_prefix, "_broad_celltypes.png"), p2, 
         width = 10, height = 8, bg = "white")
  
  # Broad with legend
  p2_legend <- DimPlot(seurat_obj, label = FALSE, label.size = 3, raster = FALSE) +
    ggtitle("Broad Cell Type Categories")
  
  ggsave(paste0(output_prefix, "_broad_celltypes_legend.png"), p2_legend, 
         width = 10, height = 8, bg = "white")
  
  return(list(detailed = p1, detailed_labeled = p1_labeled, broad = p2, broad_legend = p2_legend))
}

#' Create cell count analysis plots (exactly following original patterns)
#' @param seurat_obj Annotated Seurat object
#' @param output_prefix Prefix for output files
create_cell_count_analysis <- function(seurat_obj, output_prefix = "cell_counts") {
  cat("Creating cell count analysis...\n")
  
  # Extract metadata (exactly as in original)
  kca_cells <- seurat_obj@meta.data %>% as.data.table()
  
  # Count cells per sample and cell type (detailed) - exactly as in original
  kca_table <- kca_cells[, .N, by = c("orig.ident", "kca")]
  
  # Create dot plot for detailed cell types (exactly as in original)
  dot_plot_cell <- ggplot(kca_table, aes(x = orig.ident, y = kca, size = N, fill = N)) +
    geom_point(shape = 21, color = "darkgrey") +
    labs(x = "Sample Number", y = "Cell Type", size = "Counts", fill = "Counts") +
    scale_size_continuous(range = c(3, 10)) +
    scale_fill_gradient(low = "orange", high = "red") +
    theme(
      panel.background = element_rect(fill = "white"),
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.text.y = element_text(size = 6)
    ) +
    ggtitle("Cell Counts per Sample (Detailed Cell Types)")
  
  ggsave(paste0(output_prefix, "_detailed_per_sample.png"), dot_plot_cell, 
         width = 18, height = 13, bg = "white")
  
  # Count cells per sample and cell type (broad) - exactly as in original
  kca_large_cells <- seurat_obj@meta.data %>% as.data.table()
  kca_large_table <- kca_large_cells[, .N, by = c("orig.ident", "kca_large")]
  
  # Create dot plot for broad cell types (exactly as in original)
  dot_plot_cell_large <- ggplot(kca_large_table, aes(x = orig.ident, y = kca_large, size = N, fill = N)) +
    geom_point(shape = 21, color = "darkgrey") +
    labs(x = "Sample Number", y = "Cell Type", size = "Counts", fill = "Counts") +
    scale_size_continuous(range = c(3, 10)) +
    scale_fill_gradient(low = "pink", high = "red") +
    theme(
      panel.background = element_rect(fill = "white"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    ) +
    ggtitle("Cell Counts per Sample (Broad Cell Types)")
  
  ggsave(paste0(output_prefix, "_broad_per_sample.png"), dot_plot_cell_large, 
         width = 12, height = 8, bg = "white")
  
  # Create log-transformed version (from original)
  kca_table_log <- kca_table
  kca_table_log$log_counts <- log(kca_table_log$N + 1)
  
  dot_plot_cell_log <- ggplot(kca_table_log, aes(x = orig.ident, y = kca, size = log_counts, fill = log_counts)) +
    geom_point(shape = 21, color = "darkgrey") +
    labs(x = "Sample Number", y = "Cell Type", size = "log_counts", fill = "log_counts") +
    scale_size_continuous(range = c(3, 10)) +
    scale_fill_gradient(low = "orange", high = "red") +
    theme(
      panel.background = element_rect(fill = "white"),
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.text.y = element_text(size = 6)
    ) +
    ggtitle("Cell Counts per Sample (Log-transformed)")
  
  ggsave(paste0(output_prefix, "_detailed_per_sample_log.png"), dot_plot_cell_log, 
         width = 18, height = 13, bg = "white")
  
  # Save count tables
  write.csv(kca_table, paste0(output_prefix, "_detailed_table.csv"), row.names = FALSE)
  write.csv(kca_large_table, paste0(output_prefix, "_broad_table.csv"), row.names = FALSE)
  
  # Print summary tables (exactly as in original)
  cat("\nDetailed cell type count summary:\n")
  total_cell_type_count <- kca_table[, .(Total_Count = sum(N)), by = kca]
  print(total_cell_type_count)
  
  cat("\nBroad cell type count summary:\n")
  broad_cell_type_count <- kca_large_table[, .(Total_Count = sum(N)), by = kca_large]
  print(broad_cell_type_count)
  
  return(list(
    detailed_plot = dot_plot_cell,
    broad_plot = dot_plot_cell_large,
    log_plot = dot_plot_cell_log,
    detailed_table = kca_table,
    broad_table = kca_large_table
  ))
}

#' Create annotation quality assessment plots (exactly from original)
#' @param singler_results SingleR results object
#' @param output_prefix Prefix for output files
create_annotation_qc <- function(singler_results, output_prefix = "annotation_qc") {
  cat("Creating annotation quality assessment...\n")
  
  # Score heatmap (exactly as in original)
  p1 <- plotScoreHeatmap(singler_results, show.labels = TRUE, order.by = "label")
  ggsave(paste0(output_prefix, "_score_heatmap.png"), p1, width = 23, height = 13, bg = "white")
  
  # Delta distribution plot (exactly as in original)
  p2 <- plotDeltaDistribution(singler_results)
  ggsave(paste0(output_prefix, "_delta_distribution.png"), p2, width = 10, height = 6, bg = "white")
  
  return(list(score_heatmap = p1, delta_distribution = p2))
}

# =============================================================================
# MAIN ANNOTATION WORKFLOW
# =============================================================================

annotation_workflow <- function() {
  
  cat("=== KIDNEY CELL ATLAS ANNOTATION WORKFLOW ===\n\n")
  start_time <- Sys.time()
  
  # Create output directory
  if (!dir.exists(CONFIG$output_dir)) {
    dir.create(CONFIG$output_dir, recursive = TRUE)
  }
  
  # Set working directory to output folder
  original_wd <- getwd()
  setwd(CONFIG$output_dir)
  
  tryCatch({
    
    # -------------------------------------------------------------------------
    # STEP 1: LOAD DATA
    # -------------------------------------------------------------------------
    
    cat("STEP 1: Loading data...\n")
    
    # Load processed Seurat object
    cat("Loading processed Seurat object...\n")
    if (!file.exists(file.path(original_wd, CONFIG$seurat_object_path))) {
      stop("Seurat object not found. Please run main workflow first.")
    }
    
    kidney_integrated <- readRDS(file.path(original_wd, CONFIG$seurat_object_path))
    cat(sprintf("Loaded Seurat object: %d cells, %d genes\n", 
                ncol(kidney_integrated), nrow(kidney_integrated)))
    
    # Load kidney atlas reference
    if (!file.exists(file.path(original_wd, CONFIG$kidney_atlas_path))) {
      stop("Kidney Cell Atlas file not found. Please download from kidneycellatlas.org")
    }
    
    kidney_ref <- load_kidney_atlas_reference(file.path(original_wd, CONFIG$kidney_atlas_path))
    
    # -------------------------------------------------------------------------
    # STEP 2: SINGLER ANNOTATION
    # -------------------------------------------------------------------------
    
    cat("\nSTEP 2: Performing SingleR annotation...\n")
    
    # Perform annotation (exactly as in original)
    kca <- perform_singler_annotation(kidney_integrated, kidney_ref, "celltype")
    
    # Add annotations to Seurat object (exactly as in original)
    kidney_integrated@meta.data$kca <- kca$labels
    
    # Check for NA annotations and handle them (exactly as in original)
    na_count <- sum(is.na(kidney_integrated@meta.data$kca))
    cat(sprintf("Found %d cells with NA annotations\n", na_count))
    
    # Remove cells with NA annotations (exactly as in original)
    kidney_integrated_NA <- kidney_integrated[, !is.na(kidney_integrated@meta.data$kca)]
    cat(sprintf("Removed %d cells with NA annotations\n", 
                ncol(kidney_integrated) - ncol(kidney_integrated_NA)))
    
    # -------------------------------------------------------------------------
    # STEP 3: CREATE CELL TYPE GROUPINGS
    # -------------------------------------------------------------------------
    
    cat("\nSTEP 3: Creating cell type groupings...\n")
    
    kidney_integrated_NA <- create_cell_type_groups(kidney_integrated_NA, "kca", "kca_large")
    
    # Print annotation summary (exactly as in original)
    cat("\nDetailed cell type distribution:\n")
    print(table(kidney_integrated_NA@meta.data$kca))
    
    cat("\nBroad cell type distribution:\n")
    print(table(kidney_integrated_NA@meta.data$kca_large))
    
    # -------------------------------------------------------------------------
    # STEP 4: CREATE VISUALIZATIONS
    # -------------------------------------------------------------------------
    
    cat("\nSTEP 4: Creating visualizations...\n")
    
    # Annotation plots
    annotation_plots <- create_annotation_plots(kidney_integrated_NA)
    
    # Cell count analysis
    cell_count_analysis <- create_cell_count_analysis(kidney_integrated_NA)
    
    # Annotation quality assessment
    annotation_qc <- create_annotation_qc(kca)
    
    # -------------------------------------------------------------------------
    # STEP 5: EXPORT RESULTS
    # -------------------------------------------------------------------------
    
    cat("\nSTEP 5: Exporting results...\n")
    
    # Save annotated Seurat object
    saveRDS(kidney_integrated_NA, "kidney_annotated_final.rds")
    
    # Convert to h5ad format for Python analysis (if sceasy is available)
    cat("Attempting to convert to h5ad format...\n")
    tryCatch({
      if (requireNamespace("sceasy", quietly = TRUE)) {
        use_condaenv('base', required = TRUE)
        sceasy::convertFormat(kidney_integrated_NA, 
                             from = "seurat", 
                             to = "anndata",
                             outFile = 'kidney_annotated_final.h5ad')
        cat("Successfully exported to h5ad format\n")
      } else {
        cat("sceasy not available - skipping h5ad export\n")
      }
    }, error = function(e) {
      cat("Warning: Could not convert to h5ad format. Error:", e$message, "\n")
    })
    
    # Save gene list (exactly as in original)
    gene_names_indata <- rownames(kidney_integrated_NA@assays$RNA@counts)
    write.csv(gene_names_indata, "gene_names_in_KCAdata.csv", row.names = FALSE)
    
    # Save filtered gene list (exactly as in original logic)
    gene_names_filtered <- gene_names_indata[!grepl("^SNORA|^SNORD|^RN7SL|^RN7SK|^RNU|^RMRP|^RPPH1|^MT-|^RPL|^RPS", gene_names_indata, ignore.case = TRUE)]
    write.csv(gene_names_filtered, "filtered_gene_names.csv", row.names = FALSE)
    
    # Create summary report
    summary_stats <- data.frame(
      Metric = c(
        "Total Samples", 
        "Total Cells (annotated)", 
        "Total Genes", 
        "Detailed Cell Types", 
        "Broad Cell Type Categories",
        "Cells with NA annotations (removed)",
        "Annotation Time (minutes)"
      ),
      Value = c(
        length(unique(kidney_integrated_NA$orig.ident)),
        ncol(kidney_integrated_NA),
        nrow(kidney_integrated_NA),
        length(unique(kidney_integrated_NA$kca)),
        length(unique(kidney_integrated_NA$kca_large)),
        na_count,
        round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 2)
      )
    )
    
    write.csv(summary_stats, "annotation_summary.csv", row.names = FALSE)
    
    # Save SingleR results object for further analysis
    saveRDS(kca, "singler_results.rds")
    
    # -------------------------------------------------------------------------
    # STEP 6: SUMMARY
    # -------------------------------------------------------------------------
    
    cat("\n=== ANNOTATION WORKFLOW COMPLETE ===\n")
    print(summary_stats)
    
    cat("\nFiles created:\n")
    cat("- kidney_annotated_final.rds (annotated Seurat object)\n")
    cat("- kidney_annotated_final.h5ad (h5ad format for Python, if sceasy available)\n")
    cat("- singler_results.rds (SingleR results object)\n")
    cat("- annotation_detailed_celltypes*.png (detailed cell type plots)\n")
    cat("- annotation_broad_celltypes*.png (broad cell type plots)\n")
    cat("- cell_counts_*.png (cell count analysis plots)\n")
    cat("- annotation_qc_*.png (annotation quality plots)\n")
    cat("- *_table.csv (cell count tables)\n")
    cat("- gene_names_in_KCAdata.csv (all genes in dataset)\n")
    cat("- filtered_gene_names.csv (protein-coding genes)\n")
    cat("- annotation_summary.csv (summary statistics)\n")
    
    return(kidney_integrated_NA)
    
  }, finally = {
    # Restore original working directory
    setwd(original_wd)
  })
}

# =============================================================================
# RUN WORKFLOW
# =============================================================================

# Execute the annotation workflow
if (!interactive()) {
  # Run automatically if script is sourced non-interactively
  result <- annotation_workflow()
} else {
  # Manual execution
  cat("To run the annotation workflow, execute: result <- annotation_workflow()\n")
  cat("Make sure you have:\n")
  cat("1. Completed the main processing workflow\n")
  cat("2. Downloaded the Kidney Cell Atlas h5ad file (Mature_Full_v3.h5ad)\n")
  cat("3. Updated the CONFIG paths at the top of this script\n")
  cat("4. Installed reticulate and python with scanpy: py_install('scanpy')\n")
}
# =============================================================================
# MAIN KIDNEY SINGLE-CELL RNA-SEQ PROCESSING WORKFLOW
# =============================================================================
# 
# This script processes kidney single-cell RNA-seq data from multiple samples,
# performs quality control, normalization, integration, and basic annotation.
#
# Input: Multiple h5 files or 10X formatted directories
# Output: Integrated Seurat object with basic annotations and markers
#
# Author: Based on newRef-kidney-normals-integrated.Rmd by Aditi Kuchi
# =============================================================================

# Load required libraries (matching the original Rmd versions)
library(Seurat)
library(SeuratObject)
library(dplyr)
library(Matrix)
library(ggplot2)
library(pheatmap)
library(cowplot)
library(purrr)
library(SingleR)
library(SingleCellExperiment)
library(celldex)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define data paths (modify these paths to your data location)
DATA_PATHS <- list(
  # H5 format files (kN1-kN21)
  h5_base_path = "/Users/kuchia/Desktop/Kidney-normal-dataset1/",
  h5_samples = 1:21,
  
  # 10X format files (kN22-kN24)
  tenx_base_path = "/Users/kuchia/Desktop/kidney-normal-liao-dataset2/GSE131685_RAW/",
  tenx_samples = c("kidney-liao1", "kidney-liao2", "kidney-liao3")
)

# QC filtering parameters (from original workflow)
QC_PARAMS <- list(
  min_count_rna = 500,
  min_feature_rna = 250,
  min_log10_genes_per_umi = 0.85,
  min_cells_per_gene = 10
)

# Integration parameters (from original workflow)
INTEGRATION_PARAMS <- list(
  n_features = 2000,
  n_dims_pca = 50,
  n_dims_integration = 50,
  cluster_resolution = 0.1
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

#' Load single H5 sample data (matching original pattern)
#' @param sample_number Sample number (1-21)
#' @param base_path Base path to H5 files
#' @return Seurat object
load_h5_sample <- function(sample_number, base_path) {
  sample_name <- paste0("N", sample_number)
  data_path <- file.path(base_path, paste0("kN", sample_number), "filtered_feature_bc_matrix.h5")
  
  cat(sprintf("Loading sample: %s from %s\n", sample_name, data_path))
  
  if (!file.exists(data_path)) {
    cat(sprintf("Warning: File not found: %s\n", data_path))
    return(NULL)
  }
  
  rawcounts <- Read10X_h5(data_path, use.names = TRUE, unique.features = TRUE)
  seurat_obj <- CreateSeuratObject(counts = rawcounts, min.cells = 0, min.features = 0, project = sample_name)
  rm(rawcounts)
  gc()
  
  return(seurat_obj)
}

#' Load single 10X sample data (matching original pattern)
#' @param sample_name Sample directory name
#' @param base_path Base path to 10X directories  
#' @param project_name Project name for Seurat object
#' @return Seurat object
load_10x_sample <- function(sample_name, base_path, project_name) {
  data_path <- file.path(base_path, sample_name)
  
  cat(sprintf("Loading sample: %s from %s\n", project_name, data_path))
  
  if (!dir.exists(data_path)) {
    cat(sprintf("Warning: Directory not found: %s\n", data_path))
    return(NULL)
  }
  
  rawcounts <- Read10X(data_path, unique.features = TRUE)
  seurat_obj <- CreateSeuratObject(counts = rawcounts, min.cells = 0, min.features = 0, project = project_name)
  rm(rawcounts)
  gc()
  
  return(seurat_obj)
}

#' Merge Seurat objects in batches (following original pattern)
#' @param seurat_list List of Seurat objects
#' @param batch_size Number of objects to merge at once
#' @return Merged Seurat object
merge_seurat_objects <- function(seurat_list, batch_size = 5) {
  cat("Merging Seurat objects in batches...\n")
  
  # Remove any NULL objects
  seurat_list <- seurat_list[!sapply(seurat_list, is.null)]
  
  if (length(seurat_list) == 0) {
    stop("No valid Seurat objects to merge")
  }
  
  if (length(seurat_list) == 1) {
    return(seurat_list[[1]])
  }
  
  # Create batches
  merged_batches <- list()
  
  for (i in seq(1, length(seurat_list), batch_size)) {
    end_idx <- min(i + batch_size - 1, length(seurat_list))
    batch_objects <- seurat_list[i:end_idx]
    
    if (length(batch_objects) > 1) {
      merged_batches[[length(merged_batches) + 1]] <- merge(
        x = batch_objects[[1]], 
        y = batch_objects[-1]
      )
    } else {
      merged_batches[[length(merged_batches) + 1]] <- batch_objects[[1]]
    }
    
    cat(sprintf("Merged batch %d\n", length(merged_batches)))
  }
  
  # Final merge
  if (length(merged_batches) > 1) {
    final_merged <- merge(x = merged_batches[[1]], y = merged_batches[-1])
  } else {
    final_merged <- merged_batches[[1]]
  }
  
  return(final_merged)
}

#' Perform QC filtering (matching original method exactly)
#' @param seurat_obj Seurat object
#' @param qc_params List of QC parameters
#' @return Filtered Seurat object
perform_qc_filtering <- function(seurat_obj, qc_params) {
  cat("Performing quality control filtering...\n")
  
  # Calculate QC metrics (matching original)
  seurat_obj$log10GenesPerUMI <- log10(seurat_obj$nFeature_RNA) / log10(seurat_obj$nCount_RNA)
  seurat_obj$GenesPerUMI <- seurat_obj$nFeature_RNA / seurat_obj$nCount_RNA
  
  cat(sprintf("Before filtering: %d cells, %d genes\n", ncol(seurat_obj), nrow(seurat_obj)))
  
  # Apply filtering (exactly as in original)
  filtered_seurat <- subset(
    x = seurat_obj,
    subset = (nCount_RNA >= qc_params$min_count_rna) & 
             (nFeature_RNA >= qc_params$min_feature_rna) & 
             (log10GenesPerUMI > qc_params$min_log10_genes_per_umi)
  )
  
  # Filter genes (exactly as in original)
  counts <- GetAssayData(object = filtered_seurat, slot = "counts")
  nonzero <- counts > 0
  keep_genes <- Matrix::rowSums(nonzero) >= qc_params$min_cells_per_gene
  filtered_counts <- counts[keep_genes, ]
  
  filtered_seurat <- CreateSeuratObject(filtered_counts, meta.data = filtered_seurat@meta.data)
  
  cat(sprintf("After filtering: %d cells, %d genes\n", ncol(filtered_seurat), nrow(filtered_seurat)))
  
  return(filtered_seurat)
}

#' Create QC visualization plots
#' @param seurat_obj Seurat object with metadata
#' @param plot_prefix Prefix for saved plot files
create_qc_plots <- function(seurat_obj, plot_prefix = "QC") {
  cat("Creating QC plots...\n")
  
  metadata <- seurat_obj@meta.data
  metadata$cells <- rownames(metadata)
  seurat_obj$log10GenesPerUMI <- log10(metadata$nFeature_RNA) / log10(metadata$nCount_RNA)
  # Cell counts per sample
  p1 <- metadata %>% 
    ggplot(aes(x = orig.ident, fill = orig.ident)) + 
    geom_bar() +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
    ggtitle("Number of Cells per Sample") +
    geom_hline(yintercept = 2000, linetype = "dashed", color = "red")
  
  # Gene distribution
  p2 <- metadata %>%
    ggplot(aes(color = orig.ident, x = nFeature_RNA, fill = orig.ident)) +
    geom_density(alpha = 0.2) +
    theme_classic() +
    scale_x_log10() +
    geom_vline(xintercept = 300, linetype = "dashed", color = "red") +
    ggtitle("Genes Detected per Cell")
  
  # Gene complexity
  p3 <- metadata %>%
    ggplot(aes(x = seurat_obj$log10GenesPerUMI, color = orig.ident, fill = orig.ident)) +
    geom_density(alpha = 0.2) +
    theme_classic() +
    geom_vline(xintercept = 0.85, linetype = "dashed", color = "red") +
    ggtitle("Gene Expression Complexity")
  
  # Save plots
  ggsave(paste0(plot_prefix, "_cells_per_sample.png"), p1, width = 12, height = 8, bg = "white")
  ggsave(paste0(plot_prefix, "_genes_per_cell.png"), p2, width = 10, height = 6, bg = "white")
  ggsave(paste0(plot_prefix, "_gene_complexity.png"), p3, width = 10, height = 6, bg = "white")
  
  return(list(p1, p2, p3))
}

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

main_workflow <- function() {
  
  cat("=== KIDNEY SINGLE-CELL RNA-SEQ PROCESSING WORKFLOW ===\n\n")
  start_time <- Sys.time()
  
  # -------------------------------------------------------------------------
  # STEP 1: DATA LOADING 
  # -------------------------------------------------------------------------
  
  cat("STEP 1: Loading data...\n")
  
  seurat_objects <- list()
  
  # Load H5 format samples (kN1-kN21 -> N1-N21)
  for (i in DATA_PATHS$h5_samples) {
    seurat_obj <- load_h5_sample(i, DATA_PATHS$h5_base_path)
    if (!is.null(seurat_obj)) {
      seurat_objects[[paste0("N", i)]] <- seurat_obj
    }
  }
  
  # Load 10X format samples (kN22-kN24 -> N22-N24)
  for (i in 1:length(DATA_PATHS$tenx_samples)) {
    sample_name <- DATA_PATHS$tenx_samples[i]
    project_name <- paste0("N", length(DATA_PATHS$h5_samples) + i)
    seurat_obj <- load_10x_sample(sample_name, DATA_PATHS$tenx_base_path, project_name)
    if (!is.null(seurat_obj)) {
      seurat_objects[[project_name]] <- seurat_obj
    }
  }
  
  cat(sprintf("Successfully loaded %d samples\n", length(seurat_objects)))
  
  # -------------------------------------------------------------------------
  # STEP 2: MERGING (following original pattern exactly)
  # -------------------------------------------------------------------------
  
  cat("\nSTEP 2: Merging datasets...\n")
  
  merged_seurat <- merge_seurat_objects(seurat_objects, batch_size = 5)
  
  cat(sprintf("Successfully merged data\n"))
  cat(sprintf("Total cells: %d, Total genes: %d\n", ncol(merged_seurat), nrow(merged_seurat)))
  
  # Clean up
  rm(seurat_objects)
  gc()
  
  # -------------------------------------------------------------------------
  # STEP 3: QUALITY CONTROL AND FILTERING
  # -------------------------------------------------------------------------
  
  cat("\nSTEP 3: Quality control and filtering...\n")
  
  # Create pre-filtering QC plots
  create_qc_plots(merged_seurat, "pre_filtering")
  
  # Apply filtering
  filtered_seurat <- perform_qc_filtering(merged_seurat, QC_PARAMS)
  
  # Create post-filtering QC plots  
  create_qc_plots(filtered_seurat, "post_filtering")
  
  # Clean up
  rm(merged_seurat)
  gc()
  
  # -------------------------------------------------------------------------
  # STEP 4: INTEGRATION (following original workflow exactly)
  # -------------------------------------------------------------------------
  
  cat("\nSTEP 4: Data integration...\n")
  
  # Split object by sample
  split_seurat <- SplitObject(filtered_seurat, split.by = "orig.ident")
  
  # Process each sample (exactly as in original)
  split_seurat <- lapply(X = split_seurat, FUN = function(x) {
    x <- NormalizeData(x, verbose = FALSE)
    x <- FindVariableFeatures(x, verbose = FALSE)
    return(x)
  })
  
  # Select integration features
  features <- SelectIntegrationFeatures(object.list = split_seurat)
  
  # Scale and run PCA (exactly as in original)
  split_seurat <- lapply(X = split_seurat, FUN = function(x) {
    x <- ScaleData(x, features = features, verbose = FALSE)
    x <- RunPCA(x, features = features, verbose = FALSE)
    return(x)
  })
  
  # Find integration anchors (exactly as in original)
  anchors <- FindIntegrationAnchors(object.list = split_seurat, reduction = "rpca", dims = 1:50)
  
  # Integrate data (exactly as in original)
  seurat_integrated <- IntegrateData(anchorset = anchors, dims = 1:50)
  
  # Scale integrated data
  seurat_integrated <- ScaleData(seurat_integrated, verbose = FALSE)
  seurat_integrated <- RunPCA(seurat_integrated, verbose = FALSE)
  seurat_integrated <- RunUMAP(seurat_integrated, dims = 1:50)
  
  # Clean up
  rm(split_seurat, anchors)
  gc()
  
  # -------------------------------------------------------------------------
  # STEP 5: CLUSTERING (following original parameters)
  # -------------------------------------------------------------------------
  
  cat("\nSTEP 5: Clustering...\n")
  
  # Find neighbors and clusters (using original parameters)
  seurat_integrated <- FindNeighbors(object = seurat_integrated, dims = 1:40)
  seurat_integrated <- FindClusters(object = seurat_integrated, resolution = INTEGRATION_PARAMS$cluster_resolution)
  
  # Set active identity (exactly as in original)
  Idents(object = seurat_integrated) <- paste0("integrated_snn_res.", INTEGRATION_PARAMS$cluster_resolution)
  
  # Create integration plots
  p1 <- DimPlot(seurat_integrated, reduction = "umap", label = TRUE, label.size = 4, raster = FALSE)
  p2 <- DimPlot(seurat_integrated, reduction = "umap", group.by = "orig.ident", raster = FALSE)
  p3 <- DimPlot(seurat_integrated, reduction = "umap", split.by = "orig.ident", ncol = 4, raster = FALSE)
  
  ggsave("integration_clusters.png", p1, width = 10, height = 8, bg = "white")
  ggsave("integration_samples.png", p2, width = 12, height = 8, bg = "white")
  ggsave("integration_samples_split.png", p3, width = 20, height = 15, bg = "white")
  
  # -------------------------------------------------------------------------
  # STEP 6: MARKER IDENTIFICATION
  # -------------------------------------------------------------------------
  
  cat("\nSTEP 6: Finding cluster markers...\n")
  
  # Switch to RNA assay for marker finding (exactly as in original)
  DefaultAssay(seurat_integrated) <- "RNA"
  seurat_integrated <- NormalizeData(seurat_integrated, normalization.method = "LogNormalize", verbose = FALSE)
  
  # Find all markers (exactly as in original)
  seurat_integrated_markers <- FindAllMarkers(seurat_integrated, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25, verbose = FALSE)
  seurat_integrated_top <- seurat_integrated_markers %>% dplyr::group_by(cluster) %>% dplyr::top_n(n = 20)
  
  # Save markers
  write.csv(seurat_integrated_markers, "all_cluster_markers.csv", row.names = FALSE)
  write.csv(seurat_integrated_top, "top_cluster_markers.csv", row.names = FALSE)
  
  # -------------------------------------------------------------------------
  # STEP 7: BASIC SINGLER ANNOTATION
  # -------------------------------------------------------------------------
  
  cat("\nSTEP 7: Basic SingleR annotation with HPCA...\n")
  
  # Load HPCA reference (exactly as in original)
  hpca.ref <- celldex::HumanPrimaryCellAtlasData()
  
  # Convert to SingleCellExperiment and run SingleR (exactly as in original)
  sce <- as.SingleCellExperiment(seurat_integrated)
  hpca.main <- SingleR(test = sce, assay.type.test = 1, ref = hpca.ref, labels = hpca.ref$label.main)
  
  # Add annotations to Seurat object
  seurat_integrated@meta.data$hpca.main <- hpca.main$pruned.labels
  
  # Create annotation plots
  seurat_integrated <- SetIdent(seurat_integrated, value = "hpca.main")
  p4 <- DimPlot(seurat_integrated, label = TRUE, label.size = 3, raster = FALSE) + NoLegend()
  p5 <- DimPlot(seurat_integrated, label = FALSE, label.size = 3, raster = FALSE)
  
  ggsave("hpca_annotation_labeled.png", p4, width = 10, height = 8, bg = "white")
  ggsave("hpca_annotation_legend.png", p5, width = 12, height = 8, bg = "white")
  
  # Print annotation summary
  cat("\nHPCA annotation summary:\n")
  print(table(hpca.main$pruned.labels))
  
  # -------------------------------------------------------------------------
  # STEP 8: SAVE RESULTS
  # -------------------------------------------------------------------------
  
  cat("\nSTEP 8: Saving results...\n")
  
  # Save integrated object
  saveRDS(seurat_integrated, "kidney_integrated_processed.rds")
  
  # Create summary report
  summary_stats <- data.frame(
    Metric = c("Total Samples", "Total Cells (final)", "Total Genes (final)", 
               "Number of Clusters", "Processing Time (minutes)"),
    Value = c(
      length(unique(seurat_integrated$orig.ident)),
      ncol(seurat_integrated),
      nrow(seurat_integrated),
      length(unique(Idents(seurat_integrated))),
      round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 2)
    )
  )
  
  write.csv(summary_stats, "processing_summary.csv", row.names = FALSE)
  
  # Print summary
  cat("\n=== MAIN PROCESSING COMPLETE ===\n")
  print(summary_stats)
  cat(sprintf("\nFiles saved:\n"))
  cat("- kidney_integrated_processed.rds (main Seurat object)\n")
  cat("- all_cluster_markers.csv (all cluster markers)\n")
  cat("- top_cluster_markers.csv (top 20 markers per cluster)\n")
  cat("- processing_summary.csv (summary statistics)\n")
  cat("- Various QC and visualization plots\n")
  cat("\nTo perform detailed cell type annotation with Kidney Cell Atlas, run the annotation workflow script.\n")
  
  return(seurat_integrated)
}

# =============================================================================
# RUN WORKFLOW
# =============================================================================

# Execute the main workflow
if (!interactive()) {
  # Run automatically if script is sourced non-interactively
  result <- main_workflow()
} else {
  # Manual execution
  cat("To run the workflow, execute: result <- main_workflow()\n")
  cat("Make sure to update the DATA_PATHS configuration at the top of this script\n")
}
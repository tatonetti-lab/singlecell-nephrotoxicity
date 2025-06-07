"""
Constants and mappings used throughout the nephrotoxicity analysis pipeline.
"""

# Cell type color mapping for visualizations
CELL_TYPE_COLORS = {
    'Nephron': 'rgb(235, 110, 100)',
    'Endothelium': 'rgb(140, 180, 60)', 
    'Stroma': 'rgb(70, 170, 190)',
    'Immune': 'rgb(160, 110, 230)'
}

# RGB tuples for matplotlib
CELL_TYPE_COLORS_MPL = {
    'Nephron': (235/255, 110/255, 100/255),
    'Endothelium': (140/255, 180/255, 60/255),
    'Stroma': (70/255, 170/255, 190/255),
    'Immune': (160/255, 110/255, 230/255)
}

# Cell type mapping from detailed to broad categories
CELL_TYPE_MAPPING = {
    "Ascending vasa recta endothelium": "Endothelium",
    "Descending vasa recta endothelium": "Endothelium",
    "Glomerular endothelium": "Endothelium",
    "Peritubular capillary endothelium 1": "Endothelium",
    "Peritubular capillary endothelium 2": "Endothelium",
    "Myofibroblast": "Stroma",
    "Fibroblast": "Stroma",
    "Connecting tubule": "Nephron",
    "Distinct proximal tubule 1": "Nephron",
    "Distinct proximal tubule 2": "Nephron",
    "Epithelial progenitor cell": "Nephron",
    "Indistinct intercalated cell": "Nephron",
    "Pelvic epithelium": "Nephron",
    "Podocyte": "Nephron",
    "Principal cell": "Nephron",
    "Proliferating Proximal Tubule": "Nephron",
    "Proximal tubule": "Nephron",
    "Thick ascending limb of Loop of Henle": "Nephron",
    "Type A intercalated cell": "Nephron",
    "Type B intercalated cell": "Nephron",
    "Transitional Urothelium": "Nephron",
    "B cell": "Immune",
    "CD4 T cell": "Immune",
    "CD8 T cell": "Immune",
    "Mast cell": "Immune",
    "MNP-d/Tissue macrophage": "Immune",
    "MNP-c/dendritic cell": "Immune",
    "MNP-b/non-classical monocyte derived": "Immune",
    "MNP-a/classical monocyte derived": "Immune",
    "Neutrophil": "Immune",
    "NK cell": "Immune",
    "NKT cell": "Immune",
    "Plasmacytoid dendritic cell": "Immune"
}

# Kidney cell line IDs for LINCS analysis
KIDNEY_CELL_IDS = ['HA1E', 'HA1E.101', 'HA1E.311', 'HEK293T', 'HEKTE', 'NKDBA']

# Statistical significance thresholds
SIGNIFICANCE_THRESHOLDS = {
    'fdr': 0.05,
    'log2fc': 1.0,
    'p_value': 0.0000025,  # For simulation analysis
    'log_fc_threshold': 2.0  # For simulation analysis
}

# File extensions and formats
SUPPORTED_IMAGE_FORMATS = ['.png', '.svg', '.pdf']
SUPPORTED_DATA_FORMATS = ['.csv', '.h5ad', '.gctx', '.txt']

# Default output subdirectories
OUTPUT_SUBDIRS = {
    'drug2cell': 'drug2cell_results',
    'statistical': 'statistical_results', 
    'ml_drug': 'drug_score_ml_results',
    'differential': 'differential_expression_results',
    'simulation': 'simulation_results'
}

# Toxicity column names for DIRIL dataset
TOXICITY_COLUMNS = ['Label_Gong', 'Label_Shi', 'my_findings']

# Target condition for toxicity analysis
TARGET_CONDITION = 'acute kidney injury'

# Cross-validation settings
CV_CONFIG = {
    'cv_folds': 5,
    'test_size': 0.2,
    'random_state': 42
}

# Plotting settings
PLOT_CONFIG = {
    'dpi': 300,
    'bbox_inches': 'tight',
    'figure_formats': ['png'],
    'publication_style': {
        'svg.fonttype': 'none',
        'font.family': 'Arial',
        'axes.linewidth': 0.5,
        'axes.labelsize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    }
}

# Data validation patterns
VALIDATION_PATTERNS = {
    'drug_name': r'^[a-zA-Z0-9\s\-_().,]+$',
    'gene_symbol': r'^[A-Z0-9\-_]+$',
    'cell_type': r'^[a-zA-Z0-9\s\-_/()]+$'
}

# Error messages
ERROR_MESSAGES = {
    'file_not_found': "Required file not found: {path}",
    'invalid_config': "Invalid configuration: {error}",
    'missing_data': "Missing required data: {data_type}",
    'analysis_failed': "Analysis failed in {component}: {error}",
    'insufficient_data': "Insufficient data for analysis: {details}"
}

"""
Helper utilities for the nephrotoxicity analysis pipeline.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
import warnings
from collections import Counter

from .logging import get_logger


def format_execution_time(start_time: pd.Timestamp, end_time: pd.Timestamp) -> str:
    """
    Format execution time for display.
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp
        
    Returns:
        Formatted time string
    """
    duration = end_time - start_time
    total_seconds = duration.total_seconds()
    
    if total_seconds < 60:
        return f"{total_seconds:.1f} seconds"
    elif total_seconds < 3600:
        minutes = total_seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = total_seconds / 3600
        return f"{hours:.1f} hours"


def validate_drug_name(drug_name: str) -> bool:
    """
    Validate drug name format.
    
    Args:
        drug_name: Drug name to validate
        
    Returns:
        True if valid
    """
    if not drug_name or not isinstance(drug_name, str):
        return False
    
    # Basic pattern: alphanumeric characters, spaces, hyphens, parentheses, periods, commas
    pattern = r'^[a-zA-Z0-9\s\-_().,]+$'
    return bool(re.match(pattern, drug_name.strip()))


def validate_gene_symbol(gene_symbol: str) -> bool:
    """
    Validate gene symbol format.
    
    Args:
        gene_symbol: Gene symbol to validate
        
    Returns:
        True if valid
    """
    if not gene_symbol or not isinstance(gene_symbol, str):
        return False
    
    # Gene symbols: uppercase alphanumeric with hyphens/underscores
    pattern = r'^[A-Z0-9\-_]+$'
    return bool(re.match(pattern, gene_symbol.strip()))


def clean_drug_names(df: pd.DataFrame, drug_col: str = 'drug_name') -> pd.DataFrame:
    """
    Clean drug names in a DataFrame.
    
    Args:
        df: DataFrame containing drug names
        drug_col: Name of the drug name column
        
    Returns:
        DataFrame with cleaned drug names
    """
    logger = get_logger(__name__)
    
    if drug_col not in df.columns:
        logger.warning(f"Drug column '{drug_col}' not found in DataFrame")
        return df
    
    df = df.copy()
    original_count = len(df)
    
    # Convert to string and strip whitespace
    df[drug_col] = df[drug_col].astype(str).str.strip()
    
    # Convert to lowercase for consistency
    df[drug_col] = df[drug_col].str.lower()
    
    # Remove entries that are just 'nan' or empty
    df = df[~df[drug_col].isin(['nan', '', 'none', 'null'])]
    
    cleaned_count = len(df)
    if cleaned_count < original_count:
        logger.info(f"Removed {original_count - cleaned_count} invalid drug names")
    
    return df


def clean_gene_symbols(gene_list: List[str]) -> List[str]:
    """
    Clean a list of gene symbols.
    
    Args:
        gene_list: List of gene symbols
        
    Returns:
        Cleaned list of gene symbols
    """
    if not gene_list:
        return []
    
    cleaned = []
    for gene in gene_list:
        if isinstance(gene, str):
            gene = gene.strip().upper()
            if validate_gene_symbol(gene):
                cleaned.append(gene)
    
    return list(set(cleaned))  # Remove duplicates


def extract_base_pattern(sig_id: str) -> Optional[str]:
    """
    Extract base pattern from signature ID (for LINCS data).
    
    Args:
        sig_id: Signature ID (e.g. 'REP.A008_HA1E_24H:E01' or 'REP.A008_HA1E_24H_X1_B24:E01')
        
    Returns:
        Base pattern (e.g. 'REP.A008_HA1E_24H') or None if invalid
    """
    if pd.isna(sig_id):
        return None
    
    # Remove _X\d+ and _B\d+ patterns, then take everything before the colon
    cleaned = re.sub(r'_X\d+', '', str(sig_id))
    cleaned = re.sub(r'_B\d+', '', cleaned)
    base = cleaned.split(':')[0]
    return base


def convert_to_numeric(df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    Convert DataFrame columns to numeric where possible.
    
    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude from conversion
        
    Returns:
        DataFrame with numeric columns converted
    """
    if exclude_cols is None:
        exclude_cols = []
    
    df = df.copy()
    
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        # Try to convert to numeric
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass
    
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median', 
                         fill_value: Any = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('median', 'mean', 'mode', 'fill', 'drop')
        fill_value: Value to use for 'fill' strategy
        
    Returns:
        DataFrame with missing values handled
    """
    logger = get_logger(__name__)
    df = df.copy()
    
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        return df
    
    logger.info(f"Handling {missing_count} missing values using strategy: {strategy}")
    
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'fill':
        df = df.fillna(fill_value)
    else:
        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                if strategy == 'median':
                    fill_val = df[col].median()
                elif strategy == 'mean':
                    fill_val = df[col].mean()
                else:
                    fill_val = 0
                df[col].fillna(fill_val, inplace=True)
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                if strategy == 'mode':
                    mode_val = df[col].mode()
                    fill_val = mode_val.iloc[0] if not mode_val.empty else 'unknown'
                else:
                    fill_val = 'unknown'
                df[col].fillna(fill_val, inplace=True)
    
    return df


def calculate_log_fold_change(baseline: np.ndarray, treatment: np.ndarray, 
                            epsilon: float = 1e-10) -> np.ndarray:
    """
    Calculate log2 fold change between baseline and treatment.
    
    Args:
        baseline: Baseline values
        treatment: Treatment values
        epsilon: Small value to avoid log(0)
        
    Returns:
        Log2 fold change values
    """
    baseline_mean = np.mean(baseline, axis=0) + epsilon
    treatment_mean = np.mean(treatment, axis=0) + epsilon
    
    return np.log2(treatment_mean / baseline_mean)


def create_toxicity_dictionary(drug_df: pd.DataFrame, 
                             drug_col: str = 'drug_name',
                             toxicity_col: str = 'is_toxic',
                             genes_col: str = 'combined_genes') -> Dict[str, Dict[str, List[str]]]:
    """
    Create toxicity dictionary for drug2cell analysis.
    
    Args:
        drug_df: DataFrame with drug information
        drug_col: Name of drug name column
        toxicity_col: Name of toxicity column
        genes_col: Name of genes column
        
    Returns:
        Dictionary with toxic and non-toxic drug categories
    """
    logger = get_logger(__name__)
    
    toxicity_dict = {'toxic': {}, 'non_toxic': {}}
    toxicity_counts = Counter()
    
    for _, row in drug_df.iterrows():
        # Determine toxicity status
        is_toxic_value = str(row[toxicity_col]).lower()
        
        if is_toxic_value == 'false':
            toxicity = 'non_toxic'
        else:
            toxicity = 'toxic'
        
        toxicity_counts[is_toxic_value] += 1
        
        drug_name = row[drug_col]
        genes = []
        if pd.notna(row[genes_col]):
            genes = [g.strip() for g in str(row[genes_col]).split(',')]
        
        toxicity_dict[toxicity][drug_name] = genes
    
    logger.info(f"Created toxicity dictionary:")
    logger.info(f"  Toxic drugs: {len(toxicity_dict['toxic'])}")
    logger.info(f"  Non-toxic drugs: {len(toxicity_dict['non_toxic'])}")
    
    return toxicity_dict


def combine_gene_information(*gene_columns: pd.Series) -> pd.Series:
    """
    Combine gene information from multiple columns.
    
    Args:
        *gene_columns: Variable number of pandas Series containing gene information
        
    Returns:
        Series with combined gene information
    """
    def combine_genes(row_data):
        genes = set()
        for gene_data in row_data:
            if pd.notna(gene_data):
                gene_list = [gene.strip() for gene in str(gene_data).split(',')]
                genes.update(gene_list)
        return ','.join(sorted(genes)) if genes else np.nan
    
    # Create DataFrame from all gene columns
    gene_df = pd.DataFrame({f'genes_{i}': col for i, col in enumerate(gene_columns)})
    
    # Apply combination function row-wise
    return gene_df.apply(combine_genes, axis=1)


def filter_dataframe_by_values(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Filter DataFrame by column values.
    
    Args:
        df: Input DataFrame
        filters: Dictionary mapping column names to filter values
        
    Returns:
        Filtered DataFrame
    """
    logger = get_logger(__name__)
    filtered_df = df.copy()
    original_count = len(filtered_df)
    
    for column, filter_value in filters.items():
        if column not in filtered_df.columns:
            logger.warning(f"Filter column '{column}' not found in DataFrame")
            continue
        
        if isinstance(filter_value, list):
            # Filter by list of values
            filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
        else:
            # Filter by single value
            filtered_df = filtered_df[filtered_df[column] == filter_value]
        
        logger.debug(f"Applied filter {column}={filter_value}: {len(filtered_df)} rows remaining")
    
    filtered_count = len(filtered_df)
    logger.info(f"Filtering removed {original_count - filtered_count} rows")
    
    return filtered_df


def calculate_summary_statistics(df: pd.DataFrame, 
                               group_col: str = None) -> Dict[str, Any]:
    """
    Calculate summary statistics for a DataFrame.
    
    Args:
        df: Input DataFrame
        group_col: Optional column to group by
        
    Returns:
        Dictionary with summary statistics
    """
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats['numeric_columns'] = len(numeric_cols)
        stats['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Categorical column statistics
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        stats['categorical_columns'] = len(categorical_cols)
        stats['categorical_summary'] = {}
        for col in categorical_cols:
            stats['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_common': df[col].value_counts().head().to_dict()
            }
    
    # Group-specific statistics
    if group_col and group_col in df.columns:
        stats['group_counts'] = df[group_col].value_counts().to_dict()
    
    return stats


def validate_data_consistency(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """
    Validate consistency across multiple DataFrames.
    
    Args:
        data_dict: Dictionary mapping names to DataFrames
        
    Returns:
        Dictionary with validation issues
    """
    logger = get_logger(__name__)
    issues = {}
    
    # Check for common columns and overlapping data
    all_dataframes = list(data_dict.items())
    
    for i, (name1, df1) in enumerate(all_dataframes):
        issues[name1] = []
        
        # Check for empty DataFrames
        if df1.empty:
            issues[name1].append("DataFrame is empty")
            continue
        
        # Check for duplicate columns
        if df1.columns.duplicated().any():
            dup_cols = df1.columns[df1.columns.duplicated()].tolist()
            issues[name1].append(f"Duplicate columns: {dup_cols}")
        
        # Check data types consistency for common columns
        for j, (name2, df2) in enumerate(all_dataframes[i+1:], i+1):
            common_cols = set(df1.columns) & set(df2.columns)
            if common_cols:
                for col in common_cols:
                    if df1[col].dtype != df2[col].dtype:
                        issues[name1].append(
                            f"Column '{col}' has different dtype than in {name2}: "
                            f"{df1[col].dtype} vs {df2[col].dtype}"
                        )
    
    # Log issues
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    if total_issues > 0:
        logger.warning(f"Found {total_issues} data consistency issues")
        for name, issue_list in issues.items():
            if issue_list:
                logger.warning(f"{name}: {len(issue_list)} issues")
    else:
        logger.info("No data consistency issues found")
    
    return issues


def create_color_variations(base_color: Tuple[float, float, float], 
                          n_variations: int) -> List[str]:
    """
    Create color variations for subtypes within a group.
    
    Args:
        base_color: RGB tuple for base color (values 0-1)
        n_variations: Number of variations to create
        
    Returns:
        List of hex color strings
    """
    try:
        import seaborn as sns
        base_hex = f'#{int(base_color[0]*255):02x}{int(base_color[1]*255):02x}{int(base_color[2]*255):02x}'
        variations = sns.color_palette(f"light:{base_hex}", n_colors=n_variations)
        return [f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' for c in variations]
    except ImportError:
        # Fallback if seaborn not available
        return [f'#{int(base_color[0]*255):02x}{int(base_color[1]*255):02x}{int(base_color[2]*255):02x}'] * n_variations


def safe_division(numerator: Union[float, np.ndarray], 
                 denominator: Union[float, np.ndarray], 
                 default_value: float = 0.0) -> Union[float, np.ndarray]:
    """
    Perform safe division avoiding division by zero.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default_value: Value to return when denominator is zero
        
    Returns:
        Division result with safe handling of zero denominator
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        
        if isinstance(denominator, np.ndarray):
            result = np.divide(numerator, denominator, 
                             out=np.full_like(denominator, default_value, dtype=float), 
                             where=(denominator != 0))
        else:
            result = numerator / denominator if denominator != 0 else default_value
        
        return result


def memory_usage_mb(obj: Any) -> float:
    """
    Calculate memory usage of an object in MB.
    
    Args:
        obj: Object to measure
        
    Returns:
        Memory usage in MB
    """
    import sys
    
    if hasattr(obj, 'memory_usage'):
        # DataFrame or Series
        return obj.memory_usage(deep=True).sum() / (1024 * 1024)
    else:
        return sys.getsizeof(obj) / (1024 * 1024)


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory-optimized DataFrame
    """
    logger = get_logger(__name__)
    original_memory = memory_usage_mb(df)
    
    df = df.copy()
    
    # Optimize integer columns
    int_cols = df.select_dtypes(include=['int']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Optimize float columns
    float_cols = df.select_dtypes(include=['float']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert string columns to category if they have few unique values
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    optimized_memory = memory_usage_mb(df)
    memory_saved = original_memory - optimized_memory
    
    if memory_saved > 0:
        logger.info(f"Memory optimization saved {memory_saved:.2f} MB "
                   f"({memory_saved/original_memory*100:.1f}% reduction)")
    
    return df


def create_output_filename(base_name: str, component: str, 
                          extension: str = '.csv', 
                          timestamp: bool = False) -> str:
    """
    Create standardized output filename.
    
    Args:
        base_name: Base name for the file
        component: Component name
        extension: File extension
        timestamp: Whether to include timestamp
        
    Returns:
        Formatted filename
    """
    filename = f"{component}_{base_name}"
    
    if timestamp:
        timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp_str}"
    
    return f"{filename}{extension}"


def batch_process_dataframe(df: pd.DataFrame, process_func, 
                          batch_size: int = 1000, **kwargs) -> pd.DataFrame:
    """
    Process DataFrame in batches to manage memory usage.
    
    Args:
        df: Input DataFrame
        process_func: Function to apply to each batch
        batch_size: Size of each batch
        **kwargs: Additional arguments for process_func
        
    Returns:
        Processed DataFrame
    """
    logger = get_logger(__name__)
    
    if len(df) <= batch_size:
        return process_func(df, **kwargs)
    
    logger.info(f"Processing DataFrame in batches of {batch_size}")
    
    processed_batches = []
    n_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        processed_batch = process_func(batch, **kwargs)
        processed_batches.append(processed_batch)
        
        batch_num = (i // batch_size) + 1
        logger.debug(f"Processed batch {batch_num}/{n_batches}")
    
    return pd.concat(processed_batches, ignore_index=True)

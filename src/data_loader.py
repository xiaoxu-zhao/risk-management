"""
Data Loading and Preprocessing Module
====================================

Handles loading and initial preprocessing of credit risk datasets from Kaggle
and other sources. Supports multiple dataset formats commonly used in credit risk.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditDataLoader:
    """
    A comprehensive data loader for credit risk datasets.
    
    Supports popular Kaggle datasets:
    - Give Me Some Credit
    - Home Credit Default Risk
    - Lending Club
    - German Credit Data
    """
    
    def __init__(self, data_path: str = "data/"):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = data_path
        self.datasets = {}
        self.metadata = {}
        
    def load_give_me_credit(self, filepath: str) -> pd.DataFrame:
        """
        Load the 'Give Me Some Credit' Kaggle dataset.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Loaded and initially processed DataFrame
        """
        try:
            df = pd.read_csv(filepath)
            
            # Basic cleaning
            df = df.drop('Unnamed: 0', axis=1, errors='ignore')
            
            # Store metadata
            self.metadata['give_me_credit'] = {
                'target': 'SeriousDlqin2yrs',
                'features': list(df.columns.drop('SeriousDlqin2yrs', errors='ignore')),
                'shape': df.shape,
                'missing_values': df.isnull().sum().sum()
            }
            
            logger.info(f"Loaded Give Me Credit dataset: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Give Me Credit dataset: {e}")
            raise
            
    def load_home_credit(self, application_path: str, 
                        bureau_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load Home Credit Default Risk dataset.
        
        Args:
            application_path: Path to application_train.csv
            bureau_path: Optional path to bureau.csv for additional features
            
        Returns:
            Loaded and processed DataFrame
        """
        try:
            # Load main application data
            df = pd.read_csv(application_path)
            
            # Load bureau data if provided
            if bureau_path and os.path.exists(bureau_path):
                bureau = pd.read_csv(bureau_path)
                # Aggregate bureau data by SK_ID_CURR
                bureau_agg = self._aggregate_bureau_data(bureau)
                df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
            
            self.metadata['home_credit'] = {
                'target': 'TARGET',
                'features': list(df.columns.drop('TARGET', errors='ignore')),
                'shape': df.shape,
                'missing_values': df.isnull().sum().sum()
            }
            
            logger.info(f"Loaded Home Credit dataset: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Home Credit dataset: {e}")
            raise
            
    def load_german_credit(self, filepath: str) -> pd.DataFrame:
        """
        Load German Credit dataset.
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            Loaded and processed DataFrame
        """
        try:
            df = pd.read_csv(filepath)
            
            # Standardize target variable (assuming 1=good, 2=bad)
            if 'class' in df.columns:
                df['default'] = (df['class'] == 2).astype(int)
                df = df.drop('class', axis=1)
            
            self.metadata['german_credit'] = {
                'target': 'default',
                'features': list(df.columns.drop('default', errors='ignore')),
                'shape': df.shape,
                'missing_values': df.isnull().sum().sum()
            }
            
            logger.info(f"Loaded German Credit dataset: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading German Credit dataset: {e}")
            raise
            
    def _aggregate_bureau_data(self, bureau: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate bureau credit data by customer ID.
        
        Args:
            bureau: Bureau DataFrame
            
        Returns:
            Aggregated bureau features
        """
        # Define aggregation functions
        agg_funcs = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum', 'mean'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum']
        }
        
        # Aggregate numerical features
        bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_funcs)
        bureau_agg.columns = ['BUREAU_' + '_'.join(col) for col in bureau_agg.columns]
        
        # Add count of bureau records
        bureau_counts = bureau.groupby('SK_ID_CURR').size().to_frame('BUREAU_COUNT')
        
        # Combine aggregations
        bureau_agg = bureau_agg.join(bureau_counts)
        bureau_agg = bureau_agg.reset_index()
        
        return bureau_agg
        
    def basic_data_quality_check(self, df: pd.DataFrame, 
                                target_col: str) -> Dict:
        """
        Perform basic data quality checks.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Dictionary with quality check results
        """
        quality_report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'target_distribution': df[target_col].value_counts().to_dict(),
            'target_rate': df[target_col].mean(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Check for high cardinality categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        high_cardinality = {}
        for col in categorical_cols:
            unique_values = df[col].nunique()
            if unique_values > 50:  # Threshold for high cardinality
                high_cardinality[col] = unique_values
        
        quality_report['high_cardinality_features'] = high_cardinality
        
        return quality_report
        
    def clean_data(self, df: pd.DataFrame, 
                   target_col: str,
                   missing_threshold: float = 0.5,
                   remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Perform basic data cleaning.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            missing_threshold: Drop columns with missing % above this threshold
            remove_duplicates: Whether to remove duplicate rows
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        initial_shape = df_clean.shape
        
        # Remove duplicates
        if remove_duplicates:
            df_clean = df_clean.drop_duplicates()
            logger.info(f"Removed {initial_shape[0] - df_clean.shape[0]} duplicate rows")
        
        # Remove columns with high missing values
        missing_pct = df_clean.isnull().sum() / len(df_clean)
        cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
        
        if cols_to_drop:
            df_clean = df_clean.drop(cols_to_drop, axis=1)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >{missing_threshold*100}% missing values")
        
        # Remove rows where target is missing
        if target_col in df_clean.columns:
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=[target_col])
            logger.info(f"Removed {initial_rows - len(df_clean)} rows with missing target")
        
        logger.info(f"Data cleaning complete: {initial_shape} -> {df_clean.shape}")
        
        return df_clean
        
    def detect_outliers(self, df: pd.DataFrame, 
                       numeric_cols: List[str] = None,
                       method: str = 'iqr',
                       factor: float = 1.5) -> Dict[str, Dict]:
        """
        Detect outliers in numeric columns using various methods.
        
        Args:
            df: Input DataFrame
            numeric_cols: List of numeric columns to check (None for all numeric)
            method: Method to use ('iqr', 'zscore', 'modified_zscore')
            factor: Factor for outlier detection (1.5 for IQR, 3.0 for z-score)
            
        Returns:
            Dictionary with outlier information for each column
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_info = {}
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
                
            col_data = df[col].dropna()
            
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                
            elif method == 'zscore':
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                outliers = df[np.abs((df[col] - df[col].mean()) / df[col].std()) > factor].index
                
            elif method == 'modified_zscore':
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                modified_z_scores = 0.6745 * (col_data - median) / mad
                outliers = df[np.abs(0.6745 * (df[col] - median) / mad) > factor].index
                
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            outlier_info[col] = {
                'method': method,
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(df) * 100,
                'outlier_indices': outliers.tolist()
            }
            
            if method == 'iqr':
                outlier_info[col].update({
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'IQR': IQR
                })
        
        logger.info(f"Outlier detection completed using {method} method")
        return outlier_info
    
    def handle_outliers(self, df: pd.DataFrame,
                       outlier_info: Dict[str, Dict] = None,
                       treatment: str = 'cap',
                       numeric_cols: List[str] = None) -> pd.DataFrame:
        """
        Handle outliers in the dataset.
        
        Args:
            df: Input DataFrame
            outlier_info: Pre-computed outlier information
            treatment: Treatment method ('cap', 'remove', 'transform')
            numeric_cols: Columns to process (None for all numeric)
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        if outlier_info is None:
            outlier_info = self.detect_outliers(df, numeric_cols)
        
        for col, info in outlier_info.items():
            if col not in df_clean.columns:
                continue
                
            if treatment == 'cap' and 'lower_bound' in info and 'upper_bound' in info:
                # Cap outliers to bounds
                df_clean[col] = df_clean[col].clip(info['lower_bound'], info['upper_bound'])
                
            elif treatment == 'remove':
                # Remove rows with outliers
                outlier_indices = info['outlier_indices']
                df_clean = df_clean.drop(outlier_indices)
                
            elif treatment == 'transform':
                # Log transform for positive skewed data
                if df_clean[col].min() > 0:
                    df_clean[f'{col}_log'] = np.log1p(df_clean[col])
                
        logger.info(f"Outlier treatment completed using {treatment} method")
        return df_clean
    
    def normalize_features(self, df: pd.DataFrame,
                          numeric_cols: List[str] = None,
                          method: str = 'standard',
                          store_params: bool = True) -> pd.DataFrame:
        """
        Normalize numeric features using various methods.
        
        Args:
            df: Input DataFrame
            numeric_cols: Columns to normalize (None for all numeric)
            method: Normalization method ('standard', 'minmax', 'robust')
            store_params: Whether to store normalization parameters
            
        Returns:
            DataFrame with normalized features
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_norm = df.copy()
        
        if store_params and 'normalization_params' not in self.metadata:
            self.metadata['normalization_params'] = {}
        
        for col in numeric_cols:
            if col not in df_norm.columns:
                continue
                
            col_data = df_norm[col].dropna()
            
            if method == 'standard':
                # Z-score normalization
                mean_val = col_data.mean()
                std_val = col_data.std()
                df_norm[col] = (df_norm[col] - mean_val) / std_val
                
                if store_params:
                    self.metadata['normalization_params'][col] = {
                        'method': 'standard',
                        'mean': mean_val,
                        'std': std_val
                    }
                    
            elif method == 'minmax':
                # Min-max normalization
                min_val = col_data.min()
                max_val = col_data.max()
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
                
                if store_params:
                    self.metadata['normalization_params'][col] = {
                        'method': 'minmax',
                        'min': min_val,
                        'max': max_val
                    }
                    
            elif method == 'robust':
                # Robust scaling using median and IQR
                median_val = col_data.median()
                q75 = col_data.quantile(0.75)
                q25 = col_data.quantile(0.25)
                iqr_val = q75 - q25
                df_norm[col] = (df_norm[col] - median_val) / iqr_val
                
                if store_params:
                    self.metadata['normalization_params'][col] = {
                        'method': 'robust',
                        'median': median_val,
                        'iqr': iqr_val
                    }
        
        logger.info(f"Feature normalization completed using {method} method")
        return df_norm
    
    def aggregate_data_sources(self, data_sources: Dict[str, pd.DataFrame],
                              join_key: str,
                              aggregation_funcs: Dict[str, Dict] = None) -> pd.DataFrame:
        """
        Aggregate multiple data sources into a unified dataset.
        
        Args:
            data_sources: Dictionary of {source_name: DataFrame}
            join_key: Column name to join datasets on
            aggregation_funcs: Custom aggregation functions for each source
            
        Returns:
            Unified aggregated DataFrame
        """
        if not data_sources:
            raise ValueError("No data sources provided")
        
        # Start with the first dataset as base
        source_names = list(data_sources.keys())
        main_df = data_sources[source_names[0]].copy()
        
        logger.info(f"Starting aggregation with base dataset: {source_names[0]} ({main_df.shape})")
        
        # Join additional datasets
        for source_name in source_names[1:]:
            source_df = data_sources[source_name].copy()
            
            # Apply custom aggregation if specified
            if aggregation_funcs and source_name in aggregation_funcs:
                agg_funcs = aggregation_funcs[source_name]
                
                # Group by join key and apply aggregations
                agg_df = source_df.groupby(join_key).agg(agg_funcs)
                
                # Flatten column names
                agg_df.columns = [f"{source_name}_{col[0]}_{col[1]}" if isinstance(col, tuple) 
                                else f"{source_name}_{col}" for col in agg_df.columns]
                agg_df = agg_df.reset_index()
            else:
                # Simple merge without aggregation
                agg_df = source_df
            
            # Merge with main dataset
            main_df = main_df.merge(agg_df, on=join_key, how='left', suffixes=('', f'_{source_name}'))
            
            logger.info(f"Merged {source_name}: {main_df.shape}")
        
        # Store aggregation metadata
        self.metadata['data_aggregation'] = {
            'sources': list(data_sources.keys()),
            'join_key': join_key,
            'final_shape': main_df.shape,
            'aggregation_funcs': aggregation_funcs
        }
        
        logger.info(f"Data aggregation completed. Final shape: {main_df.shape}")
        return main_df

    def get_sample_data(self, n_samples: int = 1000, 
                       random_state: int = 42) -> pd.DataFrame:
        """
        Generate sample credit data for demonstration purposes.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
            
        Returns:
            Sample credit dataset
        """
        np.random.seed(random_state)
        
        # Generate synthetic credit data
        data = {
            'age': np.random.normal(40, 12, n_samples).clip(18, 80).astype(int),
            'income': np.random.lognormal(10.5, 0.5, n_samples).clip(20000, 500000),
            'debt_to_income': np.random.beta(2, 5, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850).astype(int),
            'employment_years': np.random.exponential(5, n_samples).clip(0, 40),
            'num_credit_lines': np.random.poisson(3, n_samples),
            'credit_utilization': np.random.beta(2, 3, n_samples),
            'num_late_payments': np.random.poisson(1, n_samples),
            'loan_amount': np.random.lognormal(9.5, 0.7, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic default probability based on features
        logit = (-1.5 +  # Further increased to get realistic default rate (~10-15%)
                -0.01 * df['age'] +
                -0.000005 * df['income'] +
                1.5 * df['debt_to_income'] +
                -0.003 * df['credit_score'] +
                -0.03 * df['employment_years'] +
                1.0 * df['credit_utilization'] +
                0.15 * df['num_late_payments'])
        
        prob_default = 1 / (1 + np.exp(-logit))
        df['default'] = np.random.binomial(1, prob_default)
        
        self.metadata['sample_data'] = {
            'target': 'default',
            'features': list(df.columns.drop('default')),
            'shape': df.shape,
            'missing_values': 0
        }
        
        logger.info(f"Generated sample dataset: {df.shape}")
        return df
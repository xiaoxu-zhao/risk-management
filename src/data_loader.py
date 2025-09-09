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
"""
Tests for data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import CreditDataLoader


class TestCreditDataLoader:
    """Test cases for CreditDataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = CreditDataLoader()
        
    def test_initialization(self):
        """Test CreditDataLoader initialization."""
        assert self.loader.data_path == "data/"
        assert self.loader.datasets == {}
        assert self.loader.metadata == {}
        
    def test_get_sample_data(self):
        """Test sample data generation."""
        sample_data = self.loader.get_sample_data(n_samples=100, random_state=42)
        
        # Check data structure
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) == 100
        assert 'default' in sample_data.columns
        
        # Check data types
        assert sample_data['age'].dtype == 'int64'
        assert sample_data['default'].dtype == 'int64'
        assert sample_data['default'].isin([0, 1]).all()
        
        # Check metadata
        assert 'sample_data' in self.loader.metadata
        assert self.loader.metadata['sample_data']['target'] == 'default'
        
    def test_basic_data_quality_check(self):
        """Test basic data quality checking."""
        sample_data = self.loader.get_sample_data(n_samples=100)
        quality_report = self.loader.basic_data_quality_check(sample_data, 'default')
        
        # Check report structure
        required_keys = ['shape', 'missing_values', 'target_distribution', 'target_rate']
        for key in required_keys:
            assert key in quality_report
            
        # Check values
        assert quality_report['shape'] == (100, 10)  # 100 rows, 10 columns
        assert isinstance(quality_report['target_rate'], float)
        assert 0 <= quality_report['target_rate'] <= 1
        
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Create data with missing values and duplicates
        sample_data = self.loader.get_sample_data(n_samples=100)
        
        # Add missing values
        sample_data.loc[0:5, 'age'] = np.nan
        sample_data.loc[10:15, 'income'] = np.nan
        
        # Add duplicates
        sample_data = pd.concat([sample_data, sample_data.iloc[0:5]], ignore_index=True)
        
        # Clean data
        cleaned_data = self.loader.clean_data(
            sample_data, 'default', missing_threshold=0.5, remove_duplicates=True
        )
        
        # Check that duplicates were removed
        assert len(cleaned_data) < len(sample_data)
        assert not cleaned_data.duplicated().any()
        
        # Check that target missing values were removed
        assert not cleaned_data['default'].isnull().any()
        
    def test_clean_data_with_high_missing_threshold(self):
        """Test data cleaning with columns having high missing values."""
        sample_data = self.loader.get_sample_data(n_samples=100)
        
        # Add column with high missing values
        sample_data['high_missing_col'] = np.nan
        sample_data.loc[0:5, 'high_missing_col'] = 1
        
        cleaned_data = self.loader.clean_data(
            sample_data, 'default', missing_threshold=0.5
        )
        
        # Column with >50% missing should be dropped
        assert 'high_missing_col' not in cleaned_data.columns
        
    def test_sample_data_reproducibility(self):
        """Test that sample data generation is reproducible."""
        data1 = self.loader.get_sample_data(n_samples=50, random_state=42)
        
        # Create new loader instance
        loader2 = CreditDataLoader()
        data2 = loader2.get_sample_data(n_samples=50, random_state=42)
        
        # Should be identical
        pd.testing.assert_frame_equal(data1, data2)
        
    def test_sample_data_realistic_ranges(self):
        """Test that sample data has realistic value ranges."""
        sample_data = self.loader.get_sample_data(n_samples=1000)
        
        # Check age range
        assert sample_data['age'].min() >= 18
        assert sample_data['age'].max() <= 80
        
        # Check income range
        assert sample_data['income'].min() >= 20000
        assert sample_data['income'].max() <= 500000
        
        # Check debt_to_income range
        assert sample_data['debt_to_income'].min() >= 0
        assert sample_data['debt_to_income'].max() <= 1
        
        # Check credit_score range
        assert sample_data['credit_score'].min() >= 300
        assert sample_data['credit_score'].max() <= 850
        
    def test_quality_check_with_categorical_data(self):
        """Test quality check with categorical variables."""
        sample_data = self.loader.get_sample_data(n_samples=100)
        
        # Add categorical column with high cardinality
        sample_data['customer_id'] = [f'CUST_{i:04d}' for i in range(100)]
        
        quality_report = self.loader.basic_data_quality_check(sample_data, 'default')
        
        # Should identify high cardinality feature
        assert 'high_cardinality_features' in quality_report
        assert 'customer_id' in quality_report['high_cardinality_features']
        assert quality_report['high_cardinality_features']['customer_id'] == 100
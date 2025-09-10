"""
Tests for Model of Credit (MoC) module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from moc import ModelOfCredit


class TestModelOfCredit:
    """Test cases for ModelOfCredit class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.moc = ModelOfCredit()
        
        # Sample data for testing
        np.random.seed(42)
        self.n_exposures = 100
        self.sample_scores = np.random.normal(0, 1, self.n_exposures)
        self.sample_defaults = np.random.binomial(1, 0.05, self.n_exposures)
        self.sample_recovery_rates = np.random.beta(3, 2, self.n_exposures)
        self.sample_exposures = np.random.lognormal(10, 1, self.n_exposures)
        self.sample_lgds = 1 - self.sample_recovery_rates
        
        # Calibration data
        self.calibration_data = {
            'scores': np.random.normal(0, 1, 1000),
            'defaults': np.random.binomial(1, 0.05, 1000)
        }
        
    def test_initialization(self):
        """Test ModelOfCredit initialization."""
        assert isinstance(self.moc.moc_params, dict)
        assert isinstance(self.moc.calibration_data, dict)
        assert isinstance(self.moc.validation_results, dict)
    
    def test_calculate_pd_components_logistic(self):
        """Test PD calculation using logistic method."""
        pd_results = self.moc.calculate_pd_components(
            self.sample_scores, self.calibration_data, method='logistic'
        )
        
        # Check structure
        assert 'pd_estimates' in pd_results
        assert 'method' in pd_results
        assert pd_results['method'] == 'logistic'
        
        # Check PD estimates
        pd_estimates = pd_results['pd_estimates']
        assert len(pd_estimates) == self.n_exposures
        assert np.all((pd_estimates >= 0) & (pd_estimates <= 1))
        
        # Check statistics
        assert isinstance(pd_results['mean_pd'], float)
        assert isinstance(pd_results['std_pd'], float)
        assert pd_results['mean_pd'] >= 0
        assert pd_results['std_pd'] >= 0
    
    def test_calculate_pd_components_isotonic(self):
        """Test PD calculation using isotonic method."""
        pd_results = self.moc.calculate_pd_components(
            self.sample_scores, self.calibration_data, method='isotonic'
        )
        
        assert pd_results['method'] == 'isotonic'
        assert len(pd_results['pd_estimates']) == self.n_exposures
        assert np.all((pd_results['pd_estimates'] >= 0) & (pd_results['pd_estimates'] <= 1))
    
    def test_calculate_pd_components_beta(self):
        """Test PD calculation using beta method."""
        pd_results = self.moc.calculate_pd_components(
            self.sample_scores, self.calibration_data, method='beta'
        )
        
        assert pd_results['method'] == 'beta'
        assert len(pd_results['pd_estimates']) == self.n_exposures
        assert np.all((pd_results['pd_estimates'] >= 0) & (pd_results['pd_estimates'] <= 1))
    
    def test_calculate_pd_components_invalid_method(self):
        """Test PD calculation with invalid method."""
        with pytest.raises(ValueError):
            self.moc.calculate_pd_components(
                self.sample_scores, self.calibration_data, method='invalid'
            )
    
    def test_estimate_lgd_distribution(self):
        """Test LGD distribution estimation."""
        lgd_results = self.moc.estimate_lgd_distribution(self.sample_recovery_rates)
        
        # Check required statistics
        required_keys = ['mean_lgd', 'median_lgd', 'std_lgd', 'min_lgd', 'max_lgd']
        for key in required_keys:
            assert key in lgd_results
            assert isinstance(lgd_results[key], float)
        
        # Check value ranges
        assert 0 <= lgd_results['mean_lgd'] <= 1
        assert 0 <= lgd_results['min_lgd'] <= lgd_results['max_lgd'] <= 1
        
        # Check percentiles
        assert 'percentiles' in lgd_results
        percentiles = lgd_results['percentiles']
        assert percentiles['p10'] <= percentiles['p25'] <= percentiles['p75'] <= percentiles['p90']
    
    def test_estimate_lgd_distribution_with_facility_types(self):
        """Test LGD distribution with facility type segmentation."""
        facility_types = np.random.choice(['secured', 'unsecured'], self.n_exposures)
        
        lgd_results = self.moc.estimate_lgd_distribution(
            self.sample_recovery_rates, facility_types=facility_types
        )
        
        assert 'facility_segmentation' in lgd_results
        facility_seg = lgd_results['facility_segmentation']
        
        # Should have results for both facility types
        assert len(facility_seg) > 0
        for ftype, stats in facility_seg.items():
            assert 'mean_lgd' in stats
            assert 'count' in stats
            assert 0 <= stats['mean_lgd'] <= 1
    
    def test_calculate_ead_conversion(self):
        """Test EAD conversion factor calculation."""
        committed_amounts = np.random.uniform(100000, 1000000, self.n_exposures)
        outstanding_amounts = committed_amounts * np.random.uniform(0.1, 0.8, self.n_exposures)
        
        ead_results = self.moc.calculate_ead_conversion(
            committed_amounts, outstanding_amounts
        )
        
        # Check required keys
        required_keys = ['ead_amounts', 'ccf_estimates', 'mean_ccf', 'total_ead']
        for key in required_keys:
            assert key in ead_results
        
        # Check calculations
        assert len(ead_results['ead_amounts']) == self.n_exposures
        assert len(ead_results['ccf_estimates']) == self.n_exposures
        assert 0 <= ead_results['mean_ccf'] <= 1
        assert ead_results['total_ead'] >= np.sum(outstanding_amounts)
        
        # Check utilization statistics
        assert 'utilization_statistics' in ead_results
        util_stats = ead_results['utilization_statistics']
        assert 0 <= util_stats['mean_utilization'] <= 1
    
    def test_calculate_expected_loss(self):
        """Test Expected Loss calculation."""
        pd_estimates = np.random.uniform(0.01, 0.2, self.n_exposures)
        
        el_results = self.moc.calculate_expected_loss(
            pd_estimates, self.sample_lgds, self.sample_exposures
        )
        
        # Check required keys
        required_keys = ['individual_el', 'total_el', 'total_exposure', 'el_rate']
        for key in required_keys:
            assert key in el_results
        
        # Check calculations
        assert len(el_results['individual_el']) == self.n_exposures
        assert el_results['total_el'] >= 0
        assert el_results['total_exposure'] == np.sum(self.sample_exposures)
        assert 0 <= el_results['el_rate'] <= 1
        
        # Verify EL calculation
        expected_individual_el = pd_estimates * self.sample_lgds * self.sample_exposures
        np.testing.assert_array_almost_equal(
            el_results['individual_el'], expected_individual_el
        )
        
        # Check statistics
        assert 'statistics' in el_results
        stats = el_results['statistics']
        assert stats['mean_pd'] == np.mean(pd_estimates)
    
    def test_calculate_economic_capital_vasicek(self):
        """Test Economic Capital calculation using Vasicek method."""
        pd_estimates = np.random.uniform(0.01, 0.2, self.n_exposures)
        
        ec_results = self.moc.calculate_economic_capital(
            pd_estimates, self.sample_lgds, self.sample_exposures,
            method='vasicek'
        )
        
        # Check required keys
        required_keys = ['economic_capital', 'worst_case_loss', 'expected_loss', 'confidence_level']
        for key in required_keys:
            assert key in ec_results
        
        # Check values
        assert ec_results['economic_capital'] >= 0
        assert ec_results['worst_case_loss'] >= ec_results['expected_loss']
        assert ec_results['confidence_level'] == 0.999
        assert ec_results['method'] == 'vasicek'
        
        # Check that EC = Worst Case - Expected Loss
        expected_ec = ec_results['worst_case_loss'] - ec_results['expected_loss']
        assert abs(ec_results['economic_capital'] - expected_ec) < 1e-10
    
    def test_calculate_economic_capital_monte_carlo(self):
        """Test Economic Capital calculation using Monte Carlo method."""
        pd_estimates = np.random.uniform(0.01, 0.2, self.n_exposures)
        
        ec_results = self.moc.calculate_economic_capital(
            pd_estimates, self.sample_lgds, self.sample_exposures,
            method='monte_carlo'
        )
        
        assert ec_results['method'] == 'monte_carlo'
        assert 'simulation_results' in ec_results
        
        sim_results = ec_results['simulation_results']
        assert 'mean_loss' in sim_results
        assert 'std_loss' in sim_results
        assert 'var_95' in sim_results
        
        # Check that VaRs are ordered correctly
        assert sim_results['var_95'] <= sim_results['var_99'] <= sim_results['var_999']
    
    def test_calculate_economic_capital_invalid_method(self):
        """Test Economic Capital calculation with invalid method."""
        pd_estimates = np.random.uniform(0.01, 0.2, self.n_exposures)
        
        with pytest.raises(ValueError):
            self.moc.calculate_economic_capital(
                pd_estimates, self.sample_lgds, self.sample_exposures,
                method='invalid'
            )
    
    def test_calculate_raroc(self):
        """Test RAROC calculation."""
        net_income = 1000000
        economic_capital = 5000000
        expected_loss = 200000
        
        raroc = self.moc.calculate_raroc(net_income, economic_capital, expected_loss)
        
        # RAROC should be (1000000 - 200000) / 5000000 * 100 = 16%
        expected_raroc = (net_income - expected_loss) / economic_capital * 100
        assert abs(raroc - expected_raroc) < 1e-10
    
    def test_calculate_raroc_zero_capital(self):
        """Test RAROC calculation with zero capital."""
        raroc = self.moc.calculate_raroc(1000000, 0)
        assert raroc == 0.0
    
    def test_validate_moc_components(self):
        """Test MoC component validation."""
        # Create validation data
        validation_data = {
            'actual_defaults': self.sample_defaults,
            'actual_lgds': self.sample_lgds
        }
        
        # Create estimates
        pd_estimates = np.random.uniform(0.01, 0.2, self.n_exposures)
        lgd_estimates = np.random.uniform(0.2, 0.8, self.n_exposures)
        
        moc_estimates = {
            'pd_estimates': pd_estimates,
            'lgd_estimates': lgd_estimates
        }
        
        validation_results = self.moc.validate_moc_components(
            validation_data, moc_estimates
        )
        
        # Check structure
        assert 'validation_summary' in validation_results
        assert 'sample_size' in validation_results['validation_summary']
        
        # Check PD validation
        if 'pd_validation' in validation_results:
            pd_val = validation_results['pd_validation']
            assert 'mae' in pd_val
            assert 'rmse' in pd_val
            assert pd_val['mae'] >= 0
            assert pd_val['rmse'] >= 0
        
        # Check LGD validation
        if 'lgd_validation' in validation_results:
            lgd_val = validation_results['lgd_validation']
            assert 'mae' in lgd_val
            assert 'rmse' in lgd_val
            assert lgd_val['mae'] >= 0
            assert lgd_val['rmse'] >= 0
    
    def test_end_to_end_moc_workflow(self):
        """Test complete MoC workflow."""
        # Step 1: Calculate PDs
        pd_results = self.moc.calculate_pd_components(
            self.sample_scores, self.calibration_data
        )
        
        # Step 2: Estimate LGDs
        lgd_results = self.moc.estimate_lgd_distribution(self.sample_recovery_rates)
        
        # Step 3: Calculate EAD
        committed_amounts = np.random.uniform(100000, 1000000, self.n_exposures)
        outstanding_amounts = committed_amounts * 0.5
        ead_results = self.moc.calculate_ead_conversion(committed_amounts, outstanding_amounts)
        
        # Step 4: Calculate Expected Loss
        el_results = self.moc.calculate_expected_loss(
            pd_results['pd_estimates'],
            np.full(self.n_exposures, lgd_results['mean_lgd']),
            ead_results['ead_amounts']
        )
        
        # Step 5: Calculate Economic Capital
        ec_results = self.moc.calculate_economic_capital(
            pd_results['pd_estimates'],
            np.full(self.n_exposures, lgd_results['mean_lgd']),
            ead_results['ead_amounts']
        )
        
        # Step 6: Calculate RAROC
        net_income = 1000000
        raroc = self.moc.calculate_raroc(
            net_income, ec_results['economic_capital'], el_results['total_el']
        )
        
        # Verify workflow completed successfully
        assert el_results['total_el'] > 0
        assert ec_results['economic_capital'] > 0
        assert isinstance(raroc, float)
        
        # Check that EC > EL (economic capital should exceed expected loss)
        assert ec_results['economic_capital'] > el_results['total_el']
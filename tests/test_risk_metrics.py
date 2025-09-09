"""
Tests for risk_metrics module.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk_metrics import RiskMetrics


class TestRiskMetrics:
    """Test cases for RiskMetrics class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.risk_calculator = RiskMetrics()
        
        # Sample data for testing
        np.random.seed(42)
        self.sample_returns = np.random.normal(-0.01, 0.05, 1000)  # Negative mean (losses)
        self.sample_pds = np.random.beta(1, 9, 100)  # PDs between 0 and 1
        self.sample_exposures = np.random.lognormal(12, 1, 100)  # Exposure amounts
        self.sample_lgds = np.random.beta(2, 3, 100)  # LGDs between 0 and 1
        
    def test_initialization(self):
        """Test RiskMetrics initialization."""
        assert isinstance(self.risk_calculator.portfolio_metrics, dict)
        assert len(self.risk_calculator.portfolio_metrics) == 0
        
    def test_calculate_var_historical(self):
        """Test VaR calculation using historical method."""
        var = self.risk_calculator.calculate_var(
            self.sample_returns, confidence_level=0.95, method='historical'
        )
        
        # VaR should be positive (representing loss)
        assert var >= 0
        assert isinstance(var, float)
        
        # 95% VaR should be greater than 90% VaR
        var_90 = self.risk_calculator.calculate_var(
            self.sample_returns, confidence_level=0.90, method='historical'
        )
        assert var >= var_90
        
    def test_calculate_var_parametric(self):
        """Test VaR calculation using parametric method."""
        var = self.risk_calculator.calculate_var(
            self.sample_returns, confidence_level=0.95, method='parametric'
        )
        
        assert var >= 0
        assert isinstance(var, float)
        
    def test_calculate_var_monte_carlo(self):
        """Test VaR calculation using Monte Carlo method."""
        var = self.risk_calculator.calculate_var(
            self.sample_returns, confidence_level=0.95, method='monte_carlo'
        )
        
        assert var >= 0
        assert isinstance(var, float)
        
    def test_calculate_var_invalid_method(self):
        """Test VaR calculation with invalid method."""
        with pytest.raises(ValueError):
            self.risk_calculator.calculate_var(
                self.sample_returns, method='invalid_method'
            )
            
    def test_calculate_expected_shortfall(self):
        """Test Expected Shortfall calculation."""
        es = self.risk_calculator.calculate_expected_shortfall(
            self.sample_returns, confidence_level=0.95
        )
        
        # Expected Shortfall should be positive and greater than VaR
        assert es >= 0
        assert isinstance(es, float)
        
        var = self.risk_calculator.calculate_var(
            self.sample_returns, confidence_level=0.95, method='historical'
        )
        assert es >= var  # ES should be >= VaR
        
    def test_calculate_portfolio_pd_average(self):
        """Test portfolio PD calculation using average method."""
        portfolio_pd = self.risk_calculator.calculate_portfolio_pd(
            self.sample_pds, method='average'
        )
        
        # Should be between 0 and 1
        assert 0 <= portfolio_pd <= 1
        # Should be approximately equal to mean of individual PDs
        assert abs(portfolio_pd - np.mean(self.sample_pds)) < 1e-10
        
    def test_calculate_portfolio_pd_weighted(self):
        """Test portfolio PD calculation using weighted method."""
        portfolio_pd = self.risk_calculator.calculate_portfolio_pd(
            self.sample_pds, method='weighted'
        )
        
        assert 0 <= portfolio_pd <= 1
        assert isinstance(portfolio_pd, float)
        
    def test_calculate_portfolio_pd_vasicek(self):
        """Test portfolio PD calculation using Vasicek model."""
        portfolio_pd = self.risk_calculator.calculate_portfolio_pd(
            self.sample_pds, method='vasicek'
        )
        
        assert 0 <= portfolio_pd <= 1
        assert isinstance(portfolio_pd, float)
        
    def test_calculate_lgd_distribution(self):
        """Test LGD distribution calculation."""
        recovery_rates = 1 - self.sample_lgds  # Convert LGDs to recovery rates
        lgd_stats = self.risk_calculator.calculate_lgd_distribution(recovery_rates)
        
        # Check required statistics
        required_keys = ['mean_lgd', 'median_lgd', 'std_lgd', 'min_lgd', 'max_lgd']
        for key in required_keys:
            assert key in lgd_stats
            assert isinstance(lgd_stats[key], float)
            
        # Check value ranges
        assert 0 <= lgd_stats['mean_lgd'] <= 1
        assert 0 <= lgd_stats['min_lgd'] <= lgd_stats['max_lgd'] <= 1
        
    def test_calculate_expected_loss(self):
        """Test Expected Loss calculation."""
        el_metrics = self.risk_calculator.calculate_expected_loss(
            self.sample_exposures, self.sample_pds, self.sample_lgds
        )
        
        # Check required keys
        required_keys = ['total_exposure', 'individual_els', 'total_el', 'el_rate']
        for key in required_keys:
            assert key in el_metrics
            
        # Check calculations
        assert el_metrics['total_exposure'] == np.sum(self.sample_exposures)
        assert el_metrics['total_el'] >= 0
        assert 0 <= el_metrics['el_rate'] <= 1
        
        # Check individual ELs calculation
        expected_individual_els = self.sample_exposures * self.sample_pds * self.sample_lgds
        np.testing.assert_array_almost_equal(
            el_metrics['individual_els'], expected_individual_els
        )
        
    def test_calculate_regulatory_capital(self):
        """Test regulatory capital calculation."""
        capital_metrics = self.risk_calculator.calculate_regulatory_capital(
            self.sample_exposures, self.sample_pds, self.sample_lgds
        )
        
        # Check required keys
        required_keys = ['total_rwa', 'capital_requirement', 'unexpected_loss']
        for key in required_keys:
            assert key in capital_metrics
            
        # Check values are positive
        assert capital_metrics['total_rwa'] >= 0
        assert capital_metrics['capital_requirement'] >= 0
        assert capital_metrics['unexpected_loss'] >= 0
        
        # Check capital requirement is 8% of RWA
        expected_capital = capital_metrics['total_rwa'] * 0.08
        assert abs(capital_metrics['capital_requirement'] - expected_capital) < 1e-10
        
    def test_calculate_raroc(self):
        """Test RAROC calculation."""
        net_income = 1000000
        economic_capital = 5000000
        expected_loss = 200000
        
        raroc = self.risk_calculator.calculate_raroc(
            net_income, economic_capital, expected_loss
        )
        
        # RAROC should be (1000000 - 200000) / 5000000 * 100 = 16%
        expected_raroc = (net_income - expected_loss) / economic_capital * 100
        assert abs(raroc - expected_raroc) < 1e-10
        
    def test_calculate_raroc_zero_capital(self):
        """Test RAROC calculation with zero capital."""
        raroc = self.risk_calculator.calculate_raroc(1000000, 0)
        assert raroc == 0.0
        
    def test_stress_test_portfolio(self):
        """Test portfolio stress testing."""
        stress_scenarios = {
            'mild_stress': {'pd_multiplier': 1.5},
            'severe_stress': {'pd_multiplier': 3.0}
        }
        
        stress_results = self.risk_calculator.stress_test_portfolio(
            self.sample_pds, stress_scenarios
        )
        
        # Check results structure
        assert 'mild_stress' in stress_results
        assert 'severe_stress' in stress_results
        
        for scenario, results in stress_results.items():
            required_keys = ['stressed_pds', 'portfolio_pd', 'pd_increase']
            for key in required_keys:
                assert key in results
                
        # Severe stress should have higher PDs than mild stress
        mild_pd = stress_results['mild_stress']['portfolio_pd']
        severe_pd = stress_results['severe_stress']['portfolio_pd']
        assert severe_pd > mild_pd
        
    def test_calculate_portfolio_concentration(self):
        """Test portfolio concentration calculation."""
        concentration_metrics = self.risk_calculator.calculate_portfolio_concentration(
            self.sample_exposures
        )
        
        # Check required keys
        required_keys = ['hhi', 'effective_number_of_exposures', 'top_1_concentration']
        for key in required_keys:
            assert key in concentration_metrics
            
        # Check HHI range
        assert 0 <= concentration_metrics['hhi'] <= 1
        
        # Check concentration percentages
        assert 0 <= concentration_metrics['top_1_concentration'] <= 1
        assert 0 <= concentration_metrics['top_5_concentration'] <= 1
        assert 0 <= concentration_metrics['top_10_concentration'] <= 1
        
        # Top 1 should be <= Top 5 should be <= Top 10
        assert (concentration_metrics['top_1_concentration'] <= 
                concentration_metrics['top_5_concentration'] <= 
                concentration_metrics['top_10_concentration'])
        
    def test_calculate_credit_var_monte_carlo(self):
        """Test Credit VaR Monte Carlo calculation."""
        credit_var_results = self.risk_calculator.calculate_credit_var_monte_carlo(
            self.sample_exposures, self.sample_pds, self.sample_lgds, n_simulations=1000
        )
        
        # Check required keys
        required_keys = ['expected_loss', 'credit_var', 'unexpected_loss', 'expected_shortfall']
        for key in required_keys:
            assert key in credit_var_results
            assert credit_var_results[key] >= 0
            
        # Check relationships
        assert credit_var_results['credit_var'] >= credit_var_results['expected_loss']
        assert credit_var_results['expected_shortfall'] >= credit_var_results['credit_var']
        
        # Check simulation parameters
        assert credit_var_results['simulation_params']['n_simulations'] == 1000
        assert len(credit_var_results['loss_distribution']) == 1000
        
    def test_calculate_portfolio_diversification_ratio(self):
        """Test portfolio diversification ratio calculation."""
        individual_vars = np.random.uniform(100, 1000, 10)  # Individual VaRs
        portfolio_var = 500  # Portfolio VaR (should be less than sum due to diversification)
        
        div_ratio = self.risk_calculator.calculate_portfolio_diversification_ratio(
            individual_vars, portfolio_var
        )
        
        # Diversification ratio should be between 0 and 1
        assert 0 <= div_ratio <= 1
        
        # Test with zero portfolio VaR
        div_ratio_zero = self.risk_calculator.calculate_portfolio_diversification_ratio(
            individual_vars, 0
        )
        assert div_ratio_zero == 0.0
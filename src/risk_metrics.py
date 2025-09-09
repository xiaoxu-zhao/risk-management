"""
Risk Metrics Module
==================

Implementation of key risk metrics used in credit risk management:
- Value at Risk (VaR)
- Expected Shortfall (Conditional VaR)
- Probability of Default (PD)
- Loss Given Default (LGD)
- Expected Loss (EL)
- Risk-Adjusted Return on Capital (RAROC)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class RiskMetrics:
    """
    Comprehensive risk metrics calculator for credit portfolios.
    """
    
    def __init__(self):
        self.portfolio_metrics = {}
        
    def calculate_var(self, returns: np.ndarray, 
                     confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR) using different methodologies.
        
        Args:
            returns: Array of portfolio returns or losses
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            method: Method to use ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            VaR value (positive number representing potential loss)
        """
        if method == 'historical':
            # Historical simulation method
            var = np.percentile(returns, (1 - confidence_level) * 100)
            
        elif method == 'parametric':
            # Parametric method assuming normal distribution
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean_return + z_score * std_return
            
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            n_simulations = 10000
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Generate random returns
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
            
        # Return as positive value (loss amount)
        return -var if var < 0 else var
        
    def calculate_expected_shortfall(self, returns: np.ndarray, 
                                   confidence_level: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Array of portfolio returns or losses
            confidence_level: Confidence level
            
        Returns:
            Expected Shortfall value
        """
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Expected value of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return 0.0
            
        expected_shortfall = np.mean(tail_returns)
        
        # Return as positive value (expected loss amount)
        return -expected_shortfall if expected_shortfall < 0 else expected_shortfall
        
    def calculate_portfolio_pd(self, default_probabilities: np.ndarray, 
                              correlation_matrix: np.ndarray = None,
                              method: str = 'average') -> float:
        """
        Calculate portfolio-level Probability of Default.
        
        Args:
            default_probabilities: Array of individual PDs
            correlation_matrix: Asset correlation matrix (optional)
            method: Calculation method ('average', 'weighted', 'vasicek')
            
        Returns:
            Portfolio PD
        """
        if method == 'average':
            # Simple average of individual PDs
            portfolio_pd = np.mean(default_probabilities)
            
        elif method == 'weighted':
            # Weighted average (assuming equal weights for simplicity)
            weights = np.ones(len(default_probabilities)) / len(default_probabilities)
            portfolio_pd = np.sum(weights * default_probabilities)
            
        elif method == 'vasicek':
            # Vasicek single-factor model
            if correlation_matrix is not None:
                # Simplified correlation-adjusted calculation
                avg_correlation = np.mean(correlation_matrix[correlation_matrix != 1])
            else:
                avg_correlation = 0.2  # Default asset correlation
                
            # Asset value model
            z_scores = stats.norm.ppf(default_probabilities)
            portfolio_z = np.mean(z_scores)
            
            # Adjust for correlation
            adjusted_std = np.sqrt(1 - avg_correlation)
            portfolio_pd = stats.norm.cdf(portfolio_z / adjusted_std)
            
        else:
            raise ValueError(f"Unknown PD calculation method: {method}")
            
        return portfolio_pd
        
    def calculate_lgd_distribution(self, recovery_rates: np.ndarray) -> Dict:
        """
        Calculate Loss Given Default statistics from recovery rate data.
        
        Args:
            recovery_rates: Array of recovery rates (0-1)
            
        Returns:
            Dictionary with LGD statistics
        """
        lgd_values = 1 - recovery_rates
        
        lgd_stats = {
            'mean_lgd': np.mean(lgd_values),
            'median_lgd': np.median(lgd_values),
            'std_lgd': np.std(lgd_values),
            'min_lgd': np.min(lgd_values),
            'max_lgd': np.max(lgd_values),
            'percentile_25': np.percentile(lgd_values, 25),
            'percentile_75': np.percentile(lgd_values, 75),
            'percentile_95': np.percentile(lgd_values, 95)
        }
        
        return lgd_stats
        
    def calculate_expected_loss(self, exposures: np.ndarray,
                              pds: np.ndarray,
                              lgds: np.ndarray) -> Dict:
        """
        Calculate Expected Loss for credit portfolio.
        
        Args:
            exposures: Exposure at Default (EAD) values
            pds: Probability of Default values
            lgds: Loss Given Default values
            
        Returns:
            Dictionary with Expected Loss metrics
        """
        # Individual expected losses
        individual_els = exposures * pds * lgds
        
        # Portfolio metrics
        el_metrics = {
            'total_exposure': np.sum(exposures),
            'individual_els': individual_els,
            'total_el': np.sum(individual_els),
            'average_el': np.mean(individual_els),
            'el_rate': np.sum(individual_els) / np.sum(exposures),
            'el_concentration': {
                'top_10_pct': np.sum(np.sort(individual_els)[-int(0.1*len(individual_els)):]) / np.sum(individual_els),
                'top_5_pct': np.sum(np.sort(individual_els)[-int(0.05*len(individual_els)):]) / np.sum(individual_els),
                'top_1_pct': np.sum(np.sort(individual_els)[-int(0.01*len(individual_els)):]) / np.sum(individual_els)
            }
        }
        
        return el_metrics
        
    def calculate_regulatory_capital(self, exposures: np.ndarray,
                                   pds: np.ndarray,
                                   lgds: np.ndarray,
                                   asset_correlations: np.ndarray = None,
                                   confidence_level: float = 0.999) -> Dict:
        """
        Calculate regulatory capital using Basel III IRB approach.
        
        Args:
            exposures: Exposure at Default values
            pds: Probability of Default values  
            lgds: Loss Given Default values
            asset_correlations: Asset correlation values (optional)
            confidence_level: Regulatory confidence level (99.9% for Basel III)
            
        Returns:
            Dictionary with regulatory capital metrics
        """
        # Default correlations if not provided (Basel III corporate formula)
        if asset_correlations is None:
            asset_correlations = 0.12 * (1 - np.exp(-50 * pds)) / (1 - np.exp(-50))
            asset_correlations += 0.24 * (1 - (1 - np.exp(-50 * pds)) / (1 - np.exp(-50)))
            
        # Calculate correlation-adjusted PDs
        z_pd = stats.norm.ppf(pds)
        z_confidence = stats.norm.ppf(confidence_level)
        
        # Vasicek formula for conditional PD
        sqrt_corr = np.sqrt(asset_correlations)
        sqrt_one_minus_corr = np.sqrt(1 - asset_correlations)
        
        conditional_pd = stats.norm.cdf(
            (sqrt_corr * z_confidence + sqrt_one_minus_corr * z_pd) / sqrt_one_minus_corr
        )
        
        # Risk-weighted assets calculation
        unexpected_loss_rate = conditional_pd * lgds - pds * lgds
        risk_weighted_assets = exposures * unexpected_loss_rate * 12.5  # 8% capital ratio
        
        # Capital requirements
        capital_metrics = {
            'individual_rwa': risk_weighted_assets,
            'total_rwa': np.sum(risk_weighted_assets),
            'capital_requirement': np.sum(risk_weighted_assets) * 0.08,  # 8% minimum
            'capital_buffer': np.sum(risk_weighted_assets) * 0.025,  # 2.5% conservation buffer
            'total_capital_needed': np.sum(risk_weighted_assets) * 0.105,  # 10.5% total
            'unexpected_loss': np.sum(exposures * unexpected_loss_rate),
            'capital_adequacy_ratio': lambda capital_held: capital_held / np.sum(risk_weighted_assets)
        }
        
        return capital_metrics
        
    def calculate_raroc(self, net_income: float,
                       economic_capital: float,
                       expected_loss: float = None) -> float:
        """
        Calculate Risk-Adjusted Return on Capital (RAROC).
        
        Args:
            net_income: Net income from the portfolio/loan
            economic_capital: Economic capital allocated
            expected_loss: Expected loss (optional, subtracted from income)
            
        Returns:
            RAROC percentage
        """
        adjusted_income = net_income
        if expected_loss is not None:
            adjusted_income -= expected_loss
            
        if economic_capital == 0:
            return 0.0
            
        raroc = (adjusted_income / economic_capital) * 100
        
        return raroc
        
    def stress_test_portfolio(self, base_pds: np.ndarray,
                            stress_scenarios: Dict[str, Dict]) -> Dict:
        """
        Perform stress testing on credit portfolio.
        
        Args:
            base_pds: Baseline probability of default values
            stress_scenarios: Dictionary of stress scenarios with PD multipliers
            
        Returns:
            Dictionary with stress test results
        """
        stress_results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            # Apply stress multipliers
            stressed_pds = base_pds * scenario_params.get('pd_multiplier', 1.0)
            stressed_pds = np.clip(stressed_pds, 0, 1)  # Keep PDs between 0 and 1
            
            # Calculate stressed portfolio PD
            stressed_portfolio_pd = self.calculate_portfolio_pd(stressed_pds)
            
            stress_results[scenario_name] = {
                'stressed_pds': stressed_pds,
                'portfolio_pd': stressed_portfolio_pd,
                'pd_increase': stressed_portfolio_pd - self.calculate_portfolio_pd(base_pds),
                'max_individual_pd': np.max(stressed_pds),
                'avg_individual_pd': np.mean(stressed_pds)
            }
            
        return stress_results
        
    def calculate_portfolio_concentration(self, exposures: np.ndarray,
                                        sectors: np.ndarray = None) -> Dict:
        """
        Calculate portfolio concentration metrics.
        
        Args:
            exposures: Exposure amounts
            sectors: Sector labels (optional)
            
        Returns:
            Dictionary with concentration metrics
        """
        total_exposure = np.sum(exposures)
        exposure_weights = exposures / total_exposure
        
        # Herfindahl-Hirschman Index (HHI)
        hhi = np.sum(exposure_weights ** 2)
        
        # Concentration metrics
        sorted_exposures = np.sort(exposures)[::-1]
        concentration_metrics = {
            'hhi': hhi,
            'effective_number_of_exposures': 1 / hhi if hhi > 0 else 0,
            'top_1_concentration': sorted_exposures[0] / total_exposure,
            'top_5_concentration': np.sum(sorted_exposures[:5]) / total_exposure,
            'top_10_concentration': np.sum(sorted_exposures[:10]) / total_exposure,
            'largest_exposure': np.max(exposures),
            'smallest_exposure': np.min(exposures),
            'median_exposure': np.median(exposures)
        }
        
        # Sector concentration if sectors provided
        if sectors is not None:
            sector_exposures = pd.Series(exposures, index=sectors).groupby(level=0).sum()
            sector_concentration = sector_exposures / total_exposure
            
            concentration_metrics['sector_concentration'] = {
                'by_sector': sector_concentration.to_dict(),
                'max_sector_concentration': sector_concentration.max(),
                'number_of_sectors': len(sector_concentration),
                'sector_hhi': np.sum((sector_concentration ** 2))
            }
            
        return concentration_metrics
        
    def calculate_credit_var_monte_carlo(self, exposures: np.ndarray,
                                       pds: np.ndarray,
                                       lgds: np.ndarray,
                                       correlations: np.ndarray = None,
                                       n_simulations: int = 10000,
                                       confidence_level: float = 0.95) -> Dict:
        """
        Calculate Credit VaR using Monte Carlo simulation.
        
        Args:
            exposures: Exposure at Default values
            pds: Probability of Default values
            lgds: Loss Given Default values
            correlations: Asset correlation matrix (optional)
            n_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level for VaR
            
        Returns:
            Dictionary with Credit VaR results
        """
        n_assets = len(exposures)
        
        # Default correlation matrix if not provided
        if correlations is None:
            correlations = np.eye(n_assets) * 0.8 + np.ones((n_assets, n_assets)) * 0.2
            
        # Cholesky decomposition for correlation
        chol_matrix = np.linalg.cholesky(correlations)
        
        # Monte Carlo simulation
        portfolio_losses = []
        
        for _ in range(n_simulations):
            # Generate correlated random variables
            z = np.random.normal(0, 1, n_assets)
            correlated_z = chol_matrix @ z
            
            # Calculate default indicators using asset value model
            asset_values = correlated_z
            default_thresholds = stats.norm.ppf(pds)
            defaults = asset_values < default_thresholds
            
            # Calculate portfolio loss
            losses = defaults * exposures * lgds
            portfolio_loss = np.sum(losses)
            portfolio_losses.append(portfolio_loss)
            
        portfolio_losses = np.array(portfolio_losses)
        
        # Calculate risk metrics
        expected_loss = np.mean(portfolio_losses)
        credit_var = np.percentile(portfolio_losses, confidence_level * 100)
        unexpected_loss = credit_var - expected_loss
        expected_shortfall = np.mean(portfolio_losses[portfolio_losses >= credit_var])
        
        credit_var_results = {
            'expected_loss': expected_loss,
            'credit_var': credit_var,
            'unexpected_loss': unexpected_loss,
            'expected_shortfall': expected_shortfall,
            'std_dev_loss': np.std(portfolio_losses),
            'max_loss': np.max(portfolio_losses),
            'loss_distribution': portfolio_losses,
            'simulation_params': {
                'n_simulations': n_simulations,
                'confidence_level': confidence_level,
                'n_assets': n_assets
            }
        }
        
        return credit_var_results
        
    def calculate_portfolio_diversification_ratio(self, individual_vars: np.ndarray,
                                                portfolio_var: float) -> float:
        """
        Calculate portfolio diversification ratio.
        
        Args:
            individual_vars: VaR of individual positions
            portfolio_var: Portfolio VaR
            
        Returns:
            Diversification ratio
        """
        sum_individual_vars = np.sum(individual_vars)
        
        if portfolio_var == 0:
            return 0.0
            
        diversification_ratio = 1 - (portfolio_var / sum_individual_vars)
        
        return diversification_ratio
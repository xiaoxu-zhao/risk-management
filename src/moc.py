"""
Model of Credit (MoC) Module
============================

Implementation of Model of Credit methodology for credit risk modeling.
MoC provides a comprehensive framework for credit risk assessment including:
- Expected Loss calculation and decomposition
- Economic Capital requirements
- Risk-adjusted pricing
- Portfolio optimization metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ModelOfCredit:
    """
    Model of Credit implementation for comprehensive credit risk assessment.
    
    The Model of Credit (MoC) is a framework that integrates:
    - Probability of Default (PD) modeling
    - Loss Given Default (LGD) estimation
    - Exposure at Default (EAD) calculation
    - Economic capital and regulatory capital requirements
    - Risk-adjusted return metrics
    """
    
    def __init__(self):
        self.moc_params = {}
        self.calibration_data = {}
        self.validation_results = {}
        
    def calculate_pd_components(self, scores: np.ndarray, 
                               calibration_data: Dict[str, np.ndarray],
                               method: str = 'logistic') -> Dict[str, Any]:
        """
        Calculate PD from model scores using various calibration methods.
        
        Args:
            scores: Model scores/predictions
            calibration_data: Historical calibration data with 'scores' and 'defaults'
            method: Calibration method ('logistic', 'isotonic', 'beta')
            
        Returns:
            Dictionary with PD calculations and calibration metrics
        """
        if method == 'logistic':
            # Logistic regression calibration
            from scipy.optimize import minimize_scalar
            
            def logistic_log_likelihood(params, scores, defaults):
                a, b = params
                prob_default = 1 / (1 + np.exp(-(a + b * scores)))
                # Avoid log(0) by clipping probabilities
                prob_default = np.clip(prob_default, 1e-15, 1-1e-15)
                ll = np.sum(defaults * np.log(prob_default) + 
                           (1 - defaults) * np.log(1 - prob_default))
                return -ll  # Minimize negative log-likelihood
            
            # Simple calibration for demonstration
            # In practice, use sklearn's CalibratedClassifierCV
            cal_scores = calibration_data['scores']
            cal_defaults = calibration_data['defaults']
            
            # Estimate parameters via method of moments as approximation
            mean_score = np.mean(cal_scores)
            mean_default = np.mean(cal_defaults)
            
            # Simple linear mapping: PD = intercept + slope * score
            slope = (mean_default - 0.5) / (mean_score)
            intercept = mean_default - slope * mean_score
            
            # Apply logistic transformation
            linear_pred = intercept + slope * scores
            pd_estimates = 1 / (1 + np.exp(-linear_pred))
            
        elif method == 'isotonic':
            # Isotonic regression calibration (simplified)
            cal_scores = calibration_data['scores']
            cal_defaults = calibration_data['defaults']
            
            # Sort by scores and compute empirical probabilities
            sorted_indices = np.argsort(cal_scores)
            sorted_scores = cal_scores[sorted_indices]
            sorted_defaults = cal_defaults[sorted_indices]
            
            # Simple binning approach for isotonic regression approximation
            n_bins = min(20, len(cal_scores) // 50)  # Adaptive number of bins
            bin_edges = np.percentile(sorted_scores, np.linspace(0, 100, n_bins + 1))
            
            pd_estimates = np.zeros_like(scores)
            for i in range(len(scores)):
                # Find which bin the score falls into
                bin_idx = np.searchsorted(bin_edges[1:], scores[i])
                bin_idx = min(bin_idx, len(bin_edges) - 2)  # Ensure valid index
                
                # Get defaults in this bin
                in_bin = ((sorted_scores >= bin_edges[bin_idx]) & 
                         (sorted_scores < bin_edges[bin_idx + 1]))
                if np.sum(in_bin) > 0:
                    pd_estimates[i] = np.mean(sorted_defaults[in_bin])
                else:
                    pd_estimates[i] = np.mean(sorted_defaults)  # Fallback to overall mean
                    
        elif method == 'beta':
            # Beta calibration (simplified implementation)
            cal_scores = calibration_data['scores']
            cal_defaults = calibration_data['defaults']
            
            # Simple beta mapping: transform scores to [0,1] then apply beta distribution
            score_min, score_max = np.min(cal_scores), np.max(cal_scores)
            norm_scores = (scores - score_min) / (score_max - score_min + 1e-8)
            norm_scores = np.clip(norm_scores, 0.001, 0.999)
            
            # Estimate beta parameters from calibration data
            cal_norm_scores = (cal_scores - score_min) / (score_max - score_min + 1e-8)
            mean_default = np.mean(cal_defaults)
            
            # Simple transformation based on empirical default rate
            pd_estimates = norm_scores * mean_default * 2  # Simple scaling
            pd_estimates = np.clip(pd_estimates, 0.001, 0.999)
            
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        # Calculate calibration metrics
        pd_estimates = np.clip(pd_estimates, 1e-6, 1-1e-6)  # Ensure valid probabilities
        
        results = {
            'pd_estimates': pd_estimates,
            'method': method,
            'mean_pd': np.mean(pd_estimates),
            'median_pd': np.median(pd_estimates),
            'std_pd': np.std(pd_estimates),
            'min_pd': np.min(pd_estimates),
            'max_pd': np.max(pd_estimates),
            'calibration_data_size': len(calibration_data['scores'])
        }
        
        logger.info(f"PD calculation completed using {method} method. Mean PD: {results['mean_pd']:.4f}")
        return results
    
    def estimate_lgd_distribution(self, recovery_data: np.ndarray,
                                 facility_types: np.ndarray = None,
                                 collateral_values: np.ndarray = None) -> Dict[str, Any]:
        """
        Estimate Loss Given Default distribution from historical recovery data.
        
        Args:
            recovery_data: Historical recovery rates (0 to 1)
            facility_types: Types of facilities (for segmentation)
            collateral_values: Collateral values for secured facilities
            
        Returns:
            Dictionary with LGD distribution parameters and statistics
        """
        lgd_data = 1 - recovery_data  # Convert recovery rates to LGD
        lgd_data = np.clip(lgd_data, 0, 1)  # Ensure valid LGD range
        
        # Basic LGD statistics
        lgd_stats = {
            'mean_lgd': np.mean(lgd_data),
            'median_lgd': np.median(lgd_data),
            'std_lgd': np.std(lgd_data),
            'min_lgd': np.min(lgd_data),
            'max_lgd': np.max(lgd_data),
            'percentiles': {
                'p10': np.percentile(lgd_data, 10),
                'p25': np.percentile(lgd_data, 25),
                'p75': np.percentile(lgd_data, 75),
                'p90': np.percentile(lgd_data, 90),
                'p95': np.percentile(lgd_data, 95)
            }
        }
        
        # Facility type segmentation if available
        if facility_types is not None:
            unique_types = np.unique(facility_types)
            facility_lgd = {}
            
            for ftype in unique_types:
                mask = facility_types == ftype
                facility_lgd_data = lgd_data[mask]
                
                if len(facility_lgd_data) > 0:
                    facility_lgd[str(ftype)] = {
                        'mean_lgd': np.mean(facility_lgd_data),
                        'median_lgd': np.median(facility_lgd_data),
                        'count': len(facility_lgd_data)
                    }
            
            lgd_stats['facility_segmentation'] = facility_lgd
        
        # Collateral analysis if available
        if collateral_values is not None:
            # Simple collateral coverage analysis
            has_collateral = collateral_values > 0
            
            lgd_stats['collateral_analysis'] = {
                'secured_mean_lgd': np.mean(lgd_data[has_collateral]) if np.any(has_collateral) else None,
                'unsecured_mean_lgd': np.mean(lgd_data[~has_collateral]) if np.any(~has_collateral) else None,
                'secured_count': np.sum(has_collateral),
                'unsecured_count': np.sum(~has_collateral)
            }
        
        logger.info(f"LGD distribution estimated. Mean LGD: {lgd_stats['mean_lgd']:.4f}")
        return lgd_stats
    
    def calculate_ead_conversion(self, committed_amounts: np.ndarray,
                                outstanding_amounts: np.ndarray,
                                utilization_rates: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate Exposure at Default (EAD) conversion factors.
        
        Args:
            committed_amounts: Committed credit line amounts
            outstanding_amounts: Current outstanding amounts
            utilization_rates: Historical utilization at default
            
        Returns:
            Dictionary with EAD calculations and conversion factors
        """
        # Calculate current utilization
        current_utilization = outstanding_amounts / (committed_amounts + 1e-8)
        current_utilization = np.clip(current_utilization, 0, 1)
        
        # Calculate undrawn amounts
        undrawn_amounts = committed_amounts - outstanding_amounts
        undrawn_amounts = np.maximum(undrawn_amounts, 0)
        
        # Credit Conversion Factor (CCF) estimation
        if utilization_rates is not None:
            # Use historical utilization at default to estimate CCF
            default_utilization = np.clip(utilization_rates, 0, 1)
            
            # CCF is the expected drawdown on undrawn portion before default
            ccf_estimates = np.minimum((default_utilization * committed_amounts - outstanding_amounts) / 
                                     (undrawn_amounts + 1e-8), 1.0)
            ccf_estimates = np.clip(ccf_estimates, 0, 1)
            
            mean_ccf = np.mean(ccf_estimates)
        else:
            # Use regulatory or industry default CCF rates
            ccf_estimates = np.full_like(committed_amounts, 0.75)  # 75% default CCF
            mean_ccf = 0.75
        
        # Calculate EAD
        ead_amounts = outstanding_amounts + ccf_estimates * undrawn_amounts
        
        ead_results = {
            'ead_amounts': ead_amounts,
            'ccf_estimates': ccf_estimates,
            'mean_ccf': mean_ccf,
            'total_committed': np.sum(committed_amounts),
            'total_outstanding': np.sum(outstanding_amounts),
            'total_ead': np.sum(ead_amounts),
            'utilization_statistics': {
                'mean_utilization': np.mean(current_utilization),
                'median_utilization': np.median(current_utilization),
                'std_utilization': np.std(current_utilization)
            }
        }
        
        logger.info(f"EAD calculation completed. Mean CCF: {mean_ccf:.4f}, Total EAD: {np.sum(ead_amounts):,.0f}")
        return ead_results
    
    def calculate_expected_loss(self, pd_estimates: np.ndarray,
                               lgd_estimates: np.ndarray,
                               ead_amounts: np.ndarray,
                               time_horizon: float = 1.0) -> Dict[str, Any]:
        """
        Calculate Expected Loss using MoC components.
        
        Args:
            pd_estimates: Probability of Default estimates
            lgd_estimates: Loss Given Default estimates
            ead_amounts: Exposure at Default amounts
            time_horizon: Time horizon in years
            
        Returns:
            Dictionary with Expected Loss calculations
        """
        # Adjust PD for time horizon if not 1 year
        if time_horizon != 1.0:
            # Simple linear adjustment (could use survival analysis for more accuracy)
            adjusted_pd = pd_estimates * time_horizon
            adjusted_pd = np.clip(adjusted_pd, 0, 1)
        else:
            adjusted_pd = pd_estimates
        
        # Calculate individual Expected Losses
        individual_el = adjusted_pd * lgd_estimates * ead_amounts
        
        # Portfolio level metrics
        total_el = np.sum(individual_el)
        total_exposure = np.sum(ead_amounts)
        el_rate = total_el / total_exposure if total_exposure > 0 else 0
        
        # Risk contribution analysis
        el_contributions = individual_el / total_el if total_el > 0 else np.zeros_like(individual_el)
        
        # Diversification metrics
        concentration_index = np.sum(el_contributions ** 2)  # Herfindahl index for EL
        effective_number = 1 / concentration_index if concentration_index > 0 else len(individual_el)
        
        el_results = {
            'individual_el': individual_el,
            'total_el': total_el,
            'total_exposure': total_exposure,
            'el_rate': el_rate,
            'el_contributions': el_contributions,
            'concentration_index': concentration_index,
            'effective_number_exposures': effective_number,
            'time_horizon': time_horizon,
            'statistics': {
                'mean_pd': np.mean(adjusted_pd),
                'mean_lgd': np.mean(lgd_estimates),
                'mean_ead': np.mean(ead_amounts),
                'max_individual_el': np.max(individual_el),
                'top_10_el_concentration': np.sum(np.sort(individual_el)[-10:]) / total_el if total_el > 0 else 0
            }
        }
        
        logger.info(f"Expected Loss calculation completed. Total EL: {total_el:,.0f}, EL Rate: {el_rate:.4%}")
        return el_results
    
    def calculate_economic_capital(self, pd_estimates: np.ndarray,
                                  lgd_estimates: np.ndarray,
                                  ead_amounts: np.ndarray,
                                  correlation_matrix: np.ndarray = None,
                                  confidence_level: float = 0.999,
                                  method: str = 'vasicek') -> Dict[str, Any]:
        """
        Calculate Economic Capital using advanced portfolio models.
        
        Args:
            pd_estimates: Probability of Default estimates
            lgd_estimates: Loss Given Default estimates  
            ead_amounts: Exposure at Default amounts
            correlation_matrix: Asset correlation matrix
            confidence_level: Confidence level for capital calculation
            method: Method to use ('vasicek', 'monte_carlo', 'granularity_adjustment')
            
        Returns:
            Dictionary with Economic Capital calculations
        """
        n_exposures = len(pd_estimates)
        
        # Calculate Expected Loss
        el_results = self.calculate_expected_loss(pd_estimates, lgd_estimates, ead_amounts)
        expected_loss = el_results['total_el']
        
        if method == 'vasicek':
            # Vasicek single-factor model
            # Assume uniform correlation or use average if matrix provided
            if correlation_matrix is not None:
                avg_correlation = np.mean(correlation_matrix[np.triu_indices(n_exposures, k=1)])
            else:
                avg_correlation = 0.15  # Typical asset correlation for corporate exposures
            
            # Transform PD to normal space
            from scipy.stats import norm
            pd_normal = norm.ppf(pd_estimates)
            conf_normal = norm.ppf(confidence_level)
            
            # Vasicek formula for conditional PD
            sqrt_rho = np.sqrt(avg_correlation)
            sqrt_one_minus_rho = np.sqrt(1 - avg_correlation)
            
            conditional_pd = norm.cdf(
                (pd_normal + sqrt_rho * conf_normal) / sqrt_one_minus_rho
            )
            
            # Worst-case losses
            worst_case_losses = conditional_pd * lgd_estimates * ead_amounts
            portfolio_worst_case = np.sum(worst_case_losses)
            
            # Economic Capital = Worst Case Loss - Expected Loss
            economic_capital = portfolio_worst_case - expected_loss
            
            capital_results = {
                'economic_capital': economic_capital,
                'worst_case_loss': portfolio_worst_case,
                'expected_loss': expected_loss,
                'unexpected_loss': economic_capital,  # Same as EC in this context
                'confidence_level': confidence_level,
                'method': method,
                'avg_correlation': avg_correlation,
                'capital_ratio': economic_capital / el_results['total_exposure'] if el_results['total_exposure'] > 0 else 0
            }
            
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            n_simulations = 10000
            portfolio_losses = np.zeros(n_simulations)
            
            # Default correlation structure (simplified)
            if correlation_matrix is not None:
                correlation = correlation_matrix
            else:
                # Simple equicorrelation matrix
                correlation = np.full((n_exposures, n_exposures), 0.15)
                np.fill_diagonal(correlation, 1.0)
            
            # Generate correlated random variables for defaults
            from scipy.stats import multivariate_normal
            
            for sim in range(n_simulations):
                # Generate correlated uniform random variables
                random_normals = multivariate_normal.rvs(
                    mean=np.zeros(n_exposures),
                    cov=correlation,
                    size=1
                ).reshape(-1)
                
                # Convert to uniform variables
                from scipy.stats import norm
                random_uniforms = norm.cdf(random_normals)
                
                # Determine defaults
                defaults = random_uniforms < pd_estimates
                
                # Calculate losses for this simulation
                sim_losses = defaults * lgd_estimates * ead_amounts
                portfolio_losses[sim] = np.sum(sim_losses)
            
            # Calculate capital metrics
            worst_case_loss = np.percentile(portfolio_losses, confidence_level * 100)
            economic_capital = worst_case_loss - expected_loss
            
            capital_results = {
                'economic_capital': economic_capital,
                'worst_case_loss': worst_case_loss,
                'expected_loss': expected_loss,
                'unexpected_loss': economic_capital,
                'confidence_level': confidence_level,
                'method': method,
                'simulation_results': {
                    'mean_loss': np.mean(portfolio_losses),
                    'std_loss': np.std(portfolio_losses),
                    'var_95': np.percentile(portfolio_losses, 95),
                    'var_99': np.percentile(portfolio_losses, 99),
                    'var_999': np.percentile(portfolio_losses, 99.9)
                },
                'capital_ratio': economic_capital / el_results['total_exposure'] if el_results['total_exposure'] > 0 else 0
            }
            
        else:
            raise ValueError(f"Unknown capital calculation method: {method}")
        
        logger.info(f"Economic Capital calculation completed using {method}. "
                   f"EC: {economic_capital:,.0f}, Capital Ratio: {capital_results['capital_ratio']:.4%}")
        
        return capital_results
    
    def calculate_raroc(self, net_income: float,
                       economic_capital: float,
                       expected_loss: float = 0.0,
                       funding_cost: float = 0.0) -> float:
        """
        Calculate Risk-Adjusted Return on Capital (RAROC).
        
        Args:
            net_income: Net income from the exposure/portfolio
            economic_capital: Economic capital requirement
            expected_loss: Expected loss amount
            funding_cost: Cost of funding
            
        Returns:
            RAROC percentage
        """
        if economic_capital == 0:
            return 0.0
        
        # RAROC = (Net Income - Expected Loss - Funding Cost) / Economic Capital
        risk_adjusted_income = net_income - expected_loss - funding_cost
        raroc = (risk_adjusted_income / economic_capital) * 100
        
        logger.info(f"RAROC calculation: {raroc:.2f}% (Income: {net_income:,.0f}, "
                   f"EC: {economic_capital:,.0f}, EL: {expected_loss:,.0f})")
        
        return raroc
    
    def validate_moc_components(self, validation_data: Dict[str, np.ndarray],
                               moc_estimates: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Validate MoC component estimates against out-of-sample data.
        
        Args:
            validation_data: Dictionary with validation dataset
            moc_estimates: Dictionary with MoC component estimates
            
        Returns:
            Dictionary with validation metrics and results
        """
        validation_results = {}
        
        # PD validation
        if 'actual_defaults' in validation_data and 'pd_estimates' in moc_estimates:
            actual_defaults = validation_data['actual_defaults']
            pd_est = moc_estimates['pd_estimates']
            
            # Hosmer-Lemeshow test approximation
            n_bins = 10
            bin_edges = np.percentile(pd_est, np.linspace(0, 100, n_bins + 1))
            
            observed_default_rates = []
            expected_default_rates = []
            bin_counts = []
            
            for i in range(n_bins):
                if i == 0:
                    mask = pd_est <= bin_edges[i + 1]
                elif i == n_bins - 1:
                    mask = pd_est > bin_edges[i]
                else:
                    mask = (pd_est > bin_edges[i]) & (pd_est <= bin_edges[i + 1])
                
                if np.sum(mask) > 0:
                    observed_rate = np.mean(actual_defaults[mask])
                    expected_rate = np.mean(pd_est[mask])
                    count = np.sum(mask)
                    
                    observed_default_rates.append(observed_rate)
                    expected_default_rates.append(expected_rate)
                    bin_counts.append(count)
            
            # Calculate calibration metrics
            observed_rates = np.array(observed_default_rates)
            expected_rates = np.array(expected_default_rates)
            
            # Mean absolute error
            pd_mae = np.mean(np.abs(observed_rates - expected_rates))
            
            # Root mean square error  
            pd_rmse = np.sqrt(np.mean((observed_rates - expected_rates) ** 2))
            
            validation_results['pd_validation'] = {
                'mae': pd_mae,
                'rmse': pd_rmse,
                'observed_rates': observed_rates.tolist(),
                'expected_rates': expected_rates.tolist(),
                'bin_counts': bin_counts,
                'overall_default_rate': np.mean(actual_defaults),
                'predicted_default_rate': np.mean(pd_est)
            }
        
        # LGD validation
        if 'actual_lgds' in validation_data and 'lgd_estimates' in moc_estimates:
            actual_lgds = validation_data['actual_lgds']
            lgd_est = moc_estimates['lgd_estimates']
            
            # Basic validation metrics
            lgd_mae = np.mean(np.abs(actual_lgds - lgd_est))
            lgd_rmse = np.sqrt(np.mean((actual_lgds - lgd_est) ** 2))
            lgd_correlation = np.corrcoef(actual_lgds, lgd_est)[0, 1]
            
            validation_results['lgd_validation'] = {
                'mae': lgd_mae,
                'rmse': lgd_rmse,
                'correlation': lgd_correlation,
                'mean_actual': np.mean(actual_lgds),
                'mean_predicted': np.mean(lgd_est),
                'bias': np.mean(lgd_est - actual_lgds)
            }
        
        # Overall validation summary
        validation_results['validation_summary'] = {
            'components_validated': list(validation_results.keys()),
            'validation_date': pd.Timestamp.now().isoformat(),
            'sample_size': len(validation_data.get('actual_defaults', []))
        }
        
        logger.info(f"MoC validation completed for {len(validation_results)} components")
        return validation_results

if __name__ == "__main__":
    # Example usage
    moc = ModelOfCredit()
    
    # Generate sample data for demonstration
    n_exposures = 1000
    np.random.seed(42)
    
    # Sample PD calibration
    scores = np.random.normal(0, 1, n_exposures)
    calibration_data = {
        'scores': np.random.normal(0, 1, 5000),
        'defaults': np.random.binomial(1, 0.05, 5000)
    }
    
    pd_results = moc.calculate_pd_components(scores, calibration_data, method='logistic')
    print(f"PD Results: Mean = {pd_results['mean_pd']:.4f}")
    
    # Sample LGD estimation
    recovery_data = np.random.beta(3, 2, 1000)  # Recovery rates
    lgd_results = moc.estimate_lgd_distribution(recovery_data)
    print(f"LGD Results: Mean = {lgd_results['mean_lgd']:.4f}")
    
    # Economic Capital calculation
    ead_amounts = np.random.lognormal(12, 1, n_exposures)
    lgd_estimates = np.full(n_exposures, lgd_results['mean_lgd'])
    
    ec_results = moc.calculate_economic_capital(
        pd_results['pd_estimates'], lgd_estimates, ead_amounts, 
        method='vasicek'
    )
    print(f"Economic Capital: {ec_results['economic_capital']:,.0f}")
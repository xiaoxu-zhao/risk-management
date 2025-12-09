"""
Risk Metrics Module
==================

Financial risk calculations based on PD, LGD, and EAD.
Implements standard Basel II/III formulas for regulatory capital and risk assessment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any

class RiskMetrics:
    def __init__(self):
        pass

    def calculate_expected_loss(self, 
                              pd_array: np.ndarray, 
                              ead_array: np.ndarray, 
                              lgd: float = 0.45) -> Dict[str, float]:
        """
        Calculates Expected Loss (EL) = PD * LGD * EAD.
        """
        # Ensure inputs are numpy arrays
        pd_array = np.asarray(pd_array)
        ead_array = np.asarray(ead_array)
        lgd_array = np.asarray(lgd)
        
        el_per_loan = pd_array * lgd_array * ead_array
        total_el = np.sum(el_per_loan)
        
        return {
            'total_el': total_el,
            'el_rate': total_el / np.sum(ead_array) if np.sum(ead_array) > 0 else 0.0
        }

    def calculate_regulatory_capital(self, 
                                   exposures: np.ndarray, 
                                   pds: np.ndarray, 
                                   lgds: np.ndarray) -> Dict[str, float]:
        """
        Calculates Regulatory Capital using Basel IRB formula.
        """
        # Call portfolio var to get capital requirement
        capital = self.calculate_portfolio_var(pds, exposures, lgds)
        return {'total_capital_needed': capital}

    def calculate_portfolio_var(self, 
                              pd_array: np.ndarray, 
                              ead_array: np.ndarray, 
                              lgd: float = 0.45, 
                              confidence_level: float = 0.999) -> float:
        """
        Calculates Credit Value at Risk (VaR) using the Vasicek Single Factor Model.
        """
        pd_array = np.asarray(pd_array)
        ead_array = np.asarray(ead_array)
        lgd_array = np.asarray(lgd)
        
        # Avoid log(0) or division by zero errors
        pd_array = np.clip(pd_array, 1e-6, 1 - 1e-6)
        
        # Asset correlation (R) - simplified Basel formula for retail exposures
        R = 0.03 + 0.13 * np.exp(-35 * pd_array)
        
        # Inverse normal of confidence level
        from scipy.stats import norm
        q = norm.ppf(confidence_level)
        
        # Conditional PD (Worst-case PD at confidence level)
        term1 = norm.ppf(pd_array)
        term2 = np.sqrt(R) * q
        term3 = np.sqrt(1 - R)
        conditional_pd = norm.cdf((term1 + term2) / term3)
        
        # Capital Requirement (K) per loan
        capital_per_loan = ead_array * lgd_array * (conditional_pd - pd_array)
        
        # VaR is the sum of capital requirements
        var = np.sum(capital_per_loan)
        return var

    def stress_test_portfolio(self, pds: np.ndarray, scenarios: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Performs stress testing by adjusting PDs based on scenarios.
        """
        results = {}
        for name, params in scenarios.items():
            multiplier = params.get('pd_multiplier', 1.0)
            stressed_pds = np.clip(pds * multiplier, 0, 1)
            results[name] = {
                'pd_increase': np.mean(stressed_pds) - np.mean(pds),
                'mean_pd': np.mean(stressed_pds)
            }
        return results

    def calculate_credit_var_monte_carlo(self, 
                                       exposures: np.ndarray, 
                                       pds: np.ndarray, 
                                       lgds: np.ndarray, 
                                       n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Calculates Credit VaR using Monte Carlo simulation.
        """
        n_loans = len(exposures)
        # Simple correlation model (single factor)
        rho = 0.1 # Asset correlation
        
        # Systemic factor Z
        Z = np.random.normal(0, 1, n_simulations)
        
        # Idiosyncratic factors
        losses = np.zeros(n_simulations)
        
        # Vectorized simulation
        # Threshold for default: N^-1(PD)
        from scipy.stats import norm
        default_thresholds = norm.ppf(pds)
        
        for i in range(n_simulations):
            # Asset value A = sqrt(rho)*Z + sqrt(1-rho)*epsilon
            epsilon = np.random.normal(0, 1, n_loans)
            asset_values = np.sqrt(rho) * Z[i] + np.sqrt(1 - rho) * epsilon
            
            defaults = asset_values < default_thresholds
            loss = np.sum(exposures * lgds * defaults)
            losses[i] = loss
            
        var_999 = np.percentile(losses, 99.9)
        expected_shortfall = np.mean(losses[losses >= var_999])
        
        return {
            'credit_var': var_999,
            'expected_shortfall': expected_shortfall,
            'loss_distribution': losses
        }

class IFRS9Calculator:
    """
    Implements IFRS 9 Impairment logic:
    - Staging (Stage 1, 2, 3) based on SICR (Significant Increase in Credit Risk)
    - 12-month vs Lifetime ECL
    - Forward-Looking Information (Macroeconomic scenarios)
    """
    
    def __init__(self):
        pass
        
    def determine_stage(self, 
                       current_pd: np.ndarray, 
                       original_pd: np.ndarray, 
                       days_past_due: Optional[np.ndarray] = None,
                       sicr_threshold: float = 2.0) -> np.ndarray:
        """
        Assigns IFRS 9 Stages (1, 2, 3).
        """
        n_samples = len(current_pd)
        stages = np.ones(n_samples, dtype=int) # Default to Stage 1
        
        # Check for SICR (Significant Increase in Credit Risk)
        # Rule 1: Relative threshold (e.g., PD doubled)
        sicr_mask = (current_pd / (original_pd + 1e-6)) > sicr_threshold
        stages[sicr_mask] = 2
        
        # Rule 2: Absolute threshold (e.g., PD > 20%)
        stages[current_pd > 0.20] = 2
        
        # Rule 3: Days Past Due (Backstop)
        if days_past_due is not None:
            stages[days_past_due > 30] = 2
            stages[days_past_due > 90] = 3
            
        # Defaulted loans are Stage 3
        stages[current_pd >= 0.99] = 3
        
        return stages

    def calculate_lifetime_pd(self, pd_12m: np.ndarray, remaining_term_years: np.ndarray) -> np.ndarray:
        """
        Approximates Lifetime PD from 12-month PD using survival probability.
        """
        survival_prob_1y = 1.0 - pd_12m
        survival_prob_lifetime = np.power(survival_prob_1y, remaining_term_years)
        return 1.0 - survival_prob_lifetime

    def apply_macro_scenarios(self, 
                            pd_12m: np.ndarray, 
                            scenarios: Dict[str, float], 
                            weights: Dict[str, float]) -> np.ndarray:
        """
        Adjusts PDs based on weighted macroeconomic scenarios.
        """
        weighted_pd = np.zeros_like(pd_12m)
        
        for name, multiplier in scenarios.items():
            weight = weights.get(name, 0.0)
            scenario_pd = np.clip(pd_12m * multiplier, 0, 1)
            weighted_pd += scenario_pd * weight
            
        return weighted_pd

    def calculate_ecl(self,
                     exposures: np.ndarray,
                     pd_12m: np.ndarray,
                     original_pd: np.ndarray,
                     lgd: np.ndarray,
                     remaining_term_years: np.ndarray,
                     days_past_due: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculates IFRS 9 Expected Credit Loss (ECL).
        """
        # 1. Determine Stage
        stages = self.determine_stage(pd_12m, original_pd, days_past_due)
        
        # 2. Calculate Lifetime PD
        pd_lifetime = self.calculate_lifetime_pd(pd_12m, remaining_term_years)
        
        # 3. Calculate ECL per loan
        ecl = np.zeros_like(exposures)
        
        # Stage 1: 12-month ECL
        mask_s1 = (stages == 1)
        ecl[mask_s1] = exposures[mask_s1] * pd_12m[mask_s1] * lgd[mask_s1]
        
        # Stage 2 & 3: Lifetime ECL
        mask_s23 = (stages >= 2)
        ecl[mask_s23] = exposures[mask_s23] * pd_lifetime[mask_s23] * lgd[mask_s23]
        
        return {
            'total_ecl': np.sum(ecl),
            'stages': stages,
            'ecl_by_stage': {
                'Stage 1': np.sum(ecl[stages == 1]),
                'Stage 2': np.sum(ecl[stages == 2]),
                'Stage 3': np.sum(ecl[stages == 3])
            },
            'count_by_stage': {
                'Stage 1': np.sum(stages == 1),
                'Stage 2': np.sum(stages == 2),
                'Stage 3': np.sum(stages == 3)
            }
        }
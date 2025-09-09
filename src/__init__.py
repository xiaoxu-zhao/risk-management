"""
Credit Risk Management Demo
==========================

A comprehensive credit risk management toolkit demonstrating:
- Credit risk modeling and validation
- Regulatory risk metrics (Basel III compliance)
- Portfolio risk analysis
- Model interpretability and fairness

Author: xiaoxu(Ivan) zhao
License: MIT
"""

__version__ = "1.0.0"
__author__ = "xiaoxu(Ivan) zhao"

from .data_loader import CreditDataLoader
from .feature_engineering import FeatureEngineer
from .models import CreditRiskModels
from .risk_metrics import RiskMetrics
from .visualization import RiskVisualizer

__all__ = [
    "CreditDataLoader",
    "FeatureEngineer", 
    "CreditRiskModels",
    "RiskMetrics",
    "RiskVisualizer"
]
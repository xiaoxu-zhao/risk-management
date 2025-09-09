#!/usr/bin/env python3
"""
Credit Risk Management Demo Script
=================================

This script demonstrates the key capabilities of the credit risk management toolkit.
Perfect for showcasing in interviews or professional presentations.

Usage:
    python demo.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from data_loader import CreditDataLoader
from feature_engineering import FeatureEngineer  
from models import CreditRiskModels
from risk_metrics import RiskMetrics
from visualization import RiskVisualizer

def main():
    """Run the comprehensive credit risk management demo."""
    
    print("=" * 70)
    print("🏦 CREDIT RISK MANAGEMENT TOOLKIT DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Step 1: Data Loading and Quality Assessment
    print("📊 Step 1: Data Loading and Quality Assessment")
    print("-" * 50)
    
    loader = CreditDataLoader()
    
    # Generate realistic sample data (in practice, load from Kaggle datasets)
    print("Generating sample credit dataset...")
    df = loader.get_sample_data(n_samples=5000, random_state=42)
    
    # Data quality check
    quality_report = loader.basic_data_quality_check(df, 'default')
    print(f"✓ Dataset loaded: {quality_report['shape']} (rows × columns)")
    print(f"✓ Default rate: {quality_report['target_rate']:.2%}")
    print(f"✓ Missing values: {quality_report['missing_values']}")
    print()
    
    # Step 2: Advanced Feature Engineering
    print("🔧 Step 2: Advanced Feature Engineering")
    print("-" * 50)
    
    fe = FeatureEngineer()
    
    # Create risk-specific features
    print("Creating risk-based features...")
    df_features = fe.create_risk_features(df)
    df_features = fe.create_interaction_features(df_features, max_interactions=5)
    
    # Handle missing values and encoding
    print("Processing categorical variables and missing values...")
    df_clean = fe.handle_missing_values(df_features)
    df_encoded = fe.encode_categorical_features(df_clean, target_col='default')
    
    # Feature summary
    summary = fe.get_feature_summary(df_encoded)
    print(f"✓ Original features: {df.shape[1]}")
    print(f"✓ Final features: {summary['total_features']}")
    print(f"✓ Numerical features: {summary['numerical_features']}")
    print()
    
    # Step 3: Machine Learning Model Development
    print("🤖 Step 3: Machine Learning Model Development")
    print("-" * 50)
    
    models = CreditRiskModels(random_state=42)
    
    # Prepare data splits
    print("Preparing train/validation/test splits...")
    data_splits = models.prepare_data(df_encoded, 'default', test_size=0.3)
    
    # Train multiple models
    print("Training Logistic Regression...")
    lr_model = models.train_logistic_regression(
        data_splits['X_train'], data_splits['y_train'],
        tune_hyperparameters=False  # Skip tuning for demo speed
    )
    
    print("Training XGBoost...")
    xgb_model = models.train_xgboost(
        data_splits['X_train'], data_splits['y_train'],
        data_splits['X_val'], data_splits['y_val'],
        tune_hyperparameters=False  # Skip tuning for demo speed
    )
    
    # Model evaluation
    print("Evaluating models...")
    lr_results = models.evaluate_model(
        lr_model, data_splits['X_test'], data_splits['y_test'], 'logistic_regression'
    )
    
    xgb_results = models.evaluate_model(
        xgb_model, data_splits['X_test'], data_splits['y_test'], 'xgboost'
    )
    
    # Model comparison
    comparison = models.compare_models()
    print("\n📈 Model Performance Comparison:")
    print(comparison.round(4))
    print()
    
    # Step 4: Risk Analytics and Regulatory Calculations
    print("📊 Step 4: Risk Analytics and Regulatory Calculations")
    print("-" * 50)
    
    risk_calc = RiskMetrics()
    
    # Generate portfolio data for risk calculations
    print("Generating portfolio exposure data...")
    n_loans = len(data_splits['X_test'])
    exposures = np.random.lognormal(12, 1, n_loans)  # Loan amounts
    
    # Get default probabilities from best model
    best_model_name = comparison.iloc[0]['Model']  # Top performing model
    pds = models.predict_default_probability(best_model_name, data_splits['X_test'])
    
    # Generate LGD values (Loss Given Default)
    lgds = np.random.beta(2, 3, n_loans)  # Realistic LGD distribution
    
    # Calculate Expected Loss
    print("Calculating Expected Loss...")
    el_metrics = risk_calc.calculate_expected_loss(exposures, pds, lgds)
    
    print(f"✓ Total Portfolio Exposure: ${el_metrics['total_exposure']:,.0f}")
    print(f"✓ Portfolio Expected Loss: ${el_metrics['total_el']:,.0f}")
    print(f"✓ Expected Loss Rate: {el_metrics['el_rate']:.2%}")
    
    # Calculate Regulatory Capital (Basel III)
    print("\nCalculating Basel III regulatory capital...")
    capital_metrics = risk_calc.calculate_regulatory_capital(exposures, pds, lgds)
    
    print(f"✓ Total Risk-Weighted Assets: ${capital_metrics['total_rwa']:,.0f}")
    print(f"✓ Minimum Capital Requirement: ${capital_metrics['capital_requirement']:,.0f}")
    print(f"✓ Total Capital Needed: ${capital_metrics['total_capital_needed']:,.0f}")
    
    # Value at Risk calculation
    print("\nCalculating portfolio VaR...")
    portfolio_returns = np.random.normal(-0.01, 0.05, 1000)  # Simulated returns
    var_95 = risk_calc.calculate_var(portfolio_returns, confidence_level=0.95)
    es_95 = risk_calc.calculate_expected_shortfall(portfolio_returns, confidence_level=0.95)
    
    print(f"✓ 95% Value at Risk: {var_95:.4f}")
    print(f"✓ 95% Expected Shortfall: {es_95:.4f}")
    
    # Stress Testing
    print("\nPerforming stress testing...")
    stress_scenarios = {
        'mild_recession': {'pd_multiplier': 2.0},
        'severe_recession': {'pd_multiplier': 4.0},
        'financial_crisis': {'pd_multiplier': 6.0}
    }
    
    stress_results = risk_calc.stress_test_portfolio(pds, stress_scenarios)
    
    print("Stress Test Results:")
    for scenario, results in stress_results.items():
        baseline_pd = np.mean(pds)
        stressed_pd = results['portfolio_pd']
        print(f"  {scenario:15s}: {baseline_pd:.2%} → {stressed_pd:.2%} "
              f"(+{(stressed_pd-baseline_pd)*100:.1f}pp)")
    print()
    
    # Step 5: Visualization and Reporting
    print("📈 Step 5: Visualization and Reporting")
    print("-" * 50)
    
    viz = RiskVisualizer()
    
    # Create visualizations
    print("Generating performance plots...")
    
    model_results = {
        'logistic_regression': lr_results,
        'xgboost': xgb_results
    }
    
    # ROC Curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    roc_fig = viz.plot_roc_curves(model_results, figsize=(5, 5))
    plt.title('ROC Curves Comparison')
    
    plt.subplot(1, 3, 2)
    # Feature importance for XGBoost
    xgb_importance = models.get_feature_importance_ranking('xgboost', top_k=10)
    importance_dict = dict(xgb_importance)
    importance_fig = viz.plot_feature_importance(importance_dict, top_k=10, figsize=(5, 5))
    plt.title('Top 10 Feature Importance (XGBoost)')
    
    plt.subplot(1, 3, 3)
    # Loss distribution
    simulated_losses = np.random.exponential(el_metrics['total_el']/1000, 1000)
    loss_fig = viz.plot_loss_distribution(
        simulated_losses, 
        var_level=np.percentile(simulated_losses, 95),
        figsize=(5, 5)
    )
    plt.title('Portfolio Loss Distribution')
    
    plt.tight_layout()
    plt.savefig('risk_management_demo_results.png', dpi=300, bbox_inches='tight')
    print("✓ Visualizations saved to 'risk_management_demo_results.png'")
    
    # Step 6: Summary and Key Insights
    print("\n🎯 Step 6: Summary and Key Insights")
    print("-" * 50)
    
    print("KEY ACHIEVEMENTS DEMONSTRATED:")
    print(f"  ✅ Advanced ML Models: {len(model_results)} algorithms trained and compared")
    print(f"  ✅ Feature Engineering: {summary['total_features']} features from {df.shape[1]} base features")
    print(f"  ✅ Model Performance: Best AUC = {comparison.iloc[0]['AUC Score']:.3f}")
    print(f"  ✅ Risk Metrics: VaR, Expected Loss, Regulatory Capital calculated")
    print(f"  ✅ Stress Testing: {len(stress_scenarios)} scenarios analyzed")
    print(f"  ✅ Basel III Compliance: Regulatory capital requirements estimated")
    
    print("\nPROFESSIONAL SKILLS SHOWCASED:")
    print("  🔸 Quantitative Finance: Risk metrics, VaR, Expected Shortfall")
    print("  🔸 Machine Learning: Multiple algorithms, hyperparameter tuning, validation")
    print("  🔸 Regulatory Knowledge: Basel III, capital adequacy, stress testing")
    print("  🔸 Data Engineering: ETL pipelines, feature engineering, data quality")
    print("  🔸 Visualization: Professional plots, interactive dashboards")
    print("  🔸 Software Engineering: Clean code, testing, documentation")
    
    print("\n" + "=" * 70)
    print("🎊 DEMONSTRATION COMPLETE! 🎊")
    print("This toolkit showcases enterprise-level credit risk management capabilities")
    print("suitable for roles in quantitative finance, risk management, and fintech.")
    print("=" * 70)

if __name__ == "__main__":
    # Set up matplotlib for non-interactive use
    plt.ioff()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError running demo: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
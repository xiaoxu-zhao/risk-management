#!/usr/bin/env python3
"""
Credit Risk Management Demo Script
=================================

This script demonstrates the key capabilities of the credit risk management toolkit.
Perfect for showcasing in interviews or professional presentations.

Usage:
    python demo.py
"""
# %% 
# main script

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Add src to path
sys.path.append('src')

from data_loader import CreditDataLoader
from lending_club_preprocessing import LendingClubPreprocessor
from feature_engineering import FeatureEngineer  
from models import CreditRiskModels
from risk_metrics import RiskMetrics
from visualization import RiskVisualizer
from moc import ModelOfCredit
from monitoring import ModelMonitor

def main():
    """Run the comprehensive credit risk management demo."""
    
    print("=" * 70)
    print("🏦 ENHANCED CREDIT RISK MANAGEMENT TOOLKIT DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Step 1: Data Loading, Processing, and Quality Assessment
    print("📊 Step 1: Data Processing & Quality Assessment")
    print("-" * 50)
    
    # Download Lending Club dataset using kagglehub
    # print("login to kagglehub")
    # kagglehub.login() # login to kagglehub for downloading the dataset
    # lc_path = kagglehub.dataset_download("wordsforthewise/lending-club")
    # print("Path to dataset files:", lc_path)
    data_path = "C:\\Users\\xzhaox\\Desktop\\OneDrive\\risk_management\\risk-management\\data"
    # Load Lending Club datasets (accepted/rejected) from path
    loader = CreditDataLoader(data_path=data_path)
    datasets = loader.load_lending_club()
    accepted_df = datasets.get('accepted')
    # rejected_df = datasets.get('rejected')
    print("Accepted loans dataset loaded:", accepted_df.shape if accepted_df is not None else "Not found")
    # print("Rejected loans dataset loaded:", rejected_df.shape if rejected_df is not None else "Not found")

    # Prepare accepted dataset using LC-specific preprocessor
    lc_prep = LendingClubPreprocessor()
    df_prepared = lc_prep.prepare_accepted(accepted_df)

    # Quality check after preparation
    quality_report = loader.basic_data_quality_check(df_prepared, 'default')
    print(f"✓ Accepted dataset prepared: {quality_report['shape']} (rows × columns)")
    print(f"✓ Default rate: {quality_report['target_rate']:.2%}")
    
    # Step 2: Advanced Feature Engineering & Data Preparation
    print("\n🔧 Step 2: Feature Engineering & Data Preparation")
    print("-" * 50)
    
    fe = FeatureEngineer()
    
    # Create comprehensive risk features
    print("Creating advanced risk features...")
    # Start from prepared LC dataset
    df_features = fe.create_risk_features(df_prepared)
    df_features = fe.create_interaction_features(df_features, max_interactions=5)
    df_features = fe.create_behavioral_features(df_features)
    
    # Handle missing values and encode categoricals
    df_imputed = fe.handle_missing_values(df_features)
    df_encoded = fe.encode_categorical_features(df_imputed, target_col='default')
    
    # Feature selection
    # Build X/y and select features
    X = df_encoded.drop(columns=['default'])
    y = df_encoded['default']
    X_selected_df, selected_features = fe.select_features(X, y, method='mutual_info', k=20)
    
    summary = fe.get_feature_summary(df_encoded)
    print(f"✓ Feature engineering complete: {summary['total_features']} features in dataset")
    print(f"✓ Selected top {len(selected_features)} most informative features")
    
    # Step 3: Model Development & Scoring Pipeline
    print("\n🤖 Step 3: Model Development & Scoring Pipeline")
    print("-" * 50)
    
    models = CreditRiskModels(random_state=42)
    
    # Prepare data with proper splits
    print("Preparing training/validation/test datasets...")
    df_for_model = X_selected_df.copy()
    df_for_model['default'] = y.values
    data_splits = models.prepare_data(df_for_model, 'default', test_size=0.3)
    
    # Train multiple models
    print("Training ensemble of ML models...")
    lr_model = models.train_logistic_regression(
        data_splits['X_train'], data_splits['y_train'],
        data_splits['X_val'], data_splits['y_val']
    )
    
    xgb_model = models.train_xgboost(
        data_splits['X_train'], data_splits['y_train'],
        data_splits['X_val'], data_splits['y_val']
    )
    
    # Enhanced model calibration
    print("Applying advanced model calibration...")
    lr_cal_results = models.calibrate_model_advanced(
        'logistic_regression', data_splits['X_val'], data_splits['y_val'], method='isotonic'
    )
    xgb_cal_results = models.calibrate_model_advanced(
        'xgboost', data_splits['X_val'], data_splits['y_val'], method='isotonic'
    )
    
    print(f"✓ Logistic Regression calibration improvement: {lr_cal_results['brier_score_improvement']:.4f}")
    print(f"✓ XGBoost calibration improvement: {xgb_cal_results['brier_score_improvement']:.4f}")
    
    # Create production scoring pipeline
    scoring_pipeline = models.create_scoring_pipeline('xgboost', selected_features)
    print("✓ Production scoring pipeline created")
    
    # Evaluate models
    lr_results = models.evaluate_model(
        lr_model, data_splits['X_test'], data_splits['y_test'], 'logistic_regression'
    )
    
    xgb_results = models.evaluate_model(
        xgb_model, data_splits['X_test'], data_splits['y_test'], 'xgboost'
    )
    
    comparison = models.compare_models()
    print(f"✓ Model evaluation complete. Best AUC: {comparison.iloc[0]['AUC Score']:.4f}")
    
    # Step 4: Model of Credit (MoC) Implementation
    print("\n💼 Step 4: Model of Credit (MoC) Implementation")
    print("-" * 50)
    
    moc = ModelOfCredit()
    
    # Generate PD estimates using model scores
    test_scores = models.predict_default_probability('xgboost', data_splits['X_test'])
    
    # Create calibration data for MoC
    calibration_data = {
        'scores': models.predict_default_probability('xgboost', data_splits['X_val']),
        'defaults': data_splits['y_val'].values
    }
    
    # Calculate MoC components
    print("Calculating MoC components...")
    
    # 1. PD Calculation
    pd_results = moc.calculate_pd_components(test_scores, calibration_data, method='logistic')
    print(f"✓ PD estimates: Mean = {pd_results['mean_pd']:.4f}, Std = {pd_results['std_pd']:.4f}")
    
    # 2. LGD Estimation
    recovery_rates = np.random.beta(3, 2, len(test_scores))  # Simulated recovery data
    lgd_results = moc.estimate_lgd_distribution(recovery_rates)
    print(f"✓ LGD distribution: Mean = {lgd_results['mean_lgd']:.4f}")
    
    # 3. EAD Calculation
    committed_amounts = np.random.uniform(50000, 500000, len(test_scores))
    outstanding_amounts = committed_amounts * np.random.uniform(0.2, 0.8, len(test_scores))
    ead_results = moc.calculate_ead_conversion(committed_amounts, outstanding_amounts)
    print(f"✓ EAD calculated: Mean CCF = {ead_results['mean_ccf']:.3f}")
    
    # 4. Expected Loss
    el_results = moc.calculate_expected_loss(
        pd_results['pd_estimates'], 
        np.full(len(test_scores), lgd_results['mean_lgd']),
        ead_results['ead_amounts']
    )
    print(f"✓ Expected Loss: ${el_results['total_el']:,.0f} ({el_results['el_rate']:.4%})")
    
    # 5. Economic Capital
    ec_results = moc.calculate_economic_capital(
        pd_results['pd_estimates'],
        np.full(len(test_scores), lgd_results['mean_lgd']),
        ead_results['ead_amounts'],
        method='vasicek'
    )
    print(f"✓ Economic Capital: ${ec_results['economic_capital']:,.0f}")
    
    # 6. RAROC Calculation
    net_income = 2000000  # Simulated income
    raroc = moc.calculate_raroc(net_income, ec_results['economic_capital'], el_results['total_el'])
    print(f"✓ RAROC: {raroc:.2f}%")
    
    # Step 5: Model Monitoring & Stability Testing
    print("\n🔍 Step 5: Model Monitoring & Stability Testing")
    print("-" * 50)
    
    monitor = ModelMonitor()
    
    # Create reference and current datasets for monitoring
    reference_scores = models.predict_default_probability('xgboost', data_splits['X_train'][:1000])
    current_scores = models.predict_default_probability('xgboost', data_splits['X_test'])
    
    # Population Stability Index
    psi_results = monitor.calculate_population_stability_index(reference_scores, current_scores)
    print(f"✓ Population Stability Index: {psi_results['psi']:.4f} ({psi_results['interpretation']})")
    
    # KS Statistic for distribution comparison
    ks_results = monitor.calculate_ks_statistic(reference_scores, current_scores)
    print(f"✓ KS Statistic: {ks_results['ks_statistic']:.4f}")
    
    # Model performance monitoring
    perf_metrics = monitor.calculate_model_performance_metrics(
        data_splits['y_test'], current_scores
    )
    print(f"✓ Current AUC: {perf_metrics['auc_roc']:.4f}, KS Power: {perf_metrics['ks_statistic']:.4f}")
    
    # Calibration error analysis
    cal_error = monitor.calculate_calibration_error(data_splits['y_test'], current_scores)
    print(f"✓ Expected Calibration Error: {cal_error['expected_calibration_error']:.4f}")
    
    # Concept drift detection
    reference_data = {
        'scores': reference_scores,
        'targets': data_splits['y_train'][:1000].values
    }
    current_data = {
        'scores': current_scores,
        'targets': data_splits['y_test'].values
    }
    
    drift_results = monitor.detect_concept_drift(reference_data, current_data)
    print(f"✓ Concept Drift: {drift_results['drift_detected']} ({drift_results['drift_strength']})")
    
    # Generate comprehensive monitoring report
    monitoring_data = {
        'psi': psi_results,
        'ks': ks_results,
        'performance': perf_metrics,
        'calibration': cal_error,
        'drift': drift_results
    }
    
    report = monitor.generate_monitoring_report('CreditRiskModel_Enhanced', monitoring_data)
    print(f"✓ Monitoring report generated with {report['executive_summary']['total_alerts']} alerts")
    
    # Step 6: Advanced Risk Analytics
    print("\n📈 Step 6: Advanced Risk Analytics")
    print("-" * 50)
    
    risk_calc = RiskMetrics()
    
    # Portfolio risk metrics using MoC outputs
    exposures = ead_results['ead_amounts']
    pds = pd_results['pd_estimates']
    lgds = np.full(len(pds), lgd_results['mean_lgd'])
    
    # Enhanced expected loss calculation
    el_metrics = risk_calc.calculate_expected_loss(exposures, pds, lgds)
    print(f"✓ Portfolio Expected Loss: ${el_metrics['total_el']:,.2f}")
    
    # Regulatory capital calculation
    capital_metrics = risk_calc.calculate_regulatory_capital(exposures, pds, lgds)
    print(f"✓ Regulatory Capital Requirement: ${capital_metrics['total_capital_needed']:,.2f}")
    
    # Advanced stress testing
    stress_scenarios = {
        'base_case': {'pd_multiplier': 1.0},
        'mild_recession': {'pd_multiplier': 2.0},
        'severe_recession': {'pd_multiplier': 4.0},
        'financial_crisis': {'pd_multiplier': 6.0}
    }
    
    stress_results = risk_calc.stress_test_portfolio(pds, stress_scenarios)
    print("✓ Stress testing results:")
    for scenario, results in stress_results.items():
        increase = results['pd_increase'] * 100
        print(f"   • {scenario.title()}: +{increase:.1f}% PD increase")
    
    # Credit VaR using Monte Carlo
    credit_var_results = risk_calc.calculate_credit_var_monte_carlo(
        exposures, pds, lgds, n_simulations=1000
    )
    print(f"✓ Credit VaR (99.9%): ${credit_var_results['credit_var']:,.0f}")
    print(f"✓ Expected Shortfall: ${credit_var_results['expected_shortfall']:,.0f}")
    
    # Step 7: Enhanced Visualization & Reporting
    print("\n📊 Step 7: Enhanced Visualization & Reporting")  
    print("-" * 50)
    
    viz = RiskVisualizer()
    
    # Create comprehensive visualizations
    print("Creating enhanced risk management visualizations...")
    plt.figure(figsize=(20, 12))
    
    # Model performance comparison
    plt.subplot(2, 4, 1)
    model_results = {
        'logistic_regression': lr_results,
        'xgboost': xgb_results
    }
    roc_fig = viz.plot_roc_curves(model_results, figsize=(5, 5))
    plt.title('Model ROC Comparison')
    
    # Feature importance
    plt.subplot(2, 4, 2)
    xgb_importance = models.get_feature_importance_ranking('xgboost', top_k=10)
    importance_dict = dict(xgb_importance)
    importance_fig = viz.plot_feature_importance(importance_dict, top_k=10, figsize=(5, 5))
    plt.title('Top 10 Feature Importance')
    
    # Calibration plot
    plt.subplot(2, 4, 3)
    calibration_fig = viz.plot_calibration_curve(
        data_splits['y_test'], current_scores, figsize=(5, 5)
    )
    plt.title('Model Calibration')
    
    # Loss distribution
    plt.subplot(2, 4, 4)
    loss_distribution = credit_var_results['loss_distribution']
    loss_fig = viz.plot_loss_distribution(
        loss_distribution,
        var_level=credit_var_results['credit_var'],
        figsize=(5, 5)
    )
    plt.title('Credit Loss Distribution')
    
    # PSI monitoring
    plt.subplot(2, 4, 5)
    psi_fig = viz.plot_psi_monitoring(
        psi_results['bin_details']['reference_percentages'],
        psi_results['bin_details']['current_percentages'],
        figsize=(5, 5)
    )
    plt.title(f'Population Stability (PSI: {psi_results["psi"]:.3f})')
    
    # Stress test results
    plt.subplot(2, 4, 6)
    scenario_names = list(stress_scenarios.keys())
    pd_increases = [stress_results[scenario]['pd_increase'] for scenario in scenario_names]
    stress_fig = viz.plot_stress_test_results(scenario_names, pd_increases, figsize=(5, 5))
    plt.title('Stress Testing Results')
    
    # Economic capital allocation
    plt.subplot(2, 4, 7)
    # Create sample allocation data
    business_lines = ['Retail', 'Corporate', 'SME', 'Cards']
    capital_allocation = [0.4, 0.3, 0.2, 0.1]
    allocation_fig = viz.plot_capital_allocation(business_lines, capital_allocation, figsize=(5, 5))
    plt.title('Economic Capital Allocation')
    
    # Risk metrics summary
    plt.subplot(2, 4, 8)
    risk_metrics_summary = {
        'Expected Loss': el_metrics['total_el'],
        'Economic Capital': ec_results['economic_capital'],
        'Credit VaR': credit_var_results['credit_var'],
        'Regulatory Capital': capital_metrics['total_capital_needed']
    }
    summary_fig = viz.plot_risk_metrics_summary(risk_metrics_summary, figsize=(5, 5))
    plt.title('Risk Metrics Summary')
    
    plt.tight_layout()
    plt.savefig('risk_management_demo_results.png', dpi=300, bbox_inches='tight')
    print("✓ Enhanced visualizations saved to 'risk_management_demo_results.png'")
    
    # Step 8: Summary and Professional Insights
    print("\n🎯 Step 8: Summary and Professional Insights")
    print("-" * 50)
    
    print("ENHANCED CAPABILITIES DEMONSTRATED:")
    print(f"  ✅ Data Processing: Outlier detection, normalization, quality assessment")
    print(f"  ✅ Feature Engineering: {summary['total_features']} features, behavioral patterns")
    print(f"  ✅ Advanced ML Models: Ensemble methods with isotonic calibration")
    print(f"  ✅ Model of Credit: Complete PD/LGD/EAD framework with RAROC")
    print(f"  ✅ Model Monitoring: PSI, KS testing, drift detection, automated alerts")
    print(f"  ✅ Risk Analytics: Credit VaR, stress testing, regulatory capital")
    print(f"  ✅ Production Pipeline: Scoring pipeline with monitoring integration")
    
    print("\nKEY PERFORMANCE METRICS:")
    print(f"  📊 Best Model AUC: {comparison.iloc[0]['AUC Score']:.4f}")
    print(f"  📊 Portfolio RAROC: {raroc:.2f}%")
    print(f"  📊 Expected Loss Rate: {el_results['el_rate']:.4%}")
    print(f"  📊 Population Stability: {psi_results['psi']:.4f}")
    print(f"  📊 Calibration Error: {cal_error['expected_calibration_error']:.4f}")
    
    print("\nENTERPRISE-LEVEL SKILLS SHOWCASED:")
    print("  🔸 Quantitative Finance: MoC framework, economic capital, RAROC")
    print("  🔸 Machine Learning: Advanced calibration, ensemble methods, monitoring")
    print("  🔸 Risk Management: Basel III, stress testing, portfolio analytics")
    print("  🔸 Model Validation: Stability testing, drift detection, backtesting")
    print("  🔸 Production Systems: Scoring pipelines, automated monitoring")
    print("  🔸 Regulatory Compliance: Capital adequacy, model governance")
    
    print("\n" + "=" * 70)
    print("🎊 ENHANCED DEMONSTRATION COMPLETE! 🎊")
    print("This toolkit showcases enterprise-grade credit risk management capabilities")
    print("demonstrating expertise in quantitative finance, ML engineering, and risk governance.")
    print("Perfect for senior roles in banking, fintech, and risk management.")
    print("=" * 70)

if __name__ == "__main__":
    # Set up matplotlib for non-interactive use
    plt.ioff()
    
    try:
        main()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages using: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Please check that all modules are properly implemented.")


# %%

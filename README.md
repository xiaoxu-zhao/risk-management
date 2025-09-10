# Credit Risk Management Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive credit risk management demonstration project showcasing advanced quantitative risk modeling techniques, machine learning applications, and regulatory compliance methodologies. Designed for job-seeking professionals to demonstrate expertise in credit risk analysis and management.

## 🎯 Project Overview

This repository demonstrates core credit risk management capabilities including:

- **Credit Risk Modeling**: Advanced ML models for default prediction
- **Regulatory Compliance**: Basel III capital calculations and stress testing
- **Portfolio Analytics**: Risk metrics, concentration analysis, and optimization
- **Model Validation**: Comprehensive backtesting and performance evaluation
- **Risk Visualization**: Interactive dashboards and reporting tools

## 🚀 Key Features

### 📊 Machine Learning Models
- **Logistic Regression**: Interpretable baseline models with regularization
- **Random Forest**: Ensemble method for non-linear relationships
- **XGBoost/LightGBM**: Gradient boosting for superior predictive performance
- **Advanced Calibration**: Isotonic regression and beta calibration for accurate risk assessment
- **Production Scoring Pipeline**: Complete scoring infrastructure with monitoring integration

### 📈 Risk Metrics & Analytics
- **Value at Risk (VaR)**: Historical, parametric, and Monte Carlo methods
- **Expected Shortfall**: Conditional VaR for tail risk assessment
- **Credit Risk Measures**: PD, LGD, EAD calculations and modeling
- **Regulatory Capital**: Basel III IRB approach implementation
- **Stress Testing**: Scenario analysis and sensitivity testing
- **Credit VaR**: Monte Carlo simulation for portfolio credit risk

### 🔍 Advanced Features
- **Enhanced Data Processing**: Outlier detection, normalization, quality assessment
- **Feature Engineering**: Risk-specific transformations, interaction terms, and behavioral patterns
- **Model of Credit (MoC)**: Complete PD/LGD/EAD framework with RAROC calculations
- **Model Monitoring**: Population Stability Index (PSI), KS testing, drift detection
- **Portfolio Optimization**: Concentration limits and diversification metrics
- **Model Interpretability**: SHAP values and feature importance analysis
- **Backtesting Framework**: Walk-forward validation and performance monitoring
- **Automated Alerts**: Real-time monitoring with configurable thresholds

## 📁 Project Structure

```
risk-management/
├── src/                          # Core source code
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Enhanced data loading and preprocessing
│   ├── feature_engineering.py   # Advanced feature creation and selection
│   ├── models.py                # ML model implementations with calibration
│   ├── risk_metrics.py          # Risk calculations and metrics
│   ├── visualization.py         # Plotting and dashboard tools
│   ├── moc.py                   # Model of Credit (MoC) framework
│   └── monitoring.py            # Model monitoring and drift detection
├── notebooks/                    # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_risk_analysis.ipynb
│   ├── 05_moc_implementation.ipynb
│   └── 06_model_monitoring.ipynb
├── tests/                        # Comprehensive test suite
│   ├── test_data_loader.py
│   ├── test_models.py
│   ├── test_risk_metrics.py
│   ├── test_moc.py
│   └── test_monitoring.py
├── data/                         # Dataset storage (gitignored)
├── docs/                         # Additional documentation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/xiaoxu-zhao/risk-management.git
   cd risk-management
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package in development mode:**
   ```bash
   pip install -e .
   ```

## 📝 Quick Start

### 1. Generate Sample Data
```python
from src.data_loader import CreditDataLoader

# Create data loader
loader = CreditDataLoader()

# Generate sample credit dataset
df = loader.get_sample_data(n_samples=10000, random_state=42)

# Perform data quality check
quality_report = loader.basic_data_quality_check(df, 'default')
print(f"Dataset shape: {quality_report['shape']}")
print(f"Default rate: {quality_report['target_rate']:.2%}")
```

### 2. Feature Engineering
```python
from src.feature_engineering import FeatureEngineer

# Initialize feature engineer
fe = FeatureEngineer()

# Create risk-specific features
df_features = fe.create_risk_features(df)
df_features = fe.create_interaction_features(df_features)

# Handle missing values and encode categorical variables
df_clean = fe.handle_missing_values(df_features)
df_encoded = fe.encode_categorical_features(df_clean, target_col='default')
```

### 3. Model Training and Evaluation
```python
from src.models import CreditRiskModels

# Initialize model trainer
models = CreditRiskModels(random_state=42)

# Prepare data splits
data_splits = models.prepare_data(df_encoded, 'default', test_size=0.3)

# Train multiple models
lr_model = models.train_logistic_regression(
    data_splits['X_train'], data_splits['y_train']
)

xgb_model = models.train_xgboost(
    data_splits['X_train'], data_splits['y_train'],
    data_splits['X_val'], data_splits['y_val']
)

# Evaluate models
lr_results = models.evaluate_model(
    lr_model, data_splits['X_test'], data_splits['y_test'], 'logistic_regression'
)

xgb_results = models.evaluate_model(
    xgb_model, data_splits['X_test'], data_splits['y_test'], 'xgboost'
)

# Compare performance
comparison = models.compare_models()
print(comparison)
```

### 4. Risk Analytics
```python
from src.risk_metrics import RiskMetrics

# Initialize risk calculator
risk_calc = RiskMetrics()

# Generate portfolio data
exposures = np.random.lognormal(12, 1, 1000)
pds = models.predict_default_probability('xgboost', data_splits['X_test'])
lgds = np.random.beta(2, 3, len(pds))

# Calculate Expected Loss
el_metrics = risk_calc.calculate_expected_loss(exposures, pds, lgds)
print(f"Portfolio Expected Loss: ${el_metrics['total_el']:,.2f}")

# Calculate regulatory capital
capital_metrics = risk_calc.calculate_regulatory_capital(exposures, pds, lgds)
print(f"Total Capital Requirement: ${capital_metrics['total_capital_needed']:,.2f}")

# Perform stress testing
stress_scenarios = {
    'mild_recession': {'pd_multiplier': 2.0},
    'severe_recession': {'pd_multiplier': 4.0}
}
stress_results = risk_calc.stress_test_portfolio(pds, stress_scenarios)
```

### 5. Visualization and Reporting
```python
from src.visualization import RiskVisualizer

# Initialize visualizer
viz = RiskVisualizer()

# Plot model performance comparison
model_results = {
    'logistic_regression': lr_results,
    'xgboost': xgb_results
}

# ROC curves
roc_fig = viz.plot_roc_curves(model_results)
plt.show()

# Feature importance
top_features = models.get_feature_importance_ranking('xgboost', top_k=15)
importance_dict = dict(top_features)
importance_fig = viz.plot_feature_importance(importance_dict)
plt.show()

# Interactive dashboard
dashboard = viz.create_interactive_risk_dashboard(model_results)
dashboard.show()
```

## 📊 Supported Datasets

The toolkit supports multiple popular credit risk datasets:

### Kaggle Datasets
- **Give Me Some Credit**: Kaggle competition dataset with 10 risk features
- **Home Credit Default Risk**: Comprehensive dataset with multiple data sources
- **Lending Club**: P2P lending historical data
- **German Credit Data**: Classic UCI dataset for binary classification

### Usage Instructions
1. Download datasets from Kaggle/UCI repositories
2. Place in `data/` directory
3. Use appropriate loader methods:
   ```python
   # For Give Me Some Credit
   df = loader.load_give_me_credit('data/cs-training.csv')
   
   # For Home Credit
   df = loader.load_home_credit('data/application_train.csv')
   ```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_models.py -v
```

## 📋 Model Performance Benchmarks

Expected performance on standard datasets:

| Model | Dataset | AUC-ROC | Precision | Recall | F1-Score |
|-------|---------|---------|-----------|---------|----------|
| Logistic Regression | Give Me Credit | 0.82+ | 0.65+ | 0.45+ | 0.53+ |
| Random Forest | Give Me Credit | 0.85+ | 0.70+ | 0.50+ | 0.58+ |
| XGBoost | Give Me Credit | 0.87+ | 0.72+ | 0.55+ | 0.62+ |

## 🔧 Configuration

### Model Hyperparameters
Key hyperparameters can be configured in the model training methods:

```python
# XGBoost configuration
xgb_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.9,
    'colsample_bytree': 0.9
}
```

### Risk Calculation Settings
```python
# VaR configuration
var_config = {
    'confidence_level': 0.95,
    'method': 'historical',  # or 'parametric', 'monte_carlo'
    'lookback_period': 252
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting PR

## 📚 Educational Resources

### Key Concepts Demonstrated
- **Credit Risk Modeling**: Statistical and ML approaches to default prediction
- **Basel III Compliance**: Regulatory capital calculations and reporting
- **Model Validation**: Backtesting, stress testing, and performance monitoring
- **Portfolio Management**: Concentration limits, diversification, and optimization

### Recommended Reading
- "Credit Risk Modeling using Excel and VBA" by Gunter Löffler
- "The Elements of Statistical Learning" by Hastie, Tibshirani & Friedman  
- "Basel III: A Global Regulatory Framework" - Bank for International Settlements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**xiaoxu (Ivan) zhao**
- GitHub: [@xiaoxu-zhao](https://github.com/xiaoxu-zhao)
- LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)

## 🙏 Acknowledgments

- Kaggle community for providing high-quality datasets
- Scikit-learn team for excellent ML library
- XGBoost and LightGBM teams for gradient boosting implementations
- Financial risk management community for methodological guidance

---

*This project is designed to showcase practical applications of credit risk management techniques for professional portfolio demonstration. It combines theoretical knowledge with hands-on implementation skills essential for quantitative finance roles.*

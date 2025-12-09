# Credit Risk Management Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive credit risk management demonstration project showcasing advanced quantitative risk modeling techniques, machine learning applications, and regulatory compliance methodologies. Designed for job-seeking professionals to demonstrate expertise in credit risk analysis and management.

## 🎯 Project Overview

This repository demonstrates core credit risk management capabilities including:

- **Credit Risk Modeling**: Advanced ML models (XGBoost, Logistic Regression) for default prediction.
- **Regulatory Compliance**: IFRS 9 ECL (Expected Credit Loss) and Basel III capital calculations.
- **Portfolio Analytics**: Risk metrics (VaR, Expected Shortfall), concentration analysis, and optimization.
- **Model Validation**: Comprehensive backtesting, calibration (Isotonic Regression), and performance evaluation.
- **Risk Visualization**: Interactive dashboards and reporting tools.

## 🚀 Key Features

### 📊 Machine Learning Models
- **Logistic Regression**: Interpretable baseline models with regularization.
- **XGBoost**: Gradient boosting for superior predictive performance (AUC ~0.70).
- **Advanced Calibration**: Isotonic regression to ensure predicted probabilities match observed default rates.
- **Feature Engineering**: Custom financial ratios (e.g., Loan-to-Income) and behavioral features.

### 📈 Risk Metrics & Analytics
- **Value at Risk (VaR)**: Historical, parametric, and Monte Carlo methods.
- **Expected Shortfall**: Conditional VaR for tail risk assessment.
- **Credit Risk Measures**: PD (Probability of Default), LGD (Loss Given Default), EAD (Exposure at Default).
- **Regulatory Capital**: Basel III IRB approach implementation.
- **IFRS 9 Compliance**: Staging logic (Stage 1/2/3) and Lifetime ECL calculations.

### 🔍 Advanced Features
- **Data Leakage Prevention**: Rigorous exclusion of future-dated variables (e.g., `recoveries`, `last_fico`).
- **Model of Credit (MoC)**: Complete PD/LGD/EAD framework with RAROC calculations.
- **Model Monitoring**: Population Stability Index (PSI), KS testing, drift detection.
- **Automated Alerts**: Real-time monitoring with configurable thresholds.

## 📁 Project Structure

```
risk-management/
├── src/                          # Core source code
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Data loading and initial quality checks
│   ├── lending_club_preprocessing.py # Specific cleaning for Lending Club data
│   ├── feature_engineering.py   # Feature creation and transformation
│   ├── models.py                # ML model implementations (XGBoost, LR)
│   ├── risk_metrics.py          # Risk calculations (VaR, EL, IFRS 9)
│   ├── moc.py                   # Model of Credit (MoC) framework
│   ├── monitoring.py            # Model monitoring (PSI, KS)
│   └── visualization.py         # Plotting utilities
├── notebooks/                    # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb  # EDA and Data Quality
│   └── 02_modelling_and_risk_analysis.ipynb # Full Pipeline (Feature Eng -> Model -> Risk Metrics)
├── data/                         # Dataset storage (gitignored)
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

### 1. Run the Analysis Pipeline
The core analysis is contained in `notebooks/02_modelling_and_risk_analysis.ipynb`. This notebook:
1.  Loads and cleans the Lending Club data.
2.  Engineers features (including interaction terms).
3.  Trains an XGBoost model (AUC: 0.703).
4.  Calibrates the model using Isotonic Regression.
5.  Calculates Portfolio Expected Loss and IFRS 9 ECL.

### 2. Example Code Usage
You can also use the modules directly in your own scripts:

```python
from src.data_loader import CreditDataLoader
from src.feature_engineering import FeatureEngineer
from src.models import CreditRiskModels

# 1. Load Data
loader = CreditDataLoader(data_path='data/')
datasets = loader.load_lending_club(accepted_only=True)
df = datasets['accepted']

# 2. Feature Engineering
fe = FeatureEngineer()
df_features = fe.create_risk_features(df)
df_encoded = fe.encode_categorical_features(df_features, target_col='default')

# 3. Train Model
models = CreditRiskModels()
splits = models.prepare_data(df_encoded, 'default')
xgb_model = models.train_xgboost(splits['X_train'], splits['y_train'])

# 4. Evaluate
results = models.evaluate_model(xgb_model, splits['X_test'], splits['y_test'])
print(f"Model AUC: {results['roc_auc']:.4f}")
```

## 📊 Results Summary

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **AUC Score** | **0.7030** | Good predictive power for application scorecard. |
| **Top Feature** | `loan_to_income` | Affordability is the primary driver of default. |
| **Calibration** | Isotonic | Probabilities are well-aligned with observed rates. |
| **Leakage** | None | Future variables (e.g., `last_fico`) strictly excluded. |

## 🧪 Testing

Run the comprehensive test suite:

```bash
pytest tests/ -v
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

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

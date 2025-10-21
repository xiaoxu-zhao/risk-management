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

> **Note:** This project is currently under active development. The modules marked as `(In Progress)` or `(Planned)` are placeholders for future implementation.

**Completed & In-Progress Modules:**
- `data_loader.py`: Complete.
- `lending_club_preprocessing.py`: Complete.
- `feature_engineering.py`: Complete.
- `visualization.py`: Initial implementation complete.
- `01_data_exploration.ipynb`: Complete.

---

```
risk-management/
├── src/                          # Core source code
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Data loading and initial quality checks
│   ├── lending_club_preprocessing.py # Specific cleaning for Lending Club data
│   ├── feature_engineering.py   # Feature creation and transformation
│   ├── visualization.py         # Plotting utilities (In Progress)
│   ├── models.py                # (Planned) ML model implementations
│   ├── risk_metrics.py          # (Planned) Risk calculations (VaR, EL, etc.)
│   ├── moc.py                   # (Planned) Model of Credit (MoC) framework
│   └── monitoring.py            # (Planned) Model monitoring (PSI, KS)
├── notebooks/                    # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb  # Complete
│   ├── 02_feature_engineering.ipynb # (Planned)
│   ├── 03_model_development.ipynb   # (Planned)
│   ├── 04_risk_analysis.ipynb       # (Planned)
│   ├── 05_moc_implementation.ipynb  # (Planned)
│   └── 06_model_monitoring.ipynb    # (Planned)
├── tests/                        # (Planned) Comprehensive test suite
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   ├── test_risk_metrics.py
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

This guide demonstrates how to use the currently completed modules for data loading, preprocessing, and initial visualization. As more modules are developed, this section will be updated.

### 1. Data Loading and Preprocessing
This example shows the primary workflow for loading the Lending Club dataset and applying the specific cleaning and feature engineering logic.

```python
from src.data_loader import CreditDataLoader
from src.lending_club_preprocessing import LendingClubPreprocessor
from src.visualization import RiskVisualizer
import matplotlib.pyplot as plt

# Define the path to your data directory
DATA_DIR = 'data/'

# Step 1: Load the raw data
print("Loading data...")
loader = CreditDataLoader(data_path=DATA_DIR)
datasets = loader.load_lending_club(accepted_only=True)
accepted_df = datasets['accepted']
print(f"Raw data loaded with shape: {accepted_df.shape}")

# Step 2: Apply Lending Club-specific preprocessing
print("Preprocessing data...")
preprocessor = LendingClubPreprocessor(include_pricing_features=False)
df_prepared = preprocessor.prepare_accepted(accepted_df)
print(f"Data prepared. New shape: {df_prepared.shape}")
print(f"Default rate in prepared data: {df_prepared['default'].mean():.2%}")

# Step 3: Visualize feature correlations
print("Generating correlation heatmap...")
visualizer = RiskVisualizer()
fig = visualizer.plot_correlation_heatmap(df_prepared, figsize=(12, 10))
plt.show()
```

### 2. Future Usage (Planned)
The following examples illustrate the intended functionality of modules that are currently planned for development.

#### Model Training and Evaluation
```python
# (Planned)
from src.models import CreditRiskModels
# ... code to train and evaluate models ...
```

#### Risk Analytics
```python
# (Planned)
from src.risk_metrics import RiskMetrics
# ... code to calculate EL, VaR, and regulatory capital ...
```

## 📊 Focus Dataset: Lending Club

The toolkit is currently focused on the **Lending Club** dataset, a popular source for P2P lending historical data. Future development will add support for other standard credit risk datasets.

### Lending Club Data Preparation Pipeline

We provide a Lending Club–specific pipeline that follows an ordered risk data preparation process:

1.  **Raw Cleaning**: Handles duplicates, removes useless text/URL columns, checks for data consistency, converts data types, and trims invalid rows.
2.  **Outlier & Missing Value Treatment**: Winsorizes key numeric fields to cap outliers and performs logical imputations for missing data while creating missing-value indicator flags.
3.  **Feature Engineering**: Creates new features such as average FICO score, credit history length, payment-to-income ratios, and derives the binary `default` target from the `loan_status` column.
4.  **Encoding & Normalization**: This final step is handled by the `FeatureEngineer` class in the modeling pipeline to prepare data for machine learning algorithms.

The primary workflow for this process is demonstrated in the [Quick Start](#-quick-start) section.

## 🧪 Testing

> **(Planned)** The test suite is under development.

Run the comprehensive test suite:

```bash
# (Planned) Run all tests
pytest tests/ -v

# (Planned) Run with coverage
pytest tests/ --cov=src --cov-report=html
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

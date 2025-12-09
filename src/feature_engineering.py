"""
Feature Engineering Utilities
=============================

Advanced feature creation and transformation for credit risk modeling.
Includes domain-specific risk features, interactions, and behavioral proxies.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Dict, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self) -> None:
        self.pipeline: Optional[Pipeline] = None
        self.feature_names_: Optional[List[str]] = None

    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates domain-specific credit risk features.
        """
        df = df.copy()
        
        # 1. Credit Line Utilization Strategy
        # Ratio of open accounts to total accounts (proxy for credit hunger)
        if {'open_acc', 'total_acc'}.issubset(df.columns):
            df['utilization_of_credit_lines'] = df['open_acc'] / df['total_acc'].replace(0, 1)

        # 2. Income Stability / Leverage
        # Loan amount relative to monthly income
        if {'loan_amnt', 'annual_inc'}.issubset(df.columns):
            df['loan_to_income'] = df['loan_amnt'] / df['annual_inc'].replace(0, 1)

        # 3. Residual Income (Monthly Income - Installment)
        if {'annual_inc', 'installment'}.issubset(df.columns):
            df['monthly_residual_income'] = (df['annual_inc'] / 12) - df['installment']

        # 4. FICO Band (Non-linear effect of credit score)
        if 'fico_avg' in df.columns:
            # Binning FICO scores into standard risk buckets
            df['fico_band'] = pd.cut(
                df['fico_avg'], 
                bins=[0, 580, 670, 740, 800, 850],
                labels=['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
            )

        return df

    def create_interaction_features(self, df: pd.DataFrame, max_interactions: int = 5) -> pd.DataFrame:
        """
        Creates interaction terms between key risk drivers.
        """
        df = df.copy()
        
        # Interaction: DTI * Utilization (High debt + High utilization = High Risk)
        if {'dti', 'revol_util'}.issubset(df.columns):
            df['dti_x_util'] = df['dti'] * df['revol_util']

        # Interaction: Loan Amount * Interest Rate (Cost of credit)
        # Note: int_rate might be dropped in preprocessing if pricing features are excluded.
        if {'loan_amnt', 'int_rate'}.issubset(df.columns):
            df['cost_of_credit'] = df['loan_amnt'] * df['int_rate']

        return df

    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates features based on past credit behavior.
        """
        df = df.copy()
        
        # Clean record flag
        delinq_cols = ['delinq_2yrs', 'pub_rec', 'tax_liens']
        present_delinq = [c for c in delinq_cols if c in df.columns]
        if present_delinq:
            df['clean_record'] = (df[present_delinq].sum(axis=1) == 0).astype(int)

        # Recent inquiry intensity
        if 'inq_last_6mths' in df.columns:
            df['high_inquiry_activity'] = (df['inq_last_6mths'] > 2).astype(int)

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic imputation for the dataframe before encoding.
        """
        df = df.copy()
        # Numeric: median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if df[c].isnull().any():
                df[c] = df[c].fillna(df[c].median())
        
        # Categorical: mode
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for c in cat_cols:
            if df[c].isnull().any():
                df[c] = df[c].fillna(df[c].mode()[0])
                
        return df

    def encode_categorical_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        One-hot encodes categorical features.
        """
        df = df.copy()
        if target_col and target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df

        X = pd.get_dummies(X, drop_first=True)
        
        if y is not None:
            X[target_col] = y
            
        return X

    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """
        Selects the top k most informative features.
        """
        logger.info(f"Selecting top {k} features using {method}...")
        
        # Ensure X only contains numeric/float columns for mutual_info
        # Drop datetime columns explicitly
        X_numeric = X.select_dtypes(include=[np.number])
        
        if method == 'mutual_info':
            # Use a subset for speed if dataset is huge
            if len(X_numeric) > 10000:
                idx = np.random.choice(len(X_numeric), 10000, replace=False)
                X_sample = X_numeric.iloc[idx]
                y_sample = y.iloc[idx]
            else:
                X_sample = X_numeric
                y_sample = y
            
            # Fill NaNs before selection (mutual_info doesn't handle NaNs)
            X_sample = X_sample.fillna(0)
                
            selector = SelectKBest(mutual_info_classif, k=k)
            selector.fit(X_sample, y_sample)
            
            selected_mask = selector.get_support()
            selected_features = X_numeric.columns[selected_mask].tolist()
        else:
            # Default to ANOVA F-value
            from sklearn.feature_selection import f_classif
            X_numeric = X_numeric.fillna(0)
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X_numeric, y)
            selected_mask = selector.get_support()
            selected_features = X_numeric.columns[selected_mask].tolist()
            
        logger.info(f"Selected features: {selected_features}")
        
        # Return original X with only selected columns
        return X[selected_features], selected_features

    def build_preprocess(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create a ColumnTransformer for numeric and categorical features.
        """
        # Strictly select numeric columns (exclude datetime)
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Identify categorical columns (object/category)
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclude datetime columns from both
        dt_cols = X.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        
        # Log if we are dropping columns
        dropped = [c for c in X.columns if c not in num_cols and c not in cat_cols]
        if dropped:
            logger.info(f"Dropping columns from preprocessing: {dropped}")

        numeric = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler(with_mean=False)),
        ])
        categorical = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ])

        ct = ColumnTransformer(
            transformers=[
                ('num', numeric, num_cols),
                ('cat', categorical, cat_cols),
            ],
            remainder='drop'
        )
        return ct

    def make_model_pipeline(self, X: pd.DataFrame, estimator) -> Pipeline:
        """
        Assemble preprocessing and an estimator into a single Pipeline.
        """
        preprocess = self.build_preprocess(X)
        pipe = Pipeline([('preprocess', preprocess), ('model', estimator)])
        return pipe

    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        return {
            'total_features': len(df.columns),
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'n_numeric': int(df.select_dtypes(include=[np.number]).shape[1]),
            'n_categorical': int(df.select_dtypes(exclude=[np.number]).shape[1]),
            'missing_values': int(df.isnull().sum().sum()),
        }
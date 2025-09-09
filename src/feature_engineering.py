"""
Feature Engineering Module
==========================

Advanced feature engineering techniques for credit risk modeling including:
- Risk-based transformations
- Interaction features  
- Behavioral patterns
- Regulatory compliance features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for credit risk datasets.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        
    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk-specific features commonly used in credit modeling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional risk features
        """
        df_features = df.copy()
        
        # Debt-to-income ratio (if components available)
        if 'debt_to_income' not in df_features.columns:
            if all(col in df_features.columns for col in ['income', 'debt']):
                df_features['debt_to_income'] = df_features['debt'] / df_features['income']
            elif all(col in df_features.columns for col in ['MonthlyIncome', 'DebtRatio']):
                # For Give Me Some Credit dataset
                df_features['debt_to_income'] = df_features['DebtRatio']
                
        # Age-based risk features
        if 'age' in df_features.columns:
            df_features['age_group'] = pd.cut(df_features['age'], 
                                            bins=[0, 25, 35, 45, 55, 65, 100],
                                            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
            df_features['is_young'] = (df_features['age'] < 25).astype(int)
            df_features['is_senior'] = (df_features['age'] > 65).astype(int)
            
        # Income-based features
        income_cols = [col for col in df_features.columns if 'income' in col.lower()]
        if income_cols:
            income_col = income_cols[0]
            df_features['log_income'] = np.log1p(df_features[income_col])
            df_features['income_quintile'] = pd.qcut(df_features[income_col], 
                                                   q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            
        # Credit utilization features
        if 'credit_utilization' in df_features.columns:
            df_features['high_utilization'] = (df_features['credit_utilization'] > 0.9).astype(int)
            df_features['low_utilization'] = (df_features['credit_utilization'] < 0.1).astype(int)
            
        # Late payment features
        late_payment_cols = [col for col in df_features.columns 
                           if any(term in col.lower() for term in ['late', 'delinq', 'past_due'])]
        
        if late_payment_cols:
            # Create binary flags for any late payments
            for col in late_payment_cols:
                df_features[f'{col}_flag'] = (df_features[col] > 0).astype(int)
            
            # Total late payment score
            df_features['total_late_payments'] = df_features[late_payment_cols].sum(axis=1)
            
        # Credit score features (if available)
        if 'credit_score' in df_features.columns:
            df_features['credit_score_band'] = pd.cut(df_features['credit_score'],
                                                    bins=[0, 580, 670, 740, 800, 850],
                                                    labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
            df_features['subprime'] = (df_features['credit_score'] < 620).astype(int)
            
        # Number of dependents risk
        if 'NumberOfDependents' in df_features.columns:
            df_features['has_dependents'] = (df_features['NumberOfDependents'] > 0).astype(int)
            df_features['many_dependents'] = (df_features['NumberOfDependents'] > 3).astype(int)
            
        # Real estate features
        real_estate_cols = [col for col in df_features.columns 
                          if any(term in col.lower() for term in ['real', 'estate', 'mortgage'])]
        if real_estate_cols:
            df_features['has_real_estate'] = (df_features[real_estate_cols[0]] > 0).astype(int)
            
        logger.info(f"Created risk features. New shape: {df_features.shape}")
        return df_features
        
    def create_interaction_features(self, df: pd.DataFrame, 
                                  max_interactions: int = 10) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: Input DataFrame
            max_interactions: Maximum number of interaction features to create
            
        Returns:
            DataFrame with interaction features
        """
        df_interact = df.copy()
        
        # Key interaction pairs for credit risk
        interaction_pairs = [
            ('age', 'income'),
            ('debt_to_income', 'credit_score'),
            ('income', 'credit_utilization'), 
            ('age', 'employment_years'),
            ('credit_score', 'num_late_payments')
        ]
        
        # Filter to existing columns
        valid_pairs = []
        for col1, col2 in interaction_pairs:
            if col1 in df_interact.columns and col2 in df_interact.columns:
                if pd.api.types.is_numeric_dtype(df_interact[col1]) and \
                   pd.api.types.is_numeric_dtype(df_interact[col2]):
                    valid_pairs.append((col1, col2))
                    
        # Create interactions
        for i, (col1, col2) in enumerate(valid_pairs[:max_interactions]):
            # Multiplication interaction
            df_interact[f'{col1}_x_{col2}'] = df_interact[col1] * df_interact[col2]
            
            # Ratio interaction (avoid division by zero)
            df_interact[f'{col1}_div_{col2}'] = (
                df_interact[col1] / (df_interact[col2] + 1e-8)
            )
            
        logger.info(f"Created {len(valid_pairs)} interaction features")
        return df_interact
        
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral and pattern-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with behavioral features
        """
        df_behavior = df.copy()
        
        # Credit behavior patterns
        if 'NumberOfTime30-59DaysPastDueNotWorse' in df_behavior.columns:
            # Create payment behavior score
            late_cols = [col for col in df_behavior.columns if 'PastDue' in col or 'Delinq' in col]
            if late_cols:
                df_behavior['payment_behavior_score'] = df_behavior[late_cols].sum(axis=1)
                df_behavior['consistent_late_payer'] = (
                    (df_behavior['payment_behavior_score'] > 2).astype(int)
                )
        
        # Credit line management
        if 'NumberOfOpenCreditLinesAndLoans' in df_behavior.columns:
            df_behavior['credit_diversification'] = df_behavior['NumberOfOpenCreditLinesAndLoans']
            
            # Credit line utilization efficiency
            if 'NumberRealEstateLoansOrLines' in df_behavior.columns:
                df_behavior['real_estate_ratio'] = (
                    df_behavior['NumberRealEstateLoansOrLines'] / 
                    (df_behavior['NumberOfOpenCreditLinesAndLoans'] + 1e-8)
                )
                
        # Risk appetite indicators
        if all(col in df_behavior.columns for col in ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio']):
            df_behavior['financial_stress'] = (
                df_behavior['RevolvingUtilizationOfUnsecuredLines'] + df_behavior['DebtRatio']
            )
            df_behavior['high_risk_behavior'] = (df_behavior['financial_stress'] > 1.5).astype(int)
            
        return df_behavior
        
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values with appropriate strategies for different feature types.
        
        Args:
            df: Input DataFrame
            strategy: Dictionary mapping column names to imputation strategies
            
        Returns:
            DataFrame with imputed missing values
        """
        df_imputed = df.copy()
        
        if strategy is None:
            strategy = {}
            
        for column in df_imputed.columns:
            if df_imputed[column].isnull().any():
                
                # Use provided strategy or default
                if column in strategy:
                    method = strategy[column]
                else:
                    # Default strategies based on data type
                    if pd.api.types.is_numeric_dtype(df_imputed[column]):
                        method = 'median'
                    else:
                        method = 'mode'
                
                # Apply imputation
                if method == 'median':
                    df_imputed[column].fillna(df_imputed[column].median(), inplace=True)
                elif method == 'mean':
                    df_imputed[column].fillna(df_imputed[column].mean(), inplace=True)
                elif method == 'mode':
                    df_imputed[column].fillna(df_imputed[column].mode().iloc[0], inplace=True)
                elif method == 'zero':
                    df_imputed[column].fillna(0, inplace=True)
                elif method == 'forward_fill':
                    df_imputed[column].fillna(method='ffill', inplace=True)
                elif method == 'backward_fill':
                    df_imputed[column].fillna(method='bfill', inplace=True)
                    
        return df_imputed
        
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  target_col: str = None) -> pd.DataFrame:
        """
        Encode categorical features for machine learning models.
        
        Args:
            df: Input DataFrame
            target_col: Target column name (excluded from encoding)
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        # Identify categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)
            
        for col in categorical_cols:
            unique_values = df_encoded[col].nunique()
            
            if unique_values == 2:
                # Binary encoding for binary categorical variables
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
                
            elif unique_values <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)
                
            else:
                # Target encoding for high cardinality (if target provided)
                if target_col and target_col in df.columns:
                    target_mean = df_encoded.groupby(col)[target_col].mean()
                    df_encoded[f'{col}_target_encoded'] = df_encoded[col].map(target_mean)
                    df_encoded.drop(col, axis=1, inplace=True)
                else:
                    # Frequency encoding as fallback
                    freq_encoding = df_encoded[col].value_counts(normalize=True)
                    df_encoded[f'{col}_frequency'] = df_encoded[col].map(freq_encoding)
                    df_encoded.drop(col, axis=1, inplace=True)
                    
        return df_encoded
        
    def scale_features(self, df: pd.DataFrame, 
                      exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Scale numerical features to standard normal distribution.
        
        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude from scaling
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        if exclude_cols is None:
            exclude_cols = []
            
        # Identify numerical columns to scale
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]
        
        if cols_to_scale:
            scaler = StandardScaler()
            df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
            self.scalers['standard_scaler'] = scaler
            
        logger.info(f"Scaled {len(cols_to_scale)} numerical features")
        return df_scaled
        
    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       method: str = 'mutual_info',
                       k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top K features using statistical methods.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Selection method ('f_classif', 'mutual_info')
            k: Number of features to select
            
        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError(f"Unknown selection method: {method}")
            
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store feature importance scores
        self.feature_importance[method] = dict(zip(
            X.columns, selector.scores_
        ))
        
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        logger.info(f"Selected {len(selected_features)} features using {method}")
        return X_selected_df, selected_features
        
    def create_polynomial_features(self, df: pd.DataFrame, 
                                 degree: int = 2,
                                 max_features: int = 50) -> pd.DataFrame:
        """
        Create polynomial features for specified numerical columns.
        
        Args:
            df: Input DataFrame
            degree: Polynomial degree
            max_features: Maximum number of polynomial features to create
            
        Returns:
            DataFrame with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        df_poly = df.copy()
        
        # Select numerical columns with low correlation to avoid explosion
        numerical_cols = df_poly.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to most important numerical features to avoid curse of dimensionality
        if len(numerical_cols) > 5:
            # Use variance as a simple importance measure
            variances = df_poly[numerical_cols].var().sort_values(ascending=False)
            numerical_cols = variances.head(5).index.tolist()
            
        if numerical_cols:
            poly = PolynomialFeatures(degree=degree, interaction_only=False, 
                                    include_bias=False)
            
            # Fit on subset to control feature explosion
            X_subset = df_poly[numerical_cols]
            X_poly = poly.fit_transform(X_subset)
            
            # Get feature names
            feature_names = poly.get_feature_names_out(numerical_cols)
            
            # Limit number of features
            if len(feature_names) > max_features:
                feature_names = feature_names[:max_features]
                X_poly = X_poly[:, :max_features]
                
            # Create DataFrame for new features
            poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df_poly.index)
            
            # Remove original features to avoid duplication
            original_feature_names = [name for name in feature_names if name in numerical_cols]
            poly_df = poly_df.drop(columns=original_feature_names, errors='ignore')
            
            # Combine with original DataFrame
            df_poly = pd.concat([df_poly, poly_df], axis=1)
            
        logger.info(f"Created polynomial features. New shape: {df_poly.shape}")
        return df_poly
        
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive summary of engineered features.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Dictionary with feature summary statistics
        """
        summary = {
            'total_features': len(df.columns),
            'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
            'missing_values': df.isnull().sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'feature_types': df.dtypes.value_counts().to_dict()
        }
        
        return summary
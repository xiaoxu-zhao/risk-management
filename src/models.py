"""
Credit Risk Models Module
========================

Machine learning model implementations for credit risk.
Includes robust training logic for Logistic Regression and XGBoost,
handling class imbalance and pipeline integration.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Try importing xgboost, handle if missing
try:
    import xgboost as xgb
except ImportError:
    xgb = None

logger = logging.getLogger(__name__)


class CreditRiskModels:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'default', test_size: float = 0.3) -> Dict[str, Any]:
        """
        Splits data into training, validation, and testing sets.
        Returns a dictionary with all splits.
        """
        X = df.drop(columns=[target_col], errors='ignore')
        y = df[target_col]
        
        # First split: Train+Val vs Test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: Train vs Val (from the remaining 70%)
        # Split 20% of the original data for validation (approx 28% of temp)
        val_size = 0.2 / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.random_state, stratify=y_temp
        )
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None) -> Any:
        """
        Trains a Logistic Regression model with class weights.
        """
        logger.info("Training Logistic Regression...")
        
        # Simple pipeline for the model itself
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=self.random_state))
        ])
        
        pipeline.fit(X_train, y_train)
        self.models['logistic_regression'] = pipeline
        return pipeline

    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None) -> Any:
        """
        Trains an XGBoost model.
        """
        if xgb is None:
            logger.warning("XGBoost not installed. Skipping.")
            return None
            
        logger.info("Training XGBoost...")
        
        # Calculate scale_pos_weight
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        ratio = float(n_neg) / n_pos if n_pos > 0 else 1.0
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=ratio,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # XGBoost handles missing values internally, but we wrap it for consistency
        pipeline = Pipeline([
            ('model', model)
        ])
        
        # Fit directly (XGBoost can handle pandas DataFrames)
        pipeline.fit(X_train, y_train)
        self.models['xgboost'] = pipeline
        return pipeline

    def calibrate_model_advanced(self, model_name: str, X_val, y_val, method: str = 'isotonic') -> Dict[str, float]:
        """
        Calibrates the model using Isotonic Regression or Platt Scaling.
        """
        if model_name not in self.models:
            return {'brier_score_improvement': 0.0}
            
        base_model = self.models[model_name]
        
        # Calculate initial Brier score
        probs_before = base_model.predict_proba(X_val)[:, 1]
        brier_before = brier_score_loss(y_val, probs_before)
        
        # Calibrate
        calibrated = CalibratedClassifierCV(base_model, method=method, cv='prefit')
        calibrated.fit(X_val, y_val)
        
        # Calculate new Brier score
        probs_after = calibrated.predict_proba(X_val)[:, 1]
        brier_after = brier_score_loss(y_val, probs_after)
        
        # Update stored model
        self.models[model_name] = calibrated
        
        return {
            'brier_score_before': brier_before,
            'brier_score_after': brier_after,
            'brier_score_improvement': brier_before - brier_after
        }

    def create_scoring_pipeline(self, model_name: str, feature_names: List[str]) -> Pipeline:
        """
        Returns the final production pipeline.
        """
        return self.models.get(model_name)

    def evaluate_model(self, model, X_test, y_test, model_name: str = 'model') -> Dict[str, Any]:
        """
        Evaluates the model and returns key risk metrics.
        """
        if model is None:
            return {}
            
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        result = {
            'roc_auc': auc,
            'pr_auc': pr_auc,
            'probabilities': y_pred_proba,
            'predictions': y_pred
        }
        
        # Store result for comparison
        self.results[model_name] = {'AUC Score': auc, 'PR AUC': pr_auc}
        
        return result

    def compare_models(self) -> pd.DataFrame:
        """
        Returns a DataFrame comparing all evaluated models.
        """
        df = pd.DataFrame(self.results).T
        df.index.name = 'Model'
        df = df.reset_index().sort_values('AUC Score', ascending=False)
        return df

    def predict_default_probability(self, model_name: str, X) -> np.ndarray:
        """
        Returns PD (Probability of Default) for the given data.
        """
        model = self.models.get(model_name)
        if model:
            return model.predict_proba(X)[:, 1]
        return np.zeros(len(X))

    def get_feature_importance_ranking(self, model_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extracts feature importance from the model.
        """
        model = self.models.get(model_name)
        if not model:
            return []
            
        # Handle CalibratedClassifierCV wrapper
        if isinstance(model, CalibratedClassifierCV):
            base_estimator = model.base_estimator
        elif isinstance(model, Pipeline):
            base_estimator = model.named_steps['model']
        else:
            base_estimator = model
            
        # Extract importance
        if hasattr(base_estimator, 'feature_importances_'):
            importances = base_estimator.feature_importances_
            # Assuming X columns are passed in order. 
            # Note: This is a simplification. In a real pipeline, we'd track feature names through transformers.
            # Here we return indices if names aren't available, or map to passed names if we had them.
            indices = np.argsort(importances)[::-1][:top_k]
            return [(f"Feature_{i}", importances[i]) for i in indices]
        elif hasattr(base_estimator, 'coef_'):
            importances = np.abs(base_estimator.coef_[0])
            indices = np.argsort(importances)[::-1][:top_k]
            return [(f"Feature_{i}", importances[i]) for i in indices]
            
        return []
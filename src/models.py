"""
Credit Risk Models Module
========================

Implementation of various machine learning models for credit risk assessment:
- Logistic Regression (baseline and interpretable)
- Random Forest (ensemble method)
- XGBoost (gradient boosting)
- Model evaluation and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, log_loss
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import logging

logger = logging.getLogger(__name__)


class CreditRiskModels:
    """
    Comprehensive credit risk modeling toolkit with multiple algorithms and evaluation methods.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the models with random state for reproducibility.
        
        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        self.models = {}
        self.model_performance = {}
        self.feature_importance = {}
        
    def prepare_data(self, df: pd.DataFrame, 
                    target_col: str,
                    test_size: float = 0.3,
                    validation_size: float = 0.2) -> Dict[str, Any]:
        """
        Prepare data for model training with train/validation/test splits.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            
        Returns:
            Dictionary containing train, validation, and test datasets
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # First split: train/validation and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size, 
            random_state=self.random_state, stratify=y_temp
        )
        
        data_splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_names': X.columns.tolist()
        }
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return data_splits
        
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame = None, y_val: pd.Series = None,
                                tune_hyperparameters: bool = True) -> LogisticRegression:
        """
        Train logistic regression model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Trained logistic regression model
        """
        if tune_hyperparameters:
            # Hyperparameter grid
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            }
            
            # Grid search with cross-validation
            lr_base = LogisticRegression(random_state=self.random_state, max_iter=1000)
            grid_search = GridSearchCV(
                lr_base, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            logger.info(f"Best LR parameters: {grid_search.best_params_}")
            
        else:
            # Default logistic regression
            best_model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                C=1.0,
                penalty='l2'
            )
            best_model.fit(X_train, y_train)
            
        self.models['logistic_regression'] = best_model
        
        # Store feature importance (coefficients)
        feature_importance = dict(zip(
            X_train.columns, 
            np.abs(best_model.coef_[0])
        ))
        self.feature_importance['logistic_regression'] = feature_importance
        
        return best_model
        
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame = None, y_val: pd.Series = None,
                          tune_hyperparameters: bool = True) -> RandomForestClassifier:
        """
        Train Random Forest model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Trained Random Forest model
        """
        if tune_hyperparameters:
            # Hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Grid search with cross-validation
            rf_base = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf_base, param_grid, cv=3, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            logger.info(f"Best RF parameters: {grid_search.best_params_}")
            
        else:
            # Default Random Forest
            best_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            best_model.fit(X_train, y_train)
            
        self.models['random_forest'] = best_model
        
        # Store feature importance
        feature_importance = dict(zip(
            X_train.columns, 
            best_model.feature_importances_
        ))
        self.feature_importance['random_forest'] = feature_importance
        
        return best_model
        
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame = None, y_val: pd.Series = None,
                     tune_hyperparameters: bool = True) -> xgb.XGBClassifier:
        """
        Train XGBoost model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Trained XGBoost model
        """
        if tune_hyperparameters:
            # Hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            # Grid search with cross-validation
            xgb_base = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            )
            grid_search = GridSearchCV(
                xgb_base, param_grid, cv=3, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            logger.info(f"Best XGB parameters: {grid_search.best_params_}")
            
        else:
            # Default XGBoost with early stopping if validation data provided
            xgb_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'random_state': self.random_state,
                'eval_metric': 'logloss'
            }
            
            best_model = xgb.XGBClassifier(**xgb_params)
            
            if X_val is not None and y_val is not None:
                # Use early stopping - removed early_stopping_rounds for compatibility
                best_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                best_model.fit(X_train, y_train)
                
        self.models['xgboost'] = best_model
        
        # Store feature importance
        feature_importance = dict(zip(
            X_train.columns, 
            best_model.feature_importances_
        ))
        self.feature_importance['xgboost'] = feature_importance
        
        return best_model
        
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame = None, y_val: pd.Series = None,
                      tune_hyperparameters: bool = True) -> lgb.LGBMClassifier:
        """
        Train LightGBM model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Trained LightGBM model
        """
        if tune_hyperparameters:
            # Hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'feature_fraction': [0.8, 0.9, 1.0],
                'bagging_fraction': [0.8, 0.9, 1.0]
            }
            
            # Grid search with cross-validation
            lgb_base = lgb.LGBMClassifier(
                random_state=self.random_state,
                verbose=-1
            )
            grid_search = GridSearchCV(
                lgb_base, param_grid, cv=3, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            logger.info(f"Best LGB parameters: {grid_search.best_params_}")
            
        else:
            # Default LightGBM
            lgb_params = {
                'n_estimators': 200,
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.9,
                'random_state': self.random_state,
                'verbose': -1
            }
            
            best_model = lgb.LGBMClassifier(**lgb_params)
            
            if X_val is not None and y_val is not None:
                # Use early stopping - removed early_stopping_rounds for compatibility
                best_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                best_model.fit(X_train, y_train)
                
        self.models['lightgbm'] = best_model
        
        # Store feature importance
        feature_importance = dict(zip(
            X_train.columns, 
            best_model.feature_importances_
        ))
        self.feature_importance['lightgbm'] = feature_importance
        
        return best_model
        
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        
        # Precision-Recall curve data
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10
        )
        
        evaluation_results = {
            'model_name': model_name,
            'auc_score': auc_score,
            'average_precision': avg_precision,
            'log_loss': logloss,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
            'pr_curve': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds},
            'calibration': {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value
            }
        }
        
        self.model_performance[model_name] = evaluation_results
        
        logger.info(f"{model_name} - AUC: {auc_score:.4f}, Avg Precision: {avg_precision:.4f}")
        
        return evaluation_results
        
    def cross_validate_models(self, X: pd.DataFrame, y: pd.Series,
                            models_to_validate: List[str] = None,
                            cv_folds: int = 5) -> Dict[str, Dict]:
        """
        Perform cross-validation for multiple models.
        
        Args:
            X: Features
            y: Target
            models_to_validate: List of model names to validate
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation results for each model
        """
        if models_to_validate is None:
            models_to_validate = list(self.models.keys())
            
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name in models_to_validate:
            if model_name in self.models:
                model = self.models[model_name]
                
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1
                )
                
                cv_results[model_name] = {
                    'cv_scores': cv_scores,
                    'mean_cv_score': cv_scores.mean(),
                    'std_cv_score': cv_scores.std(),
                    'cv_folds': cv_folds
                }
                
                logger.info(f"{model_name} CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
        return cv_results
        
    def get_feature_importance_ranking(self, model_name: str, 
                                     top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Get top K most important features for a model.
        
        Args:
            model_name: Name of the model
            top_k: Number of top features to return
            
        Returns:
            List of tuples (feature_name, importance_score)
        """
        if model_name not in self.feature_importance:
            raise ValueError(f"Feature importance not available for {model_name}")
            
        importance_dict = self.feature_importance[model_name]
        sorted_importance = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_importance[:top_k]
        
    def predict_default_probability(self, model_name: str, 
                                  X: pd.DataFrame) -> np.ndarray:
        """
        Predict default probability using specified model.
        
        Args:
            model_name: Name of the model to use
            X: Features for prediction
            
        Returns:
            Array of default probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
            
        model = self.models[model_name]
        probabilities = model.predict_proba(X)[:, 1]
        
        return probabilities
        
    def calibrate_model(self, model_name: str, X_cal: pd.DataFrame, 
                       y_cal: pd.Series, method: str = 'isotonic') -> Any:
        """
        Calibrate model probabilities using Platt scaling or isotonic regression.
        
        Args:
            model_name: Name of the model to calibrate
            X_cal: Calibration features
            y_cal: Calibration target
            method: Calibration method ('sigmoid' or 'isotonic')
            
        Returns:
            Calibrated model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
            
        base_model = self.models[model_name]
        
        # Create calibrated classifier
        calibrated_clf = CalibratedClassifierCV(
            base_model, method=method, cv='prefit'
        )
        
        # Fit on calibration data
        calibrated_clf.fit(X_cal, y_cal)
        
        # Store calibrated model
        calibrated_model_name = f"{model_name}_calibrated"
        self.models[calibrated_model_name] = calibrated_clf
        
        logger.info(f"Model {model_name} calibrated using {method} method")
        
        return calibrated_clf
        
    def compare_models(self) -> pd.DataFrame:
        """
        Create comparison table of all trained models.
        
        Returns:
            DataFrame with model comparison metrics
        """
        comparison_data = []
        
        for model_name, performance in self.model_performance.items():
            row = {
                'Model': model_name,
                'AUC Score': performance['auc_score'],
                'Average Precision': performance['average_precision'],
                'Log Loss': performance['log_loss'],
                'Precision': performance['classification_report']['1']['precision'],
                'Recall': performance['classification_report']['1']['recall'],
                'F1-Score': performance['classification_report']['1']['f1-score']
            }
            comparison_data.append(row)
            
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC Score', ascending=False)
        
        return comparison_df
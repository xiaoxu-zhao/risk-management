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
from datetime import datetime
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
        
    def calibrate_model_advanced(self, model_name: str, X_cal: pd.DataFrame,
                                y_cal: pd.Series, method: str = 'isotonic',
                                validation_split: float = 0.3) -> Dict[str, Any]:
        """
        Advanced model calibration with comprehensive evaluation.
        
        Args:
            model_name: Name of the model to calibrate
            X_cal: Calibration features
            y_cal: Calibration target
            method: Calibration method ('isotonic', 'sigmoid', 'beta')
            validation_split: Fraction for validation during calibration
            
        Returns:
            Dictionary with calibration results and metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        base_model = self.models[model_name]
        
        # Split calibration data
        X_cal_train, X_cal_val, y_cal_train, y_cal_val = train_test_split(
            X_cal, y_cal, test_size=validation_split, 
            random_state=self.random_state, stratify=y_cal
        )
        
        # Get uncalibrated predictions
        uncalibrated_probs = base_model.predict_proba(X_cal_val)[:, 1]
        
        calibration_results = {
            'method': method,
            'calibration_data_size': len(X_cal_train),
            'validation_data_size': len(X_cal_val)
        }
        
        if method in ['isotonic', 'sigmoid']:
            # Use sklearn's CalibratedClassifierCV
            calibrated_clf = CalibratedClassifierCV(
                base_model, method=method, cv='prefit'
            )
            calibrated_clf.fit(X_cal_train, y_cal_train)
            calibrated_probs = calibrated_clf.predict_proba(X_cal_val)[:, 1]
            
        elif method == 'beta':
            # Beta calibration (custom implementation)
            calibrated_clf, calibrated_probs = self._beta_calibration(
                base_model, X_cal_train, y_cal_train, X_cal_val
            )
            
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        # Evaluate calibration quality
        calibration_metrics = self._evaluate_calibration(
            y_cal_val, uncalibrated_probs, calibrated_probs
        )
        
        # Store calibrated model
        calibrated_model_name = f"{model_name}_calibrated_{method}"
        self.models[calibrated_model_name] = calibrated_clf
        
        calibration_results.update(calibration_metrics)
        calibration_results['calibrated_model_name'] = calibrated_model_name
        
        logger.info(f"Advanced calibration completed for {model_name} using {method}. "
                   f"Brier Score improvement: {calibration_metrics['brier_score_improvement']:.4f}")
        
        return calibration_results
    
    def _beta_calibration(self, base_model, X_train, y_train, X_val):
        """
        Implement beta calibration method.
        
        Args:
            base_model: Base classifier
            X_train: Training features for calibration
            y_train: Training targets for calibration
            X_val: Validation features
            
        Returns:
            Tuple of (calibrated_model, calibrated_probabilities)
        """
        # Get uncalibrated probabilities on training set
        train_probs = base_model.predict_proba(X_train)[:, 1]
        
        # Fit beta distribution parameters
        from scipy.optimize import minimize
        from scipy.stats import beta
        
        def beta_loss(params, probs, labels):
            a, b = params
            if a <= 0 or b <= 0:
                return 1e6  # Large penalty for invalid parameters
            
            # Transform probabilities to beta distribution range
            epsilon = 1e-8
            probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
            
            # Calculate beta probabilities
            try:
                beta_probs = beta.cdf(probs_clipped, a, b)
                beta_probs = np.clip(beta_probs, epsilon, 1 - epsilon)
                
                # Negative log-likelihood
                loss = -np.sum(labels * np.log(beta_probs) + 
                             (1 - labels) * np.log(1 - beta_probs))
                return loss
            except:
                return 1e6
        
        # Initial parameter guess
        initial_params = [2.0, 2.0]
        
        # Optimize beta parameters
        try:
            result = minimize(beta_loss, initial_params, 
                            args=(train_probs, y_train),
                            method='L-BFGS-B',
                            bounds=[(0.1, 100), (0.1, 100)])
            
            if result.success:
                a_opt, b_opt = result.x
            else:
                # Fallback to method of moments
                mean_prob = np.mean(train_probs)
                var_prob = np.var(train_probs)
                
                # Method of moments for beta distribution
                if var_prob > 0 and mean_prob > 0 and mean_prob < 1:
                    a_opt = mean_prob * ((mean_prob * (1 - mean_prob)) / var_prob - 1)
                    b_opt = (1 - mean_prob) * ((mean_prob * (1 - mean_prob)) / var_prob - 1)
                    a_opt = max(0.1, a_opt)
                    b_opt = max(0.1, b_opt)
                else:
                    a_opt, b_opt = 2.0, 2.0  # Default parameters
        
        except:
            a_opt, b_opt = 2.0, 2.0  # Default parameters
        
        # Apply beta calibration to validation set
        val_probs = base_model.predict_proba(X_val)[:, 1]
        epsilon = 1e-8
        val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
        
        try:
            calibrated_probs = beta.cdf(val_probs_clipped, a_opt, b_opt)
            calibrated_probs = np.clip(calibrated_probs, epsilon, 1 - epsilon)
        except:
            # Fallback to simple linear scaling
            calibrated_probs = val_probs_clipped
        
        # Create a simple calibrated model wrapper
        class BetaCalibratedModel:
            def __init__(self, base_model, a, b):
                self.base_model = base_model
                self.a = a
                self.b = b
            
            def predict_proba(self, X):
                base_probs = self.base_model.predict_proba(X)
                calibrated_pos = beta.cdf(np.clip(base_probs[:, 1], 1e-8, 1-1e-8), self.a, self.b)
                calibrated_neg = 1 - calibrated_pos
                return np.column_stack([calibrated_neg, calibrated_pos])
            
            def predict(self, X):
                probs = self.predict_proba(X)
                return (probs[:, 1] >= 0.5).astype(int)
        
        calibrated_model = BetaCalibratedModel(base_model, a_opt, b_opt)
        
        return calibrated_model, calibrated_probs
    
    def _evaluate_calibration(self, y_true, uncalibrated_probs, calibrated_probs):
        """
        Evaluate calibration quality using multiple metrics.
        
        Args:
            y_true: True labels
            uncalibrated_probs: Original probabilities
            calibrated_probs: Calibrated probabilities
            
        Returns:
            Dictionary with calibration evaluation metrics
        """
        # Brier Score
        uncalibrated_brier = np.mean((uncalibrated_probs - y_true) ** 2)
        calibrated_brier = np.mean((calibrated_probs - y_true) ** 2)
        brier_improvement = uncalibrated_brier - calibrated_brier
        
        # Expected Calibration Error (ECE)
        def calculate_ece(y_true, y_prob, n_bins=10):
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            total_samples = len(y_true)
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
        
        uncalibrated_ece = calculate_ece(y_true, uncalibrated_probs)
        calibrated_ece = calculate_ece(y_true, calibrated_probs)
        ece_improvement = uncalibrated_ece - calibrated_ece
        
        # Reliability (from Brier decomposition)
        def calculate_reliability(y_true, y_prob, n_bins=10):
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            reliability = 0
            
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                
                if np.sum(in_bin) > 0:
                    bin_weight = np.sum(in_bin) / len(y_true)
                    bin_accuracy = np.mean(y_true[in_bin])
                    bin_confidence = np.mean(y_prob[in_bin])
                    reliability += bin_weight * (bin_confidence - bin_accuracy) ** 2
            
            return reliability
        
        uncalibrated_reliability = calculate_reliability(y_true, uncalibrated_probs)
        calibrated_reliability = calculate_reliability(y_true, calibrated_probs)
        
        return {
            'uncalibrated_brier_score': uncalibrated_brier,
            'calibrated_brier_score': calibrated_brier,
            'brier_score_improvement': brier_improvement,
            'uncalibrated_ece': uncalibrated_ece,
            'calibrated_ece': calibrated_ece,
            'ece_improvement': ece_improvement,
            'uncalibrated_reliability': uncalibrated_reliability,
            'calibrated_reliability': calibrated_reliability,
            'reliability_improvement': uncalibrated_reliability - calibrated_reliability
        }
    
    def create_scoring_pipeline(self, model_name: str, 
                              feature_names: List[str] = None,
                              scaling_method: str = 'standard') -> Dict[str, Any]:
        """
        Create a comprehensive scoring pipeline for production use.
        
        Args:
            model_name: Name of the trained model
            feature_names: Expected feature names (for validation)
            scaling_method: Scaling method used during training
            
        Returns:
            Dictionary containing the scoring pipeline components
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Create scoring pipeline
        pipeline_components = {
            'model': model,
            'model_name': model_name,
            'feature_names': feature_names,
            'scaling_method': scaling_method,
            'model_metadata': {
                'training_date': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'hyperparameters': getattr(model, 'get_params', lambda: {})(),
            }
        }
        
        # Add feature importance if available
        if model_name in self.feature_importance:
            pipeline_components['feature_importance'] = self.feature_importance[model_name]
        
        # Add performance metrics if available
        if model_name in self.model_performance:
            pipeline_components['model_performance'] = self.model_performance[model_name]
        
        # Create scoring function
        def score_function(X_new: pd.DataFrame, return_probabilities: bool = True) -> Dict[str, Any]:
            """
            Score new data using the trained model.
            
            Args:
                X_new: New features to score
                return_probabilities: Whether to return probabilities or just predictions
                
            Returns:
                Dictionary with scoring results
            """
            # Validate features
            if feature_names is not None:
                missing_features = set(feature_names) - set(X_new.columns)
                if missing_features:
                    raise ValueError(f"Missing features: {missing_features}")
                
                # Reorder columns to match training order
                X_new = X_new[feature_names]
            
            # Generate predictions
            if return_probabilities:
                try:
                    probabilities = model.predict_proba(X_new)[:, 1]
                    predictions = (probabilities >= 0.5).astype(int)
                except AttributeError:
                    # Model doesn't support predict_proba
                    predictions = model.predict(X_new)
                    probabilities = predictions.astype(float)  # Convert to float
            else:
                predictions = model.predict(X_new)
                probabilities = None
            
            # Score statistics
            scoring_results = {
                'predictions': predictions,
                'probabilities': probabilities,
                'n_scored': len(X_new),
                'scoring_date': datetime.now().isoformat(),
                'model_name': model_name
            }
            
            if probabilities is not None:
                scoring_results.update({
                    'mean_probability': np.mean(probabilities),
                    'median_probability': np.median(probabilities),
                    'std_probability': np.std(probabilities),
                    'predicted_default_rate': np.mean(predictions)
                })
            
            return scoring_results
        
        pipeline_components['score_function'] = score_function
        
        logger.info(f"Scoring pipeline created for {model_name}")
        return pipeline_components
    
    def batch_score_with_monitoring(self, model_name: str, X_new: pd.DataFrame,
                                   reference_scores: np.ndarray = None) -> Dict[str, Any]:
        """
        Score new data with built-in monitoring for drift detection.
        
        Args:
            model_name: Name of the model to use
            X_new: New features to score
            reference_scores: Reference score distribution for comparison
            
        Returns:
            Dictionary with scoring results and monitoring metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Generate scores
        scoring_pipeline = self.create_scoring_pipeline(model_name)
        scoring_results = scoring_pipeline['score_function'](X_new)
        
        # Add monitoring if reference scores provided
        if reference_scores is not None:
            from .monitoring import ModelMonitor
            
            monitor = ModelMonitor()
            current_scores = scoring_results['probabilities']
            
            if current_scores is not None:
                # Calculate PSI
                psi_results = monitor.calculate_population_stability_index(
                    reference_scores, current_scores
                )
                
                # Calculate KS statistic
                ks_results = monitor.calculate_ks_statistic(
                    reference_scores, current_scores
                )
                
                monitoring_metrics = {
                    'psi': psi_results['psi'],
                    'psi_interpretation': psi_results['interpretation'],
                    'ks_statistic': ks_results['ks_statistic'],
                    'distribution_shift_detected': (
                        psi_results['psi'] > 0.25 or ks_results['ks_statistic'] > 0.1
                    )
                }
                
                scoring_results['monitoring'] = monitoring_metrics
        
        return scoring_results

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
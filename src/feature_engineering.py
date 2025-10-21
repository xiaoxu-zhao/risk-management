"""
Feature Engineering Utilities
=============================

General helpers for tabular ML:
- Build preprocessing pipeline (impute/scale/encode)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self) -> None:
        self.pipeline: Optional[Pipeline] = None
        self.feature_names_: Optional[List[str]] = None

    def build_preprocess(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create a ColumnTransformer for numeric and categorical features.
        """
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        numeric = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler(with_mean=False)),
        ])
        categorical = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=True)),
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
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'n_numeric': int(df.select_dtypes(include=[np.number]).shape[1]),
            'n_categorical': int(df.select_dtypes(exclude=[np.number]).shape[1]),
            'missing_values': int(df.isnull().sum().sum()),
        }
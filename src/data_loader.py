"""
Data Loading Module
===================

Handles loading of credit risk datasets from disk (KaggleHub or local).
Focus: Lending Club accepted loans (recursive file search).
"""

from __future__ import annotations

import os
import glob
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditDataLoader:
    """
    Credit dataset loader with helpers for public datasets.
    """

    def __init__(self, data_path: str = "data/") -> None:
        self.data_path = data_path
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, dict] = {}

    def load_give_me_credit(self, filepath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(filepath)
            df = df.drop('Unnamed: 0', axis=1, errors='ignore')
            self.metadata['give_me_credit'] = {
                'target': 'SeriousDlqin2yrs',
                'features': [c for c in df.columns if c != 'SeriousDlqin2yrs'],
                'shape': df.shape,
                'missing_values': int(df.isnull().sum().sum()),
            }
            logger.info("Loaded Give Me Credit dataset: %s", df.shape)
            return df
        except Exception as e:
            logger.error("Error loading Give Me Credit dataset: %s", e)
            raise

    def load_lending_club(
        self,
        data_dir: Optional[str] = None,
        drop_cols: Optional[List[str]] = None,
        accepted_only: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Recursively find and load Lending Club CSVs.

        It looks for files whose names contain 'accepted' and (optionally) 'rejected'.

        Args:
            data_dir: Root directory to search (defaults to self.data_path).
            drop_cols: Optional columns to drop after load.
            accepted_only: If True, only returns accepted loans.

        Returns:
            {'accepted': DataFrame} or {'accepted': DataFrame, 'rejected': DataFrame}
        """
        base = data_dir or self.data_path
        result: Dict[str, pd.DataFrame] = {}

        try:
            csv_files = glob.glob(os.path.join(base, '**', '*.csv'), recursive=True)
            accepted_file = next((f for f in csv_files if 'accepted' in os.path.basename(f).lower()), None)
            rejected_file = next((f for f in csv_files if 'rejected' in os.path.basename(f).lower()), None)

            if not accepted_file:
                raise FileNotFoundError(f"No accepted CSV found under: {base}")

            def _read(path: str) -> pd.DataFrame:
                try:
                    df_ = pd.read_csv(path, low_memory=False)
                except PermissionError as pe:
                    logger.error("Permission denied reading %s. Adjust folder permissions or run VS Code as Administrator.", path)
                    raise pe
                drop_unnamed = [c for c in df_.columns if str(c).lower().startswith('unnamed')]
                if drop_unnamed:
                    df_.drop(columns=drop_unnamed, inplace=True, errors='ignore')
                if drop_cols:
                    df_.drop(columns=[c for c in drop_cols if c in df_.columns], inplace=True, errors='ignore')
                return df_

            df_acc = _read(accepted_file)
            result['accepted'] = df_acc
            self.metadata['lending_club_accepted'] = {
                'path': accepted_file,
                'shape': df_acc.shape,
                'missing_values': int(df_acc.isnull().sum().sum()),
            }
            logger.info("Loaded Lending Club accepted: %s", df_acc.shape)

            if not accepted_only and rejected_file:
                df_rej = _read(rejected_file)
                result['rejected'] = df_rej
                self.metadata['lending_club_rejected'] = {
                    'path': rejected_file,
                    'shape': df_rej.shape,
                    'missing_values': int(df_rej.isnull().sum().sum()),
                }
                logger.info("Loaded Lending Club rejected: %s", df_rej.shape)

            self.datasets.update({k: v for k, v in result.items()})
            return result

        except Exception as e:
            logger.error("Error loading Lending Club datasets: %s", e)
            raise

    def basic_data_quality_check(self, df: pd.DataFrame, target_col: Optional[str] = None) -> dict:
        report = {
            'shape': df.shape,
            'missing_values': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'target_distribution': {},
            'target_rate': np.nan,
        }
        if target_col and target_col in df.columns:
            dist = df[target_col].value_counts(dropna=False).to_dict()
            report['target_distribution'] = dist
            denom = dist.get(0, 0) + dist.get(1, 0)
            report['target_rate'] = (dist.get(1, 0) / denom) if denom else np.nan
        return report
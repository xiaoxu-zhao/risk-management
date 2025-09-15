"""
Lending Club Data Preprocessing
===============================

Implements ordered data preparation for Lending Club accepted and rejected datasets.

Order of Operations used here (aligned with risk data prep best practices):

1) Raw Data Cleaning (always first)
    - Drop duplicates (id, member_id if present)
    - Remove mostly-useless columns: url, desc, emp_title, title, etc.
    - Consistency checks:
         funded_amnt <= loan_amnt; out_prncp = 0 when loan_status = Fully Paid
         negative values in balances/amounts -> set to NaN
    - Data type conversions:
         dates: issue_d, earliest_cr_line, last_pymnt_d -> datetime
         percentages: revol_util -> numeric within [0, 1]
    - Trim garbage rows: annual_inc <= 0, dti > 999 -> drop

2) Outlier & Missing Value Strategy (second)
    - Winsorize/cap outliers for: annual_inc, revol_bal, dti (configurable)
    - Logical imputations for delinquency months: fill NaN with 0 and add _missing flags

3) Feature Engineering (third)
    - fico_avg = (fico_range_low + fico_range_high) / 2
    - credit_history_len_months = (issue_d - earliest_cr_line) in months
    - pmt_to_inc = installment / annual_inc
    - grade_num mapping A..G -> 1..7
    - log_annual_inc = log1p(annual_inc)
    - Target creation: default from loan_status (Fully Paid=0; Charged Off/Default=1)

4) Normalization/Encoding (last)
    - Not performed inside this module: use FeatureEngineer.scale_features and
      FeatureEngineer.encode_categorical_features per modeling needs.

All steps are implemented as small, composable methods and combined in
prepare_accepted/prepare_rejected for convenience. See README for usage examples.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LendingClubPreprocessor:
    """
    End-to-end preprocessing for Lending Club datasets following risk data prep order.
    """

    GOOD_STATUSES = {
        'Fully Paid'
    }
    BAD_STATUSES = {
        'Charged Off', 'Default',
        'Does not meet the credit policy. Status:Charged Off'
    }

    def __init__(self):
        pass

    # 1) Raw Data Cleaning ----------------------------------------------------
    def clean_accepted_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Drop exact duplicates by id/member_id if available
        subset_cols = [c for c in ['id', 'member_id'] if c in df.columns]
        if subset_cols:
            before = len(df)
            df = df.drop_duplicates(subset=subset_cols)
            logger.info(f"Dropped {before - len(df)} duplicate rows by {subset_cols}")
        else:
            before = len(df)
            df = df.drop_duplicates()
            logger.info(f"Dropped {before - len(df)} exact duplicate rows")

        # Remove mostly-useless text/url columns (keep minimal core)
        drop_like = [
            'url', 'desc', 'emp_title', 'title', 'zip_code', 'policy_code',
            'application_type',  # optional depending on use-case
        ]
        to_drop = [c for c in df.columns if c.lower() in drop_like]
        if to_drop:
            df.drop(columns=to_drop, inplace=True, errors='ignore')

        # Consistency checks
        if {'funded_amnt', 'loan_amnt'} <= set(df.columns):
            mism = (df['funded_amnt'] > df['loan_amnt']).sum()
            if mism:
                logger.info(f"Capping funded_amnt to loan_amnt for {mism} rows")
            df['funded_amnt'] = np.minimum(df['funded_amnt'], df['loan_amnt'])

        if {'out_prncp', 'loan_status'} <= set(df.columns):
            fully_paid_mask = df['loan_status'] == 'Fully Paid'
            fix_cnt = (df.loc[fully_paid_mask, 'out_prncp'] != 0).sum()
            if fix_cnt:
                logger.info(f"Setting out_prncp=0 for Fully Paid on {fix_cnt} rows")
            df.loc[fully_paid_mask, 'out_prncp'] = 0.0

        # Negative balances -> NaN (for common balance/amount fields)
        neg_cols = [c for c in df.columns if any(k in c.lower() for k in ['bal', 'amt', 'amount'])]
        for c in neg_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                neg_count = (df[c] < 0).sum()
                if neg_count:
                    logger.info(f"Setting {neg_count} negatives to NaN in {c}")
                    df.loc[df[c] < 0, c] = np.nan

        # Convert data types
        date_cols = [c for c in ['issue_d', 'earliest_cr_line', 'last_pymnt_d'] if c in df.columns]
        for c in date_cols:
            df[c] = pd.to_datetime(df[c], format='%b-%Y', errors='coerce')

        if 'revol_util' in df.columns:
            # remove % and convert to [0,1]
            df['revol_util'] = (
                pd.to_numeric(df['revol_util'].astype(str).str.replace('%', '', regex=False), errors='coerce') / 100.0
            )

        # Trim garbage rows
        if 'annual_inc' in df.columns:
            before = len(df)
            df = df[df['annual_inc'] > 0]
            logger.info(f"Dropped {before - len(df)} rows with non-positive annual_inc")

        if 'dti' in df.columns:
            before = len(df)
            df = df[df['dti'] <= 999]
            logger.info(f"Dropped {before - len(df)} rows with dti > 999")

        return df

    # 2) Outliers and Missing -------------------------------------------------
    def cap_outliers(self, df: pd.DataFrame,
                     cols: Optional[List[str]] = None,
                     lower_q: float = 0.01,
                     upper_q: float = 0.99) -> pd.DataFrame:
        df = df.copy()
        if cols is None:
            cols = [c for c in ['annual_inc', 'revol_bal', 'dti'] if c in df.columns]
        for c in cols:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            lo, hi = df[c].quantile([lower_q, upper_q])
            df[c] = df[c].clip(lo, hi)
        logger.info(f"Winsorized/capped outliers for columns: {cols}")
        return df

    def logical_imputations(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        fill_zero_cols = [
            'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog'
        ]
        for c in fill_zero_cols:
            if c in df.columns:
                df[f'{c}_missing'] = df[c].isna().astype(int)
                df[c] = df[c].fillna(0)
        return df

    # 3) Feature Engineering --------------------------------------------------
    def engineer_features_accepted(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Target: binary default
        if 'loan_status' in df.columns:
            status_map = {s: 0 for s in self.GOOD_STATUSES}
            status_map.update({s: 1 for s in self.BAD_STATUSES})
            df['default'] = df['loan_status'].map(status_map)
            # Keep only labeled rows
            before = len(df)
            df = df[~df['default'].isna()].copy()
            df['default'] = df['default'].astype(int)
            logger.info(f"Filtered {before - len(df)} rows with non-final statuses")

        # fico average
        if {'fico_range_low', 'fico_range_high'} <= set(df.columns):
            df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2.0

        # credit history length in months
        if {'issue_d', 'earliest_cr_line'} <= set(df.columns):
            delta = (df['issue_d'] - df['earliest_cr_line'])
            df['credit_history_len_months'] = (delta.dt.days / 30.44).round(1)

        # payment to income
        if {'installment', 'annual_inc'} <= set(df.columns):
            with np.errstate(divide='ignore', invalid='ignore'):
                df['pmt_to_inc'] = df['installment'] / df['annual_inc']

        # grade mapping (A-G)
        if 'grade' in df.columns:
            mapping = {g: i for i, g in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G'], start=1)}
            df['grade_num'] = df['grade'].map(mapping)

        # log transforms
        if 'annual_inc' in df.columns:
            df['log_annual_inc'] = np.log1p(df['annual_inc'])

        return df

    # 4) Pipeline -------------------------------------------------------------
    def prepare_accepted(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full ordered pipeline (clean → cap/outliers & impute → engineer) for accepted data.
        Returns a DataFrame with engineered features and 'default' target.
        """
        df1 = self.clean_accepted_raw(df)
        df2 = self.cap_outliers(df1)
        df3 = self.logical_imputations(df2)
        df4 = self.engineer_features_accepted(df3)
        logger.info(f"Accepted dataset prepared. Shape: {df4.shape}")
        return df4

    # Rejected data (no target) ----------------------------------------------
    def clean_rejected_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Drop duplicates
        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"Dropped {before - len(df)} exact duplicate rows (rejected)")

        # Convert percentage-like cols if present
        for c in [c for c in df.columns if 'util' in c.lower()]:
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace('%', '', regex=False), errors='coerce') / 100.0

        # Enforce numeric types where feasible
        for c in df.columns:
            if df[c].dtype == object:
                # try to coerce common numeric strings
                df[c] = pd.to_numeric(df[c], errors='ignore')
        return df

    def engineer_features_rejected(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Amount Requested' in df.columns and 'Risk_Score' in df.columns:
            df['risk_per_dollar'] = df['Risk_Score'] / (df['Amount Requested'] + 1e-6)
        if 'Debt-To-Income Ratio' in df.columns:
            # normalize percent to [0,1]
            if df['Debt-To-Income Ratio'].dtype == object:
                df['dti_rej'] = pd.to_numeric(df['Debt-To-Income Ratio'].str.rstrip('%'), errors='coerce') / 100.0
            else:
                df['dti_rej'] = df['Debt-To-Income Ratio']
        return df

    def prepare_rejected(self, df: pd.DataFrame) -> pd.DataFrame:
        df1 = self.clean_rejected_raw(df)
        df2 = self.engineer_features_rejected(df1)
        logger.info(f"Rejected dataset prepared. Shape: {df2.shape}")
        return df2

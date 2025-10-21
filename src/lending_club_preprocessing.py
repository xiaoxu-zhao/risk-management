"""
Lending Club Accepted-Loans Preprocessing
========================================

1) Raw cleaning
2) Outliers & logical imputations
3) Feature engineering
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LendingClubPreprocessor:
    GOOD_STATUSES = {'Fully Paid'}
    BAD_STATUSES = {'Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off'}

    def __init__(self, include_pricing_features: bool = False) -> None:
        self.include_pricing_features = include_pricing_features
        self.useless_cols = ['id', 'member_id', 'url', 'desc', 'title', 'zip_code', 'emp_title', 'policy_code']
        self.leakage_cols = [
            'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
            'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
            'recoveries', 'collection_recovery_fee',
            'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d',
            'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
            'hardship_amount', 'hardship_start_date', 'hardship_end_date',
            'payment_plan_start_date', 'hardship_length', 'hardship_dpd',
            'hardship_loan_status', 'orig_projected_additional_accrued_interest',
            'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
            'debt_settlement_flag', 'debt_settlement_flag_date', 'settlement_status',
            'settlement_date', 'settlement_amount', 'settlement_percentage', 'settlement_term',
            'last_credit_pull_d',
        ]
        self.pricing_cols = ['int_rate', 'grade', 'sub_grade']

    # Step 1: Raw cleaning
    def clean_accepted_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        subset_cols = [c for c in ['id', 'member_id'] if c in df.columns]
        df = df.drop_duplicates(subset=subset_cols) if subset_cols else df.drop_duplicates()

        drop_now = [c for c in (self.useless_cols + self.leakage_cols) if c in df.columns]
        if not self.include_pricing_features:
            drop_now += [c for c in self.pricing_cols if c in df.columns]
        if drop_now:
            df.drop(columns=drop_now, inplace=True, errors='ignore')

        if 'revol_util' in df.columns:
            df['revol_util'] = pd.to_numeric(
                df['revol_util'].astype(str).str.replace('%', '', regex=False).str.strip(),
                errors='coerce'
            ) / 100.0

        for c in ['issue_d', 'earliest_cr_line', 'last_pymnt_d']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], format='%b-%Y', errors='coerce')

        if {'funded_amnt', 'loan_amnt'}.issubset(df.columns):
            df['funded_amnt'] = np.minimum(df['funded_amnt'], df['loan_amnt'])

        for c in df.select_dtypes(include=[np.number]).columns:
            df.loc[df[c] < 0, c] = np.nan

        if 'annual_inc' in df.columns:
            df = df[df['annual_inc'].isna() | (df['annual_inc'] > 0)]
        if 'dti' in df.columns:
            df = df[df['dti'].isna() | (df['dti'] <= 999)]
        return df

    # Step 2: Outliers & logical imputations
    def cap_outliers(self, df: pd.DataFrame, cols: Optional[List[str]] = None,
                     lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
        df = df.copy()
        cols = cols or [c for c in ['annual_inc', 'revol_bal', 'dti'] if c in df.columns]
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                lo, hi = df[c].quantile([lower_q, upper_q])
                df[c] = df[c].clip(lo, hi)
        return df

    def logical_imputations(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog']:
            if c in df.columns:
                df[f'{c}_missing'] = df[c].isna().astype(int)
                df[c] = df[c].fillna(0)
        for c in ['revol_util', 'dti', 'annual_inc']:
            if c in df.columns:
                df[f'{c}_missing'] = df[c].isna().astype(int)
        return df

    # Step 3: Feature engineering
    def engineer_features_accepted(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'loan_status' in df.columns:
            status_map = {s: 0 for s in self.GOOD_STATUSES}
            status_map.update({s: 1 for s in self.BAD_STATUSES})
            df['default'] = df['loan_status'].map(status_map)
            df = df[~df['default'].isna()].copy()
            df['default'] = df['default'].astype(int)
            df.drop(columns=['loan_status'], inplace=True, errors='ignore')

        if {'fico_range_low', 'fico_range_high'}.issubset(df.columns):
            df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2.0

        if {'issue_d', 'earliest_cr_line'}.issubset(df.columns):
            df['credit_history_len_months'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 30.44

        if {'installment', 'annual_inc'}.issubset(df.columns):
            with np.errstate(divide='ignore', invalid='ignore'):
                df['pmt_to_inc'] = df['installment'] / (df['annual_inc'] / 12.0)

        if 'annual_inc' in df.columns:
            df['log_annual_inc'] = np.log1p(df['annual_inc'])

        if 'grade' in df.columns and self.include_pricing_features:
            df['grade_num'] = df['grade'].map({g: i for i, g in enumerate('ABCDEFG', start=1)})

        return df

    # Orchestrator
    def prepare_accepted(self, df: pd.DataFrame) -> pd.DataFrame:
        df1 = self.clean_accepted_raw(df)
        df2 = self.cap_outliers(df1)
        df3 = self.logical_imputations(df2)
        df4 = self.engineer_features_accepted(df3)
        return df4

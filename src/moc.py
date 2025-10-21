"""
Model of Credit (MoC) Module
============================

Placeholder for broader credit risk framework (PD/LGD/EAD, pricing, capital).
Not required for accepted-loans data loading/cleaning/feature-engineering.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ModelOfCredit:
    """
    Stub for future extension. Keeps interface minimal for now.
    """

    def __init__(self) -> None:
        pass

    def calculate_pd_components(
        self,
        scores: np.ndarray,
        calibration_data: Optional[Dict[str, np.ndarray]] = None,
        method: str = 'logistic',
    ) -> Dict[str, Any]:
        """
        Placeholder for PD calibration decomposition.
        """
        return {"method": method, "n_scores": int(scores.shape[0])}

    def estimate_lgd_distribution(
        self,
        recovery_data: np.ndarray,
        facility_types: Optional[np.ndarray] = None,
        collateral_values: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Placeholder for LGD distribution estimation.
        """
        return {"n_observations": int(recovery_data.shape[0])}
"""
Model Monitoring Module
======================

Comprehensive model monitoring and stability testing for credit risk models.
Includes drift detection, performance monitoring, and automated reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Comprehensive model monitoring toolkit for credit risk models.
    
    Features:
    - Performance stability testing
    - Population stability index (PSI)
    - Kolmogorov-Smirnov (KS) test monitoring  
    - Drift detection algorithms
    - Automated monitoring reports
    - Alert generation
    """
    
    def __init__(self, alert_thresholds: Dict[str, float] = None):
        """
        Initialize model monitor with configurable alert thresholds.
        
        Args:
            alert_thresholds: Dictionary of metric thresholds for alerts
        """
        self.alert_thresholds = alert_thresholds or {
            'psi_threshold': 0.25,        # PSI > 0.25 indicates significant shift
            'ks_threshold': 0.1,          # KS statistic threshold
            'auc_degradation': 0.05,      # AUC degradation threshold
            'calibration_error': 0.1,     # Maximum acceptable calibration error
            'default_rate_shift': 0.5     # Relative change in default rate
        }
        
        self.monitoring_history = {}
        self.alerts_generated = []
        
    def calculate_population_stability_index(self, reference_scores: np.ndarray,
                                           current_scores: np.ndarray,
                                           n_bins: int = 10,
                                           method: str = 'quantile') -> Dict[str, Any]:
        """
        Calculate Population Stability Index (PSI) between reference and current score distributions.
        
        Args:
            reference_scores: Reference/baseline score distribution
            current_scores: Current score distribution to compare
            n_bins: Number of bins for PSI calculation
            method: Binning method ('quantile' or 'equal_width')
            
        Returns:
            Dictionary with PSI results and bin-level details
        """
        # Create bins based on reference distribution
        if method == 'quantile':
            bin_edges = np.percentile(reference_scores, np.linspace(0, 100, n_bins + 1))
        elif method == 'equal_width':
            min_score = np.min(reference_scores)
            max_score = np.max(reference_scores)
            bin_edges = np.linspace(min_score, max_score, n_bins + 1)
        else:
            raise ValueError(f"Unknown binning method: {method}")
        
        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) <= 1:
            return {'psi': 0.0, 'warning': 'Insufficient unique values for PSI calculation'}
        
        # Calculate bin frequencies
        ref_counts, _ = np.histogram(reference_scores, bins=bin_edges)
        cur_counts, _ = np.histogram(current_scores, bins=bin_edges)
        
        # Convert to percentages (add small constant to avoid log(0))
        ref_percentages = (ref_counts + 1e-6) / np.sum(ref_counts + 1e-6)
        cur_percentages = (cur_counts + 1e-6) / np.sum(cur_counts + 1e-6)
        
        # Calculate PSI
        psi_values = (cur_percentages - ref_percentages) * np.log(cur_percentages / ref_percentages)
        total_psi = np.sum(psi_values)
        
        # Interpretation
        if total_psi < 0.1:
            interpretation = "No significant change"
        elif total_psi < 0.25:
            interpretation = "Some minor change"
        else:
            interpretation = "Major shift in population"
        
        # Generate alert if threshold exceeded
        if total_psi > self.alert_thresholds['psi_threshold']:
            alert = {
                'type': 'PSI_ALERT',
                'value': total_psi,
                'threshold': self.alert_thresholds['psi_threshold'],
                'message': f"PSI value {total_psi:.4f} exceeds threshold",
                'timestamp': datetime.now()
            }
            self.alerts_generated.append(alert)
        
        psi_results = {
            'psi': total_psi,
            'interpretation': interpretation,
            'bin_details': {
                'bin_edges': bin_edges.tolist(),
                'reference_percentages': ref_percentages.tolist(),
                'current_percentages': cur_percentages.tolist(),
                'psi_by_bin': psi_values.tolist()
            },
            'reference_stats': {
                'mean': np.mean(reference_scores),
                'std': np.std(reference_scores),
                'count': len(reference_scores)
            },
            'current_stats': {
                'mean': np.mean(current_scores),
                'std': np.std(current_scores),
                'count': len(current_scores)
            }
        }
        
        logger.info(f"PSI calculated: {total_psi:.4f} ({interpretation})")
        return psi_results
    
    def calculate_ks_statistic(self, reference_scores: np.ndarray,
                              current_scores: np.ndarray,
                              reference_targets: np.ndarray = None,
                              current_targets: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate Kolmogorov-Smirnov statistic for distribution comparison.
        
        Args:
            reference_scores: Reference score distribution
            current_scores: Current score distribution
            reference_targets: Reference target values (for discriminatory power)
            current_targets: Current target values (for discriminatory power)
            
        Returns:
            Dictionary with KS test results
        """
        from scipy.stats import ks_2samp
        
        # Two-sample KS test for score distributions
        ks_statistic, p_value = ks_2samp(reference_scores, current_scores)
        
        ks_results = {
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'significant_change': p_value < 0.05
        }
        
        # Calculate KS for model discriminatory power if targets provided
        if reference_targets is not None and current_targets is not None:
            # KS between good and bad distributions
            ref_good_scores = reference_scores[reference_targets == 0]
            ref_bad_scores = reference_scores[reference_targets == 1]
            cur_good_scores = current_scores[current_targets == 0]
            cur_bad_scores = current_scores[current_targets == 1]
            
            if len(ref_good_scores) > 0 and len(ref_bad_scores) > 0:
                ref_ks_power, _ = ks_2samp(ref_good_scores, ref_bad_scores)
                
            if len(cur_good_scores) > 0 and len(cur_bad_scores) > 0:
                cur_ks_power, _ = ks_2samp(cur_good_scores, cur_bad_scores)
                
                ks_results.update({
                    'reference_ks_power': ref_ks_power,
                    'current_ks_power': cur_ks_power,
                    'ks_power_change': cur_ks_power - ref_ks_power,
                    'discriminatory_power_maintained': abs(cur_ks_power - ref_ks_power) < self.alert_thresholds['ks_threshold']
                })
        
        # Generate alert if significant degradation
        if ks_statistic > self.alert_thresholds['ks_threshold']:
            alert = {
                'type': 'KS_ALERT',
                'value': ks_statistic,
                'threshold': self.alert_thresholds['ks_threshold'],
                'message': f"KS statistic {ks_statistic:.4f} indicates distribution shift",
                'timestamp': datetime.now()
            }
            self.alerts_generated.append(alert)
        
        logger.info(f"KS statistic calculated: {ks_statistic:.4f} (p-value: {p_value:.6f})")
        return ks_results
    
    def calculate_model_performance_metrics(self, y_true: np.ndarray,
                                          y_pred_proba: np.ndarray,
                                          sample_weight: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive model performance metrics.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            sample_weight: Sample weights (optional)
            
        Returns:
            Dictionary with performance metrics
        """
        # Basic classification metrics
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Confusion matrix components
        if sample_weight is None:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
        else:
            tp = np.sum(sample_weight[(y_true == 1) & (y_pred == 1)])
            tn = np.sum(sample_weight[(y_true == 0) & (y_pred == 0)])
            fp = np.sum(sample_weight[(y_true == 0) & (y_pred == 1)])
            fn = np.sum(sample_weight[(y_true == 1) & (y_pred == 0)])
        
        # Basic metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # Calculate AUC-ROC
        auc_roc = self._calculate_auc(y_true, y_pred_proba, sample_weight)
        
        # Calculate KS statistic between good and bad distributions
        good_scores = y_pred_proba[y_true == 0]
        bad_scores = y_pred_proba[y_true == 1]
        
        if len(good_scores) > 0 and len(bad_scores) > 0:
            from scipy.stats import ks_2samp
            ks_stat, _ = ks_2samp(good_scores, bad_scores)
        else:
            ks_stat = 0.0
        
        # Gini coefficient (2 * AUC - 1)
        gini = 2 * auc_roc - 1
        
        metrics = {
            'auc_roc': auc_roc,
            'gini': gini,
            'ks_statistic': ks_stat,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'default_rate': np.mean(y_true) if len(y_true) > 0 else 0.0,
            'predicted_default_rate': np.mean(y_pred_proba) if len(y_pred_proba) > 0 else 0.0
        }
        
        return metrics
    
    def _calculate_auc(self, y_true: np.ndarray, y_scores: np.ndarray, 
                      sample_weight: np.ndarray = None) -> float:
        """
        Calculate AUC-ROC score with optional sample weights.
        
        Args:
            y_true: True binary labels  
            y_scores: Predicted scores/probabilities
            sample_weight: Sample weights
            
        Returns:
            AUC-ROC score
        """
        # Sort by scores (descending)
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_sorted = y_true[desc_score_indices]
        scores_sorted = y_scores[desc_score_indices]
        
        if sample_weight is not None:
            weights_sorted = sample_weight[desc_score_indices]
        else:
            weights_sorted = np.ones_like(y_sorted)
        
        # Calculate weighted TPR and FPR
        tps = np.cumsum(y_sorted * weights_sorted)
        fps = np.cumsum((1 - y_sorted) * weights_sorted)
        
        # Total positives and negatives
        total_pos = tps[-1] if len(tps) > 0 else 0
        total_neg = fps[-1] if len(fps) > 0 else 0
        
        if total_pos == 0 or total_neg == 0:
            return 0.5  # No discriminatory power
        
        # Calculate TPR and FPR
        tpr = tps / total_pos
        fpr = fps / total_neg
        
        # Add (0,0) point
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)
        return abs(auc)  # Ensure positive value
    
    def calculate_calibration_error(self, y_true: np.ndarray,
                                   y_pred_proba: np.ndarray,
                                   n_bins: int = 10) -> Dict[str, Any]:
        """
        Calculate model calibration error and reliability statistics.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration analysis
            
        Returns:
            Dictionary with calibration metrics
        """
        # Create bins based on predicted probabilities
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        bin_centers = []
        observed_rates = []
        predicted_rates = []
        bin_counts = []
        calibration_errors = []
        
        for i in range(n_bins):
            # Define bin boundaries
            if i == 0:
                mask = y_pred_proba <= bin_edges[i + 1]
            elif i == n_bins - 1:
                mask = y_pred_proba > bin_edges[i]
            else:
                mask = (y_pred_proba > bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])
            
            if np.sum(mask) > 0:
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                observed_rate = np.mean(y_true[mask])
                predicted_rate = np.mean(y_pred_proba[mask])
                count = np.sum(mask)
                
                bin_centers.append(bin_center)
                observed_rates.append(observed_rate)
                predicted_rates.append(predicted_rate)
                bin_counts.append(count)
                calibration_errors.append(abs(observed_rate - predicted_rate))
        
        # Calculate overall calibration metrics
        if len(calibration_errors) > 0:
            # Expected Calibration Error (ECE)
            total_samples = len(y_true)
            weights = np.array(bin_counts) / total_samples
            ece = np.sum(weights * calibration_errors)
            
            # Maximum Calibration Error (MCE)
            mce = np.max(calibration_errors)
            
            # Brier Score
            brier_score = np.mean((y_pred_proba - y_true) ** 2)
            
            # Brier Score decomposition
            reliability = np.sum(weights * np.array(calibration_errors) ** 2)
            resolution = np.sum(weights * (np.array(observed_rates) - np.mean(y_true)) ** 2)
            uncertainty = np.mean(y_true) * (1 - np.mean(y_true))
            
        else:
            ece = mce = brier_score = reliability = resolution = uncertainty = 0.0
        
        # Generate alert for poor calibration
        if ece > self.alert_thresholds['calibration_error']:
            alert = {
                'type': 'CALIBRATION_ALERT',
                'value': ece,
                'threshold': self.alert_thresholds['calibration_error'],
                'message': f"Expected Calibration Error {ece:.4f} exceeds threshold",
                'timestamp': datetime.now()
            }
            self.alerts_generated.append(alert)
        
        calibration_results = {
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'brier_score': brier_score,
            'reliability': reliability,
            'resolution': resolution,
            'uncertainty': uncertainty,
            'bin_details': {
                'bin_centers': bin_centers,
                'observed_rates': observed_rates,
                'predicted_rates': predicted_rates,
                'bin_counts': bin_counts,
                'calibration_errors': calibration_errors
            }
        }
        
        logger.info(f"Calibration analysis completed. ECE: {ece:.4f}, MCE: {mce:.4f}")
        return calibration_results
    
    def detect_concept_drift(self, reference_data: Dict[str, np.ndarray],
                            current_data: Dict[str, np.ndarray],
                            drift_detection_method: str = 'combined') -> Dict[str, Any]:
        """
        Detect concept drift using multiple statistical tests.
        
        Args:
            reference_data: Reference dataset with 'features', 'targets', 'scores'
            current_data: Current dataset with same structure
            drift_detection_method: Method to use ('psi', 'ks', 'combined')
            
        Returns:
            Dictionary with drift detection results
        """
        drift_results = {
            'drift_detected': False,
            'drift_strength': 'None',
            'affected_features': [],
            'test_results': {}
        }
        
        # Feature drift detection
        if 'features' in reference_data and 'features' in current_data:
            ref_features = reference_data['features']
            cur_features = current_data['features']
            
            if ref_features.shape[1] == cur_features.shape[1]:
                feature_drift_results = {}
                
                for i in range(ref_features.shape[1]):
                    feature_name = f'feature_{i}'
                    
                    # PSI for this feature
                    psi_result = self.calculate_population_stability_index(
                        ref_features[:, i], cur_features[:, i], n_bins=10
                    )
                    
                    # KS test for this feature
                    ks_result = self.calculate_ks_statistic(
                        ref_features[:, i], cur_features[:, i]
                    )
                    
                    feature_drift_results[feature_name] = {
                        'psi': psi_result['psi'],
                        'ks_statistic': ks_result['ks_statistic'],
                        'ks_p_value': ks_result['p_value']
                    }
                    
                    # Check if this feature shows drift
                    if (psi_result['psi'] > self.alert_thresholds['psi_threshold'] or 
                        ks_result['ks_statistic'] > self.alert_thresholds['ks_threshold']):
                        drift_results['affected_features'].append(feature_name)
                
                drift_results['test_results']['feature_drift'] = feature_drift_results
        
        # Score drift detection
        if 'scores' in reference_data and 'scores' in current_data:
            score_psi = self.calculate_population_stability_index(
                reference_data['scores'], current_data['scores']
            )
            
            score_ks = self.calculate_ks_statistic(
                reference_data['scores'], current_data['scores']
            )
            
            drift_results['test_results']['score_drift'] = {
                'psi': score_psi['psi'],
                'ks_statistic': score_ks['ks_statistic'],
                'ks_p_value': score_ks['p_value']
            }
        
        # Target distribution drift
        if 'targets' in reference_data and 'targets' in current_data:
            ref_default_rate = np.mean(reference_data['targets'])
            cur_default_rate = np.mean(current_data['targets'])
            
            relative_change = abs(cur_default_rate - ref_default_rate) / (ref_default_rate + 1e-8)
            
            drift_results['test_results']['target_drift'] = {
                'reference_default_rate': ref_default_rate,
                'current_default_rate': cur_default_rate,
                'absolute_change': cur_default_rate - ref_default_rate,
                'relative_change': relative_change
            }
            
            if relative_change > self.alert_thresholds['default_rate_shift']:
                drift_results['affected_features'].append('target_distribution')
        
        # Overall drift assessment
        drift_indicators = []
        
        if 'score_drift' in drift_results['test_results']:
            score_drift = drift_results['test_results']['score_drift']
            if score_drift['psi'] > self.alert_thresholds['psi_threshold']:
                drift_indicators.append('score_psi')
            if score_drift['ks_statistic'] > self.alert_thresholds['ks_threshold']:
                drift_indicators.append('score_ks')
        
        if len(drift_results['affected_features']) > 0:
            drift_indicators.append('feature_drift')
        
        # Determine drift strength
        if len(drift_indicators) == 0:
            drift_results['drift_strength'] = 'None'
        elif len(drift_indicators) <= 2:
            drift_results['drift_strength'] = 'Mild'
        else:
            drift_results['drift_strength'] = 'Severe'
        
        drift_results['drift_detected'] = len(drift_indicators) > 0
        drift_results['drift_indicators'] = drift_indicators
        
        # Generate drift alert
        if drift_results['drift_detected']:
            alert = {
                'type': 'DRIFT_ALERT',
                'strength': drift_results['drift_strength'],
                'indicators': drift_indicators,
                'affected_features': drift_results['affected_features'],
                'message': f"Concept drift detected ({drift_results['drift_strength']})",
                'timestamp': datetime.now()
            }
            self.alerts_generated.append(alert)
        
        logger.info(f"Drift detection completed. Drift detected: {drift_results['drift_detected']} "
                   f"(Strength: {drift_results['drift_strength']})")
        
        return drift_results
    
    def generate_monitoring_report(self, model_name: str,
                                 monitoring_data: Dict[str, Any],
                                 report_date: datetime = None) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report.
        
        Args:
            model_name: Name of the model being monitored
            monitoring_data: Dictionary with all monitoring results
            report_date: Date of the report (default: current date)
            
        Returns:
            Formatted monitoring report
        """
        if report_date is None:
            report_date = datetime.now()
        
        report = {
            'report_metadata': {
                'model_name': model_name,
                'report_date': report_date.isoformat(),
                'report_type': 'Model Monitoring Report',
                'generated_by': 'ModelMonitor'
            },
            'executive_summary': {},
            'detailed_results': monitoring_data,
            'alerts': [alert for alert in self.alerts_generated 
                      if alert['timestamp'].date() == report_date.date()],
            'recommendations': []
        }
        
        # Generate executive summary
        summary = {
            'total_alerts': len(report['alerts']),
            'drift_detected': False,
            'performance_degradation': False,
            'calibration_issues': False
        }
        
        # Check for specific issues
        for alert in report['alerts']:
            if alert['type'] == 'DRIFT_ALERT':
                summary['drift_detected'] = True
            elif alert['type'] in ['PSI_ALERT', 'KS_ALERT']:
                summary['performance_degradation'] = True
            elif alert['type'] == 'CALIBRATION_ALERT':
                summary['calibration_issues'] = True
        
        # Generate recommendations
        recommendations = []
        
        if summary['drift_detected']:
            recommendations.append(
                "Consider model retraining due to detected concept drift"
            )
        
        if summary['performance_degradation']:
            recommendations.append(
                "Investigate population stability and model discriminatory power"
            )
        
        if summary['calibration_issues']:
            recommendations.append(
                "Review model calibration and consider recalibration"
            )
        
        if len(recommendations) == 0:
            recommendations.append("Model performance is stable - continue monitoring")
        
        report['executive_summary'] = summary
        report['recommendations'] = recommendations
        
        # Store in monitoring history
        self.monitoring_history[f"{model_name}_{report_date.date()}"] = report
        
        logger.info(f"Monitoring report generated for {model_name}. "
                   f"Alerts: {summary['total_alerts']}")
        
        return report
    
    def get_alert_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get summary of alerts generated in the past N days.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Alert summary statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        recent_alerts = [alert for alert in self.alerts_generated 
                        if alert['timestamp'] >= cutoff_date]
        
        alert_types = {}
        for alert in recent_alerts:
            alert_type = alert['type']
            if alert_type not in alert_types:
                alert_types[alert_type] = 0
            alert_types[alert_type] += 1
        
        summary = {
            'total_alerts': len(recent_alerts),
            'alert_types': alert_types,
            'period_days': days_back,
            'most_recent_alert': recent_alerts[-1] if recent_alerts else None,
            'alert_frequency': len(recent_alerts) / days_back
        }
        
        return summary


if __name__ == "__main__":
    # Example usage
    monitor = ModelMonitor()
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Reference data (baseline)
    ref_scores = np.random.normal(0.3, 0.15, n_samples)
    ref_targets = np.random.binomial(1, 0.05, n_samples)
    
    # Current data (with some drift)
    cur_scores = np.random.normal(0.35, 0.18, n_samples)  # Slight drift
    cur_targets = np.random.binomial(1, 0.08, n_samples)  # Higher default rate
    
    # Calculate PSI
    psi_results = monitor.calculate_population_stability_index(ref_scores, cur_scores)
    print(f"PSI: {psi_results['psi']:.4f} - {psi_results['interpretation']}")
    
    # Calculate KS statistic
    ks_results = monitor.calculate_ks_statistic(ref_scores, cur_scores)
    print(f"KS Statistic: {ks_results['ks_statistic']:.4f}")
    
    # Calculate performance metrics
    perf_metrics = monitor.calculate_model_performance_metrics(cur_targets, cur_scores)
    print(f"AUC: {perf_metrics['auc_roc']:.4f}, KS: {perf_metrics['ks_statistic']:.4f}")
    
    # Detect concept drift
    reference_data = {'scores': ref_scores, 'targets': ref_targets}
    current_data = {'scores': cur_scores, 'targets': cur_targets}
    
    drift_results = monitor.detect_concept_drift(reference_data, current_data)
    print(f"Drift detected: {drift_results['drift_detected']} ({drift_results['drift_strength']})")
    
    # Generate monitoring report
    monitoring_data = {
        'psi': psi_results,
        'ks': ks_results,
        'performance': perf_metrics,
        'drift': drift_results
    }
    
    report = monitor.generate_monitoring_report("CreditRiskModel_v1", monitoring_data)
    print(f"Report generated with {report['executive_summary']['total_alerts']} alerts")
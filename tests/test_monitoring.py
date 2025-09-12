"""
Tests for monitoring module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from monitoring import ModelMonitor


class TestModelMonitor:
    """Test cases for ModelMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = ModelMonitor()
        
        # Sample data for testing
        np.random.seed(42)
        self.n_samples = 1000
        
        # Reference data (baseline)
        self.reference_scores = np.random.normal(0.3, 0.15, self.n_samples)
        self.reference_targets = np.random.binomial(1, 0.05, self.n_samples)
        
        # Current data (with some drift)
        self.current_scores = np.random.normal(0.35, 0.18, self.n_samples)
        self.current_targets = np.random.binomial(1, 0.08, self.n_samples)
        
        # Features for drift detection
        self.reference_features = np.random.normal(0, 1, (self.n_samples, 5))
        self.current_features = np.random.normal(0.2, 1.1, (self.n_samples, 5))
    
    def test_initialization(self):
        """Test ModelMonitor initialization."""
        assert isinstance(self.monitor.alert_thresholds, dict)
        assert isinstance(self.monitor.monitoring_history, dict)
        assert isinstance(self.monitor.alerts_generated, list)
        
        # Check default thresholds
        assert self.monitor.alert_thresholds['psi_threshold'] == 0.25
        assert self.monitor.alert_thresholds['ks_threshold'] == 0.1
    
    def test_calculate_population_stability_index(self):
        """Test PSI calculation."""
        psi_results = self.monitor.calculate_population_stability_index(
            self.reference_scores, self.current_scores
        )
        
        # Check structure
        required_keys = ['psi', 'interpretation', 'bin_details']
        for key in required_keys:
            assert key in psi_results
        
        # Check PSI value
        assert isinstance(psi_results['psi'], float)
        assert psi_results['psi'] >= 0
        
        # Check interpretation
        assert psi_results['interpretation'] in [
            "No significant change", "Some minor change", "Major shift in population"
        ]
        
        # Check bin details
        bin_details = psi_results['bin_details']
        assert 'bin_edges' in bin_details
        assert 'reference_percentages' in bin_details
        assert 'current_percentages' in bin_details
        
        # Check that percentages sum to approximately 1
        ref_sum = sum(bin_details['reference_percentages'])
        cur_sum = sum(bin_details['current_percentages'])
        assert abs(ref_sum - 1.0) < 1e-6
        assert abs(cur_sum - 1.0) < 1e-6
    
    def test_calculate_psi_with_different_methods(self):
        """Test PSI calculation with different binning methods."""
        # Quantile method
        psi_quantile = self.monitor.calculate_population_stability_index(
            self.reference_scores, self.current_scores, method='quantile'
        )
        
        # Equal width method
        psi_equal = self.monitor.calculate_population_stability_index(
            self.reference_scores, self.current_scores, method='equal_width'
        )
        
        # Both should return valid results
        assert psi_quantile['psi'] >= 0
        assert psi_equal['psi'] >= 0
        
        # Results might differ between methods
        # but both should be reasonable
        assert psi_quantile['psi'] < 10  # Sanity check
        assert psi_equal['psi'] < 10    # Sanity check
    
    def test_calculate_ks_statistic(self):
        """Test KS statistic calculation."""
        ks_results = self.monitor.calculate_ks_statistic(
            self.reference_scores, self.current_scores
        )
        
        # Check structure
        required_keys = ['ks_statistic', 'p_value', 'significant_change']
        for key in required_keys:
            assert key in ks_results
        
        # Check values
        assert 0 <= ks_results['ks_statistic'] <= 1
        assert 0 <= ks_results['p_value'] <= 1
        assert isinstance(ks_results['significant_change'], bool)
    
    def test_calculate_ks_with_targets(self):
        """Test KS statistic with target information for discriminatory power."""
        ks_results = self.monitor.calculate_ks_statistic(
            self.reference_scores, self.current_scores,
            self.reference_targets, self.current_targets
        )
        
        # Should include discriminatory power analysis
        if 'reference_ks_power' in ks_results:
            assert 0 <= ks_results['reference_ks_power'] <= 1
            assert 0 <= ks_results['current_ks_power'] <= 1
            assert 'ks_power_change' in ks_results
            assert 'discriminatory_power_maintained' in ks_results
    
    def test_calculate_model_performance_metrics(self):
        """Test model performance metrics calculation."""
        # Create predictions close to targets for testing
        y_pred_proba = self.current_targets + np.random.normal(0, 0.1, len(self.current_targets))
        y_pred_proba = np.clip(y_pred_proba, 0, 1)
        
        metrics = self.monitor.calculate_model_performance_metrics(
            self.current_targets, y_pred_proba
        )
        
        # Check required metrics
        required_metrics = [
            'auc_roc', 'gini', 'ks_statistic', 'precision', 'recall',
            'f1_score', 'accuracy', 'default_rate', 'predicted_default_rate'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Check value ranges
        assert 0 <= metrics['auc_roc'] <= 1
        assert -1 <= metrics['gini'] <= 1
        assert 0 <= metrics['ks_statistic'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_calculate_calibration_error(self):
        """Test calibration error calculation."""
        # Create well-calibrated predictions for testing
        y_pred_proba = np.random.uniform(0, 1, self.n_samples)
        y_true = np.random.binomial(1, y_pred_proba)
        
        cal_results = self.monitor.calculate_calibration_error(y_true, y_pred_proba)
        
        # Check structure
        required_keys = [
            'expected_calibration_error', 'maximum_calibration_error',
            'brier_score', 'reliability', 'resolution', 'uncertainty'
        ]
        
        for key in required_keys:
            assert key in cal_results
            assert isinstance(cal_results[key], float)
            assert cal_results[key] >= 0  # All should be non-negative
        
        # Check bin details
        assert 'bin_details' in cal_results
        bin_details = cal_results['bin_details']
        assert 'bin_centers' in bin_details
        assert 'observed_rates' in bin_details
        assert 'predicted_rates' in bin_details
    
    def test_detect_concept_drift(self):
        """Test concept drift detection."""
        reference_data = {
            'scores': self.reference_scores,
            'targets': self.reference_targets,
            'features': self.reference_features
        }
        
        current_data = {
            'scores': self.current_scores,
            'targets': self.current_targets,
            'features': self.current_features
        }
        
        drift_results = self.monitor.detect_concept_drift(reference_data, current_data)
        
        # Check structure
        required_keys = [
            'drift_detected', 'drift_strength', 'affected_features', 'test_results'
        ]
        
        for key in required_keys:
            assert key in drift_results
        
        # Check values
        assert isinstance(drift_results['drift_detected'], bool)
        assert drift_results['drift_strength'] in ['None', 'Mild', 'Severe']
        assert isinstance(drift_results['affected_features'], list)
        
        # Check test results
        test_results = drift_results['test_results']
        
        if 'feature_drift' in test_results:
            feature_drift = test_results['feature_drift']
            assert isinstance(feature_drift, dict)
            
            # Should have results for each feature
            for feature_name, results in feature_drift.items():
                assert 'psi' in results
                assert 'ks_statistic' in results
                assert results['psi'] >= 0
                assert 0 <= results['ks_statistic'] <= 1
        
        if 'score_drift' in test_results:
            score_drift = test_results['score_drift']
            assert 'psi' in score_drift
            assert 'ks_statistic' in score_drift
        
        if 'target_drift' in test_results:
            target_drift = test_results['target_drift']
            assert 'reference_default_rate' in target_drift
            assert 'current_default_rate' in target_drift
            assert 'relative_change' in target_drift
    
    def test_generate_monitoring_report(self):
        """Test monitoring report generation."""
        # Generate some monitoring data
        psi_results = self.monitor.calculate_population_stability_index(
            self.reference_scores, self.current_scores
        )
        
        monitoring_data = {
            'psi': psi_results
        }
        
        report = self.monitor.generate_monitoring_report(
            'TestModel', monitoring_data
        )
        
        # Check structure
        required_sections = [
            'report_metadata', 'executive_summary', 'detailed_results',
            'alerts', 'recommendations'
        ]
        
        for section in required_sections:
            assert section in report
        
        # Check metadata
        metadata = report['report_metadata']
        assert metadata['model_name'] == 'TestModel'
        assert 'report_date' in metadata
        
        # Check executive summary
        summary = report['executive_summary']
        assert 'total_alerts' in summary
        assert 'drift_detected' in summary
        
        # Check recommendations
        assert isinstance(report['recommendations'], list)
        assert len(report['recommendations']) > 0
    
    def test_alert_generation(self):
        """Test alert generation functionality."""
        initial_alert_count = len(self.monitor.alerts_generated)
        
        # Create data that should trigger alerts
        high_psi_scores = np.random.normal(0.8, 0.2, self.n_samples)  # Very different distribution
        
        # This should trigger a PSI alert
        psi_results = self.monitor.calculate_population_stability_index(
            self.reference_scores, high_psi_scores
        )
        
        # Check if alert was generated
        assert len(self.monitor.alerts_generated) > initial_alert_count
        
        # Check alert structure
        latest_alert = self.monitor.alerts_generated[-1]
        assert 'type' in latest_alert
        assert 'value' in latest_alert
        assert 'threshold' in latest_alert
        assert 'message' in latest_alert
        assert 'timestamp' in latest_alert
    
    def test_get_alert_summary(self):
        """Test alert summary functionality."""
        # Generate some alerts first
        high_psi_scores = np.random.normal(0.8, 0.2, self.n_samples)
        self.monitor.calculate_population_stability_index(
            self.reference_scores, high_psi_scores
        )
        
        # Get alert summary
        summary = self.monitor.get_alert_summary(days_back=30)
        
        # Check structure
        required_keys = [
            'total_alerts', 'alert_types', 'period_days', 'alert_frequency'
        ]
        
        for key in required_keys:
            assert key in summary
        
        # Check values
        assert summary['total_alerts'] >= 0
        assert summary['period_days'] == 30
        assert summary['alert_frequency'] >= 0
        assert isinstance(summary['alert_types'], dict)
    
    def test_custom_alert_thresholds(self):
        """Test monitor with custom alert thresholds."""
        custom_thresholds = {
            'psi_threshold': 0.1,  # Lower threshold
            'ks_threshold': 0.05   # Lower threshold
        }
        
        custom_monitor = ModelMonitor(alert_thresholds=custom_thresholds)
        
        # Should trigger more alerts with lower thresholds
        psi_results = custom_monitor.calculate_population_stability_index(
            self.reference_scores, self.current_scores
        )
        
        # Check that custom thresholds are used
        assert custom_monitor.alert_thresholds['psi_threshold'] == 0.1
        assert custom_monitor.alert_thresholds['ks_threshold'] == 0.05
    
    def test_monitoring_with_identical_data(self):
        """Test monitoring with identical reference and current data."""
        # Use same data for both reference and current
        identical_scores = self.reference_scores.copy()
        
        psi_results = self.monitor.calculate_population_stability_index(
            self.reference_scores, identical_scores
        )
        
        # PSI should be very close to 0 for identical distributions
        assert psi_results['psi'] < 0.01
        assert psi_results['interpretation'] == "No significant change"
        
        ks_results = self.monitor.calculate_ks_statistic(
            self.reference_scores, identical_scores
        )
        
        # KS statistic should be very close to 0 for identical distributions
        assert ks_results['ks_statistic'] < 0.01
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very small datasets
        small_ref = np.array([0.1, 0.2, 0.3])
        small_cur = np.array([0.15, 0.25, 0.35])
        
        # Should handle small datasets gracefully
        psi_results = self.monitor.calculate_population_stability_index(
            small_ref, small_cur
        )
        assert isinstance(psi_results['psi'], float)
        
        # Test with constant values
        constant_ref = np.full(100, 0.5)
        constant_cur = np.full(100, 0.5)
        
        psi_results = self.monitor.calculate_population_stability_index(
            constant_ref, constant_cur
        )
        
        # Should handle constant distributions
        assert 'psi' in psi_results or 'warning' in psi_results
    
    def test_monitoring_workflow_integration(self):
        """Test complete monitoring workflow."""
        # Step 1: Calculate multiple metrics
        psi_results = self.monitor.calculate_population_stability_index(
            self.reference_scores, self.current_scores
        )
        
        ks_results = self.monitor.calculate_ks_statistic(
            self.reference_scores, self.current_scores
        )
        
        y_pred_proba = np.clip(self.current_scores, 0, 1)
        perf_metrics = self.monitor.calculate_model_performance_metrics(
            self.current_targets, y_pred_proba
        )
        
        cal_metrics = self.monitor.calculate_calibration_error(
            self.current_targets, y_pred_proba
        )
        
        # Step 2: Detect drift
        reference_data = {'scores': self.reference_scores, 'targets': self.reference_targets}
        current_data = {'scores': self.current_scores, 'targets': self.current_targets}
        
        drift_results = self.monitor.detect_concept_drift(reference_data, current_data)
        
        # Step 3: Generate comprehensive report
        monitoring_data = {
            'psi': psi_results,
            'ks': ks_results,
            'performance': perf_metrics,
            'calibration': cal_metrics,
            'drift': drift_results
        }
        
        report = self.monitor.generate_monitoring_report('IntegratedModel', monitoring_data)
        
        # Verify complete workflow
        assert 'psi' in monitoring_data
        assert 'drift' in monitoring_data
        assert len(report['detailed_results']) == 5
        assert isinstance(report['executive_summary']['total_alerts'], int)
        
        # Step 4: Get alert summary
        alert_summary = self.monitor.get_alert_summary()
        assert isinstance(alert_summary['total_alerts'], int)
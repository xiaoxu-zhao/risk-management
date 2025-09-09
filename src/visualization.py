"""
Risk Visualization Module
========================

Comprehensive visualization tools for credit risk analysis including:
- Model performance plots (ROC, Precision-Recall, Calibration)
- Risk distribution plots
- Portfolio analysis charts
- Feature importance visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RiskVisualizer:
    """
    Comprehensive visualization toolkit for credit risk analysis.
    """
    
    def __init__(self, style: str = 'default'):
        """
        Initialize the visualizer with preferred style.
        
        Args:
            style: Matplotlib style ('seaborn-v0_8', 'ggplot', 'default')
        """
        try:
            if style == 'seaborn':
                plt.style.use('seaborn-v0_8')
            else:
                plt.style.use(style)
        except OSError:
            # Fallback to default if style not available
            plt.style.use('default')
        
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 8)
        
    def plot_roc_curves(self, model_results: Dict[str, Dict], 
                       figsize: Tuple[int, int] = (10, 8),
                       save_path: str = None) -> plt.Figure:
        """
        Plot ROC curves for multiple models.
        
        Args:
            model_results: Dictionary containing model evaluation results
            figsize: Figure size
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (model_name, results) in enumerate(model_results.items()):
            if 'roc_curve' in results:
                fpr = results['roc_curve']['fpr']
                tpr = results['roc_curve']['tpr']
                auc_score = results['auc_score']
                
                ax.plot(fpr, tpr, color=self.colors[i % len(self.colors)], 
                       linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_precision_recall_curves(self, model_results: Dict[str, Dict],
                                    figsize: Tuple[int, int] = (10, 8),
                                    save_path: str = None) -> plt.Figure:
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            model_results: Dictionary containing model evaluation results
            figsize: Figure size
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (model_name, results) in enumerate(model_results.items()):
            if 'pr_curve' in results:
                precision = results['pr_curve']['precision']
                recall = results['pr_curve']['recall']
                avg_precision = results['average_precision']
                
                ax.plot(recall, precision, color=self.colors[i % len(self.colors)], 
                       linewidth=2, label=f'{model_name} (AP = {avg_precision:.3f})')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_calibration_curves(self, model_results: Dict[str, Dict],
                              figsize: Tuple[int, int] = (10, 8),
                              save_path: str = None) -> plt.Figure:
        """
        Plot calibration curves for multiple models.
        
        Args:
            model_results: Dictionary containing model evaluation results
            figsize: Figure size
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (model_name, results) in enumerate(model_results.items()):
            if 'calibration' in results:
                fraction_pos = results['calibration']['fraction_of_positives']
                mean_pred = results['calibration']['mean_predicted_value']
                
                ax.plot(mean_pred, fraction_pos, 'o-', 
                       color=self.colors[i % len(self.colors)],
                       linewidth=2, label=model_name)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Model Calibration Curves', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_feature_importance(self, feature_importance: Dict[str, float],
                              top_k: int = 20,
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: str = None) -> plt.Figure:
        """
        Plot feature importance as horizontal bar chart.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            top_k: Number of top features to display
            figsize: Figure size
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:top_k]
        
        features, importances = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, color=self.colors[0], alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Top feature at the top
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {top_k} Feature Importances', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_default_rate_distribution(self, df: pd.DataFrame,
                                     target_col: str,
                                     feature_col: str,
                                     bins: int = 20,
                                     figsize: Tuple[int, int] = (12, 6),
                                     save_path: str = None) -> plt.Figure:
        """
        Plot default rate distribution across feature bins.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            feature_col: Feature column name
            bins: Number of bins
            figsize: Figure size
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Create bins
        df_copy = df.copy()
        df_copy['feature_bin'] = pd.cut(df_copy[feature_col], bins=bins)
        
        # Calculate default rates by bin
        bin_stats = df_copy.groupby('feature_bin').agg({
            target_col: ['count', 'sum', 'mean']
        }).round(3)
        
        bin_stats.columns = ['count', 'defaults', 'default_rate']
        bin_stats = bin_stats.reset_index()
        
        # Plot 1: Default rate by bin
        x_pos = range(len(bin_stats))
        ax1.bar(x_pos, bin_stats['default_rate'], color=self.colors[1], alpha=0.7)
        ax1.set_xlabel(f'{feature_col} Bins', fontsize=12)
        ax1.set_ylabel('Default Rate', fontsize=12)
        ax1.set_title(f'Default Rate by {feature_col}', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'Bin {i+1}' for i in x_pos], rotation=45)
        
        # Plot 2: Volume by bin
        ax2.bar(x_pos, bin_stats['count'], color=self.colors[2], alpha=0.7)
        ax2.set_xlabel(f'{feature_col} Bins', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title(f'Volume by {feature_col}', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'Bin {i+1}' for i in x_pos], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_correlation_heatmap(self, df: pd.DataFrame,
                               figsize: Tuple[int, int] = (12, 10),
                               save_path: str = None) -> plt.Figure:
        """
        Plot correlation heatmap of features.
        
        Args:
            df: Input DataFrame
            figsize: Figure size
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Calculate correlation matrix
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .5},
                   fmt='.2f', ax=ax)
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_loss_distribution(self, losses: np.ndarray,
                             var_level: float = None,
                             expected_shortfall: float = None,
                             figsize: Tuple[int, int] = (12, 6),
                             save_path: str = None) -> plt.Figure:
        """
        Plot loss distribution with VaR and Expected Shortfall markers.
        
        Args:
            losses: Array of loss values
            var_level: Value at Risk level (optional)
            expected_shortfall: Expected Shortfall level (optional)
            figsize: Figure size
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        n, bins, patches = ax.hist(losses, bins=50, density=True, 
                                  alpha=0.7, color=self.colors[0], 
                                  edgecolor='black', linewidth=0.5)
        
        # Add VaR line
        if var_level is not None:
            ax.axvline(var_level, color='red', linestyle='--', linewidth=2,
                      label=f'VaR: {var_level:,.0f}')
        
        # Add Expected Shortfall line
        if expected_shortfall is not None:
            ax.axvline(expected_shortfall, color='orange', linestyle='--', linewidth=2,
                      label=f'Expected Shortfall: {expected_shortfall:,.0f}')
        
        # Add mean line
        mean_loss = np.mean(losses)
        ax.axvline(mean_loss, color='green', linestyle='-', linewidth=2,
                  label=f'Expected Loss: {mean_loss:,.0f}')
        
        ax.set_xlabel('Loss Amount', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Portfolio Loss Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_interactive_risk_dashboard(self, model_results: Dict[str, Dict],
                                        portfolio_metrics: Dict = None) -> go.Figure:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            model_results: Dictionary containing model evaluation results
            portfolio_metrics: Portfolio risk metrics (optional)
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['ROC Curves', 'Precision-Recall Curves', 
                          'Feature Importance', 'Model Comparison'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # ROC Curves
        for model_name, results in model_results.items():
            if 'roc_curve' in results:
                fpr = results['roc_curve']['fpr']
                tpr = results['roc_curve']['tpr']
                auc_score = results['auc_score']
                
                fig.add_trace(
                    go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{model_name} (AUC = {auc_score:.3f})',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
        
        # Add diagonal for ROC
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(dash='dash', color='black'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Precision-Recall Curves
        for model_name, results in model_results.items():
            if 'pr_curve' in results:
                precision = results['pr_curve']['precision']
                recall = results['pr_curve']['recall']
                avg_precision = results['average_precision']
                
                fig.add_trace(
                    go.Scatter(
                        x=recall, y=precision,
                        mode='lines',
                        name=f'{model_name} (AP = {avg_precision:.3f})',
                        line=dict(width=2),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # Model Comparison Bar Chart
        models = list(model_results.keys())
        auc_scores = [results['auc_score'] for results in model_results.values()]
        
        fig.add_trace(
            go.Bar(
                x=models, y=auc_scores,
                name='AUC Scores',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Credit Risk Model Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        
        fig.update_xaxes(title_text="Model", row=2, col=2)
        fig.update_yaxes(title_text="AUC Score", row=2, col=2)
        
        return fig
        
    def plot_portfolio_concentration(self, exposures: np.ndarray,
                                   labels: List[str] = None,
                                   title: str = "Portfolio Concentration",
                                   figsize: Tuple[int, int] = (12, 8),
                                   save_path: str = None) -> plt.Figure:
        """
        Plot portfolio concentration using pie chart and Lorenz curve.
        
        Args:
            exposures: Array of exposure amounts
            labels: Labels for exposures (optional)
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Sort exposures for Lorenz curve
        sorted_exposures = np.sort(exposures)
        cumsum_exposures = np.cumsum(sorted_exposures)
        total_exposure = np.sum(exposures)
        
        # Plot 1: Top exposures pie chart
        top_n = min(10, len(exposures))
        top_indices = np.argsort(exposures)[-top_n:]
        top_exposures = exposures[top_indices]
        top_labels = [f'Asset {i+1}' for i in top_indices] if labels is None else [labels[i] for i in top_indices]
        
        # Add "Others" category
        others_exposure = total_exposure - np.sum(top_exposures)
        if others_exposure > 0:
            top_exposures = np.append(top_exposures, others_exposure)
            top_labels.append('Others')
        
        ax1.pie(top_exposures, labels=top_labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Top {top_n} Exposures')
        
        # Plot 2: Lorenz curve
        n_exposures = len(exposures)
        x_lorenz = np.arange(1, n_exposures + 1) / n_exposures
        y_lorenz = cumsum_exposures / total_exposure
        
        ax2.plot(x_lorenz, y_lorenz, 'b-', linewidth=2, label='Lorenz Curve')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality')
        ax2.set_xlabel('Cumulative Share of Exposures')
        ax2.set_ylabel('Cumulative Share of Value')
        ax2.set_title('Concentration Lorenz Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_model_performance_comparison(self, comparison_df: pd.DataFrame,
                                        figsize: Tuple[int, int] = (14, 8),
                                        save_path: str = None) -> plt.Figure:
        """
        Plot comprehensive model performance comparison.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            figsize: Figure size
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Define metrics to plot
        metrics = ['AUC Score', 'Average Precision', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                bars = axes[i].bar(comparison_df['Model'], comparison_df[metric], 
                                 color=self.colors[i % len(self.colors)], alpha=0.7)
                axes[i].set_title(metric, fontweight='bold')
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
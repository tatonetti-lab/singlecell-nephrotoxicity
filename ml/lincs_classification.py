"""
LINCS machine learning classification component - FIXED VERSION.

This component handles machine learning classification on LINCS differential expression data,
including model training, evaluation, and comprehensive reporting.

Key fixes:
1. Consistent ROC AUC key naming
2. Proper error handling for metric access
3. Safer metric extraction with fallbacks
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings

from core.base import BaseMLComponent, BaseVisualizationComponent, BaseDataProcessor
from utils.logging import ComponentLogger

warnings.filterwarnings('ignore')


class LincsMLClassifier(BaseMLComponent, BaseVisualizationComponent, BaseDataProcessor):
    """
    Machine learning classification component for LINCS differential expression data.

    This component:
    1. Loads processed differential expression data
    2. Prepares data for machine learning (feature selection, scaling)
    3. Trains multiple classification models
    4. Evaluates models with cross-validation
    5. Performs feature importance analysis
    6. Generates comprehensive reports and visualizations
    """

    def __init__(self, config: Dict[str, Any], component_name: str = 'ml_lincs'):
        super().__init__(config, component_name)

        # Get input directory
        self.input_dir = Path(config['output_base_dir']) / config['output_subdirs']['differential']

        # ML-specific configuration
        self.models_config = self.component_config.get('models', {})

        self.component_logger = ComponentLogger(component_name, verbose=self.verbose)

    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        # Try to load DE genes only first (better for ML if available)
        ml_file_de = self.input_dir / 'direct_analysis_ml_ready_de_genes_only.csv'
        ml_file_all = self.input_dir / 'direct_analysis_ml_ready_all_genes.csv'

        if not ml_file_de.exists() and not ml_file_all.exists():
            self.logger.error(f"No ML-ready files found in {self.input_dir}")
            return False

        return True

    def run(self) -> Dict[str, Any]:
        """Run LINCS machine learning classification pipeline."""
        self.component_logger.start_component(total_steps=11)

        try:
            # Step 1: Load differential expression results
            self.component_logger.step_completed("Loading differential expression results")
            ml_data, de_results, data_type = self._load_de_results()

            # Step 2: Prepare data for ML
            self.component_logger.step_completed("Preparing data for ML")
            X, y, y_encoded, label_encoder = self._prepare_ml_data(ml_data)

            # Step 3: Train-test split
            self.component_logger.step_completed("Splitting data")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_encoded
            )

            # Step 4: Scale features
            self.component_logger.step_completed("Scaling features")
            scaler, X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)

            # Step 5: Initialize models
            self.component_logger.step_completed("Initializing models")
            models = self._initialize_models()

            # Step 6: Cross-validation evaluation
            self.component_logger.step_completed("Performing cross-validation")
            cv_results = self._evaluate_models_cv(models, X_train_scaled, y_train)

            # Step 7: Train and evaluate on test set
            self.component_logger.step_completed("Training and evaluating on test set")
            model_results = self._train_and_evaluate_models(models, X_train_scaled, X_test_scaled, y_train, y_test)

            self.component_logger.finish_component(success=True)

            return {
                'model_results': model_results,
                'cv_results': cv_results,
                'best_model': self._get_best_model(model_results),
            }

        except Exception as e:
            self.component_logger.finish_component(success=False)
            self.logger.error(f"LINCS ML classification failed: {e}")
            raise

    def _load_de_results(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str]:
        """Load differential expression results."""
        self.logger.info("Loading differential expression results...")

        # Try to load DE genes only first (better for ML if available)
        try:
            ml_data = pd.read_csv(self.input_dir / 'direct_analysis_ml_ready_de_genes_only.csv', index_col=0)
            data_type = "DE genes only"
        except FileNotFoundError:
            # Fall back to all genes
            ml_data = pd.read_csv(self.input_dir / 'direct_analysis_ml_ready_all_genes.csv', index_col=0)
            data_type = "all genes"

        # Load additional data for reference
        try:
            de_results = pd.read_csv(self.input_dir / 'direct_analysis_differential_expression_results.csv', index_col=0)
        except FileNotFoundError:
            de_results = None

        self.logger.info(f"Loaded ML dataset ({data_type}): {ml_data.shape}")
        if de_results is not None:
            self.logger.info(f"DE results reference: {de_results.shape}")

        return ml_data, de_results, data_type

    def _prepare_ml_data(self, ml_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, LabelEncoder]:
        """Prepare data for machine learning."""
        self.logger.info("Preparing data for machine learning...")

        # Separate features and target
        X = ml_data.drop('toxicity_label', axis=1)
        y = ml_data['toxicity_label']

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        self.logger.info(f"Dataset shape: {X.shape}")
        self.logger.info(f"Feature columns: {X.shape[1]}")
        self.logger.info(f"Class distribution: {pd.Series(y).value_counts()}")

        # Check for any missing values
        missing_vals = X.isnull().sum().sum()
        if missing_vals > 0:
            self.logger.warning(f"Found {missing_vals} missing values. Filling with median...")
            X = X.fillna(X.median())

        # Remove constant features
        constant_features = X.columns[X.var() == 0]
        if len(constant_features) > 0:
            self.logger.info(f"Removing {len(constant_features)} constant features")
            X = X.drop(constant_features, axis=1)

        return X, y, y_encoded, label_encoder

    def _scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[StandardScaler, pd.DataFrame, pd.DataFrame]:
        """Scale features using StandardScaler."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        self.logger.info(f"Training set: {X_train_scaled.shape}")
        self.logger.info(f"Test set: {X_test_scaled.shape}")

        return scaler, X_train_scaled, X_test_scaled

    def _initialize_models(self) -> Dict:
        """Initialize machine learning models with optimal hyperparameters."""
        self.logger.info("Initializing machine learning models...")

        models = {}

        if 'logistic' in self.models_config:
            models['Logistic_Regression'] = LogisticRegression(
                **self.models_config['logistic'],
                random_state=self.random_state
            )

        if 'random_forest' in self.models_config:
            models['Random_Forest'] = RandomForestClassifier(
                **self.models_config['random_forest'],
                random_state=self.random_state
            )

        if 'svc' in self.models_config:
            models['SVM'] = SVC(
                **self.models_config['svc'],
                random_state=self.random_state
            )

        if 'xgboost' in self.models_config:
            models['XGBoost'] = xgb.XGBClassifier(
                **self.models_config['xgboost'],
                random_state=self.random_state
            )

        self.logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
        return models

    def _evaluate_models_cv(self, models: Dict, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Evaluate models using cross-validation."""
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_results = {}

        for model_name, model in models.items():
            self.logger.info(f"Cross-validating {model_name}...")

            # Calculate cross-validation scores
            cv_scores = {
                'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
                'balanced_accuracy': cross_val_score(model, X, y, cv=cv, scoring='balanced_accuracy'),
                'f1': cross_val_score(model, X, y, cv=cv, scoring='f1'),
                'precision': cross_val_score(model, X, y, cv=cv, scoring='precision'),
                'recall': cross_val_score(model, X, y, cv=cv, scoring='recall'),
                'roc_auc': cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            }

            # Calculate mean and std for each metric
            cv_summary = {}
            for metric, scores in cv_scores.items():
                cv_summary[f'{metric}_mean'] = np.mean(scores)
                cv_summary[f'{metric}_std'] = np.std(scores)

            cv_results[model_name] = cv_summary

        return cv_results

    def _train_and_evaluate_models(self, models: Dict, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                 y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train and evaluate all models."""
        model_results = {}

        for model_name, model in models.items():
            self.logger.info(f"Training {model_name}...")

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

            # Calculate metrics using our own method to ensure consistency
            metrics = self._calculate_metrics(y_test, y_pred, y_prob)

            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_[0])

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)

            # Precision-Recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

            # Store results
            model_results[model_name] = {
                'model': model,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'roc_data': (fpr, tpr),
                'pr_data': (precision_curve, recall_curve),
                'predictions': {'y_pred': y_pred, 'y_prob': y_prob}
            }

            # Print results
            self.logger.info(f"Results for {model_name}:")
            for metric, value in metrics.items():
                self.logger.info(f"  {metric}: {value:.3f}")

        return model_results

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics with consistent key names."""
        metrics = {}

        try:
            # Basic classification metrics
            metrics['accuracy'] = balanced_accuracy_score(y_true, y_pred)
            metrics['f1_score'] = f1_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)

            # ROC AUC - using consistent key name
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)

        except Exception as e:
            self.logger.warning(f"Error calculating some metrics: {e}")
            # Provide fallback values
            metrics.setdefault('accuracy', 0.0)
            metrics.setdefault('f1_score', 0.0)
            metrics.setdefault('precision', 0.0)
            metrics.setdefault('recall', 0.0)
            metrics.setdefault('roc_auc', 0.0)

        return metrics

    def _create_model_visualizations(self, model_results: Dict, y_test: np.ndarray) -> None:
        """Create comprehensive model evaluation visualizations."""
        self.logger.info("Creating model evaluation visualizations...")

        # ROC Curves and other plots
        n_models = len(model_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. ROC Curves
        ax1 = axes[0, 0]
        for model_name, results in model_results.items():
            fpr, tpr = results['roc_data']
            auc_score = results['metrics'].get('roc_auc', 0.0)  # Safe access with fallback
            ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Precision-Recall Curves
        ax2 = axes[0, 1]
        for model_name, results in model_results.items():
            precision, recall = results['pr_data']
            # Calculate average precision properly
            y_pred_prob = results['predictions']['y_prob']
            avg_precision = average_precision_score(y_test, y_pred_prob)
            ax2.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})', linewidth=2)

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Metrics comparison
        ax3 = axes[1, 0]
        metrics_df = pd.DataFrame({name: results['metrics'] for name, results in model_results.items()}).T
        metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
        available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
        metrics_df[available_metrics].plot(kind='bar', ax=ax3)
        ax3.set_title('Model Performance Comparison')
        ax3.set_ylabel('Score')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)

        # 4. Confusion matrix for best model
        best_model = self._get_best_model(model_results)
        ax4 = axes[1, 1]
        y_pred = model_results[best_model]['predictions']['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, ax=ax4)
        ax4.set_title(f'Confusion Matrix - {best_model}')
        ax4.set_ylabel('True Label')
        ax4.set_xlabel('Predicted Label')

        plt.tight_layout()
        self.save_figure(fig, 'model_evaluation_plots')
        self.close_figure(fig)

    def _get_data_info(self, data_type: str, ml_data: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Get data information for reporting."""
        return {
            'data_type': data_type,
            'n_samples': len(ml_data),
            'n_features_original': len(X.columns),
            'n_toxic': sum(y == 'Toxic'),
            'n_nontoxic': sum(y == 'Non-toxic')
        }

    def _get_best_model(self, model_results: Dict) -> str:
        """Get the name of the best performing model based on ROC AUC."""
        if not model_results:
            return "No models"

        return max(model_results.keys(),
                  key=lambda x: model_results[x]['metrics'].get('roc_auc', 0.0))

    def _get_best_roc_auc(self, model_results: Dict) -> float:
        """Safely get the best ROC AUC score from model results."""
        if not model_results:
            return 0.0

        roc_aucs = []
        for results in model_results.values():
            roc_auc = results['metrics'].get('roc_auc', 0.0)
            roc_aucs.append(roc_auc)

        return max(roc_aucs) if roc_aucs else 0.0

    def _get_best_f1(self, model_results: Dict) -> float:
        """Safely get the best F1 score from model results."""
        if not model_results:
            return 0.0

        f1_scores = []
        for results in model_results.values():
            f1_score = results['metrics'].get('f1_score', 0.0)
            f1_scores.append(f1_score)

        return max(f1_scores) if f1_scores else 0.0
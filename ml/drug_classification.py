"""
Drug toxicity machine learning classification component.

This component performs machine learning analysis on drug scores to predict
nephrotoxicity using multiple algorithms and ensemble methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, Tuple, List, Optional, Any

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, auc, classification_report, confusion_matrix
)
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb

from core.base import BaseMLComponent, BaseVisualizationAnalyzer
from utils.logging import ComponentLogger

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
pd.options.mode.chained_assignment = None


class DrugToxicityClassifier(BaseVisualizationAnalyzer):
    """
    A comprehensive machine learning classifier for drug nephrotoxicity prediction.
    
    This component:
    1. Loads drug score data from drug2cell and statistical analysis
    2. Preprocesses data for machine learning
    3. Trains multiple classification models with balanced datasets
    4. Evaluates models using cross-validation and test sets
    5. Creates ensemble models
    6. Generates comprehensive visualizations and reports
    """
    
    def __init__(self, config: Dict[str, Any], component_name: str = 'ml_drug'):
        super().__init__(config, component_name)
        
        # Add ML-specific configuration (from BaseMLComponent)
        self.test_size = self.component_config.get('test_size', 0.2)
        self.cv_folds = self.component_config.get('cv_folds', 5)
        self.random_state = self.component_config.get('random_state', 42)
        self.verbose = self.component_config.get('verbose', True)
        
        # Get input file
        self.input_file = Path(config['output_base_dir']) / 'drug2cell_results' / 'combined_drug_matrix.csv'
        
        # ML-specific configuration
        self.models_config = self.component_config.get('models', {})
        
        # Initialize model storage
        self.models = {}
        self.ensemble_model = None
        self.scaler = RobustScaler()
        self.balancer = RandomOverSampler(random_state=self.random_state)
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.feature_names = None
        
        self.component_logger = ComponentLogger(component_name, verbose=self.verbose)
    
    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        if not self.input_file.exists():
            self.logger.error(f"Combined drug matrix not found: {self.input_file}")
            return False
        
        return True
    
    def run(self) -> Dict[str, Any]:
        """Run machine learning classification pipeline."""
        self.component_logger.start_component(total_steps=8)
        
        try:
            # Step 1: Load and preprocess data
            self.component_logger.step_completed("Loading and preprocessing data")
            combined_matrix = pd.read_csv(self.input_file)
            X, Y = self._preprocess_data(combined_matrix)
            
            # Step 2: Split data
            self.component_logger.step_completed("Splitting data")
            self._split_data(X, Y)
            
            # Step 3: Train individual models
            self.component_logger.step_completed("Training individual models")
            self._train_individual_models()
            
            # Step 4: Train ensemble model
            self.component_logger.step_completed("Training ensemble model")
            self._train_ensemble_model()
            
            # Step 5: Evaluate models with cross-validation
            self.component_logger.step_completed("Performing cross-validation")
            cv_results = self._evaluate_all_models_cv()
            
            # Step 6: Evaluate on test set
            self.component_logger.step_completed("Evaluating on test set")
            test_results = self._evaluate_all_models_test()
            
            # Step 7: Create visualizations
            self.component_logger.step_completed("Creating visualizations")
            self._create_visualizations(test_results, cv_results)
            
            # Step 8: Save results and generate report
            self.component_logger.step_completed("Saving results and generating report")
            saved_files = self._save_results(test_results, cv_results)
            summary_stats = self._generate_summary_stats(test_results, cv_results)
            
            self.component_logger.finish_component(success=True)
            
            return {
                'test_results': test_results,
                'cv_results': cv_results,
                'models': self.models,
                'ensemble_model': self.ensemble_model,
                'summary_stats': summary_stats,
                'saved_files': saved_files,
                'best_model': self._get_best_model_name(test_results),
                'best_f1_score': max([results['F1_Score'] for results in test_results.values()])
            }
            
        except Exception as e:
            self.component_logger.finish_component(success=False)
            self.logger.error(f"Drug toxicity classification failed: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the combined drug matrix for ML analysis."""
        self.logger.info("Preprocessing data for ML analysis...")
        
        # Pivot data to create feature matrix
        unique_cell_types = data['cell_type'].unique()
        joined_data = pd.DataFrame()
        
        for cell_type in unique_cell_types:
            cell_type_data = data[data['cell_type'] == cell_type].copy()
            
            # Rename score columns with cell type prefix
            if 'drug_score' in cell_type_data.columns:
                cell_type_data = cell_type_data.rename(columns={
                    'drug_score': f'{cell_type}_drug_score',
                    'log_drug_score': f'{cell_type}_log_drug_score'
                })
            
            if joined_data.empty:
                joined_data = cell_type_data.drop(columns=['cell_type'])
            else:
                cell_type_data = cell_type_data.drop(columns=['group', 'cell_type'])
                joined_data = pd.merge(joined_data, cell_type_data, on='drug_id', how='inner')
        
        # Prepare features and target
        X = joined_data.drop(columns=['drug_id', 'group'])
        
        # Use only drug_score columns (not log_drug_score)
        drug_score_cols = [col for col in X.columns if 'drug_score' in col and 'log_drug_score' not in col]
        X = X[drug_score_cols]
        
        # Create binary target
        Y = joined_data['group'].apply(lambda x: 1 if x == 'nephrotoxic' else 0)
        
        # Handle infinite and NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        self.feature_names = X.columns.tolist()
        
        self.logger.info(f"Preprocessed data shape: {X.shape}")
        self.logger.info(f"Features: {len(X.columns)}")
        self.logger.info(f"Target distribution: {Y.value_counts().to_dict()}")
        
        return X, Y
    
    def _split_data(self, X: pd.DataFrame, Y: pd.Series) -> None:
        """Split data into training and testing sets."""
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state, stratify=Y
        )
        
        self.logger.info(f"Data split - Train: {len(self.X_train)}, Test: {len(self.X_test)}")
    
    def _train_individual_models(self) -> None:
        """Train individual machine learning models."""
        self.logger.info("Training individual models...")
        
        # Logistic Regression
        if 'logistic' in self.models_config:
            self.logger.info("Training Logistic Regression...")
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('balancer', self.balancer),
                ('classifier', LogisticRegression(
                    **self.models_config['logistic'],
                    random_state=self.random_state
                ))
            ])
            pipeline.fit(self.X_train, self.Y_train)
            self.models['logistic'] = pipeline
        
        # XGBoost
        if 'xgboost' in self.models_config:
            self.logger.info("Training XGBoost...")
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('balancer', self.balancer),
                ('classifier', xgb.XGBClassifier(
                    **self.models_config['xgboost'],
                    random_state=self.random_state
                ))
            ])
            pipeline.fit(self.X_train, self.Y_train)
            self.models['xgboost'] = pipeline
        
        # Random Forest
        if 'random_forest' in self.models_config:
            self.logger.info("Training Random Forest...")
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('balancer', self.balancer),
                ('classifier', RandomForestClassifier(
                    **self.models_config['random_forest'],
                    random_state=self.random_state
                ))
            ])
            pipeline.fit(self.X_train, self.Y_train)
            self.models['random_forest'] = pipeline
        
        # Support Vector Classifier
        if 'svc' in self.models_config:
            self.logger.info("Training SVC...")
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('balancer', self.balancer),
                ('classifier', SVC(
                    **self.models_config['svc'],
                    random_state=self.random_state
                ))
            ])
            pipeline.fit(self.X_train, self.Y_train)
            self.models['svc'] = pipeline
        
        self.logger.info(f"Trained {len(self.models)} individual models")
    
    def _train_ensemble_model(self) -> None:
        """Train ensemble model using individual model predictions."""
        self.logger.info("Training ensemble model...")
        
        if len(self.models) < 2:
            self.logger.warning("Need at least 2 models for ensemble. Skipping ensemble training.")
            return
        
        # Get probability predictions from individual models
        probas = np.column_stack([
            model.predict_proba(self.X_train)[:, 1] 
            for model in self.models.values()
        ])
        
        # Combine original features with model probabilities
        X_ensemble = np.column_stack([self.X_train, probas])
        
        # Train ensemble with logistic regression
        self.ensemble_model = Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', LogisticRegression(max_iter=2000, random_state=self.random_state))
        ])
        self.ensemble_model.fit(X_ensemble, self.Y_train)
        
        self.logger.info("Ensemble model trained")
    
    def _predict(self, X: pd.DataFrame, model: str = 'ensemble') -> np.ndarray:
        """Make predictions using specified model."""
        if model == 'ensemble' and self.ensemble_model is not None:
            probas = np.column_stack([
                m.predict_proba(X)[:, 1] for m in self.models.values()
            ])
            X_ensemble = np.column_stack([X, probas])
            return self.ensemble_model.predict(X_ensemble)
        elif model in self.models:
            return self.models[model].predict(X)
        else:
            raise ValueError(f"Model '{model}' not available")
    
    def _predict_proba(self, X: pd.DataFrame, model: str = 'ensemble') -> np.ndarray:
        """Get prediction probabilities."""
        if model == 'ensemble' and self.ensemble_model is not None:
            probas = np.column_stack([
                m.predict_proba(X)[:, 1] for m in self.models.values()
            ])
            X_ensemble = np.column_stack([X, probas])
            return self.ensemble_model.predict_proba(X_ensemble)[:, 1]
        elif model in self.models:
            return self.models[model].predict_proba(X)[:, 1]
        else:
            raise ValueError(f"Model '{model}' not available")
    
    def _evaluate_model_single(self, X: pd.DataFrame, Y: pd.Series, model: str = 'ensemble') -> Dict:
        """Evaluate model performance."""
        Y_pred = self._predict(X, model)
        Y_proba = self._predict_proba(X, model)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(Y, Y_proba)
        roc_auc = auc(fpr, tpr)
        
        return {
            'Accuracy': accuracy_score(Y, Y_pred),
            'Precision': precision_score(Y, Y_pred, zero_division=0),
            'Recall': recall_score(Y, Y_pred, zero_division=0),
            'F1_Score': f1_score(Y, Y_pred, zero_division=0),
            'AUROC': roc_auc,
            'FPR': fpr,
            'TPR': tpr,
            'Y_pred': Y_pred,
            'Y_proba': Y_proba
        }
    
    def _evaluate_all_models_test(self) -> Dict:
        """Evaluate all trained models on test set."""
        results = {}
        model_names = list(self.models.keys())
        if self.ensemble_model is not None:
            model_names.append('ensemble')
        
        for model_name in model_names:
            results[model_name] = self._evaluate_model_single(self.X_test, self.Y_test, model_name)
        
        return results
    
    def _evaluate_all_models_cv(self) -> Dict:
        """Evaluate all models using cross-validation."""
        cv_results = {}
        
        for model_name, model in self.models.items():
            scores = cross_val_score(model, self.X_train, self.Y_train, 
                                   cv=self.cv_folds, scoring='f1')
            cv_results[model_name] = {
                'mean_f1': scores.mean(),
                'std_f1': scores.std(),
                'scores': scores
            }
        
        # Cross-validation for ensemble
        if self.ensemble_model is not None:
            probas = np.column_stack([
                m.predict_proba(self.X_train)[:, 1] for m in self.models.values()
            ])
            X_ensemble = np.column_stack([self.X_train, probas])
            scores = cross_val_score(self.ensemble_model, X_ensemble, self.Y_train, 
                                   cv=self.cv_folds, scoring='f1')
            cv_results['ensemble'] = {
                'mean_f1': scores.mean(),
                'std_f1': scores.std(),
                'scores': scores
            }
        
        return cv_results
    
    def _create_visualizations(self, test_results: Dict, cv_results: Dict) -> None:
        """Create comprehensive visualizations."""
        self.logger.info("Creating visualizations...")
        
        # ROC Curves
        self._plot_roc_curves(test_results)
        
        # Confusion matrices
        self._create_confusion_matrices(test_results)
        
        # Model comparison
        self._plot_model_comparison(test_results, cv_results)
    
    def _plot_roc_curves(self, results: Dict) -> None:
        """Plot ROC curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, metrics in results.items():
            if 'FPR' in metrics and 'TPR' in metrics:
                ax.plot(metrics['FPR'], metrics['TPR'], 
                       linewidth=2, label=f'{model_name.title()} (AUROC = {metrics["AUROC"]:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - Drug Toxicity Classification')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'roc_curves')
        self.close_figure(fig)
    
    def _create_confusion_matrices(self, results: Dict) -> None:
        """Create confusion matrices for all models."""
        n_models = len(results)
        if n_models == 0:
            return
        
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, metrics) in enumerate(results.items()):
            row = idx // cols
            col = idx % cols
            
            if rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # Create confusion matrix
            cm = confusion_matrix(self.Y_test, metrics['Y_pred'])
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name.title()}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        self.save_figure(fig, 'confusion_matrices')
        self.close_figure(fig)
    
    def _plot_model_comparison(self, test_results: Dict, cv_results: Dict) -> None:
        """Plot model performance comparison."""
        # Test results comparison
        metrics_df = pd.DataFrame({name: {k: v for k, v in results.items() 
                                         if k not in ['FPR', 'TPR', 'Y_pred', 'Y_proba']} 
                                  for name, results in test_results.items()}).T
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Test set metrics
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUROC']
        metrics_df[metrics_to_plot].plot(kind='bar', ax=ax1)
        ax1.set_title('Test Set Performance')
        ax1.set_ylabel('Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Cross-validation F1 scores
        cv_means = [cv_results[model]['mean_f1'] for model in cv_results.keys()]
        cv_stds = [cv_results[model]['std_f1'] for model in cv_results.keys()]
        
        ax2.bar(range(len(cv_means)), cv_means, yerr=cv_stds, capsize=5)
        ax2.set_xticks(range(len(cv_means)))
        ax2.set_xticklabels(list(cv_results.keys()), rotation=45)
        ax2.set_title('Cross-Validation F1 Scores')
        ax2.set_ylabel('F1 Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'model_comparison')
        self.close_figure(fig)
    
    def _save_results(self, test_results: Dict, cv_results: Dict) -> Dict[str, Path]:
        """Save all machine learning results."""
        self.logger.info("Saving results...")
        
        saved_files = {}
        
        # Test set metrics
        test_metrics = {}
        for model, metrics in test_results.items():
            test_metrics[model] = {k: v for k, v in metrics.items() 
                                  if k not in ['FPR', 'TPR', 'Y_pred', 'Y_proba']}
        
        test_df = pd.DataFrame(test_metrics).T
        saved_files['test_results'] = self.save_data(test_df, 'test_results.csv')
        
        # Cross-validation results
        cv_df = pd.DataFrame({model: {'mean_f1': results['mean_f1'], 'std_f1': results['std_f1']} 
                             for model, results in cv_results.items()}).T
        saved_files['cv_results'] = self.save_data(cv_df, 'cross_validation_results.csv')
        
        # Predictions from best model
        best_model = self._get_best_model_name(test_results)
        predictions_df = pd.DataFrame({
            'true_label': self.Y_test,
            'predicted_label': test_results[best_model]['Y_pred'],
            'predicted_probability': test_results[best_model]['Y_proba']
        }, index=self.X_test.index)
        saved_files['predictions'] = self.save_data(predictions_df, 'best_model_predictions.csv')
        
        return saved_files
    
    def _generate_summary_stats(self, test_results: Dict, cv_results: Dict) -> Dict[str, Any]:
        """Generate summary statistics."""
        best_model = self._get_best_model_name(test_results)
        
        summary_stats = {
            'best_model': best_model,
            'best_test_f1': float(test_results[best_model]['F1_Score']),
            'best_test_auroc': float(test_results[best_model]['AUROC']),
            'n_models_trained': len(self.models),
            'ensemble_trained': self.ensemble_model is not None,
            'n_features': len(self.feature_names),
            'n_train_samples': len(self.X_train),
            'n_test_samples': len(self.X_test)
        }
        
        # Save summary statistics
        summary_path = self.output_dir / 'ml_summary.json'
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        return summary_stats
    
    def _get_best_model_name(self, test_results: Dict) -> str:
        """Get the name of the best performing model."""
        return max(test_results.keys(), key=lambda x: test_results[x]['F1_Score'])

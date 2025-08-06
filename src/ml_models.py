"""
Machine Learning Module for Wildfire Risk Prediction

This module demonstrates advanced machine learning capabilities including:
- Multiple predictive modeling algorithms
- Feature selection and engineering
- Model evaluation and validation
- Hyperparameter tuning
- Ensemble methods
- Model interpretation and explainability

Author: [Your Name]
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WildfireMLPredictor:
    """
    Advanced machine learning system for wildfire risk prediction.
    
    This class demonstrates sophisticated ML capabilities including:
    - Multiple predictive modeling algorithms
    - Advanced feature selection and engineering
    - Comprehensive model evaluation and validation
    - Hyperparameter optimization
    - Ensemble methods and model stacking
    - Model interpretation and explainability
    """
    
    def __init__(self, features_df: pd.DataFrame, target_series: pd.Series):
        """
        Initialize the ML predictor.
        
        Args:
            features_df (pd.DataFrame): Feature matrix
            target_series (pd.Series): Target variable
        """
        self.features_df = features_df.copy()
        self.target_series = target_series.copy()
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
        # Ensure data alignment
        common_index = features_df.index.intersection(target_series.index)
        self.features_df = features_df.loc[common_index]
        self.target_series = target_series.loc[common_index]
        
        logger.info(f"Initialized ML predictor with {len(self.features_df)} samples and {len(self.features_df.columns)} features")
    
    def prepare_data(self, test_size=0.2, random_state=42, handle_imbalance=True):
        """
        Prepare data for machine learning.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            handle_imbalance (bool): Whether to handle class imbalance
        """
        logger.info("Preparing data for machine learning...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features_df, self.target_series, 
            test_size=test_size, random_state=random_state, stratify=self.target_series
        )
        
        # Handle class imbalance if requested
        if handle_imbalance:
            logger.info("Handling class imbalance with SMOTE...")
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            logger.info(f"Original class distribution: {np.bincount(y_train)}")
            logger.info(f"Resampled class distribution: {np.bincount(y_train_resampled)}")
            
            self.X_train = X_train_resampled
            self.y_train = y_train_resampled
        else:
            self.X_train = X_train
            self.y_train = y_train
        
        self.X_test = X_test
        self.y_test = y_test
        
        # Store data info
        self.data_info = {
            'n_train': len(self.X_train),
            'n_test': len(self.X_test),
            'n_features': len(self.features_df.columns),
            'feature_names': list(self.features_df.columns),
            'class_distribution': {
                'train': np.bincount(self.y_train),
                'test': np.bincount(self.y_test)
            }
        }
        
        logger.info("Data preparation completed")
    
    def perform_feature_selection(self, method='mutual_info', n_features=10):
        """
        Perform feature selection to identify most important features.
        
        Args:
            method (str): Feature selection method
            n_features (int): Number of features to select
        """
        logger.info(f"Performing feature selection using {method}...")
        
        if method == 'mutual_info':
            # Mutual information-based feature selection
            selector = SelectKBest(score_func=f_classif, k=n_features)
            selector.fit(self.X_train, self.y_train)
            
            selected_features = self.X_train.columns[selector.get_support()]
            feature_scores = pd.DataFrame({
                'feature': self.X_train.columns,
                'score': selector.scores_
            }).sort_values('score', ascending=False)
            
        elif method == 'recursive':
            # Recursive feature elimination
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=n_features)
            selector.fit(self.X_train, self.y_train)
            
            selected_features = self.X_train.columns[selector.support_]
            feature_scores = pd.DataFrame({
                'feature': self.X_train.columns,
                'ranking': selector.ranking_
            }).sort_values('ranking')
        
        self.feature_selection_results = {
            'method': method,
            'selected_features': list(selected_features),
            'feature_scores': feature_scores,
            'n_selected': len(selected_features)
        }
        
        # Update training data with selected features
        self.X_train_selected = self.X_train[selected_features]
        self.X_test_selected = self.X_test[selected_features]
        
        logger.info(f"Feature selection completed. Selected {len(selected_features)} features")
        return self.feature_selection_results
    
    def train_models(self, use_selected_features=True):
        """
        Train multiple machine learning models.
        
        Args:
            use_selected_features (bool): Whether to use feature selection results
        """
        logger.info("Training multiple machine learning models...")
        
        # Choose features
        if use_selected_features and hasattr(self, 'X_train_selected'):
            X_train = self.X_train_selected
            X_test = self.X_test_selected
            feature_names = self.feature_selection_results['selected_features']
        else:
            X_train = self.X_train
            X_test = self.X_test
            feature_names = self.data_info['feature_names']
        
        # Define models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            # 'xgboost': xgb.XGBClassifier(random_state=42),  # Commented out due to OpenMP dependency
            'svm': SVC(probability=True, random_state=42)
        }
        
        # Train each model
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, self.y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Store model and results
            self.models[name] = pipeline
            self.results[name] = metrics
            
            # Extract feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            elif hasattr(model, 'coef_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(model.coef_[0])
                }).sort_values('importance', ascending=False)
        
        logger.info("Model training completed")
    
    def create_ensemble_model(self, models_to_ensemble=None):
        """
        Create an ensemble model combining multiple algorithms.
        
        Args:
            models_to_ensemble (list): List of model names to ensemble
        """
        logger.info("Creating ensemble model...")
        
        if models_to_ensemble is None:
            models_to_ensemble = ['random_forest', 'gradient_boosting']  # Removed xgboost due to OpenMP dependency
        
        # Get trained models
        estimators = [(name, self.models[name]) for name in models_to_ensemble if name in self.models]
        
        if len(estimators) < 2:
            logger.warning("Need at least 2 models for ensemble. Training individual models first.")
            self.train_models()
            estimators = [(name, self.models[name]) for name in models_to_ensemble if name in self.models]
        
        # Create voting classifier
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Choose features
        if hasattr(self, 'X_train_selected'):
            X_train = self.X_train_selected
            X_test = self.X_test_selected
        else:
            X_train = self.X_train
            X_test = self.X_test
        
        # Train ensemble
        ensemble.fit(X_train, self.y_train)
        
        # Make predictions
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        # Store ensemble
        self.models['ensemble'] = ensemble
        self.results['ensemble'] = metrics
        
        logger.info("Ensemble model created and trained")
    
    def hyperparameter_tuning(self, model_name='random_forest'):
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name (str): Name of the model to tune
        """
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            }
            # 'xgboost': {  # Commented out due to OpenMP dependency
            #     'classifier__n_estimators': [50, 100, 200],
            #     'classifier__learning_rate': [0.01, 0.1, 0.2],
            #     'classifier__max_depth': [3, 5, 7]
            # }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return
        
        # Choose features
        if hasattr(self, 'X_train_selected'):
            X_train = self.X_train_selected
        else:
            X_train = self.X_train
        
        # Create base model
        base_model = self.models[model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grids[model_name], 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, self.y_train)
        
        # Update model with best parameters
        self.models[f'{model_name}_tuned'] = grid_search.best_estimator_
        
        # Evaluate tuned model
        if hasattr(self, 'X_test_selected'):
            X_test = self.X_test_selected
        else:
            X_test = self.X_test
        
        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        self.results[f'{model_name}_tuned'] = metrics
        
        # Store tuning results
        self.tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Hyperparameter tuning completed. Best score: {grid_search.best_score_:.4f}")
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive model evaluation metrics."""
        return {
            'accuracy': (y_true == y_pred).mean(),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def evaluate_models(self):
        """
        Comprehensive model evaluation and comparison.
        
        Returns:
            dict: Evaluation results and rankings
        """
        logger.info("Evaluating all trained models...")
        
        # Create comparison dataframe
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'model': name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
        
        # Identify best model
        best_model = comparison_df.iloc[0]['model']
        
        # Generate evaluation report
        evaluation_report = {
            'model_comparison': comparison_df,
            'best_model': best_model,
            'best_metrics': self.results[best_model],
            'all_results': self.results
        }
        
        self.evaluation_results = evaluation_report
        
        logger.info(f"Model evaluation completed. Best model: {best_model}")
        return evaluation_report
    
    def generate_feature_importance_report(self):
        """Generate comprehensive feature importance analysis."""
        logger.info("Generating feature importance report...")
        
        importance_report = {}
        
        for model_name, importance_df in self.feature_importance.items():
            importance_report[model_name] = {
                'top_features': importance_df.head(10).to_dict('records'),
                'importance_summary': {
                    'mean_importance': importance_df['importance'].mean(),
                    'std_importance': importance_df['importance'].std(),
                    'max_importance': importance_df['importance'].max()
                }
            }
        
        return importance_report
    
    def create_prediction_pipeline(self, model_name=None):
        """
        Create a production-ready prediction pipeline.
        
        Args:
            model_name (str): Name of the model to use for predictions
        """
        if model_name is None:
            # Use best model from evaluation
            if hasattr(self, 'evaluation_results'):
                model_name = self.evaluation_results['best_model']
            else:
                model_name = 'random_forest'
        
        logger.info(f"Creating prediction pipeline with {model_name}...")
        
        # Get the trained model
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return None
        
        model = self.models[model_name]
        
        # Create prediction function
        def predict_wildfire_risk(features):
            """
            Predict wildfire risk for new data.
            
            Args:
                features (pd.DataFrame): Feature matrix for prediction
                
            Returns:
                tuple: (predictions, probabilities)
            """
            # Ensure correct features
            if hasattr(self, 'feature_selection_results'):
                features = features[self.feature_selection_results['selected_features']]
            
            # Make predictions
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]
            
            return predictions, probabilities
        
        self.prediction_pipeline = predict_wildfire_risk
        logger.info("Prediction pipeline created")
        
        return predict_wildfire_risk
    
    def generate_ml_report(self):
        """
        Generate comprehensive machine learning report.
        
        Returns:
            dict: Complete ML analysis report
        """
        logger.info("Generating comprehensive machine learning report...")
        
        # Ensure models are trained
        if not self.models:
            logger.info("Training models...")
            self.train_models()
        
        # Evaluate models
        evaluation = self.evaluate_models()
        
        # Generate feature importance report
        feature_importance = self.generate_feature_importance_report()
        
        # Create ensemble
        self.create_ensemble_model()
        
        # Final evaluation with ensemble
        final_evaluation = self.evaluate_models()
        
        # Create prediction pipeline
        self.create_prediction_pipeline()
        
        report = {
            'data_info': self.data_info,
            'feature_selection': getattr(self, 'feature_selection_results', {}),
            'model_evaluation': final_evaluation,
            'feature_importance': feature_importance,
            'recommendations': self._generate_ml_recommendations(final_evaluation)
        }
        
        return report
    
    def _generate_ml_recommendations(self, evaluation):
        """Generate recommendations based on ML results."""
        recommendations = []
        
        best_model = evaluation['best_model']
        best_metrics = evaluation['best_metrics']
        
        recommendations.append(f"Best performing model: {best_model}")
        recommendations.append(f"ROC AUC: {best_metrics['roc_auc']:.4f}")
        
        if best_metrics['roc_auc'] > 0.8:
            recommendations.append("Excellent model performance achieved")
        elif best_metrics['roc_auc'] > 0.7:
            recommendations.append("Good model performance - consider feature engineering")
        else:
            recommendations.append("Model performance needs improvement - consider more features or different algorithms")
        
        return recommendations

def main():
    """Example usage of the WildfireMLPredictor."""
    
    # Load processed data
    from data_processing import WildfireDataProcessor
    
    processor = WildfireDataProcessor()
    _, features_df, target_series, _ = processor.process_pipeline()
    
    # Initialize ML predictor
    predictor = WildfireMLPredictor(features_df, target_series)
    
    # Prepare data
    predictor.prepare_data()
    
    # Generate comprehensive report
    report = predictor.generate_ml_report()
    
    # Print summary
    print("\n=== Machine Learning Summary ===")
    print(f"Best model: {report['model_evaluation']['best_model']}")
    print(f"ROC AUC: {report['model_evaluation']['best_metrics']['roc_auc']:.4f}")
    print(f"Accuracy: {report['model_evaluation']['best_metrics']['accuracy']:.4f}")
    
    return report

if __name__ == "__main__":
    main() 
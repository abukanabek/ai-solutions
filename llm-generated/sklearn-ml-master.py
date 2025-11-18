"""
2_sklearn_ml_master.py
Ultimate Scikit-learn Machine Learning Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

# Import all major sklearn models
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.dummy import DummyClassifier, DummyRegressor

import warnings
warnings.filterwarnings('ignore')

class UltimateSklearnPipeline:
    """Comprehensive sklearn pipeline for all ML tasks"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        
    # ==================== CLASSIFICATION ====================
    
    def classification_pipeline(self, X, y, test_size=0.2, random_state=42):
        """Complete classification pipeline"""
        print("Starting Classification Pipeline...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Define models
        models = {
            'logistic_regression': LogisticRegression(random_state=random_state),
            'random_forest': RandomForestClassifier(random_state=random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=random_state),
            'svm': SVC(random_state=random_state, probability=True),
            'knn': KNeighborsClassifier(),
            'decision_tree': DecisionTreeClassifier(random_state=random_state),
            'naive_bayes': GaussianNB(),
            'mlp': MLPClassifier(random_state=random_state, max_iter=1000)
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Evaluate
            scores = self._evaluate_classification(y_test, y_pred, y_pred_proba)
            results[name] = {
                'model': pipeline,
                'scores': scores
            }
            
            print(f"{name} - Accuracy: {scores['accuracy']:.4f}, F1: {scores['f1']:.4f}")
        
        self.models['classification'] = results
        self.best_model = self._get_best_model(results, metric='f1')
        
        return results
    
    # ==================== REGRESSION ====================
    
    def regression_pipeline(self, X, y, test_size=0.2, random_state=42):
        """Complete regression pipeline"""
        print("Starting Regression Pipeline...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Define models
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=random_state),
            'lasso': Lasso(random_state=random_state),
            'random_forest': RandomForestRegressor(random_state=random_state),
            'gradient_boosting': GradientBoostingRegressor(random_state=random_state),
            'svr': SVR(),
            'knn_regressor': KNeighborsRegressor(),
            'decision_tree_reg': DecisionTreeRegressor(random_state=random_state),
            'mlp_regressor': MLPRegressor(random_state=random_state, max_iter=1000)
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            
            # Evaluate
            scores = self._evaluate_regression(y_test, y_pred)
            results[name] = {
                'model': pipeline,
                'scores': scores
            }
            
            print(f"{name} - R²: {scores['r2']:.4f}, RMSE: {scores['rmse']:.4f}")
        
        self.models['regression'] = results
        self.best_model = self._get_best_model(results, metric='r2')
        
        return results
    
    # ==================== CLUSTERING ====================
    
    def clustering_pipeline(self, X, n_clusters=3, methods=None):
        """Comprehensive clustering pipeline"""
        print("Starting Clustering Pipeline...")
        
        if methods is None:
            methods = ['kmeans', 'dbscan', 'hierarchical']
        
        # Preprocess data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        if 'kmeans' in methods:
            # KMeans with elbow method
            inertias = []
            k_range = range(2, 10)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            # Find optimal k (simplified)
            optimal_k = self._find_elbow(inertias, k_range)
            
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            
            results['kmeans'] = {
                'model': kmeans,
                'labels': labels,
                'inertia': kmeans.inertia_,
                'optimal_k': optimal_k
            }
        
        if 'dbscan' in methods:
            # DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(X_scaled)
            
            results['dbscan'] = {
                'model': dbscan,
                'labels': labels,
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
            }
        
        if 'hierarchical' in methods:
            # Agglomerative Clustering
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            labels = hierarchical.fit_predict(X_scaled)
            
            results['hierarchical'] = {
                'model': hierarchical,
                'labels': labels
            }
        
        self.models['clustering'] = results
        return results
    
    # ==================== HYPERPARETER TUNING ====================
    
    def hyperparameter_tuning(self, X, y, model_type='classification', 
                            method='grid', cv=5, n_iter=50):
        """Advanced hyperparameter tuning"""
        print(f"Starting Hyperparameter Tuning ({method})...")
        
        if model_type == 'classification':
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'logistic_regression': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'svm': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            }
        else:  # regression
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        
        tuning_results = {}
        
        for model_name, param_grid in param_grids.items():
            print(f"Tuning {model_name}...")
            
            if method == 'grid':
                search = GridSearchCV(
                    self._get_base_model(model_name, model_type),
                    param_grid,
                    cv=cv,
                    scoring='accuracy' if model_type == 'classification' else 'r2',
                    n_jobs=-1
                )
            else:  # randomized
                search = RandomizedSearchCV(
                    self._get_base_model(model_name, model_type),
                    param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring='accuracy' if model_type == 'classification' else 'r2',
                    n_jobs=-1,
                    random_state=42
                )
            
            search.fit(X, y)
            tuning_results[model_name] = {
                'best_model': search.best_estimator_,
                'best_params': search.best_params_,
                'best_score': search.best_score_
            }
            
            print(f"{model_name} - Best Score: {search.best_score_:.4f}")
        
        return tuning_results
    
    # ==================== ENSEMBLE METHODS ====================
    
    def ensemble_pipeline(self, X, y, model_type='classification', test_size=0.2):
        """Advanced ensemble methods"""
        print("Starting Ensemble Pipeline...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if model_type == 'classification':
            # Base estimators
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingClassifier(random_state=42)),
                ('svm', SVC(probability=True, random_state=42))
            ]
            
            # Voting Classifier
            voting = VotingClassifier(estimators=estimators, voting='soft')
            voting.fit(X_train, y_train)
            voting_score = voting.score(X_test, y_test)
            
            # Stacking Classifier
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=5
            )
            stacking.fit(X_train, y_train)
            stacking_score = stacking.score(X_test, y_test)
            
            results = {
                'voting_classifier': {'model': voting, 'score': voting_score},
                'stacking_classifier': {'model': stacking, 'score': stacking_score}
            }
            
        else:  # regression
            estimators = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(random_state=42)),
                ('lr', LinearRegression())
            ]
            
            # Voting Regressor
            voting = VotingRegressor(estimators=estimators)
            voting.fit(X_train, y_train)
            voting_score = voting.score(X_test, y_test)
            
            # Stacking Regressor
            stacking = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(),
                cv=5
            )
            stacking.fit(X_train, y_train)
            stacking_score = stacking.score(X_test, y_test)
            
            results = {
                'voting_regressor': {'model': voting, 'score': voting_score},
                'stacking_regressor': {'model': stacking, 'score': stacking_score}
            }
        
        print(f"Voting Score: {voting_score:.4f}")
        print(f"Stacking Score: {stacking_score:.4f}")
        
        return results
    
    # ==================== UTILITY METHODS ====================
    
    def _evaluate_classification(self, y_true, y_pred, y_pred_proba=None):
        """Comprehensive classification evaluation"""
        scores = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            scores['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
        return scores
    
    def _evaluate_regression(self, y_true, y_pred):
        """Comprehensive regression evaluation"""
        scores = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        return scores
    
    def _get_base_model(self, model_name, model_type):
        """Get base model for hyperparameter tuning"""
        if model_type == 'classification':
            models = {
                'random_forest': RandomForestClassifier(random_state=42),
                'logistic_regression': LogisticRegression(random_state=42),
                'svm': SVC(random_state=42),
                'gradient_boosting': GradientBoostingClassifier(random_state=42)
            }
        else:
            models = {
                'random_forest': RandomForestRegressor(random_state=42),
                'linear_regression': LinearRegression(),
                'gradient_boosting': GradientBoostingRegressor(random_state=42)
            }
        
        return models.get(model_name, models['random_forest'])
    
    def _find_elbow(self, values, range_values):
        """Simple elbow detection for clustering"""
        # Simplified implementation
        diffs = np.diff(values)
        second_diffs = np.diff(diffs)
        elbow_index = np.argmin(second_diffs) + 1
        return range_values[elbow_index]
    
    def _get_best_model(self, results, metric='f1'):
        """Get the best performing model"""
        best_score = -np.inf
        best_model_name = None
        
        for name, result in results.items():
            score = result['scores'].get(metric, -np.inf)
            if score > best_score:
                best_score = score
                best_model_name = name
        
        return best_model_name, results[best_model_name]

# Example usage
if __name__ == "__main__":
    # Create sample data
    from sklearn.datasets import make_classification, make_regression
    
    # Classification example
    X_clf, y_clf = make_classification(
        n_samples=1000, n_features=20, n_informative=15, 
        n_redundant=5, random_state=42
    )
    
    # Regression example
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=15, noise=0.1, random_state=42
    )
    
    # Initialize pipeline
    ml_pipeline = UltimateSklearnPipeline()
    
    # Run classification pipeline
    print("=" * 50)
    clf_results = ml_pipeline.classification_pipeline(X_clf, y_clf)
    
    # Run regression pipeline
    print("=" * 50)
    reg_results = ml_pipeline.regression_pipeline(X_reg, y_reg)
    
    # Run clustering
    print("=" * 50)
    cluster_results = ml_pipeline.clustering_pipeline(X_clf)
    
    # Hyperparameter tuning
    print("=" * 50)
    tuning_results = ml_pipeline.hyperparameter_tuning(X_clf, y_clf, 'classification')
    
    # Ensemble methods
    print("=" * 50)
    ensemble_results = ml_pipeline.ensemble_pipeline(X_clf, y_clf, 'classification')
    
    print("\nAll pipelines completed successfully!")

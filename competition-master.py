"""
7_ai_competition_master.py
Specialized Pipelines for AI Competition Scenarios
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedCrossValidation:
    """Advanced cross-validation strategies for competitions"""
    
    def __init__(self):
        self.cv_scores = {}
        
    def stratified_kfold(self, model, X, y, n_splits=5, scoring='accuracy'):
        """Stratified K-Fold for classification"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        self.cv_scores['stratified_kfold'] = scores
        return scores
    
    def time_series_split(self, model, X, y, n_splits=5):
        """Time Series Split for temporal data"""
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        self.cv_scores['time_series'] = scores
        return scores
    
    def group_kfold(self, model, X, y, groups, n_splits=5):
        """Group K-Fold for grouped data"""
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=n_splits)
        scores = cross_val_score(model, X, y, cv=gkf, scoring='accuracy')
        self.cv_scores['group_kfold'] = scores
        return scores
    
    def adversarial_validation(self, train_data, test_data, target_col=None):
        """Adversarial validation to check train/test distribution similarity"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        
        # Create labels: train=0, test=1
        train_data['adv_target'] = 0
        test_data['adv_target'] = 1
        
        # Combine data
        combined_data = pd.concat([train_data, test_data], axis=0)
        y_adv = combined_data['adv_target']
        X_adv = combined_data.drop('adv_target', axis=1)
        
        if target_col:
            X_adv = X_adv.drop(target_col, axis=1)
        
        # Train classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(model, X_adv, y_adv, cv=5, scoring='roc_auc')
        
        return np.mean(cv_scores), cv_scores

class FeatureEngineering:
    """Advanced feature engineering for competitions"""
    
    def __init__(self):
        self.feature_importance = {}
    
    def create_interaction_features(self, df, feature_pairs):
        """Create interaction features between pairs"""
        df_interactions = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplication interaction
                df_interactions[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                # Ratio interaction
                if df[feat2].min() > 0:  # Avoid division by zero
                    df_interactions[f'{feat1}_div_{feat2}'] = df[feat1] / df[feat2]
        
        return df_interactions
    
    def create_polynomial_features(self, df, features, degree=2):
        """Create polynomial features"""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[features])
        poly_feature_names = poly.get_feature_names_out(features)
        
        df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        return pd.concat([df, df_poly], axis=1)
    
    def target_encoding(self, df, categorical_columns, target_column, smooth=10):
        """Target encoding with smoothing"""
        df_encoded = df.copy()
        
        for col in categorical_columns:
            # Calculate mean target for each category
            means = df.groupby(col)[target_column].mean()
            counts = df.groupby(col)[target_column].count()
            
            # Smoothing: (mean * n + global_mean * smooth) / (n + smooth)
            global_mean = df[target_column].mean()
            smoothed_means = (means * counts + global_mean * smooth) / (counts + smooth)
            
            df_encoded[f'{col}_target_enc'] = df[col].map(smoothed_means)
        
        return df_encoded
    
    def frequency_encoding(self, df, categorical_columns):
        """Frequency encoding for categorical variables"""
        df_encoded = df.copy()
        
        for col in categorical_columns:
            freq_encoding = df[col].value_counts(normalize=True)
            df_encoded[f'{col}_freq_enc'] = df[col].map(freq_encoding)
        
        return df_encoded
    
    def create_aggregation_features(self, df, group_columns, agg_columns, agg_funcs=['mean', 'std', 'min', 'max']):
        """Create aggregation features for groups"""
        df_agg = df.copy()
        
        for group_col in group_columns:
            for agg_col in agg_columns:
                if agg_col in df.columns:
                    group_agg = df.groupby(group_col)[agg_col].agg(agg_funcs)
                    group_agg.columns = [f'{agg_col}_{group_col}_{func}' for func in agg_funcs]
                    
                    # Merge back
                    df_agg = df_agg.merge(group_agg, how='left', on=group_col)
        
        return df_agg

class EnsembleBuilder:
    """Advanced ensemble methods for competitions"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
    
    def create_stacking_ensemble(self, base_models, meta_model, X, y, cv=5):
        """Create stacking ensemble"""
        from sklearn.model_selection import cross_val_predict
        import numpy as np
        
        # Generate base model predictions
        base_predictions = []
        
        for name, model in base_models:
            # Get out-of-fold predictions
            oof_predictions = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
            base_predictions.append(oof_predictions)
            self.models[name] = model
        
        # Stack predictions
        X_meta = np.column_stack(base_predictions)
        
        # Train meta-model
        meta_model.fit(X_meta, y)
        self.models['meta'] = meta_model
        
        return X_meta, meta_model
    
    def create_blending_ensemble(self, models, X_train, y_train, X_val, y_val):
        """Create blending ensemble with validation set"""
        base_predictions = []
        
        for name, model in models:
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)[:, 1]
            base_predictions.append(preds)
            self.models[name] = model
        
        # Stack validation predictions
        X_blend = np.column_stack(base_predictions)
        
        # Train blender (simple average or weighted)
        self.weights = np.ones(len(models)) / len(models)  # Equal weights
        
        return X_blend
    
    def weighted_average_ensemble(self, predictions, weights=None):
        """Weighted average ensemble"""
        if weights is None:
            weights = np.ones(len(predictions)) / len(predictions)
        
        weighted_preds = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            weighted_preds += pred * weight
        
        return weighted_preds
    
    def rank_average_ensemble(self, predictions):
        """Rank averaging ensemble"""
        ranked_predictions = []
        
        for pred in predictions:
            ranks = pd.Series(pred).rank()
            ranked_predictions.append(ranks.values)
        
        avg_ranks = np.mean(ranked_predictions, axis=0)
        return avg_ranks / len(avg_ranks)

class AdvancedOptimization:
    """Hyperparameter optimization for competitions"""
    
    def __init__(self):
        self.best_params = {}
    
    def optuna_optimization(self, model_class, X, y, param_space, n_trials=100, direction='maximize'):
        """Hyperparameter optimization using Optuna"""
        import optuna
        from sklearn.model_selection import cross_val_score
        
        def objective(trial):
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
            
            model = model_class(**params)
            score = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
            return score
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params[model_class.__name__] = study.best_params
        return study.best_params
    
    def bayesian_optimization(self, model, param_bounds, X, y, n_iter=50):
        """Bayesian optimization using scikit-optimize"""
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        from sklearn.model_selection import cross_val_score
        
        # Convert bounds to skopt space
        space = []
        for param_name, bounds in param_bounds.items():
            if bounds['type'] == 'real':
                space.append(Real(bounds['low'], bounds['high'], name=param_name))
            elif bounds['type'] == 'integer':
                space.append(Integer(bounds['low'], bounds['high'], name=param_name))
            elif bounds['type'] == 'categorical':
                space.append(Categorical(bounds['categories'], name=param_name))
        
        def objective(params):
            param_dict = dict(zip(param_bounds.keys(), params))
            model.set_params(**param_dict)
            return -cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
        
        result = gp_minimize(objective, space, n_calls=n_iter, random_state=42)
        
        best_params = dict(zip(param_bounds.keys(), result.x))
        self.best_params[model.__class__.__name__] = best_params
        
        return best_params

class TimeSeriesCompetitionPipeline:
    """Specialized pipeline for time series competitions"""
    
    def __init__(self):
        self.features = []
        
    def create_time_features(self, df, date_column):
        """Create comprehensive time-based features"""
        df_temp = df.copy()
        df_temp[date_column] = pd.to_datetime(df_temp[date_column])
        
        # Basic time features
        df_temp['year'] = df_temp[date_column].dt.year
        df_temp['month'] = df_temp[date_column].dt.month
        df_temp['week'] = df_temp[date_column].dt.isocalendar().week
        df_temp['day'] = df_temp[date_column].dt.day
        df_temp['dayofweek'] = df_temp[date_column].dt.dayofweek
        df_temp['quarter'] = df_temp[date_column].dt.quarter
        df_temp['is_weekend'] = (df_temp[date_column].dt.dayofweek >= 5).astype(int)
        
        # Cyclical features
        df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['month'] / 12)
        df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['month'] / 12)
        df_temp['day_sin'] = np.sin(2 * np.pi * df_temp['day'] / 31)
        df_temp['day_cos'] = np.cos(2 * np.pi * df_temp['day'] / 31)
        
        return df_temp
    
    def create_lag_features(self, df, value_column, group_columns=None, lags=[1, 7, 30]):
        """Create lag features"""
        df_lags = df.copy()
        
        if group_columns:
            for group_col in group_columns:
                for lag in lags:
                    df_lags[f'{value_column}_lag_{lag}_{group_col}'] = df.groupby(group_col)[value_column].shift(lag)
        else:
            for lag in lags:
                df_lags[f'{value_column}_lag_{lag}'] = df[value_column].shift(lag)
        
        return df_lags
    
    def create_rolling_features(self, df, value_column, windows=[7, 30], group_columns=None):
        """Create rolling statistics"""
        df_rolling = df.copy()
        
        if group_columns:
            for group_col in group_columns:
                for window in windows:
                    grouped = df.groupby(group_col)[value_column]
                    df_rolling[f'{value_column}_roll_mean_{window}_{group_col}'] = grouped.transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    df_rolling[f'{value_column}_roll_std_{window}_{group_col}'] = grouped.transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
        else:
            for window in windows:
                df_rolling[f'{value_column}_roll_mean_{window}'] = df[value_column].rolling(window, min_periods=1).mean()
                df_rolling[f'{value_column}_roll_std_{window}'] = df[value_column].rolling(window, min_periods=1).std()
        
        return df_rolling
    
    def create_expanding_features(self, df, value_column, group_columns=None):
        """Create expanding window features"""
        df_expanding = df.copy()
        
        if group_columns:
            for group_col in group_columns:
                grouped = df.groupby(group_col)[value_column]
                df_expanding[f'{value_column}_expanding_mean_{group_col}'] = grouped.expanding().mean().values
                df_expanding[f'{value_column}_expanding_std_{group_col}'] = grouped.expanding().std().values
        else:
            df_expanding[f'{value_column}_expanding_mean'] = df[value_column].expanding().mean()
            df_expanding[f'{value_column}_expanding_std'] = df[value_column].expanding().std()
        
        return df_expanding

class TabularCompetitionPipeline:
    """Comprehensive pipeline for tabular competitions"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.cv_scores = {}
        
    def automated_feature_engineering(self, df, target_column=None):
        """Automated feature engineering for tabular data"""
        df_fe = df.copy()
        
        # Numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        # Categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create interaction features for numerical columns
        if len(numerical_cols) >= 2:
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    df_fe[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                    df_fe[f'{col1}_mul_{col2}'] = df[col1] * df[col2]
                    if df[col2].min() > 0:
                        df_fe[f'{col1}_div_{col2}'] = df[col1] / df[col2]
        
        # Create aggregation features for categorical columns
        for cat_col in categorical_cols:
            if len(numerical_cols) > 0:
                for num_col in numerical_cols:
                    # Groupby statistics
                    group_stats = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'min', 'max']).add_prefix(f'{num_col}_by_{cat_col}_')
                    df_fe = df_fe.merge(group_stats, how='left', left_on=cat_col, right_index=True)
        
        # Polynomial features for highly correlated numerical features
        if len(numerical_cols) >= 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(df[numerical_cols].fillna(0))
            df_fe['pca_1'] = pca_features[:, 0]
            df_fe['pca_2'] = pca_features[:, 1]
        
        return df_fe
    
    def advanced_model_training(self, X, y, task_type='classification'):
        """Train multiple advanced models with CV"""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        from sklearn.model_selection import cross_val_score
        
        if task_type == 'classification':
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'xgboost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
                'lightgbm': LGBMClassifier(n_estimators=100, random_state=42),
                'catboost': CatBoostClassifier(n_estimators=100, random_state=42, verbose=False),
                'logistic': LogisticRegression(random_state=42)
            }
        else:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor
            from catboost import CatBoostRegressor
            from sklearn.linear_model import LinearRegression
            
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'xgboost': XGBRegressor(n_estimators=100, random_state=42),
                'lightgbm': LGBMRegressor(n_estimators=100, random_state=42),
                'catboost': CatBoostRegressor(n_estimators=100, random_state=42, verbose=False),
                'linear': LinearRegression()
            }
        
        # Train models with CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if task_type == 'classification' else KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc' if task_type == 'classification' else 'neg_mean_squared_error')
                self.cv_scores[name] = scores
                print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                
                # Fit model on full data for feature importance
                model.fit(X, y)
                self.models[name] = model
                
                # Get feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                    
            except Exception as e:
                print(f"Error with {name}: {e}")
        
        return self.models, self.cv_scores
    
    def create_ensemble_submission(self, test_data, sample_submission, weights=None):
        """Create ensemble submission file"""
        predictions = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                predictions[name] = model.predict_proba(test_data)[:, 1]
            else:
                predictions[name] = model.predict(test_data)
        
        # Weighted average
        if weights is None:
            weights = {name: 1/len(predictions) for name in predictions.keys()}
        
        ensemble_prediction = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            ensemble_prediction += pred * weights.get(name, 1/len(predictions))
        
        # Create submission file
        submission = sample_submission.copy()
        if len(ensemble_prediction.shape) > 1:
            submission.iloc[:, 1] = ensemble_prediction[:, 1]
        else:
            submission.iloc[:, 1] = ensemble_prediction
        
        return submission

class ComputerVisionCompetitionPipeline:
    """Pipeline for computer vision competitions"""
    
    def __init__(self):
        self.models = {}
        
    def create_augmentation_pipeline(self):
        """Create comprehensive data augmentation pipeline"""
        from albumentations import (
            Compose, HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate,
            RandomBrightnessContrast, HueSaturationValue, GaussNoise, GaussianBlur,
            ElasticTransform, OpticalDistortion, GridDistortion, CoarseDropout
        )
        
        train_transform = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(p=0.5),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            GaussianBlur(blur_limit=3, p=0.3),
            CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
        ])
        
        return train_transform
    
    def create_pretrained_models(self, num_classes, model_names=['resnet50', 'efficientnet', 'densenet']):
        """Create multiple pretrained models"""
        import torch
        import torchvision.models as models
        
        model_dict = {}
        
        for name in model_names:
            if name == 'resnet50':
                model = models.resnet50(pretrained=True)
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            elif name == 'efficientnet':
                model = models.efficientnet_b0(pretrained=True)
                model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
            elif name == 'densenet':
                model = models.densenet121(pretrained=True)
                model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
            
            model_dict[name] = model
        
        self.models.update(model_dict)
        return model_dict
    
    def create_test_time_augmentation(self, model, image, augmentations, n_augmentations=10):
        """Test Time Augmentation (TTA)"""
        model.eval()
        predictions = []
        
        with torch.no_grad():
            # Original image
            original_pred = model(image.unsqueeze(0))
            predictions.append(original_pred)
            
            # Augmented images
            for _ in range(n_augmentations):
                augmented = augmentations(image=image.numpy())
                aug_tensor = torch.tensor(augmented['image']).unsqueeze(0)
                aug_pred = model(aug_tensor)
                predictions.append(aug_pred)
        
        # Average predictions
        avg_prediction = torch.mean(torch.stack(predictions), dim=0)
        return avg_prediction

class NLPCompetitionPipeline:
    """Pipeline for NLP competitions"""
    
    def __init__(self):
        self.models = {}
        
    def create_advanced_text_features(self, texts):
        """Create advanced text features for competitions"""
        import nltk
        from textstat import flesch_reading_ease, syllable_count
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        
        features = {}
        
        # Basic text statistics
        features['text_length'] = [len(text) for text in texts]
        features['word_count'] = [len(text.split()) for text in texts]
        features['avg_word_length'] = [np.mean([len(word) for word in text.split()]) if text.split() else 0 for text in texts]
        features['unique_word_ratio'] = [len(set(text.split())) / len(text.split()) if text.split() else 0 for text in texts]
        
        # Readability scores
        features['flesch_reading_ease'] = [flesch_reading_ease(text) for text in texts]
        
        # Sentiment features
        from textblob import TextBlob
        features['sentiment_polarity'] = [TextBlob(text).sentiment.polarity for text in texts]
        features['sentiment_subjectivity'] = [TextBlob(text).sentiment.subjectivity for text in texts]
        
        # POS tag ratios
        for text in texts:
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            pos_counts = nltk.FreqDist(tag for word, tag in pos_tags)
            
            total_tags = len(tokens)
            for pos in ['NN', 'VB', 'JJ', 'RB']:
                features[f'ratio_{pos}'] = features.get(f'ratio_{pos}', [])
                features[f'ratio_{pos}'].append(pos_counts[pos] / total_tags if total_tags > 0 else 0)
        
        return pd.DataFrame(features)
    
    def create_multilingual_features(self, texts):
        """Create features for multilingual text"""
        import langdetect
        from langdetect import detect
        
        features = {}
        features['language'] = []
        
        for text in texts:
            try:
                lang = detect(text)
                features['language'].append(lang)
            except:
                features['language'].append('unknown')
        
        # One-hot encode languages
        lang_df = pd.get_dummies(pd.Series(features['language']), prefix='lang')
        return lang_df

# Example competition workflow
class CompetitionWorkflow:
    """Complete competition workflow"""
    
    def __init__(self):
        self.cv = AdvancedCrossValidation()
        self.fe = FeatureEngineering()
        self.ensemble = EnsembleBuilder()
        self.optimizer = AdvancedOptimization()
        
    def run_tabular_competition(self, train_data, test_data, target_column, id_column=None):
        """Complete workflow for tabular competition"""
        print("Starting Tabular Competition Pipeline...")
        
        # 1. Feature Engineering
        print("1. Feature Engineering...")
        train_fe = self.fe.automated_feature_engineering(train_data, target_column)
        test_fe = self.fe.automated_feature_engineering(test_data, target_column)
        
        # 2. Prepare data
        X = train_fe.drop(columns=[target_column])
        if id_column:
            X = X.drop(columns=[id_column])
        y = train_data[target_column]
        
        X_test = test_fe
        if id_column and id_column in X_test.columns:
            X_test = X_test.drop(columns=[id_column])
        
        # 3. Advanced CV
        print("2. Cross-Validation...")
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = self.cv.stratified_kfold(rf, X, y, scoring='roc_auc')
        print(f"CV Scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 4. Model Training
        print("3. Model Training...")
        tabular_pipeline = TabularCompetitionPipeline()
        models, scores = tabular_pipeline.advanced_model_training(X, y)
        
        # 5. Ensemble Prediction
        print("4. Ensemble Prediction...")
        # This would require actual sample submission file
        # submission = tabular_pipeline.create_ensemble_submission(X_test, sample_submission)
        
        print("Competition pipeline completed!")
        return models, scores

# Example usage
if __name__ == "__main__":
    # Create sample competition data
    np.random.seed(42)
    n_samples = 1000
    
    train_data = pd.DataFrame({
        'id': range(n_samples),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    test_data = pd.DataFrame({
        'id': range(n_samples, n_samples + 500),
        'feature1': np.random.normal(0, 1, 500),
        'feature2': np.random.normal(0, 1, 500),
        'feature3': np.random.choice(['A', 'B', 'C'], 500),
    })
    
    # Run competition workflow
    workflow = CompetitionWorkflow()
    models, scores = workflow.run_tabular_competition(train_data, test_data, 'target', 'id')
    
    print("\nCompetition-ready pipelines are prepared!")
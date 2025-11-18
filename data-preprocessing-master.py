"""
1_data_preprocessing_master.py
Ultimate Data Preprocessing Pipeline for All Data Types
"""

import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, PowerTransformer, QuantileTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, RFE, RFECV, SelectFromModel,
    f_classif, f_regression, mutual_info_classif
)
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class UltimateDataPreprocessor:
    """Comprehensive data preprocessing for all data types"""
    
    def __init__(self):
        self.preprocessors = {}
        self.feature_names = []
        
    # ==================== NUMERICAL DATA ====================
    
    def handle_numerical_data(self, df, numerical_columns, strategy='auto'):
        """Advanced numerical data preprocessing"""
        df_processed = df.copy()
        
        # Outlier detection and treatment
        df_processed = self._handle_outliers(df_processed, numerical_columns)
        
        # Missing value imputation
        imputer = self._get_imputer(strategy)
        df_processed[numerical_columns] = imputer.fit_transform(df_processed[numerical_columns])
        
        # Feature scaling
        scaler = self._get_scaler('robust')
        df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
        
        # Feature engineering
        df_processed = self._create_numerical_features(df_processed, numerical_columns)
        
        self.preprocessors['numerical_scaler'] = scaler
        self.preprocessors['numerical_imputer'] = imputer
        
        return df_processed
    
    def _handle_outliers(self, df, numerical_columns, method='iqr'):
        """Multiple outlier handling strategies"""
        df_clean = df.copy()
        
        for col in numerical_columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
                
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean[col] = np.where(z_scores > 3, df_clean[col].median(), df_clean[col])
                
        return df_clean
    
    def _get_imputer(self, strategy):
        """Get appropriate imputer"""
        if strategy == 'knn':
            return KNNImputer(n_neighbors=5)
        elif strategy == 'iterative':
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            return IterativeImputer()
        else:
            return SimpleImputer(strategy='median')
    
    def _get_scaler(self, method):
        """Get appropriate scaler"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'power': PowerTransformer(),
            'quantile': QuantileTransformer(output_distribution='normal')
        }
        return scalers.get(method, RobustScaler())
    
    def _create_numerical_features(self, df, numerical_columns):
        """Create derived numerical features"""
        for col in numerical_columns:
            # Polynomial features
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_log'] = np.log1p(np.abs(df[col]))
            
            # Statistical features
            df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
            
        return df
    
    # ==================== CATEGORICAL DATA ====================
    
    def handle_categorical_data(self, df, categorical_columns, encoding='auto'):
        """Advanced categorical encoding"""
        df_encoded = df.copy()
        
        # Handle high cardinality features
        df_encoded = self._handle_high_cardinality(df_encoded, categorical_columns)
        
        # Multiple encoding strategies
        if encoding == 'auto':
            # Use target encoding for high cardinality, one-hot for low
            for col in categorical_columns:
                if df_encoded[col].nunique() > 10:
                    df_encoded = self._target_encode(df_encoded, col)
                else:
                    df_encoded = self._one_hot_encode(df_encoded, col)
        elif encoding == 'onehot':
            df_encoded = self._one_hot_encode(df_encoded, categorical_columns)
        elif encoding == 'target':
            for col in categorical_columns:
                df_encoded = self._target_encode(df_encoded, col)
                
        return df_encoded
    
    def _handle_high_cardinality(self, df, categorical_columns, threshold=10):
        """Group rare categories"""
        df_processed = df.copy()
        
        for col in categorical_columns:
            if df_processed[col].nunique() > threshold:
                # Group low frequency categories
                value_counts = df_processed[col].value_counts()
                mask = df_processed[col].isin(value_counts.index[value_counts < threshold])
                df_processed.loc[mask, col] = 'OTHER'
                
        return df_processed
    
    def _one_hot_encode(self, df, columns):
        """One-hot encoding with sparse optimization"""
        for col in columns:
            dummies = pd.get_dummies(df[col], prefix=col, sparse=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
            
        return df
    
    def _target_encode(self, df, column, target=None, smoothing=10):
        """Target encoding with smoothing"""
        # This would require target variable - simplified version
        df_encoded = df.copy()
        means = df_encoded.groupby(column).size() / len(df_encoded)
        df_encoded[f'{column}_target_enc'] = df_encoded[column].map(means)
        df_encoded.drop(column, axis=1, inplace=True)
        
        return df_encoded
    
    # ==================== TEXT DATA ====================
    
    def preprocess_text_data(self, texts, method='tfidf', **kwargs):
        """Comprehensive text preprocessing"""
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation, NMF
        
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=kwargs.get('max_features', 5000),
                ngram_range=kwargs.get('ngram_range', (1, 2)),
                stop_words=kwargs.get('stop_words', 'english')
            )
        elif method == 'count':
            vectorizer = CountVectorizer(
                max_features=kwargs.get('max_features', 5000),
                ngram_range=kwargs.get('ngram_range', (1, 2))
            )
        elif method == 'hashing':
            from sklearn.feature_extraction.text import HashingVectorizer
            vectorizer = HashingVectorizer(
                n_features=kwargs.get('n_features', 5000),
                ngram_range=kwargs.get('ngram_range', (1, 2))
            )
            
        X = vectorizer.fit_transform(texts)
        self.preprocessors['text_vectorizer'] = vectorizer
        
        # Dimensionality reduction for text
        if kwargs.get('reduce_dim', False):
            reducer = TruncatedSVD(n_components=kwargs.get('n_components', 100))
            X = reducer.fit_transform(X)
            self.preprocessors['text_reducer'] = reducer
            
        return X
    
    # ==================== TIME SERIES ====================
    
    def preprocess_time_series(self, series, window_size=10):
        """Time series feature engineering"""
        df = pd.DataFrame({'value': series})
        
        # Statistical features
        df['rolling_mean'] = df['value'].rolling(window=window_size).mean()
        df['rolling_std'] = df['value'].rolling(window=window_size).std()
        df['rolling_max'] = df['value'].rolling(window=window_size).max()
        df['rolling_min'] = df['value'].rolling(window=window_size).min()
        
        # Time-based features
        df['lag_1'] = df['value'].shift(1)
        df['lag_7'] = df['value'].shift(7)
        df['diff_1'] = df['value'].diff(1)
        
        # Seasonal features
        df['day_of_week'] = pd.Series(range(len(df))) % 7
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df.fillna(method='bfill')
    
    # ==================== FEATURE SELECTION ====================
    
    def feature_selection(self, X, y, method='random_forest', n_features=50):
        """Comprehensive feature selection"""
        
        if method == 'random_forest':
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42),
                max_features=n_features
            )
        elif method == 'rfe':
            selector = RFECV(
                estimator=RandomForestClassifier(n_estimators=50),
                min_features_to_select=n_features,
                cv=5
            )
        elif method == 'kbest':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            
        X_selected = selector.fit_transform(X, y)
        self.preprocessors['feature_selector'] = selector
        
        return X_selected
    
    # ==================== DIMENSIONALITY REDUCTION ====================
    
    def reduce_dimensionality(self, X, method='pca', n_components=2):
        """Multiple dimensionality reduction techniques"""
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == 'umap':
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        elif method == 'isomap':
            reducer = Isomap(n_components=n_components)
        elif method == 'kpca':
            reducer = KernelPCA(n_components=n_components, kernel='rbf', random_state=42)
            
        X_reduced = reducer.fit_transform(X)
        self.preprocessors['dimension_reducer'] = reducer
        
        return X_reduced
    
    # ==================== COMPLETE PIPELINE ====================
    
    def create_full_pipeline(self, df, target_column=None):
        """Complete automated preprocessing pipeline"""
        print("Starting comprehensive data preprocessing...")
        
        # Separate features and target
        if target_column:
            y = df[target_column]
            X = df.drop(columns=[target_column])
        else:
            X = df.copy()
            y = None
            
        # Identify column types
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Numerical columns: {numerical_cols}")
        print(f"Categorical columns: {categorical_cols}")
        
        # Process numerical data
        if numerical_cols:
            X_processed = self.handle_numerical_data(X, numerical_cols)
        else:
            X_processed = X.copy()
            
        # Process categorical data
        if categorical_cols:
            X_processed = self.handle_categorical_data(X_processed, categorical_cols)
            
        # Feature selection if target available
        if y is not None:
            X_processed = self.feature_selection(X_processed, y)
            
        print(f"Final shape: {X_processed.shape}")
        
        return X_processed, y

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'city': np.random.choice(['New York', 'London', 'Tokyo', 'Paris', 'Sydney'], n_samples),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Add some missing values and outliers
    sample_data.loc[np.random.choice(n_samples, 50), 'age'] = np.nan
    sample_data.loc[np.random.choice(n_samples, 10), 'income'] = 1000000  # outliers
    
    # Initialize preprocessor
    preprocessor = UltimateDataPreprocessor()
    
    # Run complete pipeline
    X_processed, y = preprocessor.create_full_pipeline(sample_data, 'target')
    
    print("Preprocessing completed successfully!")
    print(f"Processed data shape: {X_processed.shape}")
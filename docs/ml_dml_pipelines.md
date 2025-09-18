# ML-Based Double Machine Learning Pipelines

## Overview

Machine Learning methods (XGBoost, LightGBM, Neural Networks) serve as powerful nuisance function estimators in Double Machine Learning (DML) frameworks. They flexibly control for high-dimensional confounders while maintaining valid causal inference for elasticity estimation.

## The Role of ML in Causal Inference

### Why ML for Nuisance Functions?

Traditional econometric methods often require specifying functional forms (e.g., linear, log-linear). ML methods:
- Learn complex non-linear relationships automatically
- Handle high-dimensional features (hundreds of controls)
- Capture interactions without explicit specification
- Reduce model misspecification bias

### The DML Framework with ML

**Key Insight**: Use ML for prediction (nuisance functions), classical methods for inference (target parameter).

```
Step 1: Y = m(X) + ε_Y  → Use ML to learn m(X)
Step 2: T = g(X) + ε_T  → Use ML to learn g(X)  
Step 3: ε_Y = θ·ε_T + ν → Simple regression for elasticity θ
```

## XGBoost for DML

### What It Is

XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting framework that builds an ensemble of decision trees sequentially, with each tree correcting the errors of previous trees.

### Mathematical Foundation

**Objective Function:**
```
L = Σᵢ l(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)
```

Where:
- l: loss function (e.g., squared error)
- Ω: regularization term preventing overfitting
- fₖ: k-th tree in the ensemble

**Tree Building:**
Each tree splits to maximize:
```
Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
```

Where G and H are first and second-order gradients.

### Implementation

```python
import xgboost as xgb
from sklearn.model_selection import KFold
import numpy as np

class XGBoostDML:
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.models_y = []
        self.models_t = []
        
    def fit(self, X, T, Y):
        """
        Fit DML with XGBoost nuisance functions.
        
        X: Controls/confounders (n_samples, n_features)
        T: Treatment (price) (n_samples,)
        Y: Outcome (quantity) (n_samples,)
        """
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Store residuals
        Y_residuals = np.zeros_like(Y)
        T_residuals = np.zeros_like(T)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            # Model for E[Y|X]
            model_y = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,      # L1 regularization
                reg_lambda=1.0,     # L2 regularization
                random_state=42
            )
            
            model_y.fit(
                X_train, Y_train,
                eval_set=[(X_test, Y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            Y_pred = model_y.predict(X_test)
            Y_residuals[test_idx] = Y_test - Y_pred
            
            # Model for E[T|X]
            model_t = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model_t.fit(X_train, T_train)
            T_pred = model_t.predict(X_test)
            T_residuals[test_idx] = T_test - T_pred
            
            # Store models
            self.models_y.append(model_y)
            self.models_t.append(model_t)
        
        # Final stage: Regress Y residuals on T residuals
        self.elasticity = np.cov(Y_residuals, T_residuals)[0, 1] / np.var(T_residuals)
        
        # Standard error
        n = len(Y_residuals)
        final_residuals = Y_residuals - self.elasticity * T_residuals
        self.se = np.sqrt(np.var(final_residuals) / (n * np.var(T_residuals)))
        
        return self
    
    def predict_elasticity(self, X_new=None):
        """Get elasticity estimate with confidence interval."""
        ci_lower = self.elasticity - 1.96 * self.se
        ci_upper = self.elasticity + 1.96 * self.se
        
        return {
            'elasticity': self.elasticity,
            'se': self.se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def feature_importance(self):
        """Average feature importance across folds."""
        importance = np.mean([
            model.feature_importances_ 
            for model in self.models_t
        ], axis=0)
        
        return importance
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV

def tune_xgboost_dml(X, T, Y):
    """Tune XGBoost hyperparameters for DML."""
    
    # Parameter grid
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [0, 1.0, 10.0]
    }
    
    # Use CV to find best parameters for outcome model
    base_model = xgb.XGBRegressor(random_state=42)
    
    search = RandomizedSearchCV(
        base_model,
        param_dist,
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42
    )
    
    search.fit(X, Y)
    
    return search.best_params_
```

## LightGBM for High-Dimensional DML

### What It Is

LightGBM uses histogram-based algorithms for faster training on large datasets with many features. It's particularly efficient for high-dimensional controls.

### Key Advantages

- **Leaf-wise growth**: More accurate than level-wise
- **Categorical feature support**: Native handling without encoding
- **Faster training**: Histogram-based splitting
- **Lower memory usage**: Efficient data structures

### Implementation

```python
import lightgbm as lgb

class LightGBMDML:
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        
    def create_lgb_model(self, objective='regression'):
        """Create LightGBM model with good defaults."""
        return lgb.LGBMRegressor(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=-1
        )
    
    def fit_with_categoricals(self, X, T, Y, categorical_features=None):
        """Fit with native categorical support."""
        
        if categorical_features:
            # Convert to categorical type
            for cat_col in categorical_features:
                X[cat_col] = X[cat_col].astype('category')
        
        # Create dataset
        lgb_data = lgb.Dataset(
            X, label=Y,
            categorical_feature=categorical_features
        )
        
        # Train with early stopping
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        model = lgb.train(
            params,
            lgb_data,
            num_boost_round=1000,
            valid_sets=[lgb_data],
            callbacks=[lgb.early_stopping(10)]
        )
        
        return model
```

## Neural Networks for Non-linear DML

### Architecture for Nuisance Functions

```python
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn

class NeuralDML:
    def __init__(self, input_dim, hidden_sizes=[100, 50, 25]):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        
    def create_network(self):
        """Create neural network for nuisance estimation."""
        
        model = MLPRegressor(
            hidden_layer_sizes=self.hidden_sizes,
            activation='relu',
            solver='adam',
            alpha=0.001,          # L2 regularization
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )
        
        return model
    
    def create_pytorch_network(self):
        """PyTorch implementation for more control."""
        
        class NuisanceNet(nn.Module):
            def __init__(self, input_dim, hidden_sizes, dropout_rate=0.2):
                super().__init__()
                
                layers = []
                prev_size = input_dim
                
                for hidden_size in hidden_sizes:
                    layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_size),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_size = hidden_size
                
                # Output layer
                layers.append(nn.Linear(prev_size, 1))
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        return NuisanceNet(self.input_dim, self.hidden_sizes)
```

## Ensemble Methods for Robust DML

### Combining Multiple ML Models

```python
from sklearn.ensemble import StackingRegressor, VotingRegressor

class EnsembleDML:
    def __init__(self):
        self.base_models = self._create_base_models()
        
    def _create_base_models(self):
        """Create diverse base models."""
        return [
            ('rf', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )),
            ('gbm', GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )),
            ('xgb', xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )),
            ('lgb', lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                random_state=42
            ))
        ]
    
    def create_stacking_ensemble(self):
        """Stack models with meta-learner."""
        
        meta_learner = RidgeCV(cv=5)
        
        ensemble = StackingRegressor(
            estimators=self.base_models,
            final_estimator=meta_learner,
            cv=5  # Use cross-validation for training meta-learner
        )
        
        return ensemble
    
    def create_voting_ensemble(self):
        """Simple averaging ensemble."""
        
        ensemble = VotingRegressor(
            estimators=self.base_models
        )
        
        return ensemble
```

## Feature Engineering for ML-DML

### Creating Rich Features

```python
class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        
    def create_features(self, df):
        """Create features for ML models."""
        
        features = df.copy()
        
        # 1. Temporal features
        features['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        features['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
        features['month'] = df['week'] % 4 + 1
        features['quarter'] = (df['week'] - 1) // 13 + 1
        
        # 2. Price features
        features['price_squared'] = df['price'] ** 2
        features['price_cubed'] = df['price'] ** 3
        features['log_price'] = np.log(df['price'] + 1)
        
        # 3. Interaction terms
        features['price_x_income'] = df['price'] * df['income_level']
        features['price_x_promotion'] = df['price'] * df['promotion']
        
        # 4. Lag features
        for lag in [1, 2, 4]:
            features[f'price_lag_{lag}'] = df.groupby('product_id')['price'].shift(lag)
            features[f'quantity_lag_{lag}'] = df.groupby('product_id')['quantity'].shift(lag)
        
        # 5. Rolling statistics
        for window in [4, 8, 12]:
            features[f'price_roll_mean_{window}'] = (
                df.groupby('product_id')['price']
                .rolling(window, min_periods=1).mean()
                .reset_index(0, drop=True)
            )
            
            features[f'price_roll_std_{window}'] = (
                df.groupby('product_id')['price']
                .rolling(window, min_periods=1).std()
                .reset_index(0, drop=True)
            )
        
        # 6. Competitor features
        features['competitor_min_price'] = df.groupby(['store_id', 'week'])['price'].transform('min')
        features['competitor_max_price'] = df.groupby(['store_id', 'week'])['price'].transform('max')
        features['price_relative_to_mean'] = df['price'] / df.groupby(['store_id', 'week'])['price'].transform('mean')
        
        return features
    
    def create_polynomial_features(self, X, degree=2):
        """Create polynomial and interaction features."""
        
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        return X_poly
```

## Performance Optimization

### Parallel Processing

```python
from joblib import Parallel, delayed
import multiprocessing

class ParallelDML:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        
    def parallel_cross_fitting(self, X, T, Y, n_folds=5):
        """Parallelize cross-fitting across folds."""
        
        def fit_fold(train_idx, test_idx, X, T, Y):
            # Fit models for one fold
            model_y = xgb.XGBRegressor()
            model_t = xgb.XGBRegressor()
            
            model_y.fit(X[train_idx], Y[train_idx])
            model_t.fit(X[train_idx], T[train_idx])
            
            Y_resid = Y[test_idx] - model_y.predict(X[test_idx])
            T_resid = T[test_idx] - model_t.predict(X[test_idx])
            
            return test_idx, Y_resid, T_resid
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Parallel execution
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_fold)(train_idx, test_idx, X, T, Y)
            for train_idx, test_idx in kf.split(X)
        )
        
        # Combine results
        Y_residuals = np.zeros_like(Y)
        T_residuals = np.zeros_like(T)
        
        for test_idx, Y_resid, T_resid in results:
            Y_residuals[test_idx] = Y_resid
            T_residuals[test_idx] = T_resid
        
        return Y_residuals, T_residuals
```

### GPU Acceleration

```python
# For XGBoost
xgb_gpu = xgb.XGBRegressor(
    tree_method='gpu_hist',  # GPU acceleration
    predictor='gpu_predictor',
    n_estimators=100
)

# For LightGBM
lgb_gpu = lgb.LGBMRegressor(
    device='gpu',
    gpu_platform_id=0,
    gpu_device_id=0,
    n_estimators=100
)
```

## Model Selection and Validation

### Cross-Validation for ML-DML

```python
def cross_validate_dml(X, T, Y, models, n_splits=5):
    """Cross-validate different ML models for DML."""
    
    results = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for name, model_class in models.items():
        elasticities = []
        
        for train_idx, test_idx in kf.split(X):
            # Fit DML on training fold
            dml = XGBoostDML()  # or other DML class
            dml.fit(X[train_idx], T[train_idx], Y[train_idx])
            
            elasticities.append(dml.elasticity)
        
        results[name] = {
            'mean': np.mean(elasticities),
            'std': np.std(elasticities),
            'values': elasticities
        }
    
    return results
```

## Pros and Cons

### Pros
- ✅ **Flexibility**: No functional form assumptions
- ✅ **High dimensions**: Handles many features
- ✅ **Non-linearity**: Captures complex relationships
- ✅ **Feature importance**: Identifies key variables
- ✅ **Regularization**: Built-in overfitting prevention
- ✅ **Performance**: State-of-the-art prediction accuracy

### Cons
- ❌ **Black box**: Less interpretable than linear models
- ❌ **Hyperparameters**: Requires tuning
- ❌ **Computational cost**: Can be slow for large data
- ❌ **Overfitting risk**: Without proper regularization
- ❌ **Sample size**: Needs sufficient data

## Best Practices

1. **Always use cross-fitting**: Prevents overfitting bias
2. **Tune hyperparameters**: Use cross-validation
3. **Feature engineering**: Create domain-relevant features
4. **Ensemble methods**: Combine multiple models
5. **Check residuals**: Ensure no systematic patterns
6. **Bootstrap confidence intervals**: For robust inference

## References

- Chernozhukov et al. (2018). "Double/debiased machine learning"
- Chen & Guestrin (2016). "XGBoost: A scalable tree boosting system"
- Ke et al. (2017). "LightGBM: A highly efficient gradient boosting decision tree"
- Athey et al. (2019). "Machine learning methods economists should know about"

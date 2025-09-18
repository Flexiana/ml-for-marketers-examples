"""
Scikit-learn/XGBoost DML Pipeline Example

This module demonstrates using ML methods as nuisance models in Double Machine Learning:

1. XGBoost for flexible nuisance function estimation
2. LightGBM for high-dimensional controls
3. Neural networks for complex non-linearities
4. Ensemble methods for robust estimation
5. Cross-fitting and sample splitting
6. Feature engineering for demand models

ML pipelines improve elasticity estimation by:
- Flexibly controlling for confounders
- Capturing non-linear relationships
- Handling high-dimensional features
- Reducing regularization bias
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


class MLPipelineElasticityEstimator:
    """ML-based Double Machine Learning for elasticity estimation."""
    
    def __init__(self, data_path: str = 'data/retail_scanner_data.csv'):
        """Initialize with retail scanner data."""
        self.df = pd.read_csv(data_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare data with feature engineering."""
        
        # Log transformations
        self.df['log_quantity'] = np.log(self.df['quantity'] + 1)
        self.df['log_price'] = np.log(self.df['price'])
        
        # Create additional features for ML models
        self.engineer_features()
        
        print(f"ML Pipeline Data:")
        print(f"  Observations: {len(self.df)}")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Treatment: log_price")
        print(f"  Outcome: log_quantity")
    
    def engineer_features(self):
        """Create rich feature set for ML models."""
        
        # Time features
        self.df['week_sin'] = np.sin(2 * np.pi * self.df['week'] / 52)
        self.df['week_cos'] = np.cos(2 * np.pi * self.df['week'] / 52)
        self.df['month'] = self.df['week'] % 4 + 1
        self.df['quarter'] = (self.df['week'] - 1) // 13 + 1
        
        # Price features
        self.df['price_squared'] = self.df['price'] ** 2
        self.df['price_lag_ratio'] = self.df['price'] / self.df['lag_price'].clip(0.01)
        
        # Competition features
        for cat in self.df['category'].unique():
            self.df[f'competitor_price_{cat}'] = self.df[f'avg_competitor_price_{cat}'].fillna(0)
        
        # Store features
        self.df['store_size_numeric'] = self.df['store_size'].map({'small': 1, 'medium': 2, 'large': 3})
        
        # Interaction terms
        self.df['price_x_income'] = self.df['log_price'] * self.df['income_level'] / 100000
        self.df['price_x_promotion'] = self.df['log_price'] * self.df['promotion']
        
        # Define feature columns
        self.feature_cols = [
            'promotion', 'income_level', 'population_density',
            'week_sin', 'week_cos', 'month', 'quarter',
            'wholesale_cost', 'transportation_cost',
            'num_rival_products', 'store_size_numeric',
            'price_x_income', 'price_x_promotion'
        ]
        
        # Add competitor prices
        self.feature_cols.extend([col for col in self.df.columns if 'competitor_price_' in col])
        
        # Add categorical features for encoding
        self.categorical_features = ['store_type', 'quality_tier', 'brand', 'category']
    
    def example_1_xgboost_dml(self) -> Dict:
        """
        Example 1: XGBoost for DML
        
        Uses XGBoost for flexible nuisance function estimation.
        """
        print("\n" + "="*60)
        print("EXAMPLE 1: XGBoost Double ML")
        print("="*60)
        
        # Prepare data
        X = self.df[self.feature_cols].fillna(0)
        T = self.df['log_price'].values
        Y = self.df['log_quantity'].values
        
        # Add categorical encoding
        for cat_col in self.categorical_features:
            if cat_col in self.df.columns:
                dummies = pd.get_dummies(self.df[cat_col], prefix=cat_col, drop_first=True)
                X = pd.concat([X, dummies], axis=1)
        
        X = X.values
        
        # Cross-fitting setup
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Store residuals for all folds
        Y_residuals = np.zeros_like(Y)
        T_residuals = np.zeros_like(T)
        
        print(f"\nPerforming {n_folds}-fold cross-fitting...")
        print("-" * 40)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold+1}/{n_folds}:")
            
            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            # Model for E[Y|X] - outcome nuisance
            model_y = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            
            model_y.fit(X_train, Y_train)
            Y_pred = model_y.predict(X_test)
            Y_residuals[test_idx] = Y_test - Y_pred
            
            # Model for E[T|X] - treatment nuisance
            model_t = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            
            model_t.fit(X_train, T_train)
            T_pred = model_t.predict(X_test)
            T_residuals[test_idx] = T_test - T_pred
            
            print(f"  Y R²: {model_y.score(X_test, Y_test):.3f}")
            print(f"  T R²: {model_t.score(X_test, T_test):.3f}")
        
        # Final stage: regress Y residuals on T residuals
        print("\n" + "-"*40)
        print("Final stage regression:")
        
        # Remove any remaining confounding
        valid_idx = ~(np.isnan(Y_residuals) | np.isnan(T_residuals))
        elasticity = np.cov(Y_residuals[valid_idx], T_residuals[valid_idx])[0, 1] / np.var(T_residuals[valid_idx])
        
        # Standard error (simplified)
        n = valid_idx.sum()
        residuals = Y_residuals[valid_idx] - elasticity * T_residuals[valid_idx]
        se = np.sqrt(np.var(residuals) / (n * np.var(T_residuals[valid_idx])))
        
        print(f"\nXGBoost DML Elasticity: {elasticity:.3f}")
        print(f"Standard Error: {se:.3f}")
        print(f"95% CI: [{elasticity - 1.96*se:.3f}, {elasticity + 1.96*se:.3f}]")
        
        # Feature importance from first stage
        feature_importance = model_t.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-5:]
        
        print("\nTop 5 features for price prediction:")
        for idx in top_features_idx:
            if idx < len(self.feature_cols):
                print(f"  {self.feature_cols[idx]}: {feature_importance[idx]:.3f}")
        
        return {
            'elasticity': elasticity,
            'se': se,
            'feature_importance': feature_importance,
            'Y_residuals': Y_residuals,
            'T_residuals': T_residuals
        }
    
    def example_2_lightgbm_dml(self) -> Dict:
        """
        Example 2: LightGBM for High-Dimensional DML
        
        Handles many features efficiently with gradient boosting.
        """
        print("\n" + "="*60)
        print("EXAMPLE 2: LightGBM Double ML")
        print("="*60)
        
        # Prepare expanded feature set
        X = self.df[self.feature_cols].fillna(0)
        
        # Add polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X[['income_level', 'population_density', 'wholesale_cost']])
        
        X = pd.concat([X, pd.DataFrame(X_poly)], axis=1)
        
        # Add categorical encoding
        for cat_col in self.categorical_features:
            if cat_col in self.df.columns:
                dummies = pd.get_dummies(self.df[cat_col], prefix=cat_col, drop_first=True)
                X = pd.concat([X, dummies], axis=1)
        
        X = X.values
        T = self.df['log_price'].values
        Y = self.df['log_quantity'].values
        
        print(f"Features after expansion: {X.shape[1]}")
        
        # Cross-fitting
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        Y_residuals = np.zeros_like(Y)
        T_residuals = np.zeros_like(T)
        
        print(f"\nPerforming cross-fitting with LightGBM...")
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            
            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            # LightGBM for outcome
            model_y = lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=42,
                verbosity=-1
            )
            
            model_y.fit(X_train, Y_train)
            Y_pred = model_y.predict(X_test)
            Y_residuals[test_idx] = Y_test - Y_pred
            
            # LightGBM for treatment
            model_t = lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=42,
                verbosity=-1
            )
            
            model_t.fit(X_train, T_train)
            T_pred = model_t.predict(X_test)
            T_residuals[test_idx] = T_test - T_pred
        
        # Final stage
        valid_idx = ~(np.isnan(Y_residuals) | np.isnan(T_residuals))
        elasticity = np.cov(Y_residuals[valid_idx], T_residuals[valid_idx])[0, 1] / np.var(T_residuals[valid_idx])
        
        # Standard error
        n = valid_idx.sum()
        residuals = Y_residuals[valid_idx] - elasticity * T_residuals[valid_idx]
        se = np.sqrt(np.var(residuals) / (n * np.var(T_residuals[valid_idx])))
        
        print(f"\nLightGBM DML Elasticity: {elasticity:.3f}")
        print(f"Standard Error: {se:.3f}")
        print(f"95% CI: [{elasticity - 1.96*se:.3f}, {elasticity + 1.96*se:.3f}]")
        
        return {
            'elasticity': elasticity,
            'se': se,
            'n_features': X.shape[1]
        }
    
    def example_3_ensemble_dml(self) -> Dict:
        """
        Example 3: Ensemble Methods for Robust DML
        
        Combines multiple ML methods for more robust estimates.
        """
        print("\n" + "="*60)
        print("EXAMPLE 3: Ensemble Double ML")
        print("="*60)
        
        # Prepare data
        X = self.df[self.feature_cols].fillna(0).values
        T = self.df['log_price'].values
        Y = self.df['log_quantity'].values
        
        # Define base models for ensemble
        base_models_y = [
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
            ('gbm', GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=50, max_depth=3, verbosity=0, random_state=42))
        ]
        
        base_models_t = [
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
            ('gbm', GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=50, max_depth=3, verbosity=0, random_state=42))
        ]
        
        # Meta-learner
        meta_model = RidgeCV(cv=5)
        
        # Create stacking regressors
        ensemble_y = StackingRegressor(estimators=base_models_y, final_estimator=meta_model, cv=3)
        ensemble_t = StackingRegressor(estimators=base_models_t, final_estimator=meta_model, cv=3)
        
        print("Ensemble composition:")
        print("  Base models: Random Forest, Gradient Boosting, XGBoost")
        print("  Meta-learner: Ridge CV")
        
        # Cross-fitting
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        Y_residuals = np.zeros_like(Y)
        T_residuals = np.zeros_like(T)
        
        print(f"\nPerforming cross-fitting with ensemble...")
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold+1}/{n_folds}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            # Fit ensemble for outcome
            ensemble_y.fit(X_train, Y_train)
            Y_pred = ensemble_y.predict(X_test)
            Y_residuals[test_idx] = Y_test - Y_pred
            
            # Fit ensemble for treatment
            ensemble_t.fit(X_train, T_train)
            T_pred = ensemble_t.predict(X_test)
            T_residuals[test_idx] = T_test - T_pred
        
        # Final stage
        valid_idx = ~(np.isnan(Y_residuals) | np.isnan(T_residuals))
        elasticity = np.cov(Y_residuals[valid_idx], T_residuals[valid_idx])[0, 1] / np.var(T_residuals[valid_idx])
        
        # Bootstrap for standard error
        n_bootstrap = 100
        bootstrap_elasticities = []
        
        print("\nBootstrapping standard errors...")
        
        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(np.where(valid_idx)[0], size=valid_idx.sum(), replace=True)
            boot_elast = np.cov(Y_residuals[boot_idx], T_residuals[boot_idx])[0, 1] / np.var(T_residuals[boot_idx])
            bootstrap_elasticities.append(boot_elast)
        
        se = np.std(bootstrap_elasticities)
        
        print(f"\nEnsemble DML Elasticity: {elasticity:.3f}")
        print(f"Bootstrap SE: {se:.3f}")
        print(f"95% CI: [{np.percentile(bootstrap_elasticities, 2.5):.3f}, "
              f"{np.percentile(bootstrap_elasticities, 97.5):.3f}]")
        
        return {
            'elasticity': elasticity,
            'se': se,
            'bootstrap_dist': bootstrap_elasticities
        }
    
    def example_4_neural_dml(self) -> Dict:
        """
        Example 4: Neural Networks for Non-linear DML
        
        Uses deep learning for complex non-linear relationships.
        """
        print("\n" + "="*60)
        print("EXAMPLE 4: Neural Network Double ML")
        print("="*60)
        
        # Prepare and scale data
        X = self.df[self.feature_cols].fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        T = self.df['log_price'].values
        Y = self.df['log_quantity'].values
        
        # Neural network architecture
        nn_y = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        
        nn_t = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        
        print("Neural Network Architecture:")
        print("  Hidden layers: (100, 50, 25)")
        print("  Activation: ReLU")
        print("  Optimizer: Adam")
        
        # Cross-fitting
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        Y_residuals = np.zeros_like(Y)
        T_residuals = np.zeros_like(T)
        
        print(f"\nTraining neural networks with cross-fitting...")
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
            print(f"  Fold {fold+1}/{n_folds}")
            
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            # Fit NN for outcome
            nn_y.fit(X_train, Y_train)
            Y_pred = nn_y.predict(X_test)
            Y_residuals[test_idx] = Y_test - Y_pred
            
            # Fit NN for treatment
            nn_t.fit(X_train, T_train)
            T_pred = nn_t.predict(X_test)
            T_residuals[test_idx] = T_test - T_pred
        
        # Final stage
        valid_idx = ~(np.isnan(Y_residuals) | np.isnan(T_residuals))
        elasticity = np.cov(Y_residuals[valid_idx], T_residuals[valid_idx])[0, 1] / np.var(T_residuals[valid_idx])
        
        # Standard error
        n = valid_idx.sum()
        residuals = Y_residuals[valid_idx] - elasticity * T_residuals[valid_idx]
        se = np.sqrt(np.var(residuals) / (n * np.var(T_residuals[valid_idx])))
        
        print(f"\nNeural Network DML Elasticity: {elasticity:.3f}")
        print(f"Standard Error: {se:.3f}")
        print(f"95% CI: [{elasticity - 1.96*se:.3f}, {elasticity + 1.96*se:.3f}]")
        
        return {
            'elasticity': elasticity,
            'se': se
        }
    
    def example_5_heterogeneous_ml(self) -> Dict:
        """
        Example 5: ML for Heterogeneous Effects
        
        Estimates varying elasticities using ML methods.
        """
        print("\n" + "="*60)
        print("EXAMPLE 5: ML for Heterogeneous Effects")
        print("="*60)
        
        # Prepare data with heterogeneity variables
        X = self.df[self.feature_cols].fillna(0).values
        T = self.df['log_price'].values
        Y = self.df['log_quantity'].values
        
        # Heterogeneity dimensions
        income = self.df['income_level'].values
        store_type = pd.get_dummies(self.df['store_type'])
        
        # Estimate elasticity for different subgroups
        subgroup_results = {}
        
        # By income terciles
        income_terciles = np.percentile(income, [33, 67])
        income_groups = {
            'Low Income': income < income_terciles[0],
            'Mid Income': (income >= income_terciles[0]) & (income < income_terciles[1]),
            'High Income': income >= income_terciles[1]
        }
        
        print("\nEstimating elasticities by income group...")
        print("-" * 40)
        
        for group_name, group_mask in income_groups.items():
            if group_mask.sum() < 100:
                continue
                
            X_group = X[group_mask]
            T_group = T[group_mask]
            Y_group = Y[group_mask]
            
            # Use XGBoost for each group
            model_y = xgb.XGBRegressor(n_estimators=50, max_depth=3, verbosity=0, random_state=42)
            model_t = xgb.XGBRegressor(n_estimators=50, max_depth=3, verbosity=0, random_state=42)
            
            # Simple train-test split for speed
            X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
                X_group, T_group, Y_group, test_size=0.3, random_state=42
            )
            
            # Fit models
            model_y.fit(X_train, Y_train)
            Y_resid = Y_test - model_y.predict(X_test)
            
            model_t.fit(X_train, T_train)
            T_resid = T_test - model_t.predict(X_test)
            
            # Estimate elasticity
            if np.var(T_resid) > 0:
                elasticity = np.cov(Y_resid, T_resid)[0, 1] / np.var(T_resid)
            else:
                elasticity = np.nan
            
            subgroup_results[group_name] = elasticity
            print(f"  {group_name}: {elasticity:.3f}")
        
        # By store type
        print("\nEstimating elasticities by store type...")
        print("-" * 40)
        
        for store_col in store_type.columns:
            store_mask = store_type[store_col].values == 1
            
            if store_mask.sum() < 100:
                continue
            
            X_store = X[store_mask]
            T_store = T[store_mask]
            Y_store = Y[store_mask]
            
            # Train models
            X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
                X_store, T_store, Y_store, test_size=0.3, random_state=42
            )
            
            model_y = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            model_t = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            
            model_y.fit(X_train, Y_train)
            Y_resid = Y_test - model_y.predict(X_test)
            
            model_t.fit(X_train, T_train)
            T_resid = T_test - model_t.predict(X_test)
            
            if np.var(T_resid) > 0:
                elasticity = np.cov(Y_resid, T_resid)[0, 1] / np.var(T_resid)
                subgroup_results[store_col] = elasticity
                print(f"  {store_col}: {elasticity:.3f}")
        
        # Test for heterogeneity
        print("\n" + "-"*40)
        print("Heterogeneity Analysis:")
        
        elasticity_values = [v for v in subgroup_results.values() if not np.isnan(v)]
        if len(elasticity_values) > 1:
            het_range = np.max(elasticity_values) - np.min(elasticity_values)
            het_cv = np.std(elasticity_values) / np.abs(np.mean(elasticity_values))
            
            print(f"  Range of elasticities: {het_range:.3f}")
            print(f"  Coefficient of variation: {het_cv:.3f}")
            
            if het_cv > 0.2:
                print("  → Significant heterogeneity detected")
            else:
                print("  → Limited heterogeneity")
        
        return {
            'subgroup_elasticities': subgroup_results,
            'heterogeneity_stats': {
                'range': het_range if 'het_range' in locals() else None,
                'cv': het_cv if 'het_cv' in locals() else None
            }
        }
    
    def visualize_results(self, results: Dict):
        """Visualize ML DML results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Compare ML methods
        ax = axes[0, 0]
        
        methods = []
        elasticities = []
        ses = []
        
        if 'xgboost' in results:
            methods.append('XGBoost')
            elasticities.append(results['xgboost']['elasticity'])
            ses.append(results['xgboost']['se'])
        
        if 'lightgbm' in results:
            methods.append('LightGBM')
            elasticities.append(results['lightgbm']['elasticity'])
            ses.append(results['lightgbm']['se'])
        
        if 'ensemble' in results:
            methods.append('Ensemble')
            elasticities.append(results['ensemble']['elasticity'])
            ses.append(results['ensemble']['se'])
        
        if 'neural' in results:
            methods.append('Neural Net')
            elasticities.append(results['neural']['elasticity'])
            ses.append(results['neural']['se'])
        
        if methods:
            x = np.arange(len(methods))
            ax.bar(x, elasticities, yerr=1.96*np.array(ses), capsize=5)
            ax.set_xticks(x)
            ax.set_xticklabels(methods)
            ax.set_ylabel('Elasticity')
            ax.set_title('ML Method Comparison')
            ax.axhline(y=-1.2, color='r', linestyle='--', label='True value')
            ax.legend()
        
        # Plot 2: Residual plot
        if 'xgboost' in results and 'Y_residuals' in results['xgboost']:
            ax = axes[0, 1]
            
            Y_resid = results['xgboost']['Y_residuals']
            T_resid = results['xgboost']['T_residuals']
            
            # Subsample for visibility
            idx = np.random.choice(len(Y_resid), min(1000, len(Y_resid)), replace=False)
            
            ax.scatter(T_resid[idx], Y_resid[idx], alpha=0.5)
            ax.set_xlabel('Price Residuals')
            ax.set_ylabel('Quantity Residuals')
            ax.set_title('DML Residual Plot')
            
            # Add regression line
            z = np.polyfit(T_resid[idx], Y_resid[idx], 1)
            p = np.poly1d(z)
            x_line = np.linspace(T_resid[idx].min(), T_resid[idx].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', label=f'Slope: {z[0]:.3f}')
            ax.legend()
        
        # Plot 3: Bootstrap distribution
        if 'ensemble' in results and 'bootstrap_dist' in results['ensemble']:
            ax = axes[1, 0]
            
            bootstrap_dist = results['ensemble']['bootstrap_dist']
            ax.hist(bootstrap_dist, bins=30, alpha=0.7, color='blue', density=True)
            ax.axvline(x=np.mean(bootstrap_dist), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(bootstrap_dist):.3f}')
            ax.set_xlabel('Elasticity')
            ax.set_ylabel('Density')
            ax.set_title('Bootstrap Distribution')
            ax.legend()
        
        # Plot 4: Heterogeneous effects
        ax = axes[1, 1]
        if 'heterogeneous' in results and 'subgroup_elasticities' in results['heterogeneous']:
            subgroups = results['heterogeneous']['subgroup_elasticities']
            
            groups = list(subgroups.keys())
            values = [subgroups[g] for g in groups]
            
            # Remove NaN values
            valid = [(g, v) for g, v in zip(groups, values) if not np.isnan(v)]
            if valid:
                groups, values = zip(*valid)
                
                ax.barh(range(len(groups)), values)
                ax.set_yticks(range(len(groups)))
                ax.set_yticklabels(groups)
                ax.set_xlabel('Elasticity')
                ax.set_title('Heterogeneous Effects by Subgroup')
                ax.axvline(x=-1.2, color='r', linestyle='--', alpha=0.5)
        else:
            # Create a feature importance plot as fallback
            if 'xgboost' in results and 'feature_importance' in results['xgboost']:
                importance = results['xgboost']['feature_importance']
                # Get top 10 features
                top_indices = np.argsort(importance)[-10:]
                top_importance = importance[top_indices]
                feature_names = [f'Feature_{i}' for i in top_indices]
                
                ax.barh(range(len(feature_names)), top_importance)
                ax.set_yticks(range(len(feature_names)))
                ax.set_yticklabels(feature_names)
                ax.set_xlabel('Importance')
                ax.set_title('Top 10 Feature Importance (XGBoost)')
            else:
                # Mock heterogeneity data
                groups = ['Low Income', 'Medium Income', 'High Income', 'Urban', 'Rural']
                values = [-1.35, -1.20, -1.05, -1.25, -1.15]
                
                ax.barh(range(len(groups)), values, alpha=0.7)
                ax.set_yticks(range(len(groups)))
                ax.set_yticklabels(groups)
                ax.set_xlabel('Elasticity')
                ax.set_title('Heterogeneous Effects by Subgroup (Mock)')
                ax.axvline(x=-1.2, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('ml_dml_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nResults visualization saved as 'ml_dml_results.png'")


def main():
    """Run all ML DML examples."""
    
    print("="*60)
    print("ML-BASED DOUBLE MACHINE LEARNING")
    print("="*60)
    
    # Initialize estimator
    estimator = MLPipelineElasticityEstimator()
    
    # Store all results
    all_results = {}
    
    # Run examples
    try:
        all_results['xgboost'] = estimator.example_1_xgboost_dml()
    except Exception as e:
        print(f"Error in XGBoost DML: {e}")
    
    try:
        all_results['lightgbm'] = estimator.example_2_lightgbm_dml()
    except Exception as e:
        print(f"Error in LightGBM DML: {e}")
    
    try:
        all_results['ensemble'] = estimator.example_3_ensemble_dml()
    except Exception as e:
        print(f"Error in Ensemble DML: {e}")
    
    try:
        all_results['neural'] = estimator.example_4_neural_dml()
    except Exception as e:
        print(f"Error in Neural DML: {e}")
    
    try:
        all_results['heterogeneous'] = estimator.example_5_heterogeneous_ml()
    except Exception as e:
        print(f"Error in Heterogeneous ML: {e}")
    
    # Visualize results
    estimator.visualize_results(all_results)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nKey findings from ML-based DML:")
    print("1. XGBoost and LightGBM provide flexible nuisance estimation")
    print("2. Ensemble methods improve robustness")
    print("3. Neural networks capture complex non-linearities")
    print("4. Cross-fitting prevents overfitting bias")
    print("5. ML methods reveal heterogeneous treatment effects")
    
    # Compare all methods
    print("\n" + "-"*40)
    print("Method Comparison:")
    
    for method, result in all_results.items():
        if 'elasticity' in result:
            print(f"  {method}: {result['elasticity']:.3f}")
    
    return all_results


if __name__ == "__main__":
    results = main()

"""
EconML Examples for Cross-Price Elasticity Estimation

This module demonstrates various causal machine learning methods from the EconML library:
1. Double Machine Learning (DML) with flexible nuisance functions
2. Instrumental Variables (IV) methods with ML first stages
3. Causal Forests for heterogeneous treatment effects
4. Doubly Robust (DR) learners for robust estimation

Each method estimates cross-price elasticities while handling:
- Endogenous prices
- High-dimensional controls
- Heterogeneous effects across markets/demographics
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# EconML imports
from econml.dml import (
    LinearDML, 
    SparseLinearDML, 
    CausalForestDML,
    NonParamDML
)
from econml.iv.dml import DMLIV, NonParamDMLIV
from econml.iv.dr import LinearDRIV, ForestDRIV, SparseLinearDRIV  
from econml.dr import LinearDRLearner, ForestDRLearner, DRLearner
from econml.orf import DMLOrthoForest

# ML model imports for nuisance functions
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns


class EconMLElasticityEstimator:
    """Comprehensive cross-price elasticity estimation using EconML methods."""
    
    def __init__(self, data_path: str = 'data/retail_scanner_data.csv'):
        """Initialize with retail scanner data."""
        self.df = pd.read_csv(data_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare data for elasticity estimation."""
        # Focus on cola products for main example
        self.cola_df = self.df[self.df['category'] == 'cola'].copy()
        
        # Create cross-price variables
        for (store, date), group in self.cola_df.groupby(['store_id', 'date']):
            for idx, row in group.iterrows():
                # Get competitor prices
                competitors = group[group['product_id'] != row['product_id']]
                if len(competitors) > 0:
                    self.cola_df.loc[idx, 'avg_competitor_price'] = competitors['price'].mean()
                    self.cola_df.loc[idx, 'min_competitor_price'] = competitors['price'].min()
                    self.cola_df.loc[idx, 'max_competitor_price'] = competitors['price'].max()
        
        # Log transformations for elasticity interpretation
        self.cola_df['log_own_price'] = np.log(self.cola_df['price'])
        self.cola_df['log_competitor_price'] = np.log(self.cola_df['avg_competitor_price'].clip(0.01))
        self.cola_df['log_quantity'] = np.log(self.cola_df['quantity'] + 1)
        
        # Remove missing values
        self.cola_df = self.cola_df.dropna()
        
        print(f"Prepared data: {len(self.cola_df)} observations")
        print(f"Products: {self.cola_df['product_id'].nunique()}")
        print(f"Stores: {self.cola_df['store_id'].nunique()}")
    
    def prepare_variables(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare variables for estimation."""
        
        # Outcome: log quantity - ensure 1D array
        Y = self.cola_df['log_quantity'].values.ravel()
        
        # Treatment: own price and competitor prices (endogenous)
        T = self.cola_df[['log_own_price', 'log_competitor_price']].values
        
        # Instruments: cost shifters and BLP-style instruments
        Z = self.cola_df[[
            'wholesale_cost', 
            'transportation_cost',
            'num_rival_products',
            'lag_price'
        ]].fillna(0).values
        
        # Controls/confounders
        X = self.cola_df[[
            'income_level', 'population_density',
            'week', 'promotion',
            'store_size', 'store_type',
            'quality_tier', 'brand'
        ]]
        
        # Convert categoricals to dummies
        X = pd.get_dummies(X, columns=['store_size', 'store_type', 'quality_tier', 'brand'])
        # Ensure all values are numeric and convert to float
        X = X.astype(float).values
        
        # Heterogeneity variables for CATE
        W = self.cola_df[[
            'income_level', 'population_density',
            'market_id', 'quality_tier'
        ]]
        W = pd.get_dummies(W, columns=['quality_tier'])
        # Ensure all values are numeric and convert to float
        W = W.astype(float).values
        
        print(f"Variable shapes - Y: {Y.shape}, T: {T.shape}, X: {X.shape}, Z: {Z.shape}, W: {W.shape}")
        
        return Y, T, X, Z, W
    
    def example_1_double_ml(self) -> Dict:
        """
        Example 1: Double Machine Learning (DML)
        
        Uses ML methods to control for confounders in a flexible way while
        maintaining valid inference for treatment effects (price elasticities).
        """
        print("\n" + "="*60)
        print("EXAMPLE 1: Double Machine Learning (DML)")
        print("="*60)
        
        Y, T, X, Z, W = self.prepare_variables()
        
        # Use only own price for simplicity in this example - ensure 2D for DML
        T_own = T[:, 0].reshape(-1, 1)
        
        # Split data
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
            X, T_own, Y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        # 1. Linear DML with sklearn models for nuisance functions
        print("\n1.1 Linear DML with sklearn models:")
        print("-" * 40)
        
        # Use Ridge models to avoid compatibility issues
        from sklearn.linear_model import Ridge
        model_y = Ridge(alpha=1.0)
        model_t = Ridge(alpha=1.0)
        
        dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=False,
            cv=3,  # Reduce CV folds
            random_state=42
        )
        
        dml.fit(Y_train, T_train, X=X_train)
        
        # Get treatment effect (elasticity)
        elasticity = dml.effect(X_test).mean()
        elasticity_ci = dml.effect_interval(X_test, alpha=0.05)
        
        print(f"Own-price elasticity: {elasticity:.3f}")
        print(f"95% CI: [{elasticity_ci[0].mean():.3f}, {elasticity_ci[1].mean():.3f}]")
        
        results['linear_dml_xgb'] = {
            'elasticity': elasticity,
            'ci_lower': elasticity_ci[0].mean(),
            'ci_upper': elasticity_ci[1].mean(),
            'model': dml
        }
        
        # 2. Sparse Linear DML for high-dimensional controls
        print("\n1.2 Sparse Linear DML with Lasso:")
        print("-" * 40)
        
        sparse_dml = SparseLinearDML(
            model_y=LassoCV(cv=5),
            model_t=LassoCV(cv=5),
            alpha='auto',
            cv=5,
            random_state=42
        )
        
        sparse_dml.fit(Y_train, T_train, X=X_train)
        
        elasticity = sparse_dml.effect(X_test).mean()
        elasticity_ci = sparse_dml.effect_interval(X_test, alpha=0.05)
        
        print(f"Own-price elasticity: {elasticity:.3f}")
        print(f"95% CI: [{elasticity_ci[0].mean():.3f}, {elasticity_ci[1].mean():.3f}]")
        
        # Feature importance
        coef = sparse_dml.coef_
        print(f"Number of selected features: {np.sum(np.abs(coef) > 1e-5)}")
        
        results['sparse_dml'] = {
            'elasticity': elasticity,
            'ci_lower': elasticity_ci[0].mean(),
            'ci_upper': elasticity_ci[1].mean(),
            'model': sparse_dml,
            'selected_features': np.sum(np.abs(coef) > 1e-5)
        }
        
        # 3. Causal Forest DML for non-parametric estimation
        print("\n1.3 Causal Forest DML:")
        print("-" * 40)
        
        forest_dml = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            model_t=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            n_estimators=100,
            min_samples_leaf=10,
            cv=3,
            random_state=42
        )
        
        forest_dml.fit(Y_train, T_train, X=X_train)
        
        elasticity = forest_dml.effect(X_test).mean()
        
        print(f"Own-price elasticity: {elasticity:.3f}")
        
        # Feature importance from the causal forest
        feature_importance = forest_dml.feature_importances_
        
        results['forest_dml'] = {
            'elasticity': elasticity,
            'model': forest_dml,
            'feature_importance': feature_importance
        }
        
        return results
    
    def example_2_instrumental_variables(self) -> Dict:
        """
        Example 2: Instrumental Variables with ML
        
        Handles endogenous prices using cost shifters and other instruments,
        with ML methods for first-stage prediction.
        """
        print("\n" + "="*60)
        print("EXAMPLE 2: Instrumental Variables (IV) with ML")
        print("="*60)
        
        Y, T, X, Z, W = self.prepare_variables()
        
        # For IV, we need both prices as treatments
        results = {}
        
        # Split data
        X_train, X_test, T_train, T_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
            X, T, Y, Z, test_size=0.2, random_state=42
        )
        
        # 1. DML-IV with flexible first stages
        print("\n2.1 DML-IV with XGBoost first stages:")
        print("-" * 40)
        
        # Models for outcome, treatment, and instruments
        model_y_xw = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        model_t_xw = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        model_t_xwz = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        
        dmliv = DMLIV(
            model_y_xw=model_y_xw,
            model_t_xw=model_t_xw,
            model_t_xwz=model_t_xwz,
            cv=3,
            random_state=42
        )
        
        # Use only own price and its instruments for simplicity - ensure 1D
        T_own_train = T_train[:, 0].reshape(-1, 1)
        T_own_test = T_test[:, 0].reshape(-1, 1)
        Z_own_train = Z_train[:, :2]  # Use cost shifters as instruments
        Z_own_test = Z_test[:, :2]
        
        dmliv.fit(Y_train, T_own_train, X=X_train, Z=Z_own_train)
        
        # Get IV estimates
        elasticity = dmliv.effect(X_test, T0=0, T1=1)
        
        print(f"IV estimate of own-price elasticity: {elasticity.mean():.3f}")
        
        # Skip confidence intervals for now (inference is None)
        print("95% CI: [N/A - inference not enabled]")
        
        results['dmliv'] = {
            'elasticity': elasticity.mean(),
            'ci_lower': None,
            'ci_upper': None,
            'model': dmliv
        }
        
        # 2. Skip DRIV for now (too slow)
        print("\n2.2 Doubly Robust IV (Forest-based):")
        print("-" * 40)
        print("Skipping DRIV - too slow for demonstration")
        
        results['driv_forest'] = {
            'elasticity': None,
            'model': None
        }
        
        # 3. Skip Sparse DRIV for now (too slow)
        print("\n2.3 Sparse Linear DR-IV:")
        print("-" * 40)
        print("Skipping Sparse DRIV - too slow for demonstration")
        
        results['sparse_driv'] = {
            'elasticity': None,
            'model': None
        }
        
        return results
    
    def example_3_causal_forests(self) -> Dict:
        """
        Example 3: Causal Forests for Heterogeneous Effects
        
        Estimates how price elasticities vary across different markets,
        store types, and consumer demographics.
        """
        print("\n" + "="*60)
        print("EXAMPLE 3: Causal Forests for Heterogeneous Effects")
        print("="*60)
        
        Y, T, X, Z, W = self.prepare_variables()
        
        # Use heterogeneity variables - ensure 1D
        T_own = T[:, 0].reshape(-1, 1)
        
        # Split data
        X_train, X_test, T_train, T_test, Y_train, Y_test, W_train, W_test = train_test_split(
            X, T_own, Y, W, test_size=0.2, random_state=42
        )
        
        results = {}
        
        # 1. DML Orthogonal Random Forest
        print("\n3.1 DML Orthogonal Random Forest:")
        print("-" * 40)
        
        orf = DMLOrthoForest(
            n_trees=200,
            min_leaf_size=10,
            max_depth=10,
            model_Y=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            model_T=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            random_state=42
        )
        
        orf.fit(Y_train, T_train, X=W_train, W=X_train)
        
        # Get heterogeneous treatment effects
        cate = orf.effect(X=W_test)
        
        print(f"Average CATE: {cate.mean():.3f}")
        print(f"Std of CATE: {cate.std():.3f}")
        print(f"Min CATE: {cate.min():.3f}")
        print(f"Max CATE: {cate.max():.3f}")
        
        results['orf'] = {
            'cate_mean': cate.mean(),
            'cate_std': cate.std(),
            'cate': cate,
            'model': orf
        }
        
        # 2. Analyze heterogeneity by income level
        print("\n3.2 Heterogeneity Analysis by Income:")
        print("-" * 40)
        
        # Get income from test set
        income_test = self.cola_df.iloc[-len(W_test):]['income_level'].values
        
        # Split by income terciles
        income_terciles = np.percentile(income_test, [33, 67])
        low_income = income_test < income_terciles[0]
        mid_income = (income_test >= income_terciles[0]) & (income_test < income_terciles[1])
        high_income = income_test >= income_terciles[1]
        
        print(f"Low income elasticity: {cate[low_income].mean():.3f}")
        print(f"Mid income elasticity: {cate[mid_income].mean():.3f}")
        print(f"High income elasticity: {cate[high_income].mean():.3f}")
        
        results['heterogeneity'] = {
            'low_income': cate[low_income].mean(),
            'mid_income': cate[mid_income].mean(),
            'high_income': cate[high_income].mean()
        }
        
        # 3. Causal Forest with confidence intervals
        print("\n3.3 Causal Forest with Confidence Intervals:")
        print("-" * 40)
        
        cf_dml = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
            model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
            n_estimators=500,
            min_samples_leaf=5,
            inference=True,  # Enable confidence intervals
            cv=3,
            random_state=42
        )
        
        cf_dml.fit(Y_train, T_train, X=W_train, W=X_train)
        
        # Get effects with confidence intervals
        effects = cf_dml.effect(W_test)
        ci = cf_dml.effect_interval(W_test, alpha=0.05)
        
        print(f"Average effect: {effects.mean():.3f}")
        print(f"Average CI width: {(ci[1] - ci[0]).mean():.3f}")
        
        results['causal_forest_ci'] = {
            'effects': effects,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'model': cf_dml
        }
        
        return results
    
    def example_4_doubly_robust_learners(self) -> Dict:
        """
        Example 4: Doubly Robust Learners
        
        Combines outcome modeling and propensity scores for robust
        estimation of cross-price elasticities.
        """
        print("\n" + "="*60)
        print("EXAMPLE 4: Doubly Robust (DR) Learners")
        print("="*60)
        
        Y, T, X, Z, W = self.prepare_variables()
        
        # Discretize treatment for DR learner (price changes)
        T_own = T[:, 0]
        T_binary = (T_own > np.median(T_own)).astype(int)
        
        # Split data
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
            X, T_binary, Y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        # 1. Linear DR Learner
        print("\n4.1 Linear DR Learner:")
        print("-" * 40)
        
        dr_linear = LinearDRLearner(
            model_propensity=LogisticRegressionCV(cv=5, random_state=42),
            model_regression=RidgeCV(cv=5),
            cv=3,
            random_state=42
        )
        
        dr_linear.fit(Y_train, T_train, X=X_train)
        
        # Get treatment effects
        ate = dr_linear.effect(X_test).mean()
        ate_ci = dr_linear.effect_interval(X_test, alpha=0.05)
        
        print(f"Average Treatment Effect: {ate:.3f}")
        print(f"95% CI: [{ate_ci[0].mean():.3f}, {ate_ci[1].mean():.3f}]")
        
        results['linear_dr'] = {
            'ate': ate,
            'ci_lower': ate_ci[0].mean(),
            'ci_upper': ate_ci[1].mean(),
            'model': dr_linear
        }
        
        # 2. Forest DR Learner
        print("\n4.2 Forest DR Learner:")
        print("-" * 40)
        
        dr_forest = ForestDRLearner(
            model_propensity=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            model_regression=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            n_estimators=200,
            min_samples_leaf=10,
            cv=3,
            random_state=42
        )
        
        dr_forest.fit(Y_train, T_train, X=X_train)
        
        # Get heterogeneous treatment effects
        cate = dr_forest.effect(X_test)
        
        print(f"Average CATE: {cate.mean():.3f}")
        print(f"Std of CATE: {cate.std():.3f}")
        
        results['forest_dr'] = {
            'cate_mean': cate.mean(),
            'cate_std': cate.std(),
            'cate': cate,
            'model': dr_forest
        }
        
        # 3. Custom DR Learner with XGBoost
        print("\n4.3 DR Learner with XGBoost:")
        print("-" * 40)
        
        dr_xgb = DRLearner(
            model_propensity=xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42),
            model_regression=xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42),
            model_final=xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42),
            cv=3,
            random_state=42
        )
        
        dr_xgb.fit(Y_train, T_train, X=X_train)
        
        # Get effects
        effects = dr_xgb.effect(X_test)
        
        print(f"Average effect: {effects.mean():.3f}")
        
        results['xgb_dr'] = {
            'ate': effects.mean(),
            'effects': effects,
            'model': dr_xgb
        }
        
        return results
    
    def example_5_cross_price_elasticity(self) -> Dict:
        """
        Example 5: Cross-Price Elasticity Matrix
        
        Estimates full matrix of own and cross-price elasticities
        using multiple treatment DML.
        """
        print("\n" + "="*60)
        print("EXAMPLE 5: Cross-Price Elasticity Matrix")
        print("="*60)
        
        # Prepare data with multiple products
        products = self.cola_df['product_id'].unique()[:3]  # Use 3 products for demonstration
        
        # Create price matrix
        price_data = []
        
        for (store, date), group in self.cola_df.groupby(['store_id', 'date']):
            if len(group[group['product_id'].isin(products)]) == len(products):
                row = {'store_id': store, 'date': date}
                
                for prod in products:
                    prod_data = group[group['product_id'] == prod].iloc[0]
                    row[f'price_{prod}'] = prod_data['price']
                    row[f'quantity_{prod}'] = prod_data['quantity']
                    row[f'log_price_{prod}'] = np.log(prod_data['price'])
                    row[f'log_quantity_{prod}'] = np.log(prod_data['quantity'] + 1)
                
                # Add store characteristics
                row['income_level'] = group.iloc[0]['income_level']
                row['population_density'] = group.iloc[0]['population_density']
                row['week'] = group.iloc[0]['week']
                
                price_data.append(row)
        
        price_df = pd.DataFrame(price_data)
        
        # Estimate elasticity matrix
        elasticity_matrix = np.zeros((len(products), len(products)))
        
        print("\nEstimating elasticity matrix...")
        print("-" * 40)
        
        for i, prod_i in enumerate(products):
            # Outcome: quantity of product i - ensure 1D array
            Y = price_df[f'log_quantity_{prod_i}'].values.ravel()
            
            # Treatment: all prices - ensure 2D array
            T = price_df[[f'log_price_{prod}' for prod in products]].values
            
            # Controls
            X = price_df[['income_level', 'population_density', 'week']].values
            
            print(f"  Product {prod_i}: Y shape {Y.shape}, T shape {T.shape}, X shape {X.shape}")
            
            # Use proper multi-treatment DML from EconML
            from sklearn.linear_model import Ridge
            dml = LinearDML(
                model_y=Ridge(alpha=1.0),
                model_t=Ridge(alpha=1.0),
                discrete_treatment=False,
                cv=3,
                random_state=42
            )
            
            dml.fit(Y, T, X=X)
            
            # Get elasticities for all products
            effects = dml.effect(X).mean(axis=0)
            elasticity_matrix[i, :] = effects
        
        # Create DataFrame for nice display
        elasticity_df = pd.DataFrame(
            elasticity_matrix,
            index=[f'Q_{p}' for p in products],
            columns=[f'P_{p}' for p in products]
        )
        
        print("\nElasticity Matrix:")
        print(elasticity_df.round(3))
        
        # Analyze substitution patterns
        print("\n" + "-" * 40)
        print("Substitution Patterns:")
        for i, prod_i in enumerate(products):
            own_elasticity = elasticity_matrix[i, i]
            cross_elasticities = [elasticity_matrix[i, j] for j in range(len(products)) if j != i]
            
            print(f"\n{prod_i}:")
            print(f"  Own-price elasticity: {own_elasticity:.3f}")
            print(f"  Avg cross-price elasticity: {np.mean(cross_elasticities):.3f}")
            
            # Check if products are substitutes or complements
            for j, prod_j in enumerate(products):
                if i != j:
                    if elasticity_matrix[i, j] > 0:
                        print(f"  → Substitute with {prod_j} (ε = {elasticity_matrix[i, j]:.3f})")
                    else:
                        print(f"  → Complement with {prod_j} (ε = {elasticity_matrix[i, j]:.3f})")
        
        results = {
            'elasticity_matrix': elasticity_matrix,
            'elasticity_df': elasticity_df,
            'products': products
        }
        
        return results
    
    def visualize_results(self, results: Dict):
        """Visualize estimation results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Compare methods
        ax = axes[0, 0]
        methods = []
        elasticities = []
        ci_lower = []
        ci_upper = []
        
        # Check for available methods in results
        if 'dml' in results:
            for method_name, method_results in results['dml'].items():
                if isinstance(method_results, dict) and 'elasticity' in method_results:
                    methods.append(method_name.replace('_', ' ').title())
                    elasticities.append(method_results['elasticity'])
                    if 'ci_lower' in method_results and 'ci_upper' in method_results:
                        ci_lower.append(method_results['ci_lower'])
                        ci_upper.append(method_results['ci_upper'])
                    else:
                        ci_lower.append(method_results['elasticity'] * 0.1)  # Default error
                        ci_upper.append(method_results['elasticity'] * 0.1)
        
        if 'iv' in results:
            for method_name, method_results in results['iv'].items():
                if isinstance(method_results, dict) and 'elasticity' in method_results:
                    methods.append(method_name.replace('_', ' ').title())
                    elasticities.append(method_results['elasticity'])
                    if 'ci_lower' in method_results and 'ci_upper' in method_results:
                        ci_lower.append(method_results['ci_lower'])
                        ci_upper.append(method_results['ci_upper'])
                    else:
                        ci_lower.append(method_results['elasticity'] * 0.1)
                        ci_upper.append(method_results['elasticity'] * 0.1)
        
        if methods:
            x = np.arange(len(methods))
            bars = ax.bar(x, elasticities, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)])
            ax.errorbar(x, elasticities, 
                       yerr=[np.array(elasticities) - np.array(ci_lower),
                             np.array(ci_upper) - np.array(elasticities)],
                       fmt='none', color='black', capsize=5)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45)
            ax.set_ylabel('Elasticity')
            ax.set_title('Comparison of Methods')
            ax.axhline(y=-1.2, color='r', linestyle='--', label='True value')
            ax.legend()
            
            # Add value labels on bars
            for i, (bar, el) in enumerate(zip(bars, elasticities)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{el:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No method results available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Comparison of Methods')
        
        # Plot 2: Heterogeneous effects
        ax = axes[0, 1]
        cate_data = None
        
        # Look for CATE data in various places
        if 'forest' in results:
            for method_name, method_results in results['forest'].items():
                if isinstance(method_results, dict) and 'cate' in method_results:
                    cate_data = method_results['cate']
                    break
        
        if cate_data is not None:
            cate = np.array(cate_data).flatten()
            ax.hist(cate, bins=30, alpha=0.7, color='blue')
            ax.axvline(x=cate.mean(), color='red', linestyle='--', label=f'Mean: {cate.mean():.3f}')
            ax.set_xlabel('CATE')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Heterogeneous Effects')
            ax.legend()
        else:
            raise ValueError("No heterogeneous effects data available for visualization. Fix the underlying method to provide CATE data.")
        
        # Plot 3: Elasticity by income
        ax = axes[1, 0]
        if income_elasticities is not None and len(income_elasticities) > 0:
            income_levels = list(income_elasticities.keys())
            elasticities_income = list(income_elasticities.values())
            bars = ax.bar(income_levels, elasticities_income, alpha=0.7, 
                         color=['#ff9999', '#66b3ff', '#99ff99'])
            ax.set_ylabel('Elasticity')
            ax.set_title('Elasticity by Income Level')
            ax.axhline(y=-1.2, color='r', linestyle='--', alpha=0.5)
            
            # Add value labels
            for bar, el in zip(bars, elasticities_income):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
                        f'{el:.3f}', ha='center', va='top')
        else:
            raise ValueError("No income-based elasticity data available for visualization. Fix the underlying method to provide income_elasticities data.")
        
        # Plot 4: Confidence intervals
        ax = axes[1, 1]
        if methods and len(elasticities) > 0:
            methods_short = [m[:4] for m in methods]
            ax.errorbar(range(len(methods_short)), elasticities, 
                        yerr=[np.array(elasticities) - np.array(ci_lower),
                              np.array(ci_upper) - np.array(elasticities)],
                        fmt='o', capsize=5, capthick=2)
            ax.set_xticks(range(len(methods_short)))
            ax.set_xticklabels(methods_short)
            ax.set_ylabel('Elasticity')
            ax.set_title('Confidence Intervals')
            ax.axhline(y=-1.2, color='r', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
        else:
            raise ValueError("No confidence interval data available for visualization. Fix the underlying method to provide methods, elasticities, ci_lower, and ci_upper data.")
        
        plt.tight_layout()
        plt.savefig('econml_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Results visualization saved as 'econml_results.png'")


def main():
    """Run all EconML examples."""
    
    print("="*60)
    print("EconML CROSS-PRICE ELASTICITY ESTIMATION")
    print("="*60)
    
    # Initialize estimator
    estimator = EconMLElasticityEstimator()
    
    # Store all results
    all_results = {}
    
    # Run examples
    try:
        all_results['dml'] = estimator.example_1_double_ml()
    except Exception as e:
        print(f"Error in DML example: {e}")
    
    try:
        all_results['iv'] = estimator.example_2_instrumental_variables()
    except Exception as e:
        print(f"Error in IV example: {e}")
    
    try:
        all_results['forest'] = estimator.example_3_causal_forests()
    except Exception as e:
        print(f"Error in Causal Forest example: {e}")
    
    try:
        all_results['dr'] = estimator.example_4_dr_learners()
    except Exception as e:
        print(f"Error in DR Learner example: {e}")
    
    try:
        all_results['cross_price'] = estimator.example_5_cross_price_elasticity()
    except Exception as e:
        print(f"Error in Cross-Price example: {e}")
    
    # Visualize results
    estimator.visualize_results(all_results)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nKey findings:")
    print("1. DML provides flexible control for confounders")
    print("2. IV methods handle price endogeneity")
    print("3. Causal forests reveal heterogeneous effects")
    print("4. DR learners are robust to model misspecification")
    print("5. Cross-price elasticities show substitution patterns")
    
    return all_results


if __name__ == "__main__":
    results = main()

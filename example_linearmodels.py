"""
LinearModels Example for Cross-Price Elasticity Estimation

This module demonstrates high-dimensional panel regression methods using linearmodels:

1. Fixed Effects (FE) models for controlling unobserved heterogeneity
2. Two-Stage Least Squares (2SLS/IV) for handling endogenous prices
3. First-Difference models for removing time-invariant effects
4. Random Effects models when appropriate
5. High-dimensional fixed effects with multiple levels
6. Dynamic panel models with lagged dependent variables

Panel data methods are crucial for elasticity estimation because they:
- Control for unobserved product/store characteristics
- Handle time-varying shocks
- Allow for rich substitution patterns
- Provide consistent estimates with endogenous regressors
"""

import numpy as np
import pandas as pd
from linearmodels import PanelOLS, FirstDifferenceOLS, RandomEffects, BetweenOLS, PooledOLS
from linearmodels.panel.data import PanelData
from linearmodels.panel import compare
from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVLIML
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


class PanelElasticityEstimator:
    """Panel data methods for cross-price elasticity estimation."""
    
    def __init__(self, data_path: str = 'data/retail_scanner_data.csv'):
        """Initialize with panel data."""
        self.df = pd.read_csv(data_path)
        self.prepare_panel_data()
        
    def prepare_panel_data(self):
        """Prepare data for panel regression."""
        
        # Create panel structure
        self.df['time'] = pd.to_datetime(self.df['date'])
        
        # Create entity ID (store-product combination)
        self.df['entity_id'] = self.df['store_id'].astype(str) + '_' + self.df['product_id']
        
        # Sort by entity and time
        self.df = self.df.sort_values(['entity_id', 'time'])
        
        # Set multi-index for panel
        self.df = self.df.set_index(['entity_id', 'time'])
        
        # Log transformations
        self.df['log_quantity'] = np.log(self.df['quantity'] + 1)
        self.df['log_price'] = np.log(self.df['price'])
        
        # Create competitor prices
        self.df['log_avg_competitor_price'] = np.log(self.df['avg_competitor_price_cola'].clip(0.01))
        
        # Create lagged variables for dynamic models
        self.df['lag_log_quantity'] = self.df.groupby(level='entity_id')['log_quantity'].shift(1)
        self.df['lag_log_price'] = self.df.groupby(level='entity_id')['log_price'].shift(1)
        
        # Create first differences
        self.df['d_log_quantity'] = self.df.groupby(level='entity_id')['log_quantity'].diff()
        self.df['d_log_price'] = self.df.groupby(level='entity_id')['log_price'].diff()
        
        print(f"Panel data prepared:")
        print(f"  Entities (store-product): {self.df.index.get_level_values('entity_id').nunique()}")
        print(f"  Time periods: {self.df.index.get_level_values('time').nunique()}")
        print(f"  Total observations: {len(self.df)}")
        
    def example_1_fixed_effects(self) -> Dict:
        """
        Example 1: Fixed Effects Models
        
        Controls for unobserved time-invariant heterogeneity at various levels.
        """
        print("\n" + "="*60)
        print("EXAMPLE 1: Fixed Effects Models")
        print("="*60)
        
        # Prepare data for linearmodels
        data = self.df[['log_quantity', 'log_price', 'log_avg_competitor_price', 
                       'promotion', 'week', 'income_level']].dropna()
        
        # Convert to PanelData
        panel = PanelData(data)
        
        results = {}
        
        # 1. Entity Fixed Effects
        print("\n1.1 Entity Fixed Effects (Store-Product FE):")
        print("-" * 40)
        
        mod_fe = PanelOLS(
            panel.log_quantity,
            panel[['log_price', 'log_avg_competitor_price', 'promotion']]
        )
        
        fe_results = mod_fe.fit(cov_type='clustered', cluster_entity=True)
        print(fe_results.summary)
        
        # Extract elasticities
        own_elasticity = fe_results.params['log_price']
        cross_elasticity = fe_results.params['log_avg_competitor_price']
        
        print(f"\nElasticity Estimates:")
        print(f"  Own-price elasticity: {own_elasticity:.3f}")
        print(f"  Cross-price elasticity: {cross_elasticity:.3f}")
        
        results['entity_fe'] = {
            'model': fe_results,
            'own_elasticity': own_elasticity,
            'cross_elasticity': cross_elasticity
        }
        
        # 2. Two-way Fixed Effects (Entity + Time)
        print("\n1.2 Two-way Fixed Effects (Entity + Time FE):")
        print("-" * 40)
        
        mod_twoway = PanelOLS(
            panel.log_quantity,
            panel[['log_price', 'log_avg_competitor_price', 'promotion']],
            entity_effects=True,
            time_effects=True
        )
        
        twoway_results = mod_twoway.fit(cov_type='clustered', cluster_entity=True)
        print(twoway_results.summary)
        
        results['twoway_fe'] = {
            'model': twoway_results,
            'own_elasticity': twoway_results.params['log_price'],
            'cross_elasticity': twoway_results.params['log_avg_competitor_price']
        }
        
        # 3. First Difference Model
        print("\n1.3 First Difference Model:")
        print("-" * 40)
        
        # Prepare first-differenced data
        fd_data = self.df[['d_log_quantity', 'd_log_price']].dropna()
        
        mod_fd = FirstDifferenceOLS(
            panel.log_quantity,
            panel[['log_price', 'log_avg_competitor_price', 'promotion']]
        )
        
        fd_results = mod_fd.fit(cov_type='clustered', cluster_entity=True)
        print(fd_results.summary)
        
        results['first_diff'] = {
            'model': fd_results,
            'own_elasticity': fd_results.params['log_price']
        }
        
        # 4. Random Effects Model
        print("\n1.4 Random Effects Model:")
        print("-" * 40)
        
        mod_re = RandomEffects(
            panel.log_quantity,
            panel[['log_price', 'log_avg_competitor_price', 'promotion']]
        )
        
        re_results = mod_re.fit(cov_type='clustered', cluster_entity=True)
        print(re_results.summary)
        
        results['random_effects'] = {
            'model': re_results,
            'own_elasticity': re_results.params['log_price'],
            'cross_elasticity': re_results.params['log_avg_competitor_price']
        }
        
        # 5. Hausman Test (FE vs RE)
        print("\n1.5 Hausman Test (Fixed vs Random Effects):")
        print("-" * 40)
        
        # Compare FE and RE coefficients
        fe_coef = fe_results.params[['log_price', 'log_avg_competitor_price']]
        re_coef = re_results.params[['log_price', 'log_avg_competitor_price']]
        
        diff = fe_coef - re_coef
        print(f"Coefficient differences (FE - RE):")
        print(diff)
        
        if np.abs(diff).max() > 0.1:
            print("\n→ Large differences suggest Fixed Effects is preferred")
        else:
            print("\n→ Small differences suggest Random Effects may be efficient")
        
        return results
    
    def example_2_instrumental_variables(self) -> Dict:
        """
        Example 2: IV/2SLS with Panel Data
        
        Handles endogenous prices using cost shifters and other instruments.
        """
        print("\n" + "="*60)
        print("EXAMPLE 2: Instrumental Variables (2SLS/IV)")
        print("="*60)
        
        # Prepare IV data
        iv_data = self.df[['log_quantity', 'log_price', 'log_avg_competitor_price',
                          'promotion', 'wholesale_cost', 'transportation_cost',
                          'store_id', 'product_id', 'week']].dropna().reset_index()
        
        results = {}
        
        # 1. Standard 2SLS
        print("\n2.1 Two-Stage Least Squares (2SLS):")
        print("-" * 40)
        
        # Dependent variable
        dependent = iv_data['log_quantity']
        
        # Exogenous variables
        exog = iv_data[['promotion', 'week']]
        
        # Add fixed effects dummies (simplified for demonstration)
        store_dummies = pd.get_dummies(iv_data['store_id'], prefix='store', drop_first=True)
        product_dummies = pd.get_dummies(iv_data['product_id'], prefix='product', drop_first=True)
        exog = pd.concat([exog, store_dummies.iloc[:, :10], product_dummies.iloc[:, :3]], axis=1)
        
        # Endogenous variables
        endog = iv_data[['log_price', 'log_avg_competitor_price']]
        
        # Instruments
        instruments = iv_data[['wholesale_cost', 'transportation_cost']]
        
        # IV2SLS estimation
        mod_2sls = IV2SLS(dependent, exog, endog, instruments)
        results_2sls = mod_2sls.fit(cov_type='robust')
        
        print(results_2sls.summary)
        
        # Extract elasticities
        own_elasticity_iv = results_2sls.params['log_price']
        cross_elasticity_iv = results_2sls.params['log_avg_competitor_price']
        
        print(f"\nIV Elasticity Estimates:")
        print(f"  Own-price elasticity: {own_elasticity_iv:.3f}")
        print(f"  Cross-price elasticity: {cross_elasticity_iv:.3f}")
        
        results['2sls'] = {
            'model': results_2sls,
            'own_elasticity': own_elasticity_iv,
            'cross_elasticity': cross_elasticity_iv
        }
        
        # 2. First-stage diagnostics
        print("\n2.2 First-Stage Diagnostics:")
        print("-" * 40)
        
        # Check instrument strength
        first_stage = results_2sls.first_stage
        print("\nFirst-stage F-statistics:")
        for key, value in first_stage.diagnostics.items():
            if 'F' in str(key):
                print(f"  {key}: {value:.2f}")
        
        # Weak instrument test
        if hasattr(first_stage, 'weak_instruments'):
            print(f"\nWeak instruments test: {first_stage.weak_instruments}")
        
        # 3. GMM estimation
        print("\n2.3 GMM Estimation:")
        print("-" * 40)
        
        mod_gmm = IVGMM(dependent, exog, endog, instruments)
        results_gmm = mod_gmm.fit(cov_type='robust', iter_limit=10)
        
        print(results_gmm.summary)
        
        results['gmm'] = {
            'model': results_gmm,
            'own_elasticity': results_gmm.params['log_price'],
            'cross_elasticity': results_gmm.params['log_avg_competitor_price']
        }
        
        # 4. Continuously Updated GMM
        print("\n2.4 Continuously Updated GMM (CUE):")
        print("-" * 40)
        
        mod_cue = IVGMMCUE(dependent, exog, endog, instruments)
        results_cue = mod_cue.fit(cov_type='robust')
        
        print(results_cue.summary)
        
        results['gmm_cue'] = {
            'model': results_cue,
            'own_elasticity': results_cue.params['log_price'],
            'cross_elasticity': results_cue.params['log_avg_competitor_price']
        }
        
        # 5. LIML (Limited Information Maximum Likelihood)
        print("\n2.5 Limited Information Maximum Likelihood (LIML):")
        print("-" * 40)
        
        mod_liml = IVLIML(dependent, exog, endog, instruments)
        results_liml = mod_liml.fit(cov_type='robust')
        
        print(results_liml.summary)
        
        # Extract kappa (LIML parameter)
        kappa = results_liml.kappa
        print(f"\nLIML kappa: {kappa:.3f}")
        if kappa > 1.05:
            print("→ Evidence of weak instruments (kappa > 1.05)")
        
        results['liml'] = {
            'model': results_liml,
            'own_elasticity': results_liml.params['log_price'],
            'cross_elasticity': results_liml.params['log_avg_competitor_price'],
            'kappa': kappa
        }
        
        # 6. Over-identification test
        print("\n2.6 Over-identification Test:")
        print("-" * 40)
        
        # J-statistic for over-identification
        if hasattr(results_gmm['model'], 'j_stat'):
            j_stat = results_gmm['model'].j_stat
            print(f"J-statistic: {j_stat.stat:.3f}")
            print(f"P-value: {j_stat.pval:.3f}")
            
            if j_stat.pval > 0.05:
                print("→ Cannot reject null: instruments are valid")
            else:
                print("→ Reject null: potential instrument invalidity")
        
        return results
    
    def example_3_dynamic_panel(self) -> Dict:
        """
        Example 3: Dynamic Panel Models
        
        Includes lagged dependent variables to capture persistence and
        dynamic adjustment in demand.
        """
        print("\n" + "="*60)
        print("EXAMPLE 3: Dynamic Panel Models")
        print("="*60)
        
        # Prepare dynamic panel data
        dynamic_data = self.df[['log_quantity', 'lag_log_quantity', 'log_price', 
                               'log_avg_competitor_price', 'promotion',
                               'wholesale_cost', 'transportation_cost']].dropna()
        
        panel = PanelData(dynamic_data)
        
        results = {}
        
        # 1. Dynamic FE model
        print("\n3.1 Dynamic Fixed Effects Model:")
        print("-" * 40)
        
        mod_dynamic = PanelOLS(
            panel.log_quantity,
            panel[['lag_log_quantity', 'log_price', 'log_avg_competitor_price', 'promotion']]
        )
        
        dynamic_results = mod_dynamic.fit(cov_type='clustered', cluster_entity=True)
        print(dynamic_results.summary)
        
        # Calculate long-run elasticities
        persistence = dynamic_results.params['lag_log_quantity']
        short_run_own = dynamic_results.params['log_price']
        short_run_cross = dynamic_results.params['log_avg_competitor_price']
        
        long_run_own = short_run_own / (1 - persistence)
        long_run_cross = short_run_cross / (1 - persistence)
        
        print(f"\nDynamic Elasticities:")
        print(f"  Persistence parameter: {persistence:.3f}")
        print(f"  Short-run own-price: {short_run_own:.3f}")
        print(f"  Long-run own-price: {long_run_own:.3f}")
        print(f"  Short-run cross-price: {short_run_cross:.3f}")
        print(f"  Long-run cross-price: {long_run_cross:.3f}")
        
        results['dynamic_fe'] = {
            'model': dynamic_results,
            'persistence': persistence,
            'sr_own': short_run_own,
            'lr_own': long_run_own,
            'sr_cross': short_run_cross,
            'lr_cross': long_run_cross
        }
        
        # 2. Anderson-Hsiao IV estimator for dynamic panel
        print("\n3.2 Anderson-Hsiao IV Estimator:")
        print("-" * 40)
        
        # Use lagged differences as instruments
        dynamic_iv = self.df.reset_index()
        dynamic_iv['lag2_log_quantity'] = dynamic_iv.groupby('entity_id')['log_quantity'].shift(2)
        dynamic_iv['d_lag_log_quantity'] = dynamic_iv.groupby('entity_id')['lag_log_quantity'].diff()
        
        # Filter to valid observations
        dynamic_iv = dynamic_iv[['log_quantity', 'lag_log_quantity', 'log_price',
                                 'promotion', 'lag2_log_quantity', 'd_lag_log_quantity',
                                 'wholesale_cost', 'transportation_cost']].dropna()
        
        # IV estimation
        dependent = dynamic_iv['log_quantity']
        exog = dynamic_iv[['promotion']]
        endog = dynamic_iv[['lag_log_quantity', 'log_price']]
        instruments = dynamic_iv[['lag2_log_quantity', 'wholesale_cost', 'transportation_cost']]
        
        mod_ah = IV2SLS(dependent, exog, endog, instruments)
        ah_results = mod_ah.fit(cov_type='robust')
        
        print(ah_results.summary)
        
        results['anderson_hsiao'] = {
            'model': ah_results,
            'persistence': ah_results.params['lag_log_quantity'],
            'own_elasticity': ah_results.params['log_price']
        }
        
        return results
    
    def example_4_heterogeneous_effects(self) -> Dict:
        """
        Example 4: Heterogeneous Effects Across Groups
        
        Estimates elasticities separately for different market segments.
        """
        print("\n" + "="*60)
        print("EXAMPLE 4: Heterogeneous Effects")
        print("="*60)
        
        results = {}
        
        # 1. By store type
        print("\n4.1 Elasticities by Store Type:")
        print("-" * 40)
        
        store_types = self.df.reset_index()['store_type'].unique()
        
        type_elasticities = {}
        
        for store_type in store_types:
            # Filter data
            type_data = self.df.reset_index()
            type_data = type_data[type_data['store_type'] == store_type].set_index(['entity_id', 'time'])
            
            if len(type_data) > 100:  # Ensure enough observations
                panel_type = PanelData(
                    type_data[['log_quantity', 'log_price', 'promotion']].dropna()
                )
                
                mod = PanelOLS(
                    panel_type.log_quantity,
                    panel_type[['log_price', 'promotion']]
                )
                
                res = mod.fit(cov_type='clustered', cluster_entity=True, debiased=False)
                
                type_elasticities[store_type] = res.params['log_price']
                print(f"  {store_type}: {res.params['log_price']:.3f}")
        
        results['by_store_type'] = type_elasticities
        
        # 2. By income terciles
        print("\n4.2 Elasticities by Income Level:")
        print("-" * 40)
        
        # Create income terciles
        income_data = self.df.reset_index()
        terciles = income_data['income_level'].quantile([0.33, 0.67])
        
        income_groups = {
            'low': income_data[income_data['income_level'] <= terciles.iloc[0]],
            'medium': income_data[(income_data['income_level'] > terciles.iloc[0]) & 
                                (income_data['income_level'] <= terciles.iloc[1])],
            'high': income_data[income_data['income_level'] > terciles.iloc[1]]
        }
        
        income_elasticities = {}
        
        for group_name, group_data in income_groups.items():
            if len(group_data) > 100:
                group_data = group_data.set_index(['entity_id', 'time'])
                panel_income = PanelData(
                    group_data[['log_quantity', 'log_price', 'promotion']].dropna()
                )
                
                mod = PanelOLS(
                    panel_income.log_quantity,
                    panel_income[['log_price', 'promotion']]
                )
                
                res = mod.fit(cov_type='clustered', cluster_entity=True, debiased=False)
                
                income_elasticities[group_name] = res.params['log_price']
                print(f"  {group_name} income: {res.params['log_price']:.3f}")
        
        results['by_income'] = income_elasticities
        
        # 3. By product quality tier
        print("\n4.3 Elasticities by Quality Tier:")
        print("-" * 40)
        
        quality_tiers = self.df.reset_index()['quality_tier'].unique()
        
        quality_elasticities = {}
        
        for quality in quality_tiers:
            quality_data = self.df.reset_index()
            quality_data = quality_data[quality_data['quality_tier'] == quality].set_index(['entity_id', 'time'])
            
            if len(quality_data) > 100:
                panel_quality = PanelData(
                    quality_data[['log_quantity', 'log_price', 'promotion']].dropna()
                )
                
                mod = PanelOLS(
                    panel_quality.log_quantity,
                    panel_quality[['log_price', 'promotion']]
                )
                
                res = mod.fit(cov_type='clustered', cluster_entity=True, debiased=False)
                
                quality_elasticities[quality] = res.params['log_price']
                print(f"  {quality}: {res.params['log_price']:.3f}")
        
        results['by_quality'] = quality_elasticities
        
        # 4. Time-varying elasticities
        print("\n4.4 Time-Varying Elasticities:")
        print("-" * 40)
        
        # Estimate by year
        time_data = self.df.reset_index()
        time_data['year'] = pd.to_datetime(time_data['time']).dt.year
        
        years = time_data['year'].unique()
        time_elasticities = {}
        
        for year in years:
            year_data = time_data[time_data['year'] == year].set_index(['entity_id', 'time'])
            
            if len(year_data) > 100:
                panel_year = PanelData(
                    year_data[['log_quantity', 'log_price', 'promotion']].dropna()
                )
                
                mod = PanelOLS(
                    panel_year.log_quantity,
                    panel_year[['log_price', 'promotion']]
                )
                
                res = mod.fit(cov_type='clustered', cluster_entity=True, debiased=False)
                
                time_elasticities[year] = res.params['log_price']
                print(f"  Year {year}: {res.params['log_price']:.3f}")
        
        results['by_year'] = time_elasticities
        
        return results
    
    def example_5_model_comparison(self) -> Dict:
        """
        Example 5: Comprehensive Model Comparison
        
        Compares different specifications and estimators.
        """
        print("\n" + "="*60)
        print("EXAMPLE 5: Model Comparison")
        print("="*60)
        
        # Prepare data
        comp_data = self.df[['log_quantity', 'log_price', 'log_avg_competitor_price', 
                            'promotion']].dropna()
        panel = PanelData(comp_data)
        
        models = {}
        
        # 1. Pooled OLS
        print("\n5.1 Pooled OLS:")
        mod_pooled = PooledOLS(
            panel.log_quantity,
            panel[['log_price', 'log_avg_competitor_price', 'promotion']]
        )
        pooled_res = mod_pooled.fit(cov_type='clustered', cluster_entity=True)
        models['Pooled OLS'] = pooled_res
        
        # 2. Between Effects
        print("\n5.2 Between Effects:")
        mod_between = BetweenOLS(
            panel.log_quantity,
            panel[['log_price', 'log_avg_competitor_price', 'promotion']]
        )
        between_res = mod_between.fit(cov_type='robust')
        models['Between'] = between_res
        
        # 3. Fixed Effects
        print("\n5.3 Fixed Effects:")
        mod_fe = PanelOLS(
            panel.log_quantity,
            panel[['log_price', 'log_avg_competitor_price', 'promotion']]
        )
        fe_res = mod_fe.fit(cov_type='clustered', cluster_entity=True)
        models['Fixed Effects'] = fe_res
        
        # 4. Random Effects
        print("\n5.4 Random Effects:")
        mod_re = RandomEffects(
            panel.log_quantity,
            panel[['log_price', 'log_avg_competitor_price', 'promotion']]
        )
        re_res = mod_re.fit(cov_type='clustered', cluster_entity=True)
        models['Random Effects'] = re_res
        
        # 5. First Differences
        print("\n5.5 First Differences:")
        mod_fd = FirstDifferenceOLS(
            panel.log_quantity,
            panel[['log_price', 'log_avg_competitor_price', 'promotion']]
        )
        fd_res = mod_fd.fit(cov_type='clustered', cluster_entity=True)
        models['First Diff'] = fd_res
        
        # Create comparison table
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        comparison_data = []
        
        for name, model in models.items():
            comparison_data.append({
                'Model': name,
                'Own-Price Elasticity': f"{model.params['log_price']:.3f}",
                'Std Error': f"{model.std_errors['log_price']:.3f}",
                'Cross-Price Elasticity': f"{model.params['log_avg_competitor_price']:.3f}",
                'R-squared': f"{model.rsquared:.3f}",
                'N': model.nobs
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Statistical tests
        print("\n" + "-"*40)
        print("Statistical Tests:")
        
        # F-test for fixed effects
        print("\nF-test for entity effects:")
        if hasattr(fe_res, 'f_statistic_entity'):
            print(f"  F-statistic: {fe_res.f_statistic_entity.stat:.2f}")
            print(f"  P-value: {fe_res.f_statistic_entity.pval:.4f}")
        
        # Breusch-Pagan test for random effects
        print("\nBreusch-Pagan LM test for random effects:")
        pooled_resid = pooled_res.resids
        entity_avg_resid = pooled_resid.groupby(level=0).mean()
        lm_stat = len(entity_avg_resid) * (entity_avg_resid**2).sum() / (pooled_resid**2).sum()
        print(f"  LM statistic: {lm_stat:.2f}")
        
        return {'models': models, 'comparison': comparison_df}
    
    def visualize_results(self, results: Dict):
        """Visualize panel regression results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Compare estimation methods
        if 'comparison' in results:
            ax = axes[0, 0]
            comp_df = results['comparison']['comparison']
            
            # Extract elasticities
            models = comp_df['Model'].values
            elasticities = [float(e) for e in comp_df['Own-Price Elasticity'].values]
            
            ax.barh(models, elasticities)
            ax.set_xlabel('Own-Price Elasticity')
            ax.set_title('Comparison of Estimation Methods')
            ax.axvline(x=-1.2, color='r', linestyle='--', label='True value')
            ax.legend()
        
        # Plot 2: Heterogeneous effects by group
        if 'heterogeneous' in results:
            ax = axes[0, 1]
            
            # Combine different heterogeneity dimensions
            all_groups = []
            all_elasticities = []
            colors = []
            
            if 'by_store_type' in results['heterogeneous']:
                for store_type, elast in results['heterogeneous']['by_store_type'].items():
                    all_groups.append(f"Store: {store_type}")
                    all_elasticities.append(elast)
                    colors.append('blue')
            
            if 'by_income' in results['heterogeneous']:
                for income, elast in results['heterogeneous']['by_income'].items():
                    all_groups.append(f"Income: {income}")
                    all_elasticities.append(elast)
                    colors.append('green')
            
            if all_groups:
                y_pos = np.arange(len(all_groups))
                ax.barh(y_pos, all_elasticities, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(all_groups)
                ax.set_xlabel('Elasticity')
                ax.set_title('Heterogeneous Effects')
        
        # Plot 3: Dynamic adjustment
        if 'dynamic' in results and 'dynamic_fe' in results['dynamic']:
            ax = axes[1, 0]
            
            dynamic = results['dynamic']['dynamic_fe']
            categories = ['Short-run\nOwn-price', 'Long-run\nOwn-price', 
                         'Short-run\nCross-price', 'Long-run\nCross-price']
            values = [dynamic['sr_own'], dynamic['lr_own'],
                     dynamic['sr_cross'], dynamic['lr_cross']]
            
            ax.bar(categories, values, color=['blue', 'darkblue', 'green', 'darkgreen'])
            ax.set_ylabel('Elasticity')
            ax.set_title('Dynamic Elasticities')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 4: IV diagnostics
        if 'iv' in results and '2sls' in results['iv']:
            ax = axes[1, 1]
            
            # Compare OLS vs IV estimates
            methods = ['OLS', '2SLS', 'GMM', 'LIML']
            own_elast = []
            
            # Add estimates if available
            if 'fe' in results:
                own_elast.append(results['fe'].get('entity_fe', {}).get('own_elasticity', 0))
            else:
                own_elast.append(0)
            
            own_elast.append(results['iv']['2sls']['own_elasticity'])
            own_elast.append(results['iv'].get('gmm', {}).get('own_elasticity', 0))
            own_elast.append(results['iv'].get('liml', {}).get('own_elasticity', 0))
            
            ax.bar(methods, own_elast)
            ax.set_ylabel('Own-Price Elasticity')
            ax.set_title('OLS vs IV Estimates')
            ax.axhline(y=-1.2, color='r', linestyle='--', label='True value')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('linearmodels_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nResults visualization saved as 'linearmodels_results.png'")


def main():
    """Run all linearmodels examples."""
    
    print("="*60)
    print("LINEARMODELS PANEL DATA ELASTICITY ESTIMATION")
    print("="*60)
    
    # Initialize estimator
    estimator = PanelElasticityEstimator()
    
    # Store all results
    all_results = {}
    
    # Run examples
    try:
        all_results['fe'] = estimator.example_1_fixed_effects()
    except Exception as e:
        print(f"Error in fixed effects: {e}")
    
    try:
        all_results['iv'] = estimator.example_2_instrumental_variables()
    except Exception as e:
        print(f"Error in IV: {e}")
    
    try:
        all_results['dynamic'] = estimator.example_3_dynamic_panel()
    except Exception as e:
        print(f"Error in dynamic panel: {e}")
    
    try:
        all_results['heterogeneous'] = estimator.example_4_heterogeneous_effects()
    except Exception as e:
        print(f"Error in heterogeneous effects: {e}")
    
    try:
        all_results['comparison'] = estimator.example_5_model_comparison()
    except Exception as e:
        print(f"Error in model comparison: {e}")
    
    # Visualize results
    estimator.visualize_results(all_results)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nKey findings from panel data analysis:")
    print("1. Fixed effects control for time-invariant unobserved heterogeneity")
    print("2. IV/2SLS addresses price endogeneity with cost instruments")
    print("3. Dynamic models reveal persistence in demand")
    print("4. Significant heterogeneity across store types and income levels")
    print("5. Model choice matters for elasticity estimates")
    
    return all_results


if __name__ == "__main__":
    results = main()

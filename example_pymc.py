"""
PyMC Example: Bayesian Hierarchical Elasticity Estimation

This module demonstrates Bayesian hierarchical models for cross-price elasticity:

1. Hierarchical linear models with varying elasticities
2. Non-centered parameterization for better sampling
3. Heterogeneous effects across products and markets
4. Bayesian model averaging and comparison
5. Posterior predictive checks and diagnostics
6. Time-varying elasticities with state-space models

Bayesian methods provide:
- Full posterior distributions for uncertainty quantification
- Natural handling of hierarchical/multilevel data
- Incorporation of prior information
- Robust estimates with partial pooling
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns


class BayesianElasticityEstimator:
    """Bayesian hierarchical models for cross-price elasticity estimation."""
    
    def __init__(self, data_path: str = 'data/retail_scanner_data.csv'):
        """Initialize with retail scanner data."""
        self.df = pd.read_csv(data_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare data for Bayesian estimation."""
        
        # Focus on subset for computational efficiency
        self.df = self.df[self.df['market_id'].isin([1, 2, 3])]
        
        # Log transformations
        self.df['log_quantity'] = np.log(self.df['quantity'] + 1)
        self.df['log_price'] = np.log(self.df['price'])
        
        # Create indices for hierarchical modeling
        self.df['product_idx'] = pd.Categorical(self.df['product_id']).codes
        self.df['store_idx'] = pd.Categorical(self.df['store_id']).codes
        self.df['market_idx'] = pd.Categorical(self.df['market_id']).codes
        self.df['week_idx'] = pd.Categorical(self.df['week']).codes
        
        # Number of groups
        self.n_products = self.df['product_idx'].nunique()
        self.n_stores = self.df['store_idx'].nunique()
        self.n_markets = self.df['market_idx'].nunique()
        self.n_weeks = self.df['week_idx'].nunique()
        
        print(f"Prepared Bayesian data:")
        print(f"  Products: {self.n_products}")
        print(f"  Stores: {self.n_stores}")
        print(f"  Markets: {self.n_markets}")
        print(f"  Weeks: {self.n_weeks}")
        print(f"  Total obs: {len(self.df)}")
    
    def example_1_hierarchical_elasticity(self) -> Dict:
        """
        Example 1: Hierarchical Linear Model
        
        Elasticities vary by product with partial pooling across products.
        """
        print("\n" + "="*60)
        print("EXAMPLE 1: Hierarchical Elasticity Model")
        print("="*60)
        
        # Prepare data arrays
        log_quantity = self.df['log_quantity'].values
        log_price = self.df['log_price'].values
        product_idx = self.df['product_idx'].values
        promotion = self.df['promotion'].values
        
        print("\nBuilding hierarchical model...")
        
        with pm.Model() as hierarchical_model:
            
            # Hyperpriors for population-level parameters
            mu_elasticity = pm.Normal('mu_elasticity', mu=-1.0, sigma=0.5)
            sigma_elasticity = pm.HalfNormal('sigma_elasticity', sigma=0.5)
            
            mu_intercept = pm.Normal('mu_intercept', mu=3.0, sigma=1.0)
            sigma_intercept = pm.HalfNormal('sigma_intercept', sigma=1.0)
            
            # Product-specific elasticities (non-centered parameterization)
            elasticity_offset = pm.Normal('elasticity_offset', mu=0, sigma=1, shape=self.n_products)
            elasticity = pm.Deterministic('elasticity', 
                                         mu_elasticity + sigma_elasticity * elasticity_offset)
            
            # Product-specific intercepts
            intercept_offset = pm.Normal('intercept_offset', mu=0, sigma=1, shape=self.n_products)
            intercept = pm.Deterministic('intercept',
                                       mu_intercept + sigma_intercept * intercept_offset)
            
            # Promotion effect (shared across products)
            beta_promotion = pm.Normal('beta_promotion', mu=0.2, sigma=0.1)
            
            # Noise
            sigma = pm.HalfNormal('sigma', sigma=0.5)
            
            # Expected value
            mu = (intercept[product_idx] + 
                  elasticity[product_idx] * log_price +
                  beta_promotion * promotion)
            
            # Likelihood
            y = pm.Normal('y', mu=mu, sigma=sigma, observed=log_quantity)
        
        print("\nSampling from posterior...")
        
        with hierarchical_model:
            # Sample from posterior
            trace = pm.sample(2000, tune=1000, cores=2, random_seed=42)
            
            # Posterior predictive
            posterior_predictive = pm.sample_posterior_predictive(trace, random_seed=42)
        
        print("\nModel diagnostics:")
        print("-" * 40)
        
        # Check convergence
        summary = az.summary(trace, var_names=['mu_elasticity', 'sigma_elasticity'])
        print(summary)
        
        # Extract results
        results = {
            'model': hierarchical_model,
            'trace': trace,
            'posterior_predictive': posterior_predictive
        }
        
        # Print elasticity estimates
        print("\n" + "-"*40)
        print("Product-specific elasticities (mean ± std):")
        
        product_names = self.df.groupby('product_idx')['product_id'].first()
        elasticity_means = trace.posterior['elasticity'].mean(dim=['chain', 'draw'])
        elasticity_stds = trace.posterior['elasticity'].std(dim=['chain', 'draw'])
        
        for i in range(min(5, self.n_products)):
            print(f"  {product_names.iloc[i]}: {elasticity_means[i].values:.3f} ± {elasticity_stds[i].values:.3f}")
        
        print(f"\nPopulation mean elasticity: {trace.posterior['mu_elasticity'].mean().values:.3f}")
        print(f"Between-product std dev: {trace.posterior['sigma_elasticity'].mean().values:.3f}")
        
        return results
    
    def example_2_cross_price_hierarchical(self) -> Dict:
        """
        Example 2: Hierarchical Model with Cross-Price Effects
        
        Includes both own and cross-price elasticities with hierarchical structure.
        """
        print("\n" + "="*60)
        print("EXAMPLE 2: Hierarchical Cross-Price Model")
        print("="*60)
        
        # Focus on cola products for cross-price
        cola_df = self.df[self.df['category'] == 'cola'].copy()
        
        # Create competitor price variable
        for (store, week), group in cola_df.groupby(['store_id', 'week']):
            for idx, row in group.iterrows():
                other_prices = group[group['product_id'] != row['product_id']]['log_price']
                if len(other_prices) > 0:
                    cola_df.loc[idx, 'log_competitor_price'] = other_prices.mean()
        
        cola_df = cola_df.dropna()
        
        # Prepare arrays
        log_quantity = cola_df['log_quantity'].values
        log_own_price = cola_df['log_price'].values
        log_comp_price = cola_df['log_competitor_price'].values
        product_idx = pd.Categorical(cola_df['product_id']).codes
        n_products_cola = len(np.unique(product_idx))
        
        print(f"\nCola products: {n_products_cola}")
        print(f"Observations: {len(cola_df)}")
        
        with pm.Model() as cross_price_model:
            
            # Population-level parameters
            mu_own = pm.Normal('mu_own_elasticity', mu=-1.2, sigma=0.5)
            mu_cross = pm.Normal('mu_cross_elasticity', mu=0.3, sigma=0.3)
            
            sigma_own = pm.HalfNormal('sigma_own', sigma=0.3)
            sigma_cross = pm.HalfNormal('sigma_cross', sigma=0.2)
            
            # Product-specific elasticities
            own_elasticity = pm.Normal('own_elasticity', mu=mu_own, sigma=sigma_own, 
                                      shape=n_products_cola)
            cross_elasticity = pm.Normal('cross_elasticity', mu=mu_cross, sigma=sigma_cross,
                                        shape=n_products_cola)
            
            # Intercepts
            intercept = pm.Normal('intercept', mu=3, sigma=1, shape=n_products_cola)
            
            # Model error
            sigma = pm.HalfNormal('sigma', sigma=0.5)
            
            # Expected value
            mu = (intercept[product_idx] +
                  own_elasticity[product_idx] * log_own_price +
                  cross_elasticity[product_idx] * log_comp_price)
            
            # Likelihood
            y = pm.Normal('y', mu=mu, sigma=sigma, observed=log_quantity)
        
        print("\nSampling cross-price model...")
        
        with cross_price_model:
            trace_cross = pm.sample(2000, tune=1000, cores=2, random_seed=42)
        
        # Extract results
        print("\n" + "-"*40)
        print("Cross-price elasticity estimates:")
        
        own_means = trace_cross.posterior['own_elasticity'].mean(dim=['chain', 'draw'])
        cross_means = trace_cross.posterior['cross_elasticity'].mean(dim=['chain', 'draw'])
        
        cola_products = cola_df.groupby(product_idx)['product_id'].first()
        
        for i in range(min(3, n_products_cola)):
            print(f"\n{cola_products.iloc[i]}:")
            print(f"  Own-price: {own_means[i].values:.3f}")
            print(f"  Cross-price: {cross_means[i].values:.3f}")
        
        print(f"\nPopulation means:")
        print(f"  Own-price: {trace_cross.posterior['mu_own_elasticity'].mean().values:.3f}")
        print(f"  Cross-price: {trace_cross.posterior['mu_cross_elasticity'].mean().values:.3f}")
        
        return {
            'model': cross_price_model,
            'trace': trace_cross
        }
    
    def example_3_varying_slopes(self) -> Dict:
        """
        Example 3: Varying Slopes by Market
        
        Elasticities vary by both product and market with cross-level interactions.
        """
        print("\n" + "="*60)
        print("EXAMPLE 3: Varying Slopes Model")
        print("="*60)
        
        # Prepare data
        log_quantity = self.df['log_quantity'].values
        log_price = self.df['log_price'].values
        product_idx = self.df['product_idx'].values
        market_idx = self.df['market_idx'].values
        income_scaled = (self.df['income_level'].values - self.df['income_level'].mean()) / self.df['income_level'].std()
        
        print("\nBuilding varying slopes model...")
        
        with pm.Model() as varying_slopes_model:
            
            # Global intercept
            global_intercept = pm.Normal('global_intercept', mu=3, sigma=1)
            
            # Market-level elasticity variation
            market_elasticity_mean = pm.Normal('market_elasticity_mean', mu=-1.0, sigma=0.5)
            market_elasticity_sd = pm.HalfNormal('market_elasticity_sd', sigma=0.3)
            market_elasticity = pm.Normal('market_elasticity', 
                                         mu=market_elasticity_mean,
                                         sigma=market_elasticity_sd,
                                         shape=self.n_markets)
            
            # Product-level deviations from market elasticity
            product_deviation_sd = pm.HalfNormal('product_deviation_sd', sigma=0.2)
            product_deviation = pm.Normal('product_deviation', mu=0, 
                                         sigma=product_deviation_sd,
                                         shape=self.n_products)
            
            # Income effect on elasticity
            beta_income = pm.Normal('beta_income', mu=0.1, sigma=0.05)
            
            # Combined elasticity
            elasticity = (market_elasticity[market_idx] + 
                         product_deviation[product_idx] +
                         beta_income * income_scaled)
            
            # Product intercepts
            product_intercept = pm.Normal('product_intercept', mu=0, sigma=0.5, 
                                         shape=self.n_products)
            
            # Model error
            sigma = pm.HalfNormal('sigma', sigma=0.5)
            
            # Expected value
            mu = global_intercept + product_intercept[product_idx] + elasticity * log_price
            
            # Likelihood
            y = pm.Normal('y', mu=mu, sigma=sigma, observed=log_quantity)
        
        print("\nSampling varying slopes model...")
        
        with varying_slopes_model:
            trace_varying = pm.sample(2000, tune=1000, cores=2, random_seed=42)
        
        # Analyze variation
        print("\n" + "-"*40)
        print("Sources of variation in elasticities:")
        
        market_var = trace_varying.posterior['market_elasticity_sd'].mean().values ** 2
        product_var = trace_varying.posterior['product_deviation_sd'].mean().values ** 2
        total_var = market_var + product_var
        
        print(f"  Market-level: {100 * market_var / total_var:.1f}%")
        print(f"  Product-level: {100 * product_var / total_var:.1f}%")
        
        print(f"\nIncome effect on elasticity: {trace_varying.posterior['beta_income'].mean().values:.3f}")
        
        if trace_varying.posterior['beta_income'].mean().values > 0:
            print("  → Higher income = less price sensitive (less negative elasticity)")
        else:
            print("  → Higher income = more price sensitive")
        
        return {
            'model': varying_slopes_model,
            'trace': trace_varying
        }
    
    def example_4_model_comparison(self) -> Dict:
        """
        Example 4: Bayesian Model Comparison
        
        Compare different model specifications using WAIC/LOO.
        """
        print("\n" + "="*60)
        print("EXAMPLE 4: Bayesian Model Comparison")
        print("="*60)
        
        # Prepare data
        log_quantity = self.df['log_quantity'].values
        log_price = self.df['log_price'].values
        product_idx = self.df['product_idx'].values
        
        models = {}
        traces = {}
        
        # Model 1: Pooled (no hierarchy)
        print("\n4.1 Pooled Model:")
        print("-" * 40)
        
        with pm.Model() as pooled_model:
            intercept = pm.Normal('intercept', mu=3, sigma=1)
            elasticity = pm.Normal('elasticity', mu=-1, sigma=0.5)
            sigma = pm.HalfNormal('sigma', sigma=0.5)
            
            mu = intercept + elasticity * log_price
            y = pm.Normal('y', mu=mu, sigma=sigma, observed=log_quantity)
        
        with pooled_model:
            trace_pooled = pm.sample(1000, tune=500, cores=2, random_seed=42)
        
        models['pooled'] = pooled_model
        traces['pooled'] = trace_pooled
        
        # Model 2: No pooling (separate by product)
        print("\n4.2 No-Pooling Model:")
        print("-" * 40)
        
        with pm.Model() as no_pooling_model:
            intercept = pm.Normal('intercept', mu=3, sigma=1, shape=self.n_products)
            elasticity = pm.Normal('elasticity', mu=-1, sigma=1, shape=self.n_products)
            sigma = pm.HalfNormal('sigma', sigma=0.5)
            
            mu = intercept[product_idx] + elasticity[product_idx] * log_price
            y = pm.Normal('y', mu=mu, sigma=sigma, observed=log_quantity)
        
        with no_pooling_model:
            trace_no_pooling = pm.sample(1000, tune=500, cores=2, random_seed=42)
        
        models['no_pooling'] = no_pooling_model
        traces['no_pooling'] = trace_no_pooling
        
        # Model 3: Partial pooling (hierarchical)
        print("\n4.3 Hierarchical Model:")
        print("-" * 40)
        
        with pm.Model() as partial_pooling_model:
            # Hyperpriors
            mu_elasticity = pm.Normal('mu_elasticity', mu=-1, sigma=0.5)
            sigma_elasticity = pm.HalfNormal('sigma_elasticity', sigma=0.3)
            
            # Product elasticities
            elasticity = pm.Normal('elasticity', mu=mu_elasticity, 
                                  sigma=sigma_elasticity, shape=self.n_products)
            
            intercept = pm.Normal('intercept', mu=3, sigma=1, shape=self.n_products)
            sigma = pm.HalfNormal('sigma', sigma=0.5)
            
            mu = intercept[product_idx] + elasticity[product_idx] * log_price
            y = pm.Normal('y', mu=mu, sigma=sigma, observed=log_quantity)
        
        with partial_pooling_model:
            trace_partial = pm.sample(1000, tune=500, cores=2, random_seed=42)
        
        models['partial_pooling'] = partial_pooling_model
        traces['partial_pooling'] = trace_partial
        
        # Model comparison
        print("\n" + "="*40)
        print("MODEL COMPARISON")
        print("="*40)
        
        # Compute WAIC and LOO for each model
        comparison_dict = {}
        
        for name, trace in traces.items():
            print(f"\n{name.replace('_', ' ').title()}:")
            
            # WAIC
            waic = az.waic(trace)
            print(f"  WAIC: {waic.elpd_waic:.2f} ± {waic.se:.2f}")
            
            # LOO
            loo = az.loo(trace)
            print(f"  LOO: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
            
            comparison_dict[name] = trace
        
        # Compare models
        df_compare = az.compare(comparison_dict, ic='loo')
        
        print("\n" + "-"*40)
        print("Model Ranking (by LOO):")
        print(df_compare[['elpd_loo', 'se', 'dse', 'weight']])
        
        best_model = df_compare.index[0]
        print(f"\nBest model: {best_model}")
        
        return {
            'models': models,
            'traces': traces,
            'comparison': df_compare
        }
    
    def example_5_time_varying(self) -> Dict:
        """
        Example 5: Time-Varying Elasticities
        
        Elasticities evolve over time using a state-space model.
        """
        print("\n" + "="*60)
        print("EXAMPLE 5: Time-Varying Elasticities")
        print("="*60)
        
        # Aggregate to weekly level for one product
        product = self.df['product_id'].value_counts().index[0]
        weekly_df = (self.df[self.df['product_id'] == product]
                     .groupby('week')
                     .agg({'log_quantity': 'mean', 
                          'log_price': 'mean',
                          'promotion': 'mean'})
                     .reset_index()
                     .sort_values('week'))
        
        n_weeks = len(weekly_df)
        log_quantity = weekly_df['log_quantity'].values
        log_price = weekly_df['log_price'].values
        
        print(f"\nProduct: {product}")
        print(f"Weeks: {n_weeks}")
        
        print("\nBuilding state-space model...")
        
        with pm.Model() as time_varying_model:
            
            # Initial elasticity
            elasticity_init = pm.Normal('elasticity_init', mu=-1.0, sigma=0.5)
            
            # Innovation variance for random walk
            sigma_innovation = pm.HalfNormal('sigma_innovation', sigma=0.05)
            
            # Time-varying elasticity as random walk
            elasticity_innovations = pm.Normal('elasticity_innovations', 
                                              mu=0, sigma=1, shape=n_weeks-1)
            
            # Construct elasticity path
            elasticity = pm.Deterministic('elasticity',
                                         pm.math.concatenate([
                                             elasticity_init[None],
                                             elasticity_init + pm.math.cumsum(sigma_innovation * elasticity_innovations)
                                         ]))
            
            # Intercept (constant over time for simplicity)
            intercept = pm.Normal('intercept', mu=3, sigma=1)
            
            # Observation error
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.3)
            
            # Expected value
            mu = intercept + elasticity * log_price
            
            # Likelihood
            y = pm.Normal('y', mu=mu, sigma=sigma_obs, observed=log_quantity)
        
        print("\nSampling time-varying model...")
        
        with time_varying_model:
            trace_time = pm.sample(2000, tune=1000, cores=2, random_seed=42)
        
        # Extract time-varying elasticities
        elasticity_mean = trace_time.posterior['elasticity'].mean(dim=['chain', 'draw'])
        elasticity_hdi = az.hdi(trace_time, var_names=['elasticity'], hdi_prob=0.89)
        
        print("\n" + "-"*40)
        print("Time-varying elasticity summary:")
        print(f"  Initial: {elasticity_mean[0].values:.3f}")
        print(f"  Final: {elasticity_mean[-1].values:.3f}")
        print(f"  Change: {elasticity_mean[-1].values - elasticity_mean[0].values:.3f}")
        print(f"  Innovation SD: {trace_time.posterior['sigma_innovation'].mean().values:.4f}")
        
        # Detect structural breaks
        elasticity_diff = np.diff(elasticity_mean.values)
        large_changes = np.where(np.abs(elasticity_diff) > 0.1)[0]
        
        if len(large_changes) > 0:
            print(f"\nPotential structural breaks at weeks: {large_changes + 1}")
        
        return {
            'model': time_varying_model,
            'trace': trace_time,
            'elasticity_path': elasticity_mean.values,
            'weeks': weekly_df['week'].values
        }
    
    def visualize_results(self, results: Dict):
        """Visualize Bayesian estimation results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Posterior distribution of population elasticity
        if 'hierarchical' in results and 'trace' in results['hierarchical']:
            ax = axes[0, 0]
            trace = results['hierarchical']['trace']
            
            az.plot_posterior(
                trace,
                var_names=['mu_elasticity'],
                ax=ax,
                hdi_prob=0.89
            )
            ax.set_title('Population Mean Elasticity')
        
        # Plot 2: Product-specific elasticities
        if 'hierarchical' in results and 'trace' in results['hierarchical']:
            ax = axes[0, 1]
            trace = results['hierarchical']['trace']
            
            # Get product elasticities
            elasticities = trace.posterior['elasticity'].mean(dim=['chain', 'draw'])
            
            # Forest plot
            ax.scatter(elasticities.values, range(len(elasticities)), alpha=0.6)
            ax.axvline(x=-1.2, color='r', linestyle='--', label='True value')
            ax.set_ylabel('Product Index')
            ax.set_xlabel('Elasticity')
            ax.set_title('Product-Specific Elasticities')
            ax.legend()
        
        # Plot 3: Cross-price elasticities
        if 'cross_price' in results and 'trace' in results['cross_price']:
            ax = axes[0, 2]
            trace = results['cross_price']['trace']
            
            own = trace.posterior['mu_own_elasticity'].values.flatten()
            cross = trace.posterior['mu_cross_elasticity'].values.flatten()
            
            ax.hexbin(own, cross, gridsize=20, cmap='Blues')
            ax.set_xlabel('Own-Price Elasticity')
            ax.set_ylabel('Cross-Price Elasticity')
            ax.set_title('Joint Distribution of Elasticities')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        
        # Plot 4: Model comparison
        if 'comparison' in results and 'comparison' in results['comparison']:
            ax = axes[1, 0]
            
            df_comp = results['comparison']['comparison']
            models = df_comp.index
            elpd = df_comp['elpd_loo'].values
            se = df_comp['se'].values
            
            ax.errorbar(range(len(models)), elpd, yerr=se, fmt='o', capsize=5)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45)
            ax.set_ylabel('ELPD (LOO)')
            ax.set_title('Model Comparison')
            ax.grid(True, alpha=0.3)
        
        # Plot 5: Time-varying elasticity
        if 'time_varying' in results:
            ax = axes[1, 1]
            
            weeks = results['time_varying']['weeks']
            elasticity = results['time_varying']['elasticity_path']
            
            ax.plot(weeks, elasticity, 'b-', label='Elasticity')
            ax.fill_between(weeks, elasticity - 0.1, elasticity + 0.1, 
                           alpha=0.3, color='blue')
            ax.set_xlabel('Week')
            ax.set_ylabel('Elasticity')
            ax.set_title('Time-Varying Elasticity')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot 6: Posterior predictive check
        if 'hierarchical' in results:
            ax = axes[1, 2]
            
            trace = results['hierarchical']['trace']
            posterior_pred = results['hierarchical']['posterior_predictive']
            
            # Sample observed vs predicted
            y_obs = self.df['log_quantity'].values
            y_pred = posterior_pred.posterior_predictive['y'].mean(dim=['chain', 'draw']).values.flatten()
            
            # Subsample for visibility
            idx = np.random.choice(len(y_obs), min(500, len(y_obs)), replace=False)
            
            ax.scatter(y_obs[idx], y_pred[idx], alpha=0.5)
            ax.plot([y_obs.min(), y_obs.max()], [y_obs.min(), y_obs.max()], 
                   'r--', label='Perfect fit')
            ax.set_xlabel('Observed log(quantity)')
            ax.set_ylabel('Predicted log(quantity)')
            ax.set_title('Posterior Predictive Check')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('pymc_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nResults visualization saved as 'pymc_results.png'")


def main():
    """Run all PyMC Bayesian examples."""
    
    print("="*60)
    print("BAYESIAN HIERARCHICAL ELASTICITY ESTIMATION (PyMC)")
    print("="*60)
    
    # Initialize estimator
    estimator = BayesianElasticityEstimator()
    
    # Store all results
    all_results = {}
    
    # Run examples
    try:
        all_results['hierarchical'] = estimator.example_1_hierarchical_elasticity()
    except Exception as e:
        print(f"Error in hierarchical model: {e}")
    
    try:
        all_results['cross_price'] = estimator.example_2_cross_price_hierarchical()
    except Exception as e:
        print(f"Error in cross-price model: {e}")
    
    try:
        all_results['varying_slopes'] = estimator.example_3_varying_slopes()
    except Exception as e:
        print(f"Error in varying slopes: {e}")
    
    try:
        all_results['comparison'] = estimator.example_4_model_comparison()
    except Exception as e:
        print(f"Error in model comparison: {e}")
    
    try:
        all_results['time_varying'] = estimator.example_5_time_varying()
    except Exception as e:
        print(f"Error in time-varying model: {e}")
    
    # Visualize results
    estimator.visualize_results(all_results)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nKey findings from Bayesian analysis:")
    print("1. Hierarchical models provide natural uncertainty quantification")
    print("2. Partial pooling balances bias-variance tradeoff")
    print("3. Cross-level interactions explain heterogeneity")
    print("4. Model comparison guides specification choice")
    print("5. Time-varying parameters capture dynamics")
    
    return all_results


if __name__ == "__main__":
    results = main()

# PyMC: Bayesian Hierarchical Models for Elasticity Estimation

## Overview

PyMC provides a probabilistic programming framework for Bayesian inference, particularly powerful for hierarchical models that naturally handle grouped data structures common in elasticity estimation (products within categories, stores within markets, etc.).

## Bayesian Approach to Elasticity

### What It Is

Bayesian methods treat parameters as random variables with probability distributions, providing:
- Full posterior distributions (not just point estimates)
- Natural uncertainty quantification
- Incorporation of prior knowledge
- Hierarchical modeling for partial pooling

### Mathematical Foundation

#### Bayes' Theorem

```
P(θ|Data) ∝ P(Data|θ) × P(θ)
```

Where:
- P(θ|Data): Posterior distribution of parameters
- P(Data|θ): Likelihood function
- P(θ): Prior distribution

#### Hierarchical Structure

For elasticity estimation with multiple products:

**Level 1 - Observations:**
```
log(Q_it) ~ Normal(μ_it, σ)
μ_it = α_i + β_i × log(P_it) + γ × X_it
```

**Level 2 - Product-specific parameters:**
```
β_i ~ Normal(μ_β, σ_β)  # Product elasticities
α_i ~ Normal(μ_α, σ_α)  # Product intercepts
```

**Level 3 - Hyperpriors:**
```
μ_β ~ Normal(-1, 0.5)   # Population mean elasticity
σ_β ~ HalfNormal(0.3)   # Between-product variation
```

## Core Models

### 1. Basic Hierarchical Elasticity Model

```python
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

class BayesianElasticity:
    def __init__(self, data):
        """
        data should contain:
        - log_quantity: outcome
        - log_price: treatment
        - product_id: grouping variable
        - covariates: control variables
        """
        self.data = data
        self.trace = None
        self.model = None
        
    def build_hierarchical_model(self):
        """Build hierarchical elasticity model."""
        
        # Prepare data
        product_idx = pd.Categorical(self.data['product_id']).codes
        n_products = len(np.unique(product_idx))
        
        with pm.Model() as model:
            
            # Hyperpriors - population level
            mu_elasticity = pm.Normal(
                'mu_elasticity', 
                mu=-1.0,      # Prior mean
                sigma=0.5     # Prior uncertainty
            )
            
            sigma_elasticity = pm.HalfNormal(
                'sigma_elasticity',
                sigma=0.3     # Prior on between-product SD
            )
            
            mu_intercept = pm.Normal(
                'mu_intercept',
                mu=3.0,
                sigma=1.0
            )
            
            sigma_intercept = pm.HalfNormal(
                'sigma_intercept',
                sigma=1.0
            )
            
            # Product-specific parameters (non-centered parameterization)
            # This improves sampling efficiency
            elasticity_offset = pm.Normal(
                'elasticity_offset',
                mu=0,
                sigma=1,
                shape=n_products
            )
            
            elasticity = pm.Deterministic(
                'elasticity',
                mu_elasticity + sigma_elasticity * elasticity_offset
            )
            
            intercept_offset = pm.Normal(
                'intercept_offset',
                mu=0,
                sigma=1,
                shape=n_products
            )
            
            intercept = pm.Deterministic(
                'intercept',
                mu_intercept + sigma_intercept * intercept_offset
            )
            
            # Covariates (shared across products)
            if 'promotion' in self.data.columns:
                beta_promotion = pm.Normal(
                    'beta_promotion',
                    mu=0.2,
                    sigma=0.1
                )
            
            # Observation noise
            sigma = pm.HalfNormal('sigma', sigma=0.5)
            
            # Expected value
            mu = (
                intercept[product_idx] +
                elasticity[product_idx] * self.data['log_price'].values
            )
            
            if 'promotion' in self.data.columns:
                mu += beta_promotion * self.data['promotion'].values
            
            # Likelihood
            y = pm.Normal(
                'y',
                mu=mu,
                sigma=sigma,
                observed=self.data['log_quantity'].values
            )
            
        self.model = model
        return model
    
    def sample(self, draws=2000, tune=1000, chains=4):
        """Sample from posterior using MCMC."""
        
        with self.model:
            # Use NUTS sampler (No-U-Turn Sampler)
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=2,
                random_seed=42,
                target_accept=0.9  # Higher for difficult posteriors
            )
            
            # Sample posterior predictive
            self.posterior_predictive = pm.sample_posterior_predictive(
                self.trace,
                random_seed=42
            )
        
        return self.trace
```

### 2. Cross-Price Elasticity Model

```python
def build_cross_price_model(self):
    """Model with own and cross-price elasticities."""
    
    with pm.Model() as model:
        
        # Population-level parameters
        mu_own = pm.Normal('mu_own_elasticity', mu=-1.2, sigma=0.5)
        mu_cross = pm.Normal('mu_cross_elasticity', mu=0.3, sigma=0.3)
        
        sigma_own = pm.HalfNormal('sigma_own', sigma=0.3)
        sigma_cross = pm.HalfNormal('sigma_cross', sigma=0.2)
        
        # Correlation between own and cross elasticities
        rho = pm.Uniform('rho', lower=-1, upper=1)
        
        # Covariance matrix
        cov = pm.math.stack([
            [sigma_own**2, rho*sigma_own*sigma_cross],
            [rho*sigma_own*sigma_cross, sigma_cross**2]
        ])
        
        # Product-specific elasticities (multivariate normal)
        elasticities = pm.MvNormal(
            'elasticities',
            mu=pm.math.stack([mu_own, mu_cross]),
            cov=cov,
            shape=(n_products, 2)
        )
        
        own_elasticity = elasticities[:, 0]
        cross_elasticity = elasticities[:, 1]
        
        # Linear predictor
        mu = (
            intercept[product_idx] +
            own_elasticity[product_idx] * log_own_price +
            cross_elasticity[product_idx] * log_competitor_price
        )
        
        # Likelihood
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=log_quantity)
    
    return model
```

### 3. Time-Varying Elasticities

```python
def build_time_varying_model(self):
    """State-space model for time-varying elasticities."""
    
    n_periods = len(self.data['time'].unique())
    
    with pm.Model() as model:
        
        # Initial elasticity
        elasticity_init = pm.Normal(
            'elasticity_init',
            mu=-1.0,
            sigma=0.5
        )
        
        # Innovation variance (how much elasticity changes)
        sigma_innovation = pm.HalfNormal(
            'sigma_innovation',
            sigma=0.05  # Small changes over time
        )
        
        # Random walk for elasticity
        elasticity_innovations = pm.Normal(
            'elasticity_innovations',
            mu=0,
            sigma=sigma_innovation,
            shape=n_periods-1
        )
        
        # Construct elasticity path
        def elasticity_path(init, innovations):
            return pm.math.concatenate([
                init[None],
                init + pm.math.cumsum(innovations)
            ])
        
        elasticity = pm.Deterministic(
            'elasticity',
            elasticity_path(elasticity_init, elasticity_innovations)
        )
        
        # Map time periods to observations
        time_idx = pd.Categorical(self.data['time']).codes
        
        # Expected value
        mu = intercept + elasticity[time_idx] * log_price
        
        # Likelihood
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=log_quantity)
    
    return model
```

### 4. Heterogeneous Effects Model

```python
def build_heterogeneous_model(self):
    """Model with elasticities varying by demographics."""
    
    with pm.Model() as model:
        
        # Base elasticity
        beta_0 = pm.Normal('beta_0', mu=-1.0, sigma=0.5)
        
        # Income effect on elasticity
        beta_income = pm.Normal(
            'beta_income',
            mu=0.1,  # Higher income = less price sensitive
            sigma=0.05
        )
        
        # Store type effects
        beta_urban = pm.Normal('beta_urban', mu=0, sigma=0.2)
        beta_suburban = pm.Normal('beta_suburban', mu=0, sigma=0.2)
        
        # Individual elasticities
        income_scaled = (income - income.mean()) / income.std()
        
        elasticity = (
            beta_0 +
            beta_income * income_scaled +
            beta_urban * is_urban +
            beta_suburban * is_suburban
        )
        
        # Can also model as random effects
        store_effects = pm.Normal(
            'store_effects',
            mu=0,
            sigma=sigma_store,
            shape=n_stores
        )
        
        mu = intercept + (elasticity + store_effects[store_idx]) * log_price
        
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=log_quantity)
    
    return model
```

## Model Diagnostics

### Convergence Checks

```python
def check_convergence(trace):
    """Comprehensive convergence diagnostics."""
    
    # 1. R-hat (potential scale reduction factor)
    # Should be < 1.01 for all parameters
    summary = az.summary(trace)
    print("R-hat values:")
    print(summary[['r_hat']])
    
    problematic = summary[summary['r_hat'] > 1.01]
    if len(problematic) > 0:
        print(f"Warning: {len(problematic)} parameters have R-hat > 1.01")
    
    # 2. Effective sample size
    # Should be > 100 for each parameter
    print("\nEffective sample sizes:")
    print(summary[['ess_bulk', 'ess_tail']])
    
    # 3. Visual diagnostics
    # Trace plots
    az.plot_trace(trace, var_names=['mu_elasticity', 'sigma_elasticity'])
    
    # 4. Energy plot (for HMC/NUTS)
    az.plot_energy(trace)
    
    # 5. Divergences
    divergences = trace.sample_stats['diverging'].sum()
    print(f"\nNumber of divergences: {divergences}")
    
    if divergences > 0:
        print("Consider:")
        print("- Increasing target_accept")
        print("- Reparameterizing model")
        print("- Using more informative priors")
    
    return summary
```

### Posterior Predictive Checks

```python
def posterior_predictive_check(model, trace, data):
    """Check model fit using posterior predictive."""
    
    with model:
        # Sample from posterior predictive
        pp = pm.sample_posterior_predictive(trace, random_seed=42)
    
    # 1. Compare distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Observed vs predicted
    axes[0].hist(data['log_quantity'], bins=30, alpha=0.5, label='Observed')
    axes[0].hist(pp.posterior_predictive['y'].mean(dim=['chain', 'draw']), 
                 bins=30, alpha=0.5, label='Predicted')
    axes[0].legend()
    axes[0].set_title('Posterior Predictive Check')
    
    # 2. Residual analysis
    residuals = (
        data['log_quantity'] - 
        pp.posterior_predictive['y'].mean(dim=['chain', 'draw'])
    )
    
    axes[1].scatter(pp.posterior_predictive['y'].mean(dim=['chain', 'draw']), 
                   residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    
    # 3. Bayesian p-value
    # Proportion of times predicted > observed
    p_value = (pp.posterior_predictive['y'] > data['log_quantity']).mean()
    print(f"Bayesian p-value: {p_value:.3f} (should be near 0.5)")
    
    return pp
```

## Model Comparison

### Information Criteria

```python
def compare_models(models_dict):
    """Compare multiple models using information criteria."""
    
    # Calculate WAIC and LOO for each model
    comparison_data = {}
    
    for name, (model, trace) in models_dict.items():
        with model:
            # WAIC (Widely Applicable Information Criterion)
            waic = az.waic(trace)
            
            # LOO (Leave-One-Out cross-validation)
            loo = az.loo(trace)
            
            comparison_data[name] = trace
            
            print(f"\n{name}:")
            print(f"  WAIC: {waic.elpd_waic:.2f} ± {waic.se:.2f}")
            print(f"  LOO: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
            print(f"  p_waic: {waic.p_waic:.2f}")  # Effective parameters
    
    # Compare models
    df_compare = az.compare(comparison_data, ic='loo')
    print("\nModel Comparison:")
    print(df_compare)
    
    # Plot comparison
    az.plot_compare(df_compare)
    
    return df_compare
```

## Pros and Cons

### Pros
- ✅ **Full uncertainty quantification**: Entire posterior distribution
- ✅ **Natural hierarchical modeling**: Partial pooling
- ✅ **Flexible priors**: Incorporate domain knowledge
- ✅ **Missing data handling**: Natural imputation
- ✅ **Model checking**: Posterior predictive checks
- ✅ **Small sample properties**: Better than asymptotic methods

### Cons
- ❌ **Computationally intensive**: MCMC can be slow
- ❌ **Prior sensitivity**: Results depend on prior choice
- ❌ **Convergence issues**: Complex models may not converge
- ❌ **Interpretability**: Posteriors harder to explain than p-values
- ❌ **Scalability**: Challenging for very large datasets

## Practical Implementation

### Complete Example

```python
import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# Load and prepare data
data = pd.read_csv('retail_data.csv')
data['log_quantity'] = np.log(data['quantity'] + 1)
data['log_price'] = np.log(data['price'])
data['product_idx'] = pd.Categorical(data['product_id']).codes

n_products = data['product_idx'].nunique()

# Build model
with pm.Model() as hierarchical_model:
    
    # Hyperpriors
    mu_elast = pm.Normal('mu_elasticity', mu=-1.0, sigma=0.5)
    sigma_elast = pm.HalfNormal('sigma_elasticity', sigma=0.3)
    
    # Product elasticities
    elasticity = pm.Normal(
        'elasticity',
        mu=mu_elast,
        sigma=sigma_elast,
        shape=n_products
    )
    
    # Intercepts
    intercept = pm.Normal('intercept', mu=3, sigma=1, shape=n_products)
    
    # Promotion effect
    beta_promo = pm.Normal('beta_promotion', mu=0.2, sigma=0.1)
    
    # Model error
    sigma = pm.HalfNormal('sigma', sigma=0.5)
    
    # Expected value
    mu = (
        intercept[data['product_idx']] +
        elasticity[data['product_idx']] * data['log_price'] +
        beta_promo * data['promotion']
    )
    
    # Likelihood
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data['log_quantity'])

# Sample
with hierarchical_model:
    trace = pm.sample(2000, tune=1000, cores=4, random_seed=42)
    posterior_predictive = pm.sample_posterior_predictive(trace)

# Results
summary = az.summary(trace, var_names=['mu_elasticity', 'sigma_elasticity'])
print(summary)

# Extract elasticities
elasticities = trace.posterior['elasticity']
mean_elast = elasticities.mean(dim=['chain', 'draw'])

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Posterior of population mean
az.plot_posterior(trace, var_names=['mu_elasticity'], ax=axes[0,0])
axes[0,0].set_title('Population Mean Elasticity')

# Product-specific elasticities
axes[0,1].barh(range(n_products), mean_elast)
axes[0,1].set_ylabel('Product')
axes[0,1].set_xlabel('Elasticity')
axes[0,1].set_title('Product-Specific Elasticities')

# Trace plot
az.plot_trace(trace, var_names=['mu_elasticity'], axes=[axes[1,0], axes[1,1]])

plt.tight_layout()
plt.show()

# Predictions for new data
new_data = pd.DataFrame({
    'log_price': np.linspace(-1, 1, 50),
    'product_idx': 0,
    'promotion': 0
})

with hierarchical_model:
    # Set observed data to None for prediction
    pm.set_data({
        'log_price': new_data['log_price'],
        'product_idx': new_data['product_idx'],
        'promotion': new_data['promotion']
    })
    
    predictions = pm.sample_posterior_predictive(
        trace,
        predictions=True,
        random_seed=42
    )

# Plot predictions with uncertainty
pred_mean = predictions.predictions['y'].mean(dim=['chain', 'draw'])
pred_hdi = az.hdi(predictions.predictions, hdi_prob=0.89)

plt.figure(figsize=(10, 6))
plt.plot(new_data['log_price'], pred_mean, 'b-', label='Mean prediction')
plt.fill_between(
    new_data['log_price'],
    pred_hdi['y'][:, 0],
    pred_hdi['y'][:, 1],
    alpha=0.3,
    label='89% HDI'
)
plt.xlabel('Log Price')
plt.ylabel('Log Quantity')
plt.title('Demand Curve with Uncertainty')
plt.legend()
plt.show()
```

## Advanced Topics

### Prior Sensitivity Analysis

```python
def prior_sensitivity(model, data, prior_specs):
    """Test sensitivity to prior specification."""
    
    results = {}
    
    for name, priors in prior_specs.items():
        with pm.Model() as m:
            # Build model with specified priors
            mu_elast = pm.Normal('mu_elasticity', **priors['mu'])
            sigma_elast = pm.HalfNormal('sigma_elasticity', **priors['sigma'])
            # ... rest of model
            
        with m:
            trace = pm.sample(1000, tune=500)
            results[name] = trace.posterior['mu_elasticity'].mean()
    
    # Compare results
    for name, estimate in results.items():
        print(f"{name}: {estimate:.3f}")
```

### Handling Missing Data

```python
with pm.Model() as missing_data_model:
    # Impute missing prices
    price_mu = pm.Normal('price_mu', mu=data['price'].mean(), sigma=1)
    price_sigma = pm.HalfNormal('price_sigma', sigma=1)
    
    price_observed = data['price'].values.copy()
    missing_idx = np.isnan(price_observed)
    
    price_imputed = pm.Normal(
        'price_imputed',
        mu=price_mu,
        sigma=price_sigma,
        shape=missing_idx.sum()
    )
    
    price = tt.set_subtensor(price_observed[missing_idx], price_imputed)
    
    # Continue with model using imputed prices
    # ...
```

## References

- Gelman et al. (2013). "Bayesian Data Analysis"
- McElreath (2020). "Statistical Rethinking"
- Kruschke (2014). "Doing Bayesian Data Analysis"
- Rossi et al. (2005). "Bayesian Statistics and Marketing"

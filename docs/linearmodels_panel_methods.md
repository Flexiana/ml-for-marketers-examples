# LinearModels: Panel Data Methods for Elasticity Estimation

## Overview

LinearModels provides state-of-the-art panel data econometrics for Python, implementing fixed effects, random effects, instrumental variables, and dynamic panel models. These methods are essential for elasticity estimation when you have repeated observations of the same entities over time.

## Core Panel Data Models

### 1. Fixed Effects (FE) Model

#### What It Is

Fixed Effects controls for all time-invariant unobserved heterogeneity by including entity-specific intercepts. This is crucial when unobserved product quality or store characteristics affect both price and quantity.

#### Mathematical Foundation

**Model Specification:**
```
y_it = β·x_it + α_i + ε_it
```

Where:
- y_it: log quantity for entity i at time t
- x_it: log price and other time-varying variables
- α_i: entity-specific fixed effect
- ε_it: idiosyncratic error

**Within Transformation:**
```
(y_it - ȳ_i) = β·(x_it - x̄_i) + (ε_it - ε̄_i)
```

This removes α_i and any time-invariant variables.

#### Pros
- ✅ Controls for all time-invariant unobserved heterogeneity
- ✅ Consistent even with correlation between α_i and x_it
- ✅ No distributional assumptions on α_i
- ✅ Can include time fixed effects

#### Cons
- ❌ Cannot estimate time-invariant coefficients
- ❌ Inefficient if α_i uncorrelated with x_it
- ❌ Requires within-entity variation
- ❌ Degrees of freedom loss with many entities

#### Usage

```python
from linearmodels import PanelOLS
from linearmodels.panel import PanelData
import pandas as pd
import numpy as np

# Data preparation
panel_df = pd.DataFrame({
    'entity': [...],      # Store-product identifier
    'time': [...],        # Time period
    'log_quantity': [...],
    'log_price': [...],
    'promotion': [...],
    'competitor_price': [...]
})

# Set multi-index
panel_df = panel_df.set_index(['entity', 'time'])

# Convert to PanelData
data = PanelData(panel_df)

# Entity Fixed Effects
model_fe = PanelOLS(
    dependent=data.log_quantity,
    exog=data[['log_price', 'promotion', 'competitor_price']],
    entity_effects=True,
    time_effects=False
)

results_fe = model_fe.fit(cov_type='clustered', cluster_entity=True)
print(results_fe)

# Two-way Fixed Effects (entity + time)
model_twoway = PanelOLS(
    dependent=data.log_quantity,
    exog=data[['log_price', 'promotion']],
    entity_effects=True,
    time_effects=True
)

results_twoway = model_twoway.fit(cov_type='clustered')
```

### 2. Random Effects (RE) Model

#### What It Is

Random Effects treats entity-specific effects as random draws from a distribution, allowing for both within and between entity variation.

#### Mathematical Foundation

```
y_it = β·x_it + α_i + ε_it
α_i ~ N(0, σ²_α)
ε_it ~ N(0, σ²_ε)
```

**GLS Transformation:**
```
y_it - θ·ȳ_i = β·(x_it - θ·x̄_i) + (1-θ)·α_i + ε_it - θ·ε̄_i
```

Where θ = 1 - σ_ε/√(σ²_ε + T·σ²_α)

#### Pros
- ✅ More efficient than FE if assumptions hold
- ✅ Can estimate time-invariant coefficients
- ✅ Uses both within and between variation

#### Cons
- ❌ Inconsistent if α_i correlated with x_it
- ❌ Requires distributional assumptions
- ❌ Hausman test often rejects RE

#### Usage

```python
from linearmodels import RandomEffects

model_re = RandomEffects(
    dependent=data.log_quantity,
    exog=data[['log_price', 'promotion', 'store_size']]
)

results_re = model_re.fit(cov_type='clustered', cluster_entity=True)

# Hausman test (FE vs RE)
from linearmodels.panel import compare

comparison = compare({
    'Fixed Effects': results_fe,
    'Random Effects': results_re
})
print(comparison.summary)
```

### 3. First Differences (FD) Model

#### What It Is

First Differences removes fixed effects by differencing adjacent time periods.

#### Mathematical Foundation

```
Δy_it = β·Δx_it + Δε_it
```

Where Δy_it = y_it - y_i,t-1

#### Pros
- ✅ Removes fixed effects
- ✅ Simple interpretation as changes
- ✅ Good for trending data

#### Cons
- ❌ Loses first time period
- ❌ Can amplify measurement error
- ❌ Serial correlation in errors

#### Usage

```python
from linearmodels import FirstDifferenceOLS

model_fd = FirstDifferenceOLS(
    dependent=data.log_quantity,
    exog=data[['log_price', 'promotion']]
)

results_fd = model_fd.fit(cov_type='robust')
```

## Instrumental Variables for Panel Data

### 4. Panel IV/2SLS

#### What It Is

Two-Stage Least Squares for panel data handles endogenous regressors (like price) using instrumental variables.

#### Mathematical Foundation

**First Stage:**
```
x_it = π·z_it + γ·w_it + ν_it
```

**Second Stage:**
```
y_it = β·x̂_it + δ·w_it + α_i + ε_it
```

Where z_it are instruments (e.g., cost shifters).

#### Usage

```python
from linearmodels.iv import IV2SLS

# Prepare IV data
iv_data = panel_df.reset_index()

# Dependent variable
y = iv_data['log_quantity']

# Endogenous variables
endog = iv_data[['log_price']]

# Exogenous variables
exog = iv_data[['promotion', 'week']]

# Add fixed effects as dummies (simplified)
entity_dummies = pd.get_dummies(iv_data['entity'], prefix='entity')
exog = pd.concat([exog, entity_dummies], axis=1)

# Instruments
instruments = iv_data[['wholesale_cost', 'transportation_cost']]

# 2SLS estimation
model_iv = IV2SLS(
    dependent=y,
    exog=exog,
    endog=endog,
    instruments=instruments
)

results_iv = model_iv.fit(cov_type='robust')

# First-stage diagnostics
print(f"First-stage F-stat: {results_iv.first_stage.f_statistic.stat}")
print(f"Weak instruments: {results_iv.first_stage.f_statistic.stat < 10}")
```

### 5. Panel GMM

#### What It Is

Generalized Method of Moments for panel data, more efficient than 2SLS with heteroskedasticity.

#### Usage

```python
from linearmodels.iv import IVGMM

model_gmm = IVGMM(
    dependent=y,
    exog=exog,
    endog=endog,
    instruments=instruments
)

results_gmm = model_gmm.fit(
    cov_type='robust',
    iter_limit=100
)

# J-statistic for overidentification
print(f"J-statistic: {results_gmm.j_stat.stat}")
print(f"P-value: {results_gmm.j_stat.pval}")
```

## Dynamic Panel Models

### 6. Dynamic Panel with Lagged Dependent Variable

#### What It Is

Includes lagged dependent variable to capture persistence in demand.

#### Mathematical Foundation

```
y_it = ρ·y_i,t-1 + β·x_it + α_i + ε_it
```

**Long-run elasticity:**
```
β_LR = β / (1 - ρ)
```

#### Challenges

- Nickell bias: FE biased with lagged dependent variable
- Need IV methods (Anderson-Hsiao, Arellano-Bond)

#### Usage

```python
# Prepare dynamic panel
panel_df['lag_log_quantity'] = panel_df.groupby('entity')['log_quantity'].shift(1)

# Anderson-Hsiao estimator
# Use second lag as instrument for first difference
panel_df['lag2_log_quantity'] = panel_df.groupby('entity')['log_quantity'].shift(2)
panel_df['d_log_quantity'] = panel_df.groupby('entity')['log_quantity'].diff()
panel_df['d_lag_log_quantity'] = panel_df.groupby('entity')['lag_log_quantity'].diff()
panel_df['d_log_price'] = panel_df.groupby('entity')['log_price'].diff()

# IV estimation
iv_dynamic = IV2SLS(
    dependent=panel_df['d_log_quantity'].dropna(),
    exog=panel_df[['d_log_price']].dropna(),
    endog=panel_df[['d_lag_log_quantity']].dropna(),
    instruments=panel_df[['lag2_log_quantity']].dropna()
)

results_dynamic = iv_dynamic.fit()

# Calculate long-run elasticity
rho = results_dynamic.params['d_lag_log_quantity']
beta_sr = results_dynamic.params['d_log_price']
beta_lr = beta_sr / (1 - rho)

print(f"Persistence: {rho:.3f}")
print(f"Short-run elasticity: {beta_sr:.3f}")
print(f"Long-run elasticity: {beta_lr:.3f}")
```

## Model Selection and Testing

### Specification Tests

```python
# 1. Hausman Test (FE vs RE)
def hausman_test(fe_results, re_results):
    b_fe = fe_results.params
    b_re = re_results.params
    
    # Variance of difference
    var_diff = fe_results.cov - re_results.cov
    
    # Test statistic
    diff = b_fe - b_re
    chi2 = diff.T @ np.linalg.inv(var_diff) @ diff
    
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(chi2, df=len(diff))
    
    return chi2, p_value

# 2. Test for serial correlation
def serial_correlation_test(residuals, panel_structure):
    # Wooldridge test for AR(1) in panels
    pass

# 3. Test for cross-sectional dependence
def pesaran_cd_test(residuals, panel_structure):
    # Pesaran CD test
    pass
```

## Best Practices

### Data Preparation

```python
# 1. Create balanced panel if possible
def balance_panel(df, entity_col, time_col):
    idx = pd.MultiIndex.from_product([
        df[entity_col].unique(),
        df[time_col].unique()
    ], names=[entity_col, time_col])
    
    return df.set_index([entity_col, time_col]).reindex(idx)

# 2. Handle missing data
def handle_missing(df):
    # Forward fill for small gaps
    df = df.groupby(level=0).ffill(limit=1)
    
    # Drop if too many missing
    df = df.dropna(thresh=0.8*len(df.columns))
    
    return df

# 3. Create lags and differences
def create_dynamics(df, vars_to_lag):
    for var in vars_to_lag:
        df[f'lag_{var}'] = df.groupby(level=0)[var].shift(1)
        df[f'd_{var}'] = df.groupby(level=0)[var].diff()
    
    return df
```

### Choosing Between Methods

| Data Characteristic | Recommended Method |
|--------------------|-------------------|
| Unobserved heterogeneity likely | Fixed Effects |
| Need time-invariant coefficients | Random Effects |
| Price endogeneity | Panel IV/GMM |
| Dynamic demand | Dynamic panel with IV |
| Serial correlation | First Differences or AR model |
| Many entities, few periods | Fixed Effects |
| Few entities, many periods | Consider time-series methods |

## Common Pitfalls

1. **Insufficient Variation**: Need within-entity price variation for FE
2. **Weak Instruments**: Check first-stage F > 10
3. **Nickell Bias**: Don't use FE with lagged dependent variable
4. **Incidental Parameters**: FE biased with short panels (T < 10)
5. **Serial Correlation**: Use clustered standard errors

## Performance Optimization

```python
# For large panels
import dask.dataframe as dd

# Convert to Dask for parallel processing
ddf = dd.from_pandas(panel_df, npartitions=4)

# Or use sparse matrices for many fixed effects
from linearmodels.iv import IVGMMCUE
from scipy.sparse import csr_matrix

# Convert dummies to sparse
X_sparse = csr_matrix(entity_dummies.values)
```

## References

- Wooldridge (2010). "Econometric Analysis of Cross Section and Panel Data"
- Baltagi (2021). "Econometric Analysis of Panel Data"
- Cameron & Trivedi (2005). "Microeconometrics: Methods and Applications"

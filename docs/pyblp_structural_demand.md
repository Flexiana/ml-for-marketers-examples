# PyBLP: Structural Demand Estimation with Random Coefficients

## Overview

PyBLP implements the Berry, Levinsohn, and Pakes (1995) random-coefficients logit demand model, the workhorse of structural demand estimation in industrial organization. It estimates flexible substitution patterns while maintaining economic structure.

## The BLP Model

### What It Is

BLP is a structural model of demand that allows for:
- Consumer heterogeneity in preferences
- Flexible substitution patterns beyond IIA (Independence of Irrelevant Alternatives)
- Rich own- and cross-price elasticities
- Welfare analysis capabilities

### Mathematical Foundation

#### Utility Specification

Consumer i's utility from product j in market t:

```
u_ijt = δ_jt + μ_ijt + ε_ijt
```

Where:
- **δ_jt**: Mean utility (common to all consumers)
  ```
  δ_jt = x_jt·β - α·p_jt + ξ_jt
  ```
  - x_jt: observed product characteristics
  - p_jt: price
  - ξ_jt: unobserved product quality

- **μ_ijt**: Individual deviation from mean utility
  ```
  μ_ijt = Σ_k σ_k·x_jkt·ν_ik + σ_p·p_jt·ν_ip
  ```
  - ν_i: individual random draws (normal or other distributions)
  - σ: standard deviations of random coefficients

- **ε_ijt**: Type I extreme value error

#### Market Share Equation

Predicted market share:
```
s_jt = ∫ exp(δ_jt + μ_ijt) / (1 + Σ_k exp(δ_kt + μ_ikt)) dF(ν)
```

This integral is computed via simulation or quadrature.

#### Estimation

**Two-Step GMM Procedure:**

1. **Inner Loop (Contraction Mapping):**
   - For given θ₂ = (σ, π), solve for mean utilities δ
   - Use Berry inversion: δ^(h+1) = δ^h + log(s_observed) - log(s_predicted)

2. **Outer Loop (GMM):**
   - Minimize GMM objective: min_θ g(θ)'W g(θ)
   - Where g(θ) = Z'ξ(θ) (moment conditions)

### Supply Side (Optional)

#### Firm's Problem

Multi-product firms maximize profits:
```
max_p Σ_{j∈F_f} (p_j - mc_j)·s_j(p)·M
```

#### First-Order Conditions

Yields pricing equation:
```
p = mc + Ω^(-1)·s
```

Where Ω is the ownership matrix weighted by demand derivatives.

## Pros and Cons

### Pros
- ✅ **Flexible substitution patterns**: Not restricted by IIA
- ✅ **Microfounded**: Based on utility maximization
- ✅ **Welfare analysis**: Can compute consumer surplus changes
- ✅ **Rich heterogeneity**: Demographics, random coefficients
- ✅ **Supply side**: Joint estimation of demand and costs
- ✅ **Handles endogeneity**: Via instrumental variables

### Cons
- ❌ **Computationally intensive**: Nested optimization
- ❌ **Multiple equilibria**: Possible in pricing game
- ❌ **Identification challenges**: Requires good instruments
- ❌ **Large data requirements**: Many markets needed
- ❌ **Functional form assumptions**: Logit errors, distributions
- ❌ **Convergence issues**: May fail to converge

## Usage

### Data Requirements

```python
import pyblp
import pandas as pd
import numpy as np

# Required data structure
product_data = pd.DataFrame({
    # Identifiers
    'market_ids': [...],        # Market identifier (e.g., city-time)
    'product_ids': [...],       # Product identifier
    'firm_ids': [...],          # Firm identifier (for supply)
    
    # Demand variables
    'shares': [...],            # Market shares (0 to 1)
    'prices': [...],            # Product prices
    
    # Product characteristics (enter linearly)
    'x1': [...],                # Characteristic 1
    'x2': [...],                # Characteristic 2
    
    # Demographics (optional)
    'income': [...],            # Market-level income
    
    # Instruments
    'demand_instruments0': [...], # Cost shifters
    'demand_instruments1': [...], # BLP instruments
    'supply_instruments0': [...], # Demand shifters (if supply)
})

# Agent data (optional, for demographics)
agent_data = pd.DataFrame({
    'market_ids': [...],
    'weights': [...],           # Agent weights
    'nodes0': [...],            # Integration nodes
    'income': [...],            # Agent income
})
```

### Basic BLP Estimation

```python
# 1. Define formulations
X1 = pyblp.Formulation('1 + x1 + x2 + prices')  # Linear characteristics
X2 = pyblp.Formulation('1 + prices')            # Random coefficients
X3 = pyblp.Formulation('1 + cost_shifter')      # Supply (optional)

# 2. Configure problem
problem = pyblp.Problem(
    product_formulations=(X1, X2, X3),
    product_data=product_data,
    agent_formulation=pyblp.Formulation('income'),  # Demographics
    agent_data=agent_data,
    rc_type='random'  # or 'nested' for nested logit
)

# 3. Initial values
initial_sigma = np.array([
    [0.5],  # SD of constant
    [0.3],  # SD of price coefficient
])

initial_pi = np.array([
    [0.0],  # Constant not interacted
    [0.1],  # Price-income interaction
])

# 4. Solve
results = problem.solve(
    sigma=initial_sigma,
    pi=initial_pi,
    optimization=pyblp.Optimization('l-bfgs-b'),
    costs_type='linear',  # For supply side
    method='1s'  # One-step or '2s' for two-step GMM
)

print(results)
```

### Computing Elasticities

```python
# Own and cross-price elasticities
elasticities = results.compute_elasticities()

# Extract for specific market
market = 'market_1'
market_elasticities = elasticities[product_data.market_ids == market]

# Aggregate elasticities
aggregate_elasticities = results.compute_aggregate_elasticities(
    factor=0.1  # 10% price change
)

# Diversion ratios
diversions = results.compute_diversion_ratios()

# Long-run effects
long_run = results.compute_long_run_diversion_ratios()
```

### Model Extensions

#### 1. Nested Logit
```python
# Add nesting structure
product_data['nesting_ids'] = [...]  # Nest identifier

problem = pyblp.Problem(
    product_formulations=(X1,),
    product_data=product_data,
    rc_type='nested'
)

# Estimate with nesting parameter
results = problem.solve(
    rho=np.array([0.7]),  # Initial nesting parameter
)
```

#### 2. Optimal Instruments
```python
# First-stage estimation
initial_results = problem.solve(sigma=initial_sigma)

# Compute optimal instruments
opt_instruments = initial_results.compute_optimal_instruments()

# Re-estimate with optimal instruments
product_data_optimal = product_data.copy()
for i, inst in enumerate(opt_instruments.T):
    product_data_optimal[f'demand_instruments_opt{i}'] = inst

problem_optimal = pyblp.Problem(
    product_formulations=(X1, X2),
    product_data=product_data_optimal
)

final_results = problem_optimal.solve(sigma=initial_sigma)
```

#### 3. Micro Moments
```python
# Add micro moments (e.g., from survey data)
micro_moments = [
    pyblp.DemographicCovarianceMoment(
        X2_index=0,  # Which random coefficient
        demographics_index=0,  # Which demographic
        value=0.5,  # Observed covariance
        market_ids=None  # All markets
    )
]

results = problem.solve(
    sigma=initial_sigma,
    micro_moments=micro_moments
)
```

### Welfare Analysis

```python
# Consumer surplus
cs = results.compute_consumer_surpluses()

# Price change simulation
new_prices = product_data['prices'] * 1.1
new_cs = results.compute_consumer_surpluses(prices=new_prices)

# Welfare change
welfare_change = new_cs - cs

# Merger simulation
# Change ownership matrix
product_data_merger = product_data.copy()
product_data_merger.loc[
    product_data_merger.product_ids.isin(['prod1', 'prod2']), 
    'firm_ids'
] = 'merged_firm'

# Compute new equilibrium
merger_results = results.compute_prices(
    firm_ids=product_data_merger.firm_ids,
    costs=results.compute_costs()
)
```

### Serialization

```python
import pickle

# Save results
results.to_pickle('blp_results.pkl')

# Save problem (for re-estimation)
with open('blp_problem.pkl', 'wb') as f:
    pickle.dump(problem, f)

# Load
results_loaded = pyblp.read_pickle('blp_results.pkl')

with open('blp_problem.pkl', 'rb') as f:
    problem_loaded = pickle.load(f)
```

## Performance Tips

1. **Starting Values**: Use nested logit or simpler model for initial values
2. **Integration**: More nodes improve accuracy but increase computation
3. **Instruments**: BLP instruments (characteristics of other products) often work well
4. **Optimization**: Try different optimizers if convergence fails
5. **Bounds**: Set reasonable bounds on parameters

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Non-convergence | Try different starting values or optimization algorithms |
| Negative variance | Check data scaling, use log-normal distribution |
| Implausible elasticities | Check instruments strength, add micro moments |
| Slow computation | Reduce integration nodes, use parallel processing |
| Multiple equilibria | Check supply-side assumptions, use tatonnement |

## Model Selection

### When to Use BLP

✅ Use when:
- Need flexible substitution patterns
- Have market-level data (not individual purchases)
- Want to do counterfactual analysis
- Have good instruments for price

❌ Avoid when:
- Have individual choice data (use mixed logit instead)
- Limited markets (< 20)
- No price variation
- Need quick results

## References

- Berry, Levinsohn, and Pakes (1995). "Automobile Prices in Market Equilibrium"
- Berry and Haile (2014). "Identification in Differentiated Products Markets"
- Conlon and Gortmaker (2020). "Best Practices for Differentiated Products Demand Estimation"

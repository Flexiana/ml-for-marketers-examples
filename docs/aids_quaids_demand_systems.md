# AIDS/QUAIDS: Complete Demand System Estimation

## Overview

The Almost Ideal Demand System (AIDS) and its quadratic extension (QUAIDS) are complete demand systems that model how consumers allocate expenditure across multiple goods. These models are particularly valuable for cross-price elasticity estimation as they provide a full matrix of elasticities while maintaining theoretical consistency.

## The AIDS Model

### What It Is

AIDS models the complete system of budget shares as functions of prices and total expenditure, derived from a specific indirect utility function that provides a flexible approximation to any demand system.

### Mathematical Foundation

#### Cost Function

The AIDS model starts with the cost function:
```
log c(u, p) = α₀ + Σₖ αₖ log pₖ + ½ΣₖΣⱼ γₖⱼ log pₖ log pⱼ + u·β₀·Πₖ pₖ^βₖ
```

#### Budget Share Equations

The budget share for good i:
```
wᵢ = αᵢ + Σⱼ γᵢⱼ log pⱼ + βᵢ log(x/P)
```

Where:
- wᵢ: budget share of good i (pᵢqᵢ/x)
- pⱼ: price of good j
- x: total expenditure
- P: price index

#### Price Index

**Nonlinear AIDS:**
```
log P = α₀ + Σₖ αₖ log pₖ + ½ΣₖΣⱼ γₖⱼ log pₖ log pⱼ
```

**Linear Approximate AIDS (LA-AIDS):**
```
log P* = Σₖ wₖ log pₖ  (Stone's price index)
```

### Theoretical Restrictions

1. **Adding-up**: Ensures budget shares sum to 1
   ```
   Σᵢ αᵢ = 1, Σᵢ γᵢⱼ = 0, Σᵢ βᵢ = 0
   ```

2. **Homogeneity**: No money illusion (degree 0 in prices and income)
   ```
   Σⱼ γᵢⱼ = 0 for all i
   ```

3. **Symmetry**: From Slutsky symmetry
   ```
   γᵢⱼ = γⱼᵢ
   ```

### Elasticity Formulas

#### Marshallian (Uncompensated) Price Elasticities

```
εᵢⱼ = -δᵢⱼ + (γᵢⱼ - βᵢwⱼ)/wᵢ
```

Where δᵢⱼ = 1 if i=j, 0 otherwise

#### Expenditure Elasticities

```
ηᵢ = 1 + βᵢ/wᵢ
```

#### Hicksian (Compensated) Price Elasticities

Using Slutsky equation:
```
εᵢⱼʰ = εᵢⱼ + ηᵢwⱼ
```

## The QUAIDS Model

### Extension to AIDS

QUAIDS adds a quadratic income term to capture non-linear Engel curves:

```
wᵢ = αᵢ + Σⱼ γᵢⱼ log pⱼ + βᵢ log(x/P) + λᵢ/b(p)·[log(x/P)]²
```

Where b(p) = Πₖ pₖ^βₖ

### Why QUAIDS?

- Allows goods to be luxuries at some income levels and necessities at others
- Better fits empirical Engel curves
- More flexible income responses

## Implementation with Statsmodels

### Data Preparation

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

class AIDSModel:
    def __init__(self, data):
        """
        data should contain:
        - shares: wᵢ for each good
        - log_prices: log(pⱼ) for each good
        - log_expenditure: log(x)
        - demographics (optional)
        """
        self.data = data
        self.n_goods = None
        self.categories = None
        
    def prepare_data(self):
        """Prepare data for AIDS estimation."""
        # Calculate Stone's price index for LA-AIDS
        self.data['log_P_stone'] = 0
        for good in self.categories:
            self.data['log_P_stone'] += (
                self.data[f'share_{good}'] * 
                self.data[f'log_price_{good}']
            )
        
        # Real expenditure
        self.data['log_real_exp'] = (
            self.data['log_expenditure'] - 
            self.data['log_P_stone']
        )
        
    def estimate_aids(self, restrictions=None):
        """Estimate AIDS model."""
        results = {}
        
        # Estimate n-1 equations (last determined by adding-up)
        for i, good in enumerate(self.categories[:-1]):
            # Dependent variable
            y = self.data[f'share_{good}']
            
            # Independent variables
            X_list = [np.ones(len(y))]  # Constant (αᵢ)
            
            # Log prices (γᵢⱼ)
            for other_good in self.categories:
                X_list.append(self.data[f'log_price_{other_good}'])
            
            # Real expenditure (βᵢ)
            X_list.append(self.data['log_real_exp'])
            
            X = np.column_stack(X_list)
            
            # Apply restrictions if specified
            if restrictions == 'homogeneity':
                # Drop last price, use relative prices
                X = self._impose_homogeneity(X)
            
            # Estimate
            model = OLS(y, X)
            results[good] = model.fit()
        
        # Recover parameters for last good using adding-up
        results[self.categories[-1]] = self._recover_last_equation(results)
        
        return results
```

### QUAIDS Extension

```python
def estimate_quaids(self):
    """Estimate QUAIDS model with quadratic income term."""
    results = {}
    
    # Add quadratic term
    self.data['log_real_exp_sq'] = self.data['log_real_exp'] ** 2
    
    for i, good in enumerate(self.categories[:-1]):
        y = self.data[f'share_{good}']
        
        X_list = [
            np.ones(len(y)),  # αᵢ
            *[self.data[f'log_price_{g}'] for g in self.categories],  # γᵢⱼ
            self.data['log_real_exp'],  # βᵢ
            self.data['log_real_exp_sq']  # λᵢ (quadratic term)
        ]
        
        X = np.column_stack(X_list)
        model = OLS(y, X)
        results[good] = model.fit()
    
    return results
```

### Elasticity Calculation

```python
def calculate_elasticities(self, params):
    """Calculate full elasticity matrices."""
    n = len(self.categories)
    
    # Extract parameters
    alpha = np.array([params[g]['alpha'] for g in self.categories])
    beta = np.array([params[g]['beta'] for g in self.categories])
    gamma = np.zeros((n, n))
    for i, good_i in enumerate(self.categories):
        for j, good_j in enumerate(self.categories):
            gamma[i, j] = params[good_i].get(f'gamma_{good_j}', 0)
    
    # Mean shares
    shares = np.array([
        self.data[f'share_{g}'].mean() 
        for g in self.categories
    ])
    
    # Marshallian elasticities
    marshallian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if shares[i] > 0:
                if i == j:
                    marshallian[i, j] = -1 + gamma[i, j]/shares[i] - beta[i]
                else:
                    marshallian[i, j] = gamma[i, j]/shares[i] - beta[i]*shares[j]/shares[i]
    
    # Expenditure elasticities
    expenditure = 1 + beta/shares
    
    # Hicksian elasticities (compensated)
    hicksian = marshallian + np.outer(expenditure, shares)
    
    return {
        'marshallian': marshallian,
        'hicksian': hicksian,
        'expenditure': expenditure
    }
```

### Demographic Scaling

```python
def aids_with_demographics(self):
    """AIDS with demographic translating."""
    # Ray's (1983) demographic translating
    # m₀(p, x, d) = m̄₀(p) + Σₖ θₖ dₖ
    
    for good in self.categories[:-1]:
        y = self.data[f'share_{good}']
        
        X_list = [
            np.ones(len(y)),
            *[self.data[f'log_price_{g}'] for g in self.categories],
            self.data['log_real_exp'],
            self.data['household_size'],  # Demographic
            self.data['num_children'],    # Demographic
            self.data['log_real_exp'] * self.data['household_size']  # Interaction
        ]
        
        X = np.column_stack(X_list)
        model = OLS(y, X)
        result = model.fit()
```

### Welfare Analysis

```python
def welfare_analysis(self, initial_prices, new_prices, params):
    """Calculate compensating and equivalent variation."""
    
    # Cost function at initial utility
    def cost_function(prices, utility, params):
        log_P = (params['alpha_0'] + 
                sum(params[f'alpha_{i}'] * np.log(prices[i]) for i in range(n)) +
                0.5 * sum(sum(params[f'gamma_{i}{j}'] * np.log(prices[i]) * np.log(prices[j])
                         for j in range(n)) for i in range(n)))
        
        b_p = np.prod([prices[i]**params[f'beta_{i}'] for i in range(n)])
        
        return np.exp(log_P + utility * np.log(b_p))
    
    # Initial utility
    initial_cost = self.data['expenditure'].mean()
    u_initial = self._indirect_utility(initial_prices, initial_cost, params)
    
    # Compensating Variation
    # How much money needed at new prices to maintain initial utility
    cv_cost = cost_function(new_prices, u_initial, params)
    cv = cv_cost - initial_cost
    
    # Equivalent Variation
    # How much money at initial prices equivalent to price change
    u_new = self._indirect_utility(new_prices, initial_cost, params)
    ev_cost = cost_function(initial_prices, u_new, params)
    ev = initial_cost - ev_cost
    
    return {'cv': cv, 'ev': ev}
```

## Pros and Cons

### Pros
- ✅ **Theoretically consistent**: Derived from utility maximization
- ✅ **Complete system**: All own- and cross-price elasticities
- ✅ **Flexible functional form**: Good approximation to any demand system
- ✅ **Adding-up satisfied**: Budget shares always sum to 1
- ✅ **Testable restrictions**: Can test and impose theory
- ✅ **Welfare analysis**: Direct welfare measurement

### Cons
- ❌ **Simultaneity**: All equations estimated together
- ❌ **Degrees of freedom**: Many parameters to estimate
- ❌ **Multicollinearity**: Prices often move together
- ❌ **Aggregation issues**: Assumes representative consumer
- ❌ **Zero shares**: Problems with zero expenditures
- ❌ **Endogeneity**: Prices may be endogenous

## Practical Implementation

### Full Example

```python
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load data
data = pd.read_csv('expenditure_data.csv')

# Categories
categories = ['food', 'clothing', 'transport', 'other']

# Calculate shares
for cat in categories:
    data[f'share_{cat}'] = data[f'exp_{cat}'] / data['total_exp']
    data[f'log_price_{cat}'] = np.log(data[f'price_{cat}'])

data['log_total_exp'] = np.log(data['total_exp'])

# Stone's price index
data['log_P_stone'] = sum(
    data[f'share_{cat}'] * data[f'log_price_{cat}']
    for cat in categories
)

# Real expenditure
data['log_real_exp'] = data['log_total_exp'] - data['log_P_stone']

# Estimate AIDS for each good (except last)
from statsmodels.api import add_constant

results = {}
for cat in categories[:-1]:
    # Build design matrix
    X = data[[f'log_price_{c}' for c in categories] + ['log_real_exp']]
    X = add_constant(X)
    y = data[f'share_{cat}']
    
    # Estimate
    model = OLS(y, X)
    results[cat] = model.fit()
    
    print(f"\n{cat} equation:")
    print(results[cat].summary())

# Calculate elasticities at mean
mean_shares = {cat: data[f'share_{cat}'].mean() for cat in categories}

# Extract coefficients and build elasticity matrix
n = len(categories)
elasticity_matrix = np.zeros((n, n))

for i, cat_i in enumerate(categories[:-1]):
    params = results[cat_i].params
    beta_i = params['log_real_exp']
    
    for j, cat_j in enumerate(categories):
        gamma_ij = params.get(f'log_price_{cat_j}', 0)
        
        if i == j:
            elasticity_matrix[i, j] = -1 + gamma_ij/mean_shares[cat_i] - beta_i
        else:
            elasticity_matrix[i, j] = gamma_ij/mean_shares[cat_i] - beta_i*mean_shares[cat_j]/mean_shares[cat_i]

print("\nElasticity Matrix:")
print(pd.DataFrame(elasticity_matrix, 
                   index=categories[:-1], 
                   columns=categories))
```

### Testing Restrictions

```python
from scipy.stats import chi2

def test_homogeneity(results, categories):
    """Test homogeneity restriction."""
    test_stats = []
    
    for cat in categories[:-1]:
        params = results[cat].params
        gamma_sum = sum(params.get(f'log_price_{c}', 0) 
                       for c in categories)
        se_sum = np.sqrt(sum(results[cat].bse.get(f'log_price_{c}', 0)**2 
                            for c in categories))
        
        t_stat = gamma_sum / se_sum
        test_stats.append(t_stat**2)
    
    # Chi-square test
    chi2_stat = sum(test_stats)
    p_value = 1 - chi2.cdf(chi2_stat, df=len(categories)-1)
    
    return chi2_stat, p_value

def test_symmetry(results, categories):
    """Test symmetry restriction."""
    violations = []
    
    for i, cat_i in enumerate(categories[:-1]):
        for j, cat_j in enumerate(categories[:-1]):
            if i < j:
                gamma_ij = results[cat_i].params.get(f'log_price_{cat_j}', 0)
                gamma_ji = results[cat_j].params.get(f'log_price_{cat_i}', 0)
                violations.append((gamma_ij - gamma_ji)**2)
    
    return sum(violations)
```

## Model Selection

### When to Use AIDS/QUAIDS

✅ **Use when:**
- Need complete demand system
- Want theoretically consistent elasticities
- Have expenditure/budget share data
- Need welfare analysis
- Want to test demand theory

❌ **Avoid when:**
- Have individual choice data
- Limited product categories (< 3)
- Many zero purchases
- Prices are highly endogenous

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Zero shares | Use Tobit or two-step procedures |
| Multicollinearity | Use principal components of prices |
| Endogeneity | Use lagged prices or cost shifters as instruments |
| Too many parameters | Impose symmetry and homogeneity |
| Poor fit | Try QUAIDS for non-linear Engel curves |

## References

- Deaton & Muellbauer (1980). "An Almost Ideal Demand System"
- Banks, Blundell & Lewbel (1997). "Quadratic Engel Curves and Consumer Demand" (QUAIDS)
- Ray (1983). "Measuring the Costs of Children" (Demographics)
- Poi (2012). "Easy Demand-System Estimation with quaids" (Stata implementation)

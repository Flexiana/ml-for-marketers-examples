# EconML: Double Machine Learning for Cross-Price Elasticity

## Overview

EconML implements causal machine learning methods for estimating treatment effects, particularly useful for price elasticity estimation where prices are endogenous (correlated with unobserved demand shocks).

## Methods Implemented

### 1. Double Machine Learning (DML)

#### What It Is
Double Machine Learning is a method for estimating causal effects when you have high-dimensional confounders. It uses machine learning to flexibly control for confounders while maintaining valid statistical inference.

#### How It Works

**Mathematical Foundation:**

The core DML model for elasticity estimation:
```
log(Q) = θ·log(P) + g(X) + ε
log(P) = m(X) + ν
```

Where:
- Q = quantity demanded
- P = price
- X = control variables (store characteristics, demographics, etc.)
- θ = elasticity parameter (our target)
- g(X) = nuisance function for outcome
- m(X) = nuisance function for treatment

**Two-Stage Procedure:**

1. **First Stage - Residualization:**
   - Predict E[Y|X] using ML model → get residuals Ỹ = Y - E[Y|X]
   - Predict E[T|X] using ML model → get residuals T̃ = T - E[T|X]

2. **Second Stage - Elasticity Estimation:**
   - Regress Ỹ on T̃: `Ỹ = θ·T̃ + η`
   - θ is the debiased elasticity estimate

**Cross-Fitting:** To avoid overfitting bias, the sample is split into K folds. Each fold is used for prediction while the model is trained on the other K-1 folds.

#### Pros
- ✅ Handles high-dimensional confounders
- ✅ Flexible functional forms via ML
- ✅ Valid inference (confidence intervals)
- ✅ Robust to model misspecification
- ✅ No need to specify functional form of g(X)

#### Cons
- ❌ Requires large samples
- ❌ Assumes unconfoundedness
- ❌ Computationally intensive
- ❌ May be sensitive to ML algorithm choice

#### Usage

```python
from econml.dml import LinearDML
import xgboost as xgb
import numpy as np
import pickle

# Input data requirements
# Y: outcome (log quantity) - shape (n, 1)
# T: treatment (log price) - shape (n, 1) or (n, k) for multiple treatments
# X: controls - shape (n, p)

# Create nuisance models
model_y = xgb.XGBRegressor(n_estimators=100, max_depth=5)
model_t = xgb.XGBRegressor(n_estimators=100, max_depth=5)

# Initialize DML
dml = LinearDML(
    model_y=model_y,     # ML model for outcome
    model_t=model_t,     # ML model for treatment
    discrete_treatment=False,
    cv=5,                # Number of cross-fitting folds
    random_state=42
)

# Fit the model
dml.fit(Y, T, X=X)

# Get elasticity estimate
elasticity = dml.effect(X).mean()
print(f"Elasticity: {elasticity:.3f}")

# Get confidence intervals
ci_lower, ci_upper = dml.effect_interval(X, alpha=0.05)
print(f"95% CI: [{ci_lower.mean():.3f}, {ci_upper.mean():.3f}]")

# Heterogeneous effects
heterogeneous_effects = dml.effect(X)  # Effect for each observation

# Serialization
with open('dml_model.pkl', 'wb') as f:
    pickle.dump(dml, f)

# Deserialization
with open('dml_model.pkl', 'rb') as f:
    dml_loaded = pickle.load(f)
```

### 2. Instrumental Variables with ML (DMLIV)

#### What It Is
DMLIV extends DML to handle endogenous treatments using instrumental variables, combining the flexibility of ML with the identification strategy of IV.

#### How It Works

**Mathematical Foundation:**
```
Y = θ·T + g(X) + ε
T = f(Z, X) + ν
E[ε|Z, X] = 0  (exclusion restriction)
```

Where Z are instruments (e.g., cost shifters, competitor prices).

**Procedure:**
1. Use ML to predict T from (Z, X)
2. Use ML to predict Y from X
3. Apply 2SLS on residuals

#### Pros
- ✅ Handles endogenous prices
- ✅ Flexible first stage via ML
- ✅ Weaker functional form assumptions

#### Cons
- ❌ Requires valid instruments
- ❌ Weak instruments problem persists
- ❌ More complex than standard DML

#### Usage

```python
from econml.iv.dml import DMLIV

# Additional requirement: Z - instruments
# Z: shape (n, m) where m is number of instruments

dmliv = DMLIV(
    model_y_xw=xgb.XGBRegressor(),   # E[Y|X]
    model_t_xw=xgb.XGBRegressor(),   # E[T|X]
    model_t_xwz=xgb.XGBRegressor(),  # E[T|X,Z]
    cv=3
)

dmliv.fit(Y, T, X=X, Z=Z)

# Get IV estimate of elasticity
elasticity_iv = dmliv.effect(X)
```

### 3. Causal Forests

#### What It Is
Causal Forests estimate heterogeneous treatment effects using an ensemble of causal trees, revealing how elasticities vary across different market segments.

#### How It Works

**Algorithm:**
1. Build many trees, each on a bootstrap sample
2. Each split maximizes heterogeneity in treatment effect
3. Average predictions across trees

**Splitting Criterion:**
Maximize variance of treatment effects:
```
max Var(τ(left)) + Var(τ(right))
```

#### Pros
- ✅ Discovers heterogeneous effects automatically
- ✅ Non-parametric
- ✅ Handles interactions naturally
- ✅ Provides confidence intervals

#### Cons
- ❌ Requires very large samples
- ❌ Can be slow to train
- ❌ Less interpretable than linear models

#### Usage

```python
from econml.dml import CausalForestDML

# Causal Forest for heterogeneous elasticities
cf_dml = CausalForestDML(
    model_y=RandomForestRegressor(),
    model_t=RandomForestRegressor(),
    n_estimators=500,
    min_samples_leaf=10,
    max_depth=10,
    cv=3
)

cf_dml.fit(Y, T, X=X_heterogeneity, W=X_controls)

# Get heterogeneous effects
effects = cf_dml.effect(X_heterogeneity)

# Feature importance for heterogeneity
importance = cf_dml.feature_importances_
```

### 4. Doubly Robust Learners

#### What It Is
DR learners combine outcome regression and propensity scores for robust estimation, providing consistent estimates even if one model is misspecified.

#### How It Works

**Double Robustness Property:**
The estimator is consistent if either:
- The outcome model E[Y|T,X] is correct, OR
- The propensity model P(T|X) is correct

**Augmented IPW Estimator:**
```
θ̂ = 1/n Σ[μ₁(X) - μ₀(X) + T·(Y-μ₁(X))/e(X) - (1-T)·(Y-μ₀(X))/(1-e(X))]
```

#### Pros
- ✅ Robust to partial model misspecification
- ✅ Efficient under correct specification
- ✅ Works well with binary treatments

#### Cons
- ❌ More complex implementation
- ❌ Requires good overlap in propensity scores
- ❌ Less suitable for continuous treatments

#### Usage

```python
from econml.dr import ForestDRLearner

# For binary treatment (high vs low price)
T_binary = (T > np.median(T)).astype(int)

dr = ForestDRLearner(
    model_propensity=RandomForestClassifier(),
    model_regression=RandomForestRegressor(),
    n_estimators=200,
    min_samples_leaf=10
)

dr.fit(Y, T_binary, X=X)
ate = dr.effect(X).mean()
```

## Input Data Requirements

### Standard Format
```python
# Required data structure
data = {
    'log_quantity': np.array([...]),      # Outcome (n,)
    'log_price': np.array([...]),         # Treatment (n,)
    'controls': np.array([[...]]),        # Confounders (n, p)
    'instruments': np.array([[...]]),     # For IV only (n, m)
}

# Data preprocessing
from sklearn.preprocessing import StandardScaler

# Scale continuous variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_continuous)

# Encode categoricals
X_encoded = pd.get_dummies(X_categorical)

# Combine
X = np.hstack([X_scaled, X_encoded])
```

## Model Selection Guidelines

| Scenario | Recommended Method |
|----------|-------------------|
| Clean randomization | Linear DML |
| Endogenous prices | DMLIV |
| Heterogeneous effects needed | Causal Forest DML |
| Binary treatment | DR Learner |
| High dimensions | Sparse Linear DML |

## Performance Considerations

- **Sample Size**: Minimum 1,000 observations, ideally 10,000+
- **Cross-fitting folds**: 5-10 folds typically sufficient
- **ML algorithms**: XGBoost/LightGBM often best for tabular data
- **Computation time**: O(K × ML complexity) where K = number of folds

## Common Pitfalls

1. **Weak Instruments**: Check first-stage F-statistic > 10
2. **No Variation**: Ensure sufficient price variation in data
3. **Perfect Multicollinearity**: Remove redundant features
4. **Extrapolation**: Be cautious about effects outside data support

## References

- Chernozhukov et al. (2018). "Double/debiased machine learning for treatment and structural parameters"
- Athey et al. (2019). "Generalized random forests"
- Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects"

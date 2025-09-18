# Cross-Price Elasticity Estimation: Modern Methods Showcase

A comprehensive demonstration of state-of-the-art methods for estimating cross-price elasticities using Python's leading econometric and machine learning libraries.

## Overview

This project implements working examples for cross-price elasticity estimation using:

- **EconML**: Double ML, IV methods, Causal Forests, DR learners
- **PyBLP**: BLP (Berry-Levinsohn-Pakes) random-coefficient logit demand models
- **LinearModels**: High-dimensional panel regressions with fixed effects and IV/2SLS
- **Statsmodels**: AIDS/QUAIDS demand systems
- **PyMC**: Bayesian hierarchical elasticity models
- **Scikit-learn/XGBoost**: DML pipelines with ML nuisance models

## Installation

**Recommended: Use a clean virtual environment to avoid dependency conflicts**

```bash
# Create virtual environment
python -m venv elasticity_env

# Activate environment
source elasticity_env/bin/activate  # On macOS/Linux
# OR
elasticity_env\Scripts\activate     # On Windows

# Install packages
pip install -r requirements.txt
```

**Quick activation (after setup):**
```bash
# Use the provided script
source activate_env.sh
```

## Quick Start

```bash
# Run all examples with comparison
python main_cross_price_elasticity.py

# Or run individual examples:
python data_preparation.py           # Generate synthetic data
python example_econml.py            # EconML causal ML methods
python example_pyblp.py             # PyBLP demand estimation
python example_linearmodels.py      # Panel data methods
python example_statsmodels_aids.py  # AIDS/QUAIDS systems
python example_pymc.py              # Bayesian hierarchical models
python example_sklearn_xgb_dml.py   # ML-based DML
```

## Data

The project uses synthetic retail scanner data with:
- Multiple products (substitutes and complements)
- Multiple stores and markets
- Time periods with seasonality
- Price variations (promotions, regular prices)
- Instrumental variables (cost shifters, competitor prices)
- Consumer demographics for heterogeneous effects

## Methods Demonstrated

### 1. EconML (Causal Machine Learning)
- **Double ML (DML)**: Flexible control for confounders using ML
- **IV Methods**: Handling endogenous prices with ML first stages
- **Causal Forests**: Heterogeneous treatment effects
- **DR Learners**: Doubly robust estimation

### 2. PyBLP (Structural Demand)
- **Random Coefficients Logit**: Flexible substitution patterns
- **Supply Side**: Joint demand-supply estimation
- **Demographics**: Consumer heterogeneity
- **Nested Logit**: Category-based substitution

### 3. LinearModels (Panel Methods)
- **Fixed Effects**: Entity and time fixed effects
- **2SLS/IV**: Instrumental variables for panels
- **Dynamic Panels**: Lagged dependent variables
- **Heterogeneous Effects**: Varying elasticities by group

### 4. Statsmodels (Demand Systems)
- **Linear AIDS**: Almost Ideal Demand System
- **QUAIDS**: Quadratic AIDS with flexible Engel curves
- **Restrictions**: Homogeneity, symmetry, adding-up
- **Welfare Analysis**: Consumer surplus calculations

### 5. PyMC (Bayesian Methods)
- **Hierarchical Models**: Partial pooling across products
- **Cross-Price Effects**: Full elasticity matrices
- **Varying Slopes**: Market-specific elasticities
- **Time-Varying**: Dynamic elasticity evolution

### 6. ML-Based DML
- **XGBoost DML**: Gradient boosting for nuisances
- **LightGBM**: High-dimensional features
- **Ensemble Methods**: Combining multiple ML models
- **Neural Networks**: Deep learning for non-linearities

## Key Features

### Elasticity Types Estimated
- **Own-price elasticities**: Response of quantity to own price
- **Cross-price elasticities**: Response to competitor prices
- **Income elasticities**: Response to consumer income
- **Promotional elasticities**: Impact of marketing

### Methodological Advances
- **Endogeneity handling**: IV, control functions, panel methods
- **Heterogeneity**: Random coefficients, hierarchical models
- **Non-linearities**: ML methods, polynomial terms
- **Uncertainty quantification**: Bayesian posteriors, bootstrap

## Output Files

Each method produces:
- Elasticity estimates with confidence/credible intervals
- Model diagnostics and convergence checks
- Visualization plots (saved as PNG)
- Comparison tables (CSV format)

Main outputs:
- `data/`: Generated datasets in various formats
- `*.png`: Visualization plots for each method
- `elasticity_comparison.csv`: Summary comparison table
- `summary_comparison.png`: Overall comparison plot

## Results Interpretation

### True Values (from data generation)
- Own-price elasticity: -1.2
- Within-category cross-price: 0.4 (substitutes)
- Complement cross-price: -0.15
- Unrelated products: 0.02

### Method Selection Guide

Choose based on your needs:

| Scenario | Recommended Method |
|----------|-------------------|
| Causal inference focus | EconML (DML, IV) |
| Rich substitution patterns | PyBLP |
| Panel data | LinearModels |
| Complete demand system | AIDS/QUAIDS |
| Uncertainty quantification | PyMC |
| High-dimensional controls | XGBoost/LightGBM DML |
| Limited assumptions | ML-based methods |

## Performance Notes

- Full execution takes ~10-15 minutes on a modern machine
- PyBLP and PyMC examples are computationally intensive
- Reduce data size or iterations for faster testing

## Technical Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- Multi-core processor for parallel sampling

## Troubleshooting

Common issues and solutions:

1. **Memory errors**: Reduce number of markets/products in examples
2. **Convergence warnings**: Increase iterations or adjust priors
3. **Import errors**: Ensure all packages installed via `pip install -r requirements.txt`

## Extensions

Possible extensions to explore:
- Real data applications (scanner data, e-commerce)
- Dynamic pricing strategies
- Competition analysis
- Demand forecasting
- Optimal pricing algorithms

## References

Key papers for each method:
- **DML**: Chernozhukov et al. (2018) "Double/debiased machine learning"
- **BLP**: Berry, Levinsohn & Pakes (1995) "Automobile prices in market equilibrium"
- **AIDS**: Deaton & Muellbauer (1980) "An almost ideal demand system"
- **Causal Forests**: Athey et al. (2019) "Generalized random forests"

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

*This showcase demonstrates modern econometric methods for elasticity estimation. The synthetic data and true parameters allow for method validation and comparison.*

"""
PyBLP Example for Cross-Price Elasticity Estimation

This module demonstrates BLP (Berry, Levinsohn, Pakes) random-coefficient logit
demand estimation using the PyBLP library. BLP models are particularly useful for:

1. Estimating flexible substitution patterns
2. Accounting for consumer heterogeneity 
3. Handling endogenous prices with instruments
4. Recovering full elasticity matrices

The BLP model allows for rich substitution patterns by incorporating:
- Random coefficients on product characteristics
- Consumer demographics interacted with characteristics
- Unobserved product characteristics
"""

import numpy as np
import pandas as pd
import pyblp
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns


class BLPElasticityEstimator:
    """BLP random-coefficient logit demand estimation for cross-price elasticities."""
    
    def __init__(self, data_path: str = 'data/blp_data.csv'):
        """Initialize with BLP-formatted data."""
        self.df = pd.read_csv(data_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare data in PyBLP format."""
        
        # Ensure proper data types
        self.df['market_ids'] = self.df['market_ids'].astype(str)
        self.df['product_ids'] = self.df['product_id'].astype(str)
        
        # Create firm IDs from brands
        self.df['firm_ids'] = self.df['brand'].astype(str)
        
        # Ensure we have prices
        self.df['prices'] = self.df['price']
        
        # Create shares properly
        self.df['shares'] = self.df['market_share']
        
        # Add node data for integration
        self.df['nodes0'] = 0
        self.df['nodes1'] = 0
        self.df['nodes2'] = 0
        self.df['nodes3'] = 0
        
        print(f"Prepared BLP data: {len(self.df)} observations")
        print(f"Markets: {self.df['market_ids'].nunique()}")
        print(f"Products: {self.df['product_ids'].nunique()}")
        print(f"Firms: {self.df['firm_ids'].nunique()}")
    
    def example_1_basic_blp(self) -> Dict:
        """
        Example 1: Basic BLP Model
        
        Estimates demand with random coefficients on price and product characteristics.
        """
        print("\n" + "="*60)
        print("EXAMPLE 1: Basic BLP Model")
        print("="*60)
        
        # Select a subset of markets for computational efficiency
        markets_subset = self.df['market_ids'].unique()[:50]
        data = self.df[self.df['market_ids'].isin(markets_subset)].copy()
        
        print(f"\nUsing {len(markets_subset)} markets for estimation")
        print(f"Total observations: {len(data)}")
        
        # Configure PyBLP formulation
        # X1: Linear characteristics (mean utility)
        # X2: Characteristics with random coefficients
        # X3: Cost shifters for supply side
        
        X1_formulation = pyblp.Formulation('1 + prices + promotion + quality_premium + quality_value')
        X2_formulation = pyblp.Formulation('1 + prices')
        
        # Add instruments
        data['demand_instruments0'] = data['wholesale_cost']
        data['demand_instruments1'] = data['transportation_cost']
        data['demand_instruments2'] = data['num_rival_products']
        
        # Configure problem
        problem = pyblp.Problem(
            product_formulations=(X1_formulation, X2_formulation),
            product_data=data,
            agent_formulation=pyblp.Formulation('1'),
            rc_type='random'
        )
        
        print("\nProblem configuration:")
        print(problem)
        
        # Configure optimization
        optimization = pyblp.Optimization('l-bfgs-b', {'ftol': 1e-8})
        
        # Initial parameter values
        # Sigma: standard deviations of random coefficients
        initial_sigma = np.array([
            [0.5],  # Constant
            [0.3],  # Price
        ])
        
        print("\nEstimating BLP model...")
        
        # Solve the problem
        results = problem.solve(
            sigma=initial_sigma,
            optimization=optimization,
            method='1s',  # One-step GMM
        )
        
        print("\nEstimation Results:")
        print(results)
        
        # Extract parameters
        beta = results.beta
        sigma = results.sigma
        
        print("\n" + "-"*40)
        print("Parameter Estimates:")
        print(f"Beta (mean coefficients):")
        for i, name in enumerate(['Constant', 'Price', 'Promotion', 'Quality_Premium', 'Quality_Value']):
            if i < len(beta):
                print(f"  {name}: {beta[i]:.4f}")
        
        print(f"\nSigma (std dev of random coefficients):")
        for i, name in enumerate(['Constant', 'Price']):
            if i < len(sigma):
                print(f"  {name}: {sigma[i, 0]:.4f}")
        
        # Compute elasticities
        print("\n" + "-"*40)
        print("Computing Elasticities...")
        
        elasticities = results.compute_elasticities()
        
        # Get own-price elasticities
        own_elasticities = []
        for market_id in markets_subset[:10]:  # Sample of markets
            market_data = elasticities[data['market_ids'] == market_id]
            if len(market_data) > 0:
                # Diagonal elements are own-price elasticities
                for i in range(len(market_data)):
                    if i < len(market_data):
                        own_elasticities.append(market_data[i, i])
        
        if own_elasticities:
            print(f"Average own-price elasticity: {np.mean(own_elasticities):.3f}")
            print(f"Std dev: {np.std(own_elasticities):.3f}")
        
        results_dict = {
            'problem': problem,
            'results': results,
            'elasticities': elasticities,
            'beta': beta,
            'sigma': sigma,
            'own_elasticities': own_elasticities
        }
        
        return results_dict
    
    def example_2_supply_side_blp(self) -> Dict:
        """
        Example 2: BLP with Supply Side
        
        Jointly estimates demand and supply (marginal costs) for more
        accurate elasticity estimates.
        """
        print("\n" + "="*60)
        print("EXAMPLE 2: BLP with Supply Side")
        print("="*60)
        
        # Use subset for efficiency
        markets_subset = self.df['market_ids'].unique()[:30]
        data = self.df[self.df['market_ids'].isin(markets_subset)].copy()
        
        # Demand side formulations
        X1_formulation = pyblp.Formulation('1 + prices + promotion')
        X2_formulation = pyblp.Formulation('1 + prices')
        
        # Supply side formulation (marginal cost shifters)
        X3_formulation = pyblp.Formulation('1 + wholesale_cost + transportation_cost')
        
        # Add instruments
        data['demand_instruments0'] = data['wholesale_cost']
        data['demand_instruments1'] = data['transportation_cost']
        
        # Configure problem with supply side
        problem = pyblp.Problem(
            product_formulations=(X1_formulation, X2_formulation, X3_formulation),
            product_data=data,
            agent_formulation=pyblp.Formulation('1'),
            rc_type='random'
        )
        
        print("Problem with supply side:")
        print(problem)
        
        # Initial values
        initial_sigma = np.array([
            [0.5],  # Constant
            [0.3],  # Price
        ])
        
        print("\nEstimating BLP with supply side...")
        
        # Solve with supply side
        results = problem.solve(
            sigma=initial_sigma,
            costs_type='linear',
            optimization=pyblp.Optimization('l-bfgs-b', {'ftol': 1e-8}),
            method='1s'
        )
        
        print("\nResults with Supply Side:")
        print(results)
        
        # Extract marginal costs
        costs = results.compute_costs()
        
        # Compute markups
        markups = data['prices'].values - costs
        markup_pct = (markups / data['prices'].values) * 100
        
        print("\n" + "-"*40)
        print("Markup Analysis:")
        print(f"Average markup: ${np.mean(markups):.2f}")
        print(f"Average markup %: {np.mean(markup_pct):.1f}%")
        print(f"Median markup %: {np.median(markup_pct):.1f}%")
        
        # Compute profit elasticities (useful for understanding firm behavior)
        profit_elasticities = results.compute_profit_elasticities()
        
        results_dict = {
            'problem': problem,
            'results': results,
            'costs': costs,
            'markups': markups,
            'markup_pct': markup_pct,
            'profit_elasticities': profit_elasticities
        }
        
        return results_dict
    
    def example_3_demographic_interactions(self) -> Dict:
        """
        Example 3: BLP with Demographic Interactions
        
        Allows preferences to vary with observed consumer demographics,
        capturing heterogeneous price sensitivities.
        """
        print("\n" + "="*60)
        print("EXAMPLE 3: BLP with Demographics")
        print("="*60)
        
        # Use subset
        markets_subset = self.df['market_ids'].unique()[:30]
        data = self.df[self.df['market_ids'].isin(markets_subset)].copy()
        
        # Create demographic data for agents
        # Each market has distribution of consumer types
        agent_data = []
        
        for market in markets_subset:
            # Create agents with different income levels
            for income_level in [30000, 60000, 90000, 120000]:
                agent_data.append({
                    'market_ids': market,
                    'weights': 0.25,  # Equal weights
                    'nodes0': income_level / 100000,  # Normalized income
                    'income': income_level
                })
        
        agent_data = pd.DataFrame(agent_data)
        
        # Formulations with demographic interactions
        X1_formulation = pyblp.Formulation('1 + prices + promotion')
        
        # Random coefficients and demographic interactions
        # Prices interacted with income
        X2_formulation = pyblp.Formulation('1 + prices')
        demographics_formulation = pyblp.Formulation('income')
        
        # Add instruments
        data['demand_instruments0'] = data['wholesale_cost']
        data['demand_instruments1'] = data['transportation_cost']
        
        # Configure problem
        problem = pyblp.Problem(
            product_formulations=(X1_formulation, X2_formulation),
            product_data=data,
            agent_formulation=demographics_formulation,
            agent_data=agent_data,
            rc_type='random'
        )
        
        print("Problem with demographics:")
        print(problem)
        
        # Initial values
        initial_sigma = np.array([
            [0.5],  # Constant
            [0.3],  # Price
        ])
        
        initial_pi = np.array([
            [0.0],  # Constant not interacted with demographics
            [0.1],  # Price interacted with income (higher income = less price sensitive)
        ])
        
        print("\nEstimating with demographic interactions...")
        
        # Solve
        results = problem.solve(
            sigma=initial_sigma,
            pi=initial_pi,
            optimization=pyblp.Optimization('l-bfgs-b', {'ftol': 1e-8}),
            method='1s'
        )
        
        print("\nResults with Demographics:")
        print(results)
        
        # Extract parameters
        pi = results.pi
        
        print("\n" + "-"*40)
        print("Demographic Interaction Parameters (Pi):")
        print(f"Price × Income: {pi[1, 0]:.4f}")
        
        if pi[1, 0] > 0:
            print("→ Higher income consumers are less price sensitive")
        else:
            print("→ Higher income consumers are more price sensitive")
        
        # Compute elasticities by demographic group
        print("\n" + "-"*40)
        print("Elasticities by Income Level:")
        
        elasticities = results.compute_elasticities()
        
        # Sample elasticities for different markets (proxy for demographics)
        for i, market in enumerate(markets_subset[:5]):
            market_elasticities = elasticities[data['market_ids'] == market]
            if len(market_elasticities) > 0:
                own_elast = np.diag(market_elasticities).mean()
                print(f"Market {market}: {own_elast:.3f}")
        
        results_dict = {
            'problem': problem,
            'results': results,
            'pi': pi,
            'elasticities': elasticities
        }
        
        return results_dict
    
    def example_4_nested_logit(self) -> Dict:
        """
        Example 4: Nested Logit as Special Case
        
        Estimates a nested logit model where products in the same category
        are closer substitutes.
        """
        print("\n" + "="*60)
        print("EXAMPLE 4: Nested Logit Model")
        print("="*60)
        
        # Use subset
        markets_subset = self.df['market_ids'].unique()[:50]
        data = self.df[self.df['market_ids'].isin(markets_subset)].copy()
        
        # Create nesting structure based on product categories
        data['nesting_ids'] = data['category']
        
        # Formulation for nested logit
        X1_formulation = pyblp.Formulation('1 + prices + promotion')
        
        # Add instruments
        data['demand_instruments0'] = data['wholesale_cost']
        data['demand_instruments1'] = data['transportation_cost']
        data['demand_instruments2'] = data['num_rival_products']
        
        # Configure nested logit problem
        problem = pyblp.Problem(
            product_formulations=(X1_formulation,),
            product_data=data,
            # Rho parameter for nesting
            rc_type='nested',
            epsilon_scale=1.0
        )
        
        print("Nested Logit Problem:")
        print(problem)
        
        # Initial nesting parameter (0 < rho < 1)
        # Higher rho = more correlation within nest
        initial_rho = np.array([0.7])
        
        print("\nEstimating nested logit...")
        
        # Solve
        results = problem.solve(
            rho=initial_rho,
            optimization=pyblp.Optimization('l-bfgs-b', {'ftol': 1e-8})
        )
        
        print("\nNested Logit Results:")
        print(results)
        
        # Extract nesting parameter
        rho = results.rho[0]
        
        print("\n" + "-"*40)
        print(f"Nesting parameter (rho): {rho:.3f}")
        
        if rho > 0.5:
            print("→ Strong within-nest correlation (products in same category are close substitutes)")
        else:
            print("→ Weak within-nest correlation")
        
        # Compute elasticities
        elasticities = results.compute_elasticities()
        
        # Analyze substitution patterns within vs across nests
        print("\n" + "-"*40)
        print("Substitution Patterns:")
        
        # Get products by category
        categories = data['category'].unique()
        
        within_nest_elasticities = []
        across_nest_elasticities = []
        
        for market in markets_subset[:10]:
            market_data = data[data['market_ids'] == market]
            market_elast = elasticities[data['market_ids'] == market]
            
            if len(market_elast) > 0:
                for i, row_i in enumerate(market_data.itertuples()):
                    for j, row_j in enumerate(market_data.itertuples()):
                        if i != j:  # Cross-price elasticity
                            elast = market_elast[i, j]
                            if row_i.category == row_j.category:
                                within_nest_elasticities.append(elast)
                            else:
                                across_nest_elasticities.append(elast)
        
        if within_nest_elasticities and across_nest_elasticities:
            print(f"Average within-category cross elasticity: {np.mean(within_nest_elasticities):.3f}")
            print(f"Average across-category cross elasticity: {np.mean(across_nest_elasticities):.3f}")
            
            ratio = np.mean(within_nest_elasticities) / np.mean(across_nest_elasticities) if np.mean(across_nest_elasticities) != 0 else 0
            print(f"Ratio (within/across): {ratio:.2f}")
            
            if ratio > 2:
                print("→ Strong category-based substitution patterns")
        
        results_dict = {
            'problem': problem,
            'results': results,
            'rho': rho,
            'within_nest_elast': within_nest_elasticities,
            'across_nest_elast': across_nest_elasticities
        }
        
        return results_dict
    
    def example_5_optimal_instruments(self) -> Dict:
        """
        Example 5: Optimal Instruments
        
        Uses PyBLP's optimal instrument functionality to improve
        efficiency of estimates.
        """
        print("\n" + "="*60)
        print("EXAMPLE 5: Optimal Instruments")
        print("="*60)
        
        # Use subset
        markets_subset = self.df['market_ids'].unique()[:30]
        data = self.df[self.df['market_ids'].isin(markets_subset)].copy()
        
        # Formulations
        X1_formulation = pyblp.Formulation('1 + prices')
        X2_formulation = pyblp.Formulation('1 + prices')
        
        # Initial instruments
        data['demand_instruments0'] = data['wholesale_cost']
        data['demand_instruments1'] = data['transportation_cost']
        
        # Configure problem
        initial_problem = pyblp.Problem(
            product_formulations=(X1_formulation, X2_formulation),
            product_data=data,
            agent_formulation=pyblp.Formulation('1'),
            rc_type='random'
        )
        
        print("Initial problem:")
        print(initial_problem)
        
        # First-stage estimation
        print("\nFirst-stage estimation...")
        
        initial_sigma = np.array([
            [0.5],
            [0.3]
        ])
        
        initial_results = initial_problem.solve(
            sigma=initial_sigma,
            optimization=pyblp.Optimization('l-bfgs-b', {'ftol': 1e-8}),
            method='1s'
        )
        
        print("First-stage results:")
        print(initial_results)
        
        # Compute optimal instruments
        print("\n" + "-"*40)
        print("Computing optimal instruments...")
        
        # Get Jacobian for optimal instruments
        xi_jacobian = initial_results.compute_xi_by_theta_jacobian()
        
        # Update problem with optimal instruments
        data_with_optimal = data.copy()
        
        # Add optimal instruments
        # PyBLP computes these based on the Jacobian
        instrument_results = initial_results.compute_optimal_instruments()
        
        for i, name in enumerate(['demand_instruments_optimal' + str(j) for j in range(instrument_results.shape[1])]):
            if i < instrument_results.shape[1]:
                data_with_optimal[name] = instrument_results[:, i]
        
        # Re-estimate with optimal instruments
        optimal_problem = pyblp.Problem(
            product_formulations=(X1_formulation, X2_formulation),
            product_data=data_with_optimal,
            agent_formulation=pyblp.Formulation('1'),
            rc_type='random'
        )
        
        print("\nRe-estimating with optimal instruments...")
        
        optimal_results = optimal_problem.solve(
            sigma=initial_sigma,
            optimization=pyblp.Optimization('l-bfgs-b', {'ftol': 1e-8}),
            method='1s'
        )
        
        print("\nOptimal instrument results:")
        print(optimal_results)
        
        # Compare standard errors
        print("\n" + "-"*40)
        print("Efficiency Gains:")
        
        initial_se = initial_results.sigma_se
        optimal_se = optimal_results.sigma_se
        
        print("Standard Errors Comparison:")
        print(f"Initial sigma SE: {initial_se}")
        print(f"Optimal sigma SE: {optimal_se}")
        
        if np.all(optimal_se < initial_se):
            print("→ Optimal instruments improved efficiency (lower SEs)")
        
        # Compute elasticities with optimal estimates
        elasticities = optimal_results.compute_elasticities()
        
        results_dict = {
            'initial_results': initial_results,
            'optimal_results': optimal_results,
            'elasticities': elasticities,
            'efficiency_gain': (initial_se - optimal_se) / initial_se * 100
        }
        
        return results_dict
    
    def analyze_substitution_patterns(self, results: pyblp.ProblemResults) -> pd.DataFrame:
        """Analyze detailed substitution patterns from BLP results."""
        
        print("\n" + "="*60)
        print("SUBSTITUTION PATTERN ANALYSIS")
        print("="*60)
        
        # Compute full elasticity matrices
        elasticities = results.compute_elasticities()
        
        # Compute diversion ratios
        diversion_ratios = results.compute_diversion_ratios()
        
        # Get sample market for detailed analysis
        markets = self.df['market_ids'].unique()[:5]
        
        substitution_data = []
        
        for market in markets:
            market_products = self.df[self.df['market_ids'] == market]
            market_elast = elasticities[self.df['market_ids'] == market]
            market_diversion = diversion_ratios[self.df['market_ids'] == market]
            
            if len(market_elast) > 0:
                for i, prod_i in enumerate(market_products.itertuples()):
                    # Own-price elasticity
                    own_elast = market_elast[i, i]
                    
                    # Find closest substitute
                    cross_elasts = market_elast[i, :]
                    cross_elasts[i] = -np.inf  # Exclude own
                    
                    if len(cross_elasts) > 1:
                        closest_sub_idx = np.argmax(cross_elasts)
                        closest_sub = market_products.iloc[closest_sub_idx]['product_id']
                        closest_sub_elast = cross_elasts[closest_sub_idx]
                        
                        # Diversion ratio to closest substitute
                        diversion_to_closest = market_diversion[i, closest_sub_idx] if closest_sub_idx < len(market_diversion) else 0
                        
                        substitution_data.append({
                            'market': market,
                            'product': prod_i.product_id,
                            'category': prod_i.category,
                            'quality': prod_i.quality_tier,
                            'own_elasticity': own_elast,
                            'closest_substitute': closest_sub,
                            'cross_elasticity_to_closest': closest_sub_elast,
                            'diversion_to_closest': diversion_to_closest
                        })
        
        sub_df = pd.DataFrame(substitution_data)
        
        if not sub_df.empty:
            print("\nSubstitution Patterns Summary:")
            print("-" * 40)
            
            # By category
            print("\nBy Product Category:")
            for cat in sub_df['category'].unique():
                cat_data = sub_df[sub_df['category'] == cat]
                print(f"\n{cat}:")
                print(f"  Avg own-price elasticity: {cat_data['own_elasticity'].mean():.3f}")
                print(f"  Avg cross-elasticity to closest: {cat_data['cross_elasticity_to_closest'].mean():.3f}")
                print(f"  Avg diversion to closest: {cat_data['diversion_to_closest'].mean():.3f}")
            
            # By quality tier
            print("\nBy Quality Tier:")
            for quality in sub_df['quality'].unique():
                qual_data = sub_df[sub_df['quality'] == quality]
                print(f"\n{quality}:")
                print(f"  Avg own-price elasticity: {qual_data['own_elasticity'].mean():.3f}")
                print(f"  Avg cross-elasticity: {qual_data['cross_elasticity_to_closest'].mean():.3f}")
        
        return sub_df
    
    def visualize_results(self, results: Dict):
        """Visualize BLP estimation results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Own-price elasticity distribution
        if 'basic' in results and 'own_elasticities' in results['basic']:
            ax = axes[0, 0]
            own_elast = results['basic']['own_elasticities']
            ax.hist(own_elast, bins=20, alpha=0.7, color='blue')
            ax.axvline(x=np.mean(own_elast), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(own_elast):.2f}')
            ax.set_xlabel('Own-Price Elasticity')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Own-Price Elasticities')
            ax.legend()
        
        # Plot 2: Markups
        if 'supply' in results and 'markup_pct' in results['supply']:
            ax = axes[0, 1]
            markups = results['supply']['markup_pct']
            ax.hist(markups, bins=20, alpha=0.7, color='green')
            ax.axvline(x=np.median(markups), color='red', linestyle='--',
                      label=f'Median: {np.median(markups):.1f}%')
            ax.set_xlabel('Markup (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Price-Cost Markups')
            ax.legend()
        
        # Plot 3: Random coefficients
        if 'basic' in results and 'sigma' in results['basic']:
            ax = axes[1, 0]
            sigma = results['basic']['sigma']
            params = ['Constant', 'Price'][:len(sigma)]
            ax.bar(params, sigma.flatten()[:len(params)])
            ax.set_ylabel('Sigma (Std Dev)')
            ax.set_title('Random Coefficient Standard Deviations')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 4: Substitution patterns (if nested logit)
        if 'nested' in results:
            ax = axes[1, 1]
            if 'within_nest_elast' in results['nested'] and 'across_nest_elast' in results['nested']:
                within = results['nested']['within_nest_elast']
                across = results['nested']['across_nest_elast']
                
                if within and across:
                    data_to_plot = [within[:100], across[:100]]  # Limit for visualization
                    bp = ax.boxplot(data_to_plot, labels=['Within Category', 'Across Category'])
                    ax.set_ylabel('Cross-Price Elasticity')
                    ax.set_title('Substitution Patterns: Within vs Across Categories')
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('pyblp_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nResults visualization saved as 'pyblp_results.png'")


def main():
    """Run all PyBLP examples."""
    
    print("="*60)
    print("PyBLP CROSS-PRICE ELASTICITY ESTIMATION")
    print("="*60)
    
    # Initialize estimator
    estimator = BLPElasticityEstimator()
    
    # Store all results
    all_results = {}
    
    # Run examples
    try:
        all_results['basic'] = estimator.example_1_basic_blp()
    except Exception as e:
        print(f"Error in basic BLP: {e}")
    
    try:
        all_results['supply'] = estimator.example_2_supply_side_blp()
    except Exception as e:
        print(f"Error in supply-side BLP: {e}")
    
    try:
        all_results['demographics'] = estimator.example_3_demographic_interactions()
    except Exception as e:
        print(f"Error in demographic BLP: {e}")
    
    try:
        all_results['nested'] = estimator.example_4_nested_logit()
    except Exception as e:
        print(f"Error in nested logit: {e}")
    
    try:
        all_results['optimal'] = estimator.example_5_optimal_instruments()
    except Exception as e:
        print(f"Error in optimal instruments: {e}")
    
    # Analyze substitution patterns if we have results
    if 'basic' in all_results and 'results' in all_results['basic']:
        substitution_df = estimator.analyze_substitution_patterns(all_results['basic']['results'])
    
    # Visualize results
    estimator.visualize_results(all_results)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nKey findings from PyBLP estimation:")
    print("1. Random coefficients capture unobserved heterogeneity")
    print("2. Supply-side estimation reveals markup patterns")
    print("3. Demographics explain variation in price sensitivity")
    print("4. Nesting structure shows category-based substitution")
    print("5. Optimal instruments improve estimation efficiency")
    
    return all_results


if __name__ == "__main__":
    results = main()

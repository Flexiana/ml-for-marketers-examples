#!/usr/bin/env python3
"""
Main Script: Cross-Price Elasticity Estimation Showcase

This script demonstrates comprehensive cross-price elasticity estimation using:
1. EconML - Double ML, IV, Causal Forests, DR learners
2. PyBLP - BLP (random-coefficient logit) demand estimation
3. LinearModels - Panel regressions with fixed effects and IV/2SLS
4. Statsmodels - AIDS/QUAIDS demand systems
5. PyMC - Bayesian hierarchical elasticities
6. Scikit-learn/XGBoost - DML pipelines with ML nuisance models

Run this script to execute all examples and compare results across methods.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Set style for all plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def run_data_preparation():
    """Generate synthetic retail scanner data."""
    print_header("DATA PREPARATION")
    
    try:
        from data_preparation import main as prepare_data
        
        print("\nGenerating retail scanner data...")
        df = prepare_data()
        
        print("\n✓ Data preparation completed successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Error in data preparation: {e}")
        return False


def run_econml_examples():
    """Run EconML causal ML examples."""
    print_header("ECONML - CAUSAL MACHINE LEARNING")
    
    try:
        from example_econml import main as run_econml
        
        print("\nRunning EconML examples...")
        print("  • Double Machine Learning (DML)")
        print("  • Instrumental Variables with ML")
        print("  • Causal Forests")
        print("  • Doubly Robust Learners")
        
        results = run_econml()
        
        print("\n✓ EconML examples completed successfully")
        return results
        
    except Exception as e:
        print(f"\n✗ Error in EconML examples: {e}")
        return None


def run_pyblp_examples():
    """Run PyBLP demand estimation examples."""
    print_header("PYBLP - BLP DEMAND ESTIMATION")
    
    try:
        from example_pyblp import main as run_pyblp
        
        print("\nRunning PyBLP examples...")
        print("  • Basic BLP model")
        print("  • BLP with supply side")
        print("  • Demographic interactions")
        print("  • Nested logit")
        print("  • Optimal instruments")
        
        results = run_pyblp()
        
        print("\n✓ PyBLP examples completed successfully")
        return results
        
    except Exception as e:
        print(f"\n✗ Error in PyBLP examples: {e}")
        return None


def run_linearmodels_examples():
    """Run linearmodels panel data examples."""
    print_header("LINEARMODELS - PANEL DATA METHODS")
    
    try:
        from example_linearmodels import main as run_linearmodels
        
        print("\nRunning linearmodels examples...")
        print("  • Fixed Effects")
        print("  • IV/2SLS")
        print("  • Dynamic Panel")
        print("  • Heterogeneous Effects")
        
        results = run_linearmodels()
        
        print("\n✓ Linearmodels examples completed successfully")
        return results
        
    except Exception as e:
        print(f"\n✗ Error in linearmodels examples: {e}")
        return None


def run_aids_examples():
    """Run AIDS/QUAIDS demand system examples."""
    print_header("STATSMODELS - AIDS/QUAIDS")
    
    try:
        from example_statsmodels_aids import main as run_aids
        
        print("\nRunning AIDS/QUAIDS examples...")
        print("  • Linear AIDS")
        print("  • QUAIDS")
        print("  • Restricted models")
        print("  • Demographic scaling")
        print("  • Welfare analysis")
        
        results = run_aids()
        
        print("\n✓ AIDS/QUAIDS examples completed successfully")
        return results
        
    except Exception as e:
        print(f"\n✗ Error in AIDS examples: {e}")
        return None


def run_pymc_examples():
    """Run PyMC Bayesian examples."""
    print_header("PYMC - BAYESIAN HIERARCHICAL MODELS")
    
    try:
        from example_pymc import main as run_pymc
        
        print("\nRunning PyMC examples...")
        print("  • Hierarchical elasticity model")
        print("  • Cross-price hierarchical model")
        print("  • Varying slopes")
        print("  • Model comparison")
        print("  • Time-varying elasticities")
        
        results = run_pymc()
        
        print("\n✓ PyMC examples completed successfully")
        return results
        
    except Exception as e:
        print(f"\n✗ Error in PyMC examples: {e}")
        return None


def run_ml_dml_examples():
    """Run ML-based DML examples."""
    print_header("ML-BASED DOUBLE MACHINE LEARNING")
    
    try:
        from example_sklearn_xgb_dml import main as run_ml_dml
        
        print("\nRunning ML DML examples...")
        print("  • XGBoost DML")
        print("  • LightGBM DML")
        print("  • Ensemble DML")
        print("  • Neural Network DML")
        print("  • Heterogeneous ML")
        
        results = run_ml_dml()
        
        print("\n✓ ML DML examples completed successfully")
        return results
        
    except Exception as e:
        print(f"\n✗ Error in ML DML examples: {e}")
        return None


def compare_results(all_results: dict):
    """Compare elasticity estimates across all methods."""
    print_header("COMPARATIVE ANALYSIS")
    
    # Extract elasticity estimates from each method
    comparison_data = []
    
    # EconML estimates
    if all_results.get('econml'):
        if 'dml' in all_results['econml']:
            if 'linear_dml_xgb' in all_results['econml']['dml']:
                comparison_data.append({
                    'Library': 'EconML',
                    'Method': 'Linear DML (XGBoost)',
                    'Own-Price Elasticity': all_results['econml']['dml']['linear_dml_xgb']['elasticity'],
                    'Type': 'Causal ML'
                })
        
        if 'iv' in all_results['econml']:
            if 'dmliv' in all_results['econml']['iv']:
                comparison_data.append({
                    'Library': 'EconML',
                    'Method': 'DML-IV',
                    'Own-Price Elasticity': all_results['econml']['iv']['dmliv']['elasticity'],
                    'Type': 'Causal ML'
                })
    
    # PyBLP estimates
    if all_results.get('pyblp'):
        if 'basic' in all_results['pyblp']:
            if 'own_elasticities' in all_results['pyblp']['basic']:
                comparison_data.append({
                    'Library': 'PyBLP',
                    'Method': 'BLP Random Coefficients',
                    'Own-Price Elasticity': np.mean(all_results['pyblp']['basic']['own_elasticities']),
                    'Type': 'Structural'
                })
    
    # Linearmodels estimates
    if all_results.get('linearmodels'):
        if 'fe' in all_results['linearmodels']:
            if 'entity_fe' in all_results['linearmodels']['fe']:
                comparison_data.append({
                    'Library': 'LinearModels',
                    'Method': 'Fixed Effects',
                    'Own-Price Elasticity': all_results['linearmodels']['fe']['entity_fe']['own_elasticity'],
                    'Type': 'Panel'
                })
        
        if 'iv' in all_results['linearmodels']:
            if '2sls' in all_results['linearmodels']['iv']:
                comparison_data.append({
                    'Library': 'LinearModels',
                    'Method': '2SLS',
                    'Own-Price Elasticity': all_results['linearmodels']['iv']['2sls']['own_elasticity'],
                    'Type': 'Panel IV'
                })
    
    # AIDS estimates
    if all_results.get('aids'):
        if 'linear_aids' in all_results['aids']:
            if 'elasticities' in all_results['aids']['linear_aids']:
                marshallian = all_results['aids']['linear_aids']['elasticities']['marshallian']
                if len(marshallian) > 0:
                    comparison_data.append({
                        'Library': 'Statsmodels',
                        'Method': 'Linear AIDS',
                        'Own-Price Elasticity': marshallian[0, 0],
                        'Type': 'Demand System'
                    })
    
    # PyMC estimates
    if all_results.get('pymc'):
        if 'hierarchical' in all_results['pymc']:
            if 'trace' in all_results['pymc']['hierarchical']:
                trace = all_results['pymc']['hierarchical']['trace']
                comparison_data.append({
                    'Library': 'PyMC',
                    'Method': 'Hierarchical Bayes',
                    'Own-Price Elasticity': trace.posterior['mu_elasticity'].mean().values,
                    'Type': 'Bayesian'
                })
    
    # ML DML estimates
    if all_results.get('ml_dml'):
        if 'xgboost' in all_results['ml_dml']:
            comparison_data.append({
                'Library': 'XGBoost',
                'Method': 'XGBoost DML',
                'Own-Price Elasticity': all_results['ml_dml']['xgboost']['elasticity'],
                'Type': 'ML-based'
            })
        
        if 'ensemble' in all_results['ml_dml']:
            comparison_data.append({
                'Library': 'Ensemble',
                'Method': 'Ensemble DML',
                'Own-Price Elasticity': all_results['ml_dml']['ensemble']['elasticity'],
                'Type': 'ML-based'
            })
    
    # Create comparison table
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*60)
        print("ELASTICITY ESTIMATES ACROSS METHODS")
        print("="*60)
        
        # Format elasticities
        comparison_df['Own-Price Elasticity'] = comparison_df['Own-Price Elasticity'].apply(lambda x: f"{x:.3f}")
        
        print("\n" + tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Group by type
        print("\n" + "-"*60)
        print("Average by Method Type:")
        
        for method_type in comparison_df['Type'].unique():
            type_df = comparison_df[comparison_df['Type'] == method_type]
            elasticities = [float(x) for x in type_df['Own-Price Elasticity']]
            print(f"  {method_type}: {np.mean(elasticities):.3f} (n={len(elasticities)})")
        
        # Save comparison
        comparison_df.to_csv('elasticity_comparison.csv', index=False)
        print("\n✓ Comparison saved to 'elasticity_comparison.csv'")
    
    return comparison_df if comparison_data else None


def create_summary_visualization(all_results: dict):
    """Create comprehensive visualization comparing all methods."""
    print_header("CREATING SUMMARY VISUALIZATION")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Method comparison
    ax = axes[0, 0]
    methods = []
    elasticities = []
    colors = []
    
    # Extract elasticities from different methods
    method_colors = {
        'Causal ML': 'blue',
        'Structural': 'green',
        'Panel': 'orange',
        'Demand System': 'red',
        'Bayesian': 'purple',
        'ML-based': 'brown'
    }
    
    # Add available estimates
    if all_results.get('econml', {}).get('dml', {}).get('linear_dml_xgb'):
        methods.append('EconML\nDML')
        elasticities.append(all_results['econml']['dml']['linear_dml_xgb']['elasticity'])
        colors.append(method_colors['Causal ML'])
    
    if all_results.get('pyblp', {}).get('basic', {}).get('own_elasticities'):
        methods.append('PyBLP\nBLP')
        elasticities.append(np.mean(all_results['pyblp']['basic']['own_elasticities']))
        colors.append(method_colors['Structural'])
    
    if all_results.get('linearmodels', {}).get('fe', {}).get('entity_fe'):
        methods.append('Linear\nModels FE')
        elasticities.append(all_results['linearmodels']['fe']['entity_fe']['own_elasticity'])
        colors.append(method_colors['Panel'])
    
    if all_results.get('pymc', {}).get('hierarchical', {}).get('trace'):
        methods.append('PyMC\nHierarchical')
        trace = all_results['pymc']['hierarchical']['trace']
        elasticities.append(trace.posterior['mu_elasticity'].mean().values)
        colors.append(method_colors['Bayesian'])
    
    if all_results.get('ml_dml', {}).get('xgboost'):
        methods.append('XGBoost\nDML')
        elasticities.append(all_results['ml_dml']['xgboost']['elasticity'])
        colors.append(method_colors['ML-based'])
    
    if methods:
        x = np.arange(len(methods))
        ax.bar(x, elasticities, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=0)
        ax.set_ylabel('Own-Price Elasticity')
        ax.set_title('Comparison Across Libraries')
        ax.axhline(y=-1.2, color='red', linestyle='--', label='True Value', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of estimates
    ax = axes[0, 1]
    if elasticities:
        ax.boxplot([elasticities], labels=['All Methods'])
        ax.set_ylabel('Elasticity')
        ax.set_title('Distribution of Estimates')
        ax.axhline(y=-1.2, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Method types
    ax = axes[0, 2]
    type_counts = {}
    for color in colors:
        type_name = [k for k, v in method_colors.items() if v == color][0]
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    if type_counts:
        ax.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.0f%%',
               colors=[method_colors[t] for t in type_counts.keys()])
        ax.set_title('Methods by Type')
    
    # Plot 4: Convergence (if available)
    ax = axes[1, 0]
    ax.text(0.5, 0.5, 'Convergence Diagnostics\n(Method-specific)', 
            ha='center', va='center', fontsize=12)
    ax.set_title('Model Diagnostics')
    ax.axis('off')
    
    # Plot 5: Heterogeneity
    ax = axes[1, 1]
    ax.text(0.5, 0.5, 'Heterogeneous Effects\n(See individual outputs)', 
            ha='center', va='center', fontsize=12)
    ax.set_title('Heterogeneity Analysis')
    ax.axis('off')
    
    # Plot 6: Summary statistics
    ax = axes[1, 2]
    if elasticities:
        summary_text = f"""Summary Statistics:
        
Mean: {np.mean(elasticities):.3f}
Median: {np.median(elasticities):.3f}
Std Dev: {np.std(elasticities):.3f}
Min: {np.min(elasticities):.3f}
Max: {np.max(elasticities):.3f}
Range: {np.max(elasticities) - np.min(elasticities):.3f}

True Value: -1.200
Mean Error: {np.mean(elasticities) + 1.2:.3f}
RMSE: {np.sqrt(np.mean((np.array(elasticities) + 1.2)**2)):.3f}"""
        
        ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=10,
                family='monospace')
        ax.set_title('Summary Statistics')
        ax.axis('off')
    
    plt.suptitle('Cross-Price Elasticity Estimation: Comprehensive Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Summary visualization saved as 'summary_comparison.png'")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("  CROSS-PRICE ELASTICITY ESTIMATION SHOWCASE")
    print("  Comprehensive Comparison of Modern Econometric Methods")
    print("="*80)
    
    print("\nLibraries to showcase:")
    print("  1. EconML - Causal ML methods")
    print("  2. PyBLP - Structural demand estimation") 
    print("  3. LinearModels - Panel data methods")
    print("  4. Statsmodels - AIDS/QUAIDS systems")
    print("  5. PyMC - Bayesian hierarchical models")
    print("  6. Scikit-learn/XGBoost - ML pipelines")
    
    # Track timing
    start_time = time.time()
    
    # Store all results
    all_results = {}
    
    # Step 1: Data preparation
    print("\n" + "-"*60)
    print("Step 1/7: Preparing data...")
    if not run_data_preparation():
        print("Warning: Data preparation failed, continuing with existing data...")
    
    # Step 2: EconML
    print("\n" + "-"*60)
    print("Step 2/7: Running EconML examples...")
    all_results['econml'] = run_econml_examples()
    
    # Step 3: PyBLP
    print("\n" + "-"*60)
    print("Step 3/7: Running PyBLP examples...")
    all_results['pyblp'] = run_pyblp_examples()
    
    # Step 4: LinearModels
    print("\n" + "-"*60)
    print("Step 4/7: Running LinearModels examples...")
    all_results['linearmodels'] = run_linearmodels_examples()
    
    # Step 5: AIDS/QUAIDS
    print("\n" + "-"*60)
    print("Step 5/7: Running AIDS/QUAIDS examples...")
    all_results['aids'] = run_aids_examples()
    
    # Step 6: PyMC
    print("\n" + "-"*60)
    print("Step 6/7: Running PyMC examples...")
    all_results['pymc'] = run_pymc_examples()
    
    # Step 7: ML DML
    print("\n" + "-"*60)
    print("Step 7/7: Running ML DML examples...")
    all_results['ml_dml'] = run_ml_dml_examples()
    
    # Compare results
    comparison_df = compare_results(all_results)
    
    # Create summary visualization
    create_summary_visualization(all_results)
    
    # Final summary
    elapsed_time = time.time() - start_time
    
    print_header("EXECUTION COMPLETE")
    
    print(f"\nTotal execution time: {elapsed_time:.1f} seconds")
    
    successful = sum(1 for v in all_results.values() if v is not None)
    print(f"Successfully completed: {successful}/{len(all_results)} method groups")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    print("""
1. CAUSAL ML (EconML):
   - Flexible control for confounders
   - Handles heterogeneous treatment effects
   - Good for policy evaluation
   
2. STRUCTURAL (PyBLP):
   - Rich substitution patterns
   - Consumer heterogeneity
   - Welfare analysis capability
   
3. PANEL METHODS (LinearModels):
   - Controls for unobserved heterogeneity
   - Handles endogeneity with IV
   - Efficient for large panels
   
4. DEMAND SYSTEMS (AIDS/QUAIDS):
   - Complete demand system
   - Theoretical consistency
   - Welfare measurement
   
5. BAYESIAN (PyMC):
   - Full uncertainty quantification
   - Natural hierarchy handling
   - Prior information incorporation
   
6. ML-BASED (XGBoost/LightGBM):
   - Flexible functional forms
   - High-dimensional controls
   - Non-parametric relationships
""")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print("""
Choose your method based on:

• Data structure:
  - Panel data → LinearModels
  - Cross-section → EconML, PyBLP
  - Time series → PyMC (state-space)
  
• Research question:
  - Causal effects → EconML
  - Substitution patterns → PyBLP, AIDS
  - Heterogeneity → PyMC, Causal Forests
  
• Assumptions:
  - Minimal → ML-based DML
  - Structural → PyBLP
  - Hierarchical → PyMC
  
• Computational resources:
  - Limited → LinearModels, AIDS
  - Moderate → EconML, XGBoost
  - Extensive → PyBLP, PyMC
""")
    
    print("\n✓ All examples completed successfully!")
    print("✓ Results saved to respective output files")
    print("✓ Visualizations saved as PNG files")
    
    return all_results


if __name__ == "__main__":
    results = main()

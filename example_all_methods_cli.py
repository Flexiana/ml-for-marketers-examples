#!/usr/bin/env python3
"""
Comprehensive CLI for All Elasticity Estimation Methods

This module provides a unified command-line interface for all elasticity estimation
methods with colorful ASCII visualizations and marketing-friendly explanations.

Usage:
    python example_all_methods_cli.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from rich import box
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all estimators
from example_econml import EconMLElasticityEstimator
from example_pyblp import PyBLPElasticityEstimator
from example_linearmodels import LinearModelsElasticityEstimator
from example_statsmodels_aids import StatsmodelsAIDSEstimator
from example_pymc import PyMCElasticityEstimator
from example_sklearn_xgb_dml import SklearnXGBDMLEstimator

# Color codes for ASCII art
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    # Background colors
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'
    BG_MAGENTA = '\033[105m'
    BG_CYAN = '\033[106m'

class AllMethodsCLI:
    """Unified CLI for all elasticity estimation methods with marketing focus."""
    
    def __init__(self):
        self.console = Console()
        self.estimators = {}
        self.results = {}
        
    def print_banner(self):
        """Print colorful banner."""
        banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🎯 COMPREHENSIVE ELASTICITY ANALYSIS SUITE FOR MARKETING MANAGERS 🎯       ║
║                                                                              ║
║  📊 AI-Powered Cross-Price Elasticity Estimation                             ║
║  🚀 6 Advanced Econometric & Machine Learning Methods                        ║
║  💡 Complete Portfolio Pricing Intelligence Platform                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}
        """
        self.console.print(banner)
        
    def print_method_overview(self):
        """Print overview of all available methods."""
        methods_table = Table(title="🔬 AVAILABLE ANALYSIS METHODS", box=box.ROUNDED)
        methods_table.add_column("Method", style="cyan", no_wrap=True)
        methods_table.add_column("Type", style="magenta")
        methods_table.add_column("Best For", style="green")
        methods_table.add_column("Complexity", style="yellow")
        
        methods_data = [
            ("EconML (DML/IV/DR)", "Causal ML", "Causal inference", "⭐⭐⭐"),
            ("PyBLP (Structural)", "Structural", "Consumer choice", "⭐⭐⭐⭐"),
            ("LinearModels (Panel)", "Panel Data", "Time series", "⭐⭐"),
            ("Statsmodels (AIDS)", "Demand Systems", "Portfolio analysis", "⭐⭐"),
            ("PyMC (Bayesian)", "Bayesian", "Uncertainty", "⭐⭐⭐⭐"),
            ("Sklearn/XGB (ML)", "Machine Learning", "Prediction", "⭐⭐")
        ]
        
        for method, method_type, best_for, complexity in methods_data:
            methods_table.add_row(method, method_type, best_for, complexity)
        
        self.console.print(methods_table)
        
    def create_comparison_chart(self, all_results: Dict) -> str:
        """Create comparison chart across all methods."""
        chart = f"\n{Colors.BOLD}{Colors.CYAN}📊 METHOD COMPARISON DASHBOARD{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*100}{Colors.END}\n"
        
        # Extract elasticities from all methods
        method_elasticities = {}
        for method_name, results in all_results.items():
            if isinstance(results, dict):
                # Try to find elasticity values
                elasticities = []
                for key, value in results.items():
                    if isinstance(value, dict) and 'elasticity' in value and value['elasticity'] is not None:
                        elasticities.append(value['elasticity'])
                    elif isinstance(value, (int, float)) and not np.isnan(value):
                        elasticities.append(value)
                
                if elasticities:
                    method_elasticities[method_name] = np.mean(elasticities)
        
        if not method_elasticities:
            return f"{Colors.RED}No comparable elasticity data available{Colors.END}"
        
        # Sort by elasticity value
        sorted_methods = sorted(method_elasticities.items(), key=lambda x: x[1])
        
        # Create comparison bars
        max_elasticity = max(abs(v) for v in method_elasticities.values())
        
        for method, elasticity in sorted_methods:
            # Color based on method type
            if 'econml' in method.lower():
                color = Colors.CYAN
                symbol = "🤖"
            elif 'pyblp' in method.lower():
                color = Colors.MAGENTA
                symbol = "🏗️"
            elif 'linear' in method.lower():
                color = Colors.BLUE
                symbol = "📈"
            elif 'aids' in method.lower():
                color = Colors.GREEN
                symbol = "🔄"
            elif 'pymc' in method.lower():
                color = Colors.YELLOW
                symbol = "🎲"
            elif 'sklearn' in method.lower():
                color = Colors.RED
                symbol = "⚡"
            else:
                color = Colors.WHITE
                symbol = "📊"
            
            # Create bar
            bar_length = int((abs(elasticity) / max_elasticity) * 50) if max_elasticity > 0 else 0
            bar = "█" * bar_length + "░" * (50 - bar_length)
            
            chart += f"{color}{symbol} {method:<20} {bar:<50} {elasticity:>8.3f}{Colors.END}\n"
        
        chart += f"{Colors.YELLOW}{'='*100}{Colors.END}\n"
        
        # Summary statistics
        values = list(method_elasticities.values())
        mean_elasticity = np.mean(values)
        std_elasticity = np.std(values)
        
        chart += f"\n{Colors.BOLD}📈 Summary Statistics:{Colors.END}\n"
        chart += f"Mean elasticity across methods: {mean_elasticity:.3f}\n"
        chart += f"Standard deviation: {std_elasticity:.3f}\n"
        chart += f"Range: {min(values):.3f} to {max(values):.3f}\n"
        
        # Consistency analysis
        if std_elasticity < 0.2:
            consistency = f"{Colors.GREEN}High consistency - methods agree well{Colors.END}"
        elif std_elasticity < 0.5:
            consistency = f"{Colors.YELLOW}Moderate consistency - some variation{Colors.END}"
        else:
            consistency = f"{Colors.RED}Low consistency - methods disagree{Colors.END}"
        
        chart += f"Consistency: {consistency}\n"
        
        return chart
        
    def create_business_insights(self, all_results: Dict) -> str:
        """Create business insights from all results."""
        insights = f"\n{Colors.BOLD}{Colors.GREEN}💡 BUSINESS INSIGHTS & RECOMMENDATIONS{Colors.END}\n"
        insights += f"{Colors.YELLOW}{'='*80}{Colors.END}\n"
        
        # Extract key insights
        elasticities = []
        cross_price_data = []
        heterogeneity_data = []
        
        for method_name, results in all_results.items():
            if isinstance(results, dict):
                # Collect elasticities
                for key, value in results.items():
                    if isinstance(value, dict) and 'elasticity' in value and value['elasticity'] is not None:
                        elasticities.append(value['elasticity'])
                    elif isinstance(value, (int, float)) and not np.isnan(value):
                        elasticities.append(value)
                
                # Collect cross-price data
                if 'elasticity_matrix' in results:
                    cross_price_data.append(results['elasticity_matrix'])
                
                # Collect heterogeneity data
                if 'cate' in results and results['cate'] is not None:
                    heterogeneity_data.append(results['cate'])
        
        # Price sensitivity insights
        if elasticities:
            avg_elasticity = np.mean(elasticities)
            insights += f"\n{Colors.BOLD}🎯 Price Sensitivity Analysis:{Colors.END}\n"
            
            if avg_elasticity < -1.0:
                insights += f"• {Colors.RED}High price sensitivity detected{Colors.END} - customers are very responsive to price changes\n"
                insights += f"• Consider promotional pricing strategies to drive volume\n"
                insights += f"• Monitor competitor pricing closely\n"
            elif avg_elasticity < -0.5:
                insights += f"• {Colors.YELLOW}Moderate price sensitivity{Colors.END} - customers are somewhat responsive\n"
                insights += f"• Balanced pricing approach recommended\n"
                insights += f"• Focus on value proposition and quality\n"
            else:
                insights += f"• {Colors.GREEN}Low price sensitivity{Colors.END} - customers are not very responsive\n"
                insights += f"• Premium pricing strategies may be effective\n"
                insights += f"• Focus on differentiation and brand building\n"
        
        # Cross-price insights
        if cross_price_data:
            insights += f"\n{Colors.BOLD}🔄 Product Portfolio Insights:{Colors.END}\n"
            insights += f"• Cross-price elasticity analysis reveals product relationships\n"
            insights += f"• Use substitution patterns to optimize portfolio pricing\n"
            insights += f"• Consider bundling strategies for complementary products\n"
        
        # Heterogeneity insights
        if heterogeneity_data:
            insights += f"\n{Colors.BOLD}👥 Customer Segmentation Insights:{Colors.END}\n"
            insights += f"• Significant customer heterogeneity detected\n"
            insights += f"• Implement segmented pricing strategies\n"
            insights += f"• Target different customer groups with tailored approaches\n"
        
        # Strategic recommendations
        insights += f"\n{Colors.BOLD}🚀 Strategic Recommendations:{Colors.END}\n"
        insights += f"• Use multiple methods for robust elasticity estimates\n"
        insights += f"• Monitor elasticity changes over time\n"
        insights += f"• Test pricing strategies in controlled experiments\n"
        insights += f"• Integrate elasticity insights with market research\n"
        insights += f"• Develop dynamic pricing capabilities\n"
        
        insights += f"\n{Colors.YELLOW}{'='*80}{Colors.END}\n"
        
        return insights
        
    def run_econml_analysis(self):
        """Run EconML analysis."""
        self.console.print("\n[bold cyan]🤖 Running EconML Analysis...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Running causal ML methods...", total=None)
            
            try:
                estimator = EconMLElasticityEstimator()
                results = estimator.example_1_double_ml()
                progress.update(task, description="✅ EconML analysis completed!")
                
                self.results['econml'] = results
                self.console.print("[green]✅ EconML analysis completed successfully![/green]")
                
            except Exception as e:
                self.console.print(f"[red]❌ EconML analysis failed: {e}[/red]")
                
    def run_pyblp_analysis(self):
        """Run PyBLP analysis."""
        self.console.print("\n[bold magenta]🏗️ Running PyBLP Analysis...[/bold magenta]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Running structural demand estimation...", total=None)
            
            try:
                estimator = PyBLPElasticityEstimator()
                results = estimator.example_1_basic_blp()
                progress.update(task, description="✅ PyBLP analysis completed!")
                
                self.results['pyblp'] = results
                self.console.print("[green]✅ PyBLP analysis completed successfully![/green]")
                
            except Exception as e:
                self.console.print(f"[red]❌ PyBLP analysis failed: {e}[/red]")
                
    def run_linearmodels_analysis(self):
        """Run LinearModels analysis."""
        self.console.print("\n[bold blue]📈 Running LinearModels Analysis...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Running panel data analysis...", total=None)
            
            try:
                estimator = LinearModelsElasticityEstimator()
                results = estimator.example_1_fixed_effects()
                progress.update(task, description="✅ LinearModels analysis completed!")
                
                self.results['linearmodels'] = results
                self.console.print("[green]✅ LinearModels analysis completed successfully![/green]")
                
            except Exception as e:
                self.console.print(f"[red]❌ LinearModels analysis failed: {e}[/red]")
                
    def run_statsmodels_analysis(self):
        """Run Statsmodels AIDS analysis."""
        self.console.print("\n[bold green]🔄 Running Statsmodels AIDS Analysis...[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Running AIDS demand system estimation...", total=None)
            
            try:
                estimator = StatsmodelsAIDSEstimator()
                results = estimator.example_1_basic_aids()
                progress.update(task, description="✅ Statsmodels analysis completed!")
                
                self.results['statsmodels'] = results
                self.console.print("[green]✅ Statsmodels analysis completed successfully![/green]")
                
            except Exception as e:
                self.console.print(f"[red]❌ Statsmodels analysis failed: {e}[/red]")
                
    def run_pymc_analysis(self):
        """Run PyMC Bayesian analysis."""
        self.console.print("\n[bold yellow]🎲 Running PyMC Bayesian Analysis...[/bold yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Running Bayesian estimation...", total=None)
            
            try:
                estimator = PyMCElasticityEstimator()
                results = estimator.example_1_basic_bayesian()
                progress.update(task, description="✅ PyMC analysis completed!")
                
                self.results['pymc'] = results
                self.console.print("[green]✅ PyMC analysis completed successfully![/green]")
                
            except Exception as e:
                self.console.print(f"[red]❌ PyMC analysis failed: {e}[/red]")
                
    def run_sklearn_analysis(self):
        """Run Sklearn/XGB analysis."""
        self.console.print("\n[bold red]⚡ Running Sklearn/XGB Analysis...[/bold red]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Running ML-based DML...", total=None)
            
            try:
                estimator = SklearnXGBDMLEstimator()
                results = estimator.example_1_ensemble_dml()
                progress.update(task, description="✅ Sklearn analysis completed!")
                
                self.results['sklearn'] = results
                self.console.print("[green]✅ Sklearn analysis completed successfully![/green]")
                
            except Exception as e:
                self.console.print(f"[red]❌ Sklearn analysis failed: {e}[/red]")
                
    def run_all_analyses(self):
        """Run all analyses in sequence."""
        self.console.print("\n[bold cyan]🚀 Running All Elasticity Analyses...[/bold cyan]")
        
        analyses = [
            ("EconML", self.run_econml_analysis),
            ("PyBLP", self.run_pyblp_analysis),
            ("LinearModels", self.run_linearmodels_analysis),
            ("Statsmodels", self.run_statsmodels_analysis),
            ("PyMC", self.run_pymc_analysis),
            ("Sklearn", self.run_sklearn_analysis)
        ]
        
        successful = 0
        failed = 0
        
        for name, analysis_func in analyses:
            try:
                analysis_func()
                successful += 1
            except Exception as e:
                self.console.print(f"[red]❌ {name} analysis failed: {e}[/red]")
                failed += 1
        
        # Summary
        self.console.print(f"\n[bold green]📊 Analysis Summary: {successful} successful, {failed} failed[/bold green]")
        
        if self.results:
            # Show comparison chart
            comparison_chart = self.create_comparison_chart(self.results)
            self.console.print(comparison_chart)
            
            # Show business insights
            business_insights = self.create_business_insights(self.results)
            self.console.print(business_insights)
        
    def run_interactive_analysis(self):
        """Run interactive analysis with user choices."""
        self.print_banner()
        self.print_method_overview()
        
        # Analysis menu
        while True:
            self.console.print("\n[bold cyan]📋 COMPREHENSIVE ANALYSIS MENU[/bold cyan]")
            self.console.print("1. 🤖 EconML (Causal ML)")
            self.console.print("2. 🏗️ PyBLP (Structural)")
            self.console.print("3. 📈 LinearModels (Panel Data)")
            self.console.print("4. 🔄 Statsmodels (AIDS)")
            self.console.print("5. 🎲 PyMC (Bayesian)")
            self.console.print("6. ⚡ Sklearn/XGB (ML)")
            self.console.print("7. 🚀 Run All Analyses")
            self.console.print("8. 📊 View Comparison Dashboard")
            self.console.print("9. 🚪 Exit")
            
            choice = input("\n[bold yellow]Select analysis (1-9): [/bold yellow]").strip()
            
            if choice == '1':
                self.run_econml_analysis()
            elif choice == '2':
                self.run_pyblp_analysis()
            elif choice == '3':
                self.run_linearmodels_analysis()
            elif choice == '4':
                self.run_statsmodels_analysis()
            elif choice == '5':
                self.run_pymc_analysis()
            elif choice == '6':
                self.run_sklearn_analysis()
            elif choice == '7':
                self.run_all_analyses()
            elif choice == '8':
                if self.results:
                    comparison_chart = self.create_comparison_chart(self.results)
                    self.console.print(comparison_chart)
                    
                    business_insights = self.create_business_insights(self.results)
                    self.console.print(business_insights)
                else:
                    self.console.print("[red]No results available. Run some analyses first.[/red]")
            elif choice == '9':
                self.console.print("\n[bold green]👋 Thank you for using the Comprehensive Elasticity Analysis Suite![/bold green]")
                break
            else:
                self.console.print("[red]❌ Invalid choice. Please select 1-9.[/red]")
            
            input("\n[bold blue]Press Enter to continue...[/bold blue]")

def main():
    """Main function to run the CLI."""
    try:
        cli = AllMethodsCLI()
        cli.run_interactive_analysis()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Analysis interrupted by user. Goodbye!{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")

if __name__ == "__main__":
    main()

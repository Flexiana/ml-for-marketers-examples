#!/usr/bin/env python3
"""
PyMC Bayesian CLI with Colorful ASCII Visualizations for Marketing Managers

This module provides an interactive command-line interface for Bayesian
hierarchical elasticity estimation using PyMC, with colorful ASCII charts 
and marketing-friendly explanations.

Usage:
    python example_pymc_cli.py
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

from example_pymc import BayesianElasticityEstimator

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

class PyMCCLI:
    """Interactive CLI for PyMC Bayesian analysis with marketing focus."""
    
    def __init__(self):
        self.console = Console()
        self.estimator = None
        self.results = {}
        
    def print_banner(self):
        """Print colorful banner."""
        banner = f"""
{Colors.MAGENTA}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🎯 BAYESIAN HIERARCHICAL ANALYSIS FOR MARKETING MANAGERS 🎯               ║
║                                                                              ║
║  📊 Uncertainty-Aware Elasticity Estimation                                 ║
║  🚀 Powered by PyMC Library                                                 ║
║  💡 Get confidence intervals and uncertainty quantification                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}
        """
        self.console.print(banner)
        
    def print_method_explanation(self, method: str, description: str, business_value: str, real_world_example: str):
        """Print method explanation in marketing terms."""
        panel = Panel(
            f"[bold magenta]{method}[/bold magenta]\n\n"
            f"[yellow]What it tells you:[/yellow] {description}\n\n"
            f"[green]Why you care:[/green] {business_value}\n\n"
            f"[blue]Real example:[/blue] {real_world_example}",
            title="🎯 What This Means for Your Business",
            border_style="magenta",
            padding=(1, 2)
        )
        self.console.print(panel)
        
    def create_credible_interval_chart(self, results: Dict, title: str) -> str:
        """Create ASCII credible interval chart."""
        if not results:
            return f"{Colors.RED}No credible interval data available for {title}{Colors.END}"
            
        chart = f"\n{Colors.BOLD}{Colors.MAGENTA}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*70}{Colors.END}\n"
        
        # Credible intervals
        if 'credible_intervals' in results:
            intervals = results['credible_intervals']
            chart += f"{Colors.CYAN}Elasticity Estimates with Uncertainty:{Colors.END}\n"
            
            for product, interval in intervals.items():
                mean = interval['mean']
                lower = interval['lower']
                upper = interval['upper']
                width = upper - lower
                
                # Color based on uncertainty width
                if width < 0.2:
                    color = Colors.GREEN
                    symbol = "✅"
                    confidence = "High confidence"
                elif width < 0.5:
                    color = Colors.YELLOW
                    symbol = "⚠️"
                    confidence = "Moderate confidence"
                else:
                    color = Colors.RED
                    symbol = "❌"
                    confidence = "Low confidence"
                
                chart += f"  {color}{symbol} {product}:{Colors.END}\n"
                chart += f"    Mean: {mean:>6.3f}\n"
                chart += f"    95% CI: [{lower:>6.3f}, {upper:>6.3f}]\n"
                chart += f"    Width: {width:>6.3f} ({confidence})\n\n"
        
        # Overall uncertainty
        if 'overall_uncertainty' in results:
            uncertainty = results['overall_uncertainty']
            chart += f"{Colors.CYAN}Overall Uncertainty Assessment:{Colors.END}\n"
            chart += f"  Average credible interval width: {uncertainty:.3f}\n"
            
            if uncertainty < 0.3:
                chart += f"  {Colors.GREEN}✅ High confidence in estimates{Colors.END}\n"
                chart += f"  You can make pricing decisions with confidence\n"
            elif uncertainty < 0.6:
                chart += f"  {Colors.YELLOW}⚠️ Moderate confidence in estimates{Colors.END}\n"
                chart += f"  Consider additional data or longer observation periods\n"
            else:
                chart += f"  {Colors.RED}❌ Low confidence in estimates{Colors.END}\n"
                chart += f"  Need more data or different approach\n"
        
        chart += f"{Colors.YELLOW}{'='*70}{Colors.END}\n"
        return chart
        
    def create_hierarchical_effects_chart(self, results: Dict, title: str) -> str:
        """Create ASCII hierarchical effects chart."""
        if not results:
            return f"{Colors.RED}No hierarchical effects data available for {title}{Colors.END}"
            
        chart = f"\n{Colors.BOLD}{Colors.MAGENTA}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        # Group-level effects
        if 'group_effects' in results:
            group_effects = results['group_effects']
            chart += f"{Colors.CYAN}Group-Level Elasticity Variations:{Colors.END}\n"
            
            for group, effect in group_effects.items():
                mean = effect['mean']
                std = effect['std']
                
                # Color based on variation
                if std < 0.1:
                    color = Colors.GREEN
                    symbol = "🔒"
                    interpretation = "Low variation - groups are similar"
                elif std < 0.3:
                    color = Colors.YELLOW
                    symbol = "📊"
                    interpretation = "Moderate variation - some differences"
                else:
                    color = Colors.RED
                    symbol = "📈"
                    interpretation = "High variation - groups are very different"
                
                chart += f"  {color}{symbol} {group}: {mean:>6.3f} ± {std:>6.3f} - {interpretation}{Colors.END}\n"
        
        # Shrinkage effects
        if 'shrinkage' in results:
            shrinkage = results['shrinkage']
            chart += f"\n{Colors.CYAN}Bayesian Shrinkage Effects:{Colors.END}\n"
            chart += f"  Shrinkage factor: {shrinkage:.3f}\n"
            
            if shrinkage > 0.7:
                chart += f"  {Colors.GREEN}✅ Strong shrinkage - estimates pulled toward group mean{Colors.END}\n"
                chart += f"  This reduces overfitting and improves generalization\n"
            elif shrinkage > 0.3:
                chart += f"  {Colors.YELLOW}⚠️ Moderate shrinkage - balanced approach{Colors.END}\n"
                chart += f"  Good balance between group and individual estimates\n"
            else:
                chart += f"  {Colors.RED}❌ Low shrinkage - estimates mostly individual{Colors.END}\n"
                chart += f"  May be overfitting to individual groups\n"
        
        chart += f"\n{Colors.BOLD}💡 Business Insights:{Colors.END}\n"
        chart += f"• Groups with low variation: Use similar pricing strategies\n"
        chart += f"• Groups with high variation: Customize pricing by group\n"
        chart += f"• Shrinkage helps prevent overfitting to small groups\n"
        chart += f"• Uncertainty estimates help assess pricing risk\n"
        
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        return chart
        
    def create_posterior_distribution_chart(self, results: Dict, title: str) -> str:
        """Create ASCII posterior distribution chart."""
        if not results:
            return f"{Colors.RED}No posterior distribution data available for {title}{Colors.END}"
            
        chart = f"\n{Colors.BOLD}{Colors.MAGENTA}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*70}{Colors.END}\n"
        
        # Posterior statistics
        if 'posterior_stats' in results:
            stats = results['posterior_stats']
            chart += f"{Colors.CYAN}Posterior Distribution Statistics:{Colors.END}\n"
            
            for product, stat in stats.items():
                mean = stat['mean']
                std = stat['std']
                skew = stat.get('skew', 0)
                kurtosis = stat.get('kurtosis', 0)
                
                # Color based on distribution shape
                if abs(skew) < 0.5 and abs(kurtosis) < 0.5:
                    color = Colors.GREEN
                    symbol = "📊"
                    shape = "Normal distribution"
                elif abs(skew) > 1:
                    color = Colors.RED
                    symbol = "📈"
                    shape = "Skewed distribution"
                else:
                    color = Colors.YELLOW
                    symbol = "📉"
                    shape = "Slightly skewed"
                
                chart += f"  {color}{symbol} {product}:{Colors.END}\n"
                chart += f"    Mean: {mean:>6.3f}, Std: {std:>6.3f}\n"
                chart += f"    Skew: {skew:>6.3f}, Kurtosis: {kurtosis:>6.3f}\n"
                chart += f"    Shape: {shape}\n\n"
        
        # Convergence diagnostics
        if 'convergence' in results:
            convergence = results['convergence']
            chart += f"{Colors.CYAN}MCMC Convergence Diagnostics:{Colors.END}\n"
            
            if convergence.get('rhat', 1.0) < 1.1:
                chart += f"  {Colors.GREEN}✅ R-hat < 1.1: Good convergence{Colors.END}\n"
            else:
                chart += f"  {Colors.RED}❌ R-hat > 1.1: Poor convergence{Colors.END}\n"
            
            if convergence.get('effective_samples', 0) > 1000:
                chart += f"  {Colors.GREEN}✅ Effective samples > 1000: Good mixing{Colors.END}\n"
            else:
                chart += f"  {Colors.RED}❌ Effective samples < 1000: Poor mixing{Colors.END}\n"
        
        chart += f"\n{Colors.BOLD}💡 What This Means:{Colors.END}\n"
        chart += f"• Normal distributions: Reliable estimates\n"
        chart += f"• Skewed distributions: Be cautious with estimates\n"
        chart += f"• Good convergence: Trust the results\n"
        chart += f"• Poor convergence: Need more samples or different model\n"
        
        chart += f"{Colors.YELLOW}{'='*70}{Colors.END}\n"
        return chart
        
    def run_basic_bayesian_analysis(self):
        """Run basic Bayesian analysis with visualizations."""
        self.console.print("\n[bold magenta]🎯 Running Basic Bayesian Analysis...[/bold magenta]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Sampling from posterior distributions...", total=None)
            
            try:
                results = self.estimator.example_1_basic_bayesian()
                progress.update(task, description="✅ Basic Bayesian analysis completed!")
                
                # Display results
                interval_chart = self.create_credible_interval_chart(results, "🎯 BAYESIAN ELASTICITY ESTIMATES")
                self.console.print(interval_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Basic Bayesian Elasticity Estimation",
                    "Gives you elasticity estimates with confidence intervals that tell you how uncertain you should be about each number. It shows the range of likely values, not just a single point estimate.",
                    "Helps you make pricing decisions with full knowledge of the uncertainty. You can see which estimates are reliable and which need more data or different approaches.",
                    "Instead of just saying 'elasticity is -1.2', Bayesian analysis says 'elasticity is -1.2 with 95% confidence it's between -1.5 and -0.9'. Now you know how confident to be in your pricing decisions."
                )
                
                self.results['basic_bayesian'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Basic Bayesian analysis failed: {e}[/red]")
                
    def run_hierarchical_analysis(self):
        """Run hierarchical Bayesian analysis with visualizations."""
        self.console.print("\n[bold magenta]🏗️ Running Hierarchical Bayesian Analysis...[/bold magenta]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Estimating hierarchical model...", total=None)
            
            try:
                results = self.estimator.example_2_hierarchical()
                progress.update(task, description="✅ Hierarchical analysis completed!")
                
                # Display results
                hierarchical_chart = self.create_hierarchical_effects_chart(results, "🏗️ HIERARCHICAL ELASTICITY ANALYSIS")
                self.console.print(hierarchical_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Hierarchical Bayesian Model",
                    "Estimates elasticity for different groups (stores, regions, customer segments) while sharing information between groups. It prevents overfitting to small groups and improves estimates for all groups.",
                    "Gives you reliable elasticity estimates for each group while using information from all groups. Small groups benefit from information in large groups, and large groups get more stable estimates.",
                    "The analysis might reveal: 'Store A has elasticity -1.3, Store B has -0.8, but both are pulled toward the overall average of -1.1. This prevents Store A from being too extreme due to limited data.'"
                )
                
                self.results['hierarchical'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Hierarchical analysis failed: {e}[/red]")
                
    def run_time_varying_analysis(self):
        """Run time-varying Bayesian analysis with visualizations."""
        self.console.print("\n[bold magenta]📈 Running Time-Varying Analysis...[/bold magenta]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Estimating time-varying model...", total=None)
            
            try:
                results = self.estimator.example_3_time_varying()
                progress.update(task, description="✅ Time-varying analysis completed!")
                
                # Display results
                if 'time_series' in results:
                    chart = f"\n{Colors.BOLD}{Colors.MAGENTA}📈 TIME-VARYING ELASTICITY ANALYSIS{Colors.END}\n"
                    chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
                    
                    time_series = results['time_series']
                    chart += f"{Colors.CYAN}Elasticity Changes Over Time:{Colors.END}\n"
                    
                    for period, elasticity in time_series.items():
                        if elasticity < -1.5:
                            color = Colors.RED
                            symbol = "🔴"
                        elif elasticity < -1.0:
                            color = Colors.YELLOW
                            symbol = "🟡"
                        elif elasticity < -0.5:
                            color = Colors.GREEN
                            symbol = "🟢"
                        else:
                            color = Colors.BLUE
                            symbol = "🔵"
                        
                        chart += f"  {color}{symbol} {period}: {elasticity:>6.3f}{Colors.END}\n"
                    
                    chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
                    self.console.print(chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Time-Varying Bayesian Model",
                    "Shows how elasticity changes over time, capturing trends, seasonality, and structural breaks in customer price sensitivity.",
                    "Helps you understand when customers are most and least price-sensitive, so you can adjust your pricing strategy over time and plan for seasonal changes.",
                    "The analysis might reveal: 'Elasticity is -1.5 in December (customers very price-sensitive during holidays), -0.8 in June (less sensitive in summer), and trending upward over time (customers becoming less price-sensitive).' Now you can plan seasonal pricing."
                )
                
                self.results['time_varying'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Time-varying analysis failed: {e}[/red]")
                
    def run_uncertainty_quantification(self):
        """Run uncertainty quantification analysis with visualizations."""
        self.console.print("\n[bold magenta]📊 Running Uncertainty Quantification...[/bold magenta]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Quantifying uncertainty...", total=None)
            
            try:
                results = self.estimator.example_4_uncertainty()
                progress.update(task, description="✅ Uncertainty quantification completed!")
                
                # Display results
                posterior_chart = self.create_posterior_distribution_chart(results, "📊 POSTERIOR DISTRIBUTION ANALYSIS")
                self.console.print(posterior_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Uncertainty Quantification",
                    "Provides detailed analysis of how uncertain your elasticity estimates are, including distribution shapes, convergence diagnostics, and reliability measures.",
                    "Helps you assess the quality and reliability of your estimates. You can see which estimates are trustworthy and which need more data or different modeling approaches.",
                    "The analysis might reveal: 'Elasticity estimates are normally distributed with good convergence (R-hat < 1.1), so you can trust them for pricing decisions. But some estimates are skewed, indicating potential issues.'"
                )
                
                self.results['uncertainty'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Uncertainty quantification failed: {e}[/red]")
                
    def create_summary_dashboard(self):
        """Create a summary dashboard of all results."""
        if not self.results:
            self.console.print("[red]No results to display in summary dashboard.[/red]")
            return
            
        # Create summary table
        table = Table(title="🎯 BAYESIAN ANALYSIS SUMMARY", box=box.ROUNDED)
        table.add_column("Analysis", style="cyan", no_wrap=True)
        table.add_column("Key Insight", style="magenta")
        table.add_column("Business Impact", style="green")
        
        insights = {
            "Basic Bayesian": "Elasticity estimates with uncertainty",
            "Hierarchical": "Group-specific estimates with shrinkage",
            "Time-Varying": "Elasticity changes over time",
            "Uncertainty": "Reliability and convergence assessment"
        }
        
        impacts = {
            "Basic Bayesian": "Confidence-aware pricing decisions",
            "Hierarchical": "Group-specific pricing strategies",
            "Time-Varying": "Time-based pricing adjustments",
            "Uncertainty": "Quality assessment of estimates"
        }
        
        for analysis in insights:
            table.add_row(
                analysis,
                insights[analysis],
                impacts[analysis]
            )
        
        self.console.print(table)
        
        # Business recommendations
        recommendations = Panel(
            "[bold green]💡 IMMEDIATE ACTION ITEMS:[/bold green]\n\n"
            "[yellow]🎯 Pricing Strategy:[/yellow]\n"
            "• Use credible intervals to assess pricing risk\n"
            "• Focus on high-confidence estimates for major decisions\n"
            "• Consider uncertainty when setting price ranges\n\n"
            "[blue]📊 Group Management:[/blue]\n"
            "• Apply group-specific pricing based on hierarchical estimates\n"
            "• Use shrinkage to improve estimates for small groups\n"
            "• Monitor group differences over time\n\n"
            "[magenta]📈 Time-Based Strategy:[/magenta]\n"
            "• Adjust pricing based on time-varying elasticity\n"
            "• Plan for seasonal changes in price sensitivity\n"
            "• Monitor trends in customer behavior\n\n"
            "[red]⚠️ Quality Control:[/red]\n"
            "• Check convergence diagnostics before using estimates\n"
            "• Be cautious with skewed or uncertain distributions\n"
            "• Collect more data for low-confidence estimates",
            title="🎯 Your Next Steps",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(recommendations)
        
    def run_interactive_analysis(self):
        """Run interactive analysis with user choices."""
        self.print_banner()
        
        # Initialize estimator
        self.console.print("\n[bold blue]🚀 Initializing Bayesian analysis...[/bold blue]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Loading data and models...", total=None)
            self.estimator = BayesianElasticityEstimator()
            progress.update(task, description="✅ Ready for analysis!")
        
        # Analysis menu
        while True:
            self.console.print("\n[bold cyan]📋 BAYESIAN ANALYSIS MENU[/bold cyan]")
            self.console.print("1. 🎯 Basic Bayesian (Uncertainty)")
            self.console.print("2. 🏗️ Hierarchical (Group Effects)")
            self.console.print("3. 📈 Time-Varying (Trends)")
            self.console.print("4. 📊 Uncertainty Quantification")
            self.console.print("5. 📊 View Summary Dashboard")
            self.console.print("6. 🚪 Exit")
            
            choice = input("\n[bold yellow]Select analysis (1-6): [/bold yellow]").strip()
            
            if choice == '1':
                self.run_basic_bayesian_analysis()
            elif choice == '2':
                self.run_hierarchical_analysis()
            elif choice == '3':
                self.run_time_varying_analysis()
            elif choice == '4':
                self.run_uncertainty_quantification()
            elif choice == '5':
                self.create_summary_dashboard()
            elif choice == '6':
                self.console.print("\n[bold green]👋 Thank you for using the Bayesian Analysis Tool![/bold green]")
                break
            else:
                self.console.print("[red]❌ Invalid choice. Please select 1-6.[/red]")
            
            input("\n[bold blue]Press Enter to continue...[/bold blue]")

def main():
    """Main function to run the CLI."""
    try:
        cli = PyMCCLI()
        cli.run_interactive_analysis()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Analysis interrupted by user. Goodbye!{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}[/red]")

if __name__ == "__main__":
    main()

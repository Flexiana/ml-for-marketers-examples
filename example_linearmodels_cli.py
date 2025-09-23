#!/usr/bin/env python3
"""
LinearModels CLI with Colorful ASCII Visualizations for Marketing Managers

This module provides an interactive command-line interface for panel data
regression with fixed effects and IV/2SLS using linearmodels, with colorful 
ASCII charts and marketing-friendly explanations.

Usage:
    python example_linearmodels_cli.py
"""

import sys
import os
import time
import argparse
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

from example_linearmodels import PanelElasticityEstimator

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

class LinearModelsCLI:
    """Interactive CLI for LinearModels panel data analysis with marketing focus."""
    
    def __init__(self):
        self.console = Console()
        self.estimator = None
        self.results = {}
        
    def print_banner(self):
        """Print colorful banner."""
        banner = f"""
{Colors.BLUE}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  📈 PANEL DATA ANALYSIS FOR MARKETING MANAGERS 📈                           ║
║                                                                              ║
║  📊 Time Series & Cross-Sectional Data Analysis                             ║
║  🚀 Powered by LinearModels Library                                         ║
║  💡 Understand how pricing effects change over time and across markets      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}
        """
        self.console.print(banner)
        
    def print_method_explanation(self, method: str, description: str, business_value: str, real_world_example: str):
        """Print method explanation in marketing terms."""
        panel = Panel(
            f"[bold blue]{method}[/bold blue]\n\n"
            f"[yellow]What it tells you:[/yellow] {description}\n\n"
            f"[green]Why you care:[/green] {business_value}\n\n"
            f"[blue]Real example:[/blue] {real_world_example}",
            title="📈 What This Means for Your Business",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
        
    def create_elasticity_trend_chart(self, results: Dict, title: str) -> str:
        """Create ASCII elasticity trend chart."""
        if not results:
            return f"{Colors.RED}No data available for {title}{Colors.END}"
            
        chart = f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*70}{Colors.END}\n"
        
        # Extract elasticities by time period
        if 'elasticities_by_period' in results:
            periods = results['elasticities_by_period']
            chart += f"{Colors.CYAN}Elasticity Trends Over Time:{Colors.END}\n"
            
            for period, elasticity in periods.items():
                # Color based on elasticity value
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
        
        # Overall elasticity
        if 'elasticity' in results and results['elasticity'] is not None:
            elasticity = results['elasticity']
            chart += f"\n{Colors.BOLD}Overall Elasticity: {elasticity:.3f}{Colors.END}\n"
            
            if elasticity < -1.0:
                interpretation = f"{Colors.RED}🚨 HIGH SENSITIVITY: Customers are very responsive to price changes{Colors.END}\n"
                interpretation += f"{Colors.RED}   → Use promotions to drive volume{Colors.END}\n"
                interpretation += f"{Colors.RED}   → Be careful with price increases{Colors.END}"
            elif elasticity < -0.5:
                interpretation = f"{Colors.YELLOW}⚠️ MODERATE SENSITIVITY: Customers notice price changes but won't abandon you{Colors.END}\n"
                interpretation += f"{Colors.YELLOW}   → You can adjust prices but monitor sales{Colors.END}\n"
                interpretation += f"{Colors.YELLOW}   → Focus on value proposition{Colors.END}"
            else:
                interpretation = f"{Colors.GREEN}✅ LOW SENSITIVITY: Customers are loyal and less affected by price{Colors.END}\n"
                interpretation += f"{Colors.GREEN}   → You have pricing power{Colors.END}\n"
                interpretation += f"{Colors.GREEN}   → Focus on quality and brand{Colors.END}"
            
            chart += f"\n{Colors.BOLD}📊 What This Means:{Colors.END}\n{interpretation}\n"
        
        chart += f"{Colors.YELLOW}{'='*70}{Colors.END}\n"
        return chart
        
    def create_fixed_effects_chart(self, results: Dict, title: str) -> str:
        """Create ASCII fixed effects chart."""
        if not results:
            return f"{Colors.RED}No data available for {title}{Colors.END}"
            
        chart = f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        # Store fixed effects
        if 'store_effects' in results:
            store_effects = results['store_effects']
            chart += f"{Colors.CYAN}Store-Specific Price Sensitivity:{Colors.END}\n"
            
            # Sort by effect size
            sorted_stores = sorted(store_effects.items(), key=lambda x: x[1])
            
            for store_id, effect in sorted_stores[:10]:  # Show top 10
                if effect < -1.5:
                    color = Colors.RED
                    symbol = "🔴"
                elif effect < -1.0:
                    color = Colors.YELLOW
                    symbol = "🟡"
                elif effect < -0.5:
                    color = Colors.GREEN
                    symbol = "🟢"
                else:
                    color = Colors.BLUE
                    symbol = "🔵"
                
                chart += f"  {color}{symbol} Store {store_id}: {effect:>6.3f}{Colors.END}\n"
        
        # Time fixed effects
        if 'time_effects' in results:
            time_effects = results['time_effects']
            chart += f"\n{Colors.CYAN}Seasonal Price Sensitivity:{Colors.END}\n"
            
            for period, effect in time_effects.items():
                if effect < -1.5:
                    color = Colors.RED
                    symbol = "🔴"
                elif effect < -1.0:
                    color = Colors.YELLOW
                    symbol = "🟡"
                elif effect < -0.5:
                    color = Colors.GREEN
                    symbol = "🟢"
                else:
                    color = Colors.BLUE
                    symbol = "🔵"
                
                chart += f"  {color}{symbol} {period}: {effect:>6.3f}{Colors.END}\n"
        
        chart += f"\n{Colors.BOLD}💡 Business Insights:{Colors.END}\n"
        chart += f"• Some stores are more price-sensitive than others\n"
        chart += f"• Price sensitivity varies by season/time period\n"
        chart += f"• Use this data to set store-specific pricing strategies\n"
        chart += f"• Plan promotions during high-sensitivity periods\n"
        
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        return chart
        
    def create_iv_analysis_chart(self, results: Dict, title: str) -> str:
        """Create ASCII IV analysis chart."""
        if not results:
            return f"{Colors.RED}No data available for {title}{Colors.END}"
            
        chart = f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*70}{Colors.END}\n"
        
        # IV results
        if 'iv_elasticity' in results and 'ols_elasticity' in results:
            iv_elasticity = results['iv_elasticity']
            ols_elasticity = results['ols_elasticity']
            
            chart += f"{Colors.CYAN}Comparison of Methods:{Colors.END}\n"
            chart += f"  OLS (Simple regression): {ols_elasticity:>6.3f}\n"
            chart += f"  IV (Instrumental Variables): {iv_elasticity:>6.3f}\n"
            chart += f"  Difference: {abs(iv_elasticity - ols_elasticity):>6.3f}\n\n"
            
            # Interpretation
            if abs(iv_elasticity - ols_elasticity) > 0.2:
                chart += f"{Colors.RED}⚠️ Large difference detected!{Colors.END}\n"
                chart += f"   The simple regression is biased. Use IV results for pricing decisions.\n"
            else:
                chart += f"{Colors.GREEN}✅ Small difference - both methods agree{Colors.END}\n"
                chart += f"   Either method gives reliable results.\n"
        
        # First stage results
        if 'first_stage_r2' in results:
            r2 = results['first_stage_r2']
            chart += f"\n{Colors.CYAN}Instrument Strength:{Colors.END}\n"
            chart += f"  First stage R²: {r2:.3f}\n"
            
            if r2 > 0.1:
                chart += f"  {Colors.GREEN}✅ Strong instrument - reliable results{Colors.END}\n"
            elif r2 > 0.05:
                chart += f"  {Colors.YELLOW}⚠️ Moderate instrument - use with caution{Colors.END}\n"
            else:
                chart += f"  {Colors.RED}❌ Weak instrument - results may be unreliable{Colors.END}\n"
        
        chart += f"\n{Colors.BOLD}💡 What This Means:{Colors.END}\n"
        chart += f"• IV analysis gives you the most reliable price sensitivity estimates\n"
        chart += f"• It removes bias from your pricing strategy decisions\n"
        chart += f"• Use these numbers for confident pricing decisions\n"
        
        chart += f"{Colors.YELLOW}{'='*70}{Colors.END}\n"
        return chart
        
    def run_fixed_effects_analysis(self):
        """Run fixed effects analysis with visualizations."""
        self.console.print("\n[bold blue]📈 Running Fixed Effects Analysis...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Analyzing store and time effects...", total=None)
            
            try:
                results = self.estimator.example_1_fixed_effects()
                progress.update(task, description="✅ Fixed effects analysis completed!")
                
                # Display results
                chart = self.create_elasticity_trend_chart(results, "📈 FIXED EFFECTS ELASTICITY ANALYSIS")
                self.console.print(chart)
                
                # Fixed effects details
                effects_chart = self.create_fixed_effects_chart(results, "🏪 STORE & TIME EFFECTS")
                self.console.print(effects_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Fixed Effects Panel Regression",
                    "Shows you how price sensitivity varies across different stores and time periods, while controlling for everything else that affects demand.",
                    "Helps you identify which stores are most price-sensitive and when customers are most responsive to price changes, so you can tailor your pricing strategy.",
                    "The analysis might reveal: 'Store A customers are very price-sensitive (15% sales drop for 10% price increase), but Store B customers barely notice (3% sales drop). Also, customers are more sensitive in December than in June.' Now you can price by store and season."
                )
                
                self.results['fixed_effects'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Fixed effects analysis failed: {e}[/red]")
                
    def run_iv_analysis(self):
        """Run IV/2SLS analysis with visualizations."""
        self.console.print("\n[bold blue]🎯 Running IV/2SLS Analysis...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Estimating causal effects with instruments...", total=None)
            
            try:
                results = self.estimator.example_2_instrumental_variables()
                progress.update(task, description="✅ IV analysis completed!")
                
                # Display results
                chart = self.create_iv_analysis_chart(results, "🎯 IV/2SLS ELASTICITY ANALYSIS")
                self.console.print(chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Instrumental Variables (IV/2SLS)",
                    "Uses cost changes and other external factors to measure the true effect of price on demand, removing bias from your pricing strategy decisions.",
                    "Gives you the most reliable price sensitivity estimates because it uses 'natural experiments' - when costs change, you have to adjust prices, and we can see how customers really respond.",
                    "When supplier costs increase 20% and you raise prices 15%, IV analysis shows: 'Customers will buy 18% less for every 10% price increase.' This is your true, unbiased price sensitivity."
                )
                
                self.results['iv'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ IV analysis failed: {e}[/red]")
                
    def run_random_effects_analysis(self):
        """Run random effects analysis with visualizations."""
        self.console.print("\n[bold blue]🎲 Running Random Effects Analysis...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Analyzing random effects...", total=None)
            
            try:
                results = self.estimator.example_3_dynamic_panel()
                progress.update(task, description="✅ Random effects analysis completed!")
                
                # Display results
                chart = self.create_elasticity_trend_chart(results, "🎲 RANDOM EFFECTS ELASTICITY ANALYSIS")
                self.console.print(chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Random Effects Panel Regression",
                    "Assumes that store differences are random and estimates the average price sensitivity across all stores, while accounting for store-specific characteristics.",
                    "Gives you a single, reliable estimate of price sensitivity that applies to your typical store, making it easier to set company-wide pricing policies.",
                    "The analysis gives you one number like 'customers are 12% sensitive to price changes' that you can use for pricing decisions across all your stores, while still accounting for store differences."
                )
                
                self.results['random_effects'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Random effects analysis failed: {e}[/red]")
                
    def create_summary_dashboard(self):
        """Create a summary dashboard of all results."""
        if not self.results:
            self.console.print("[red]No results to display in summary dashboard.[/red]")
            return
            
        # Create summary table
        table = Table(title="📈 PANEL DATA ANALYSIS SUMMARY", box=box.ROUNDED)
        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Elasticity", style="magenta", justify="center")
        table.add_column("Key Insight", style="green")
        table.add_column("Business Impact", style="yellow")
        
        insights = {
            "Fixed Effects": "Store and time variations identified",
            "IV/2SLS": "Unbiased causal estimates obtained",
            "Random Effects": "Average elasticity across stores"
        }
        
        impacts = {
            "Fixed Effects": "Store-specific pricing strategies",
            "IV/2SLS": "Reliable pricing decisions",
            "Random Effects": "Company-wide pricing policies"
        }
        
        for method_name, results in self.results.items():
            if isinstance(results, dict) and 'elasticity' in results and results['elasticity'] is not None:
                elasticity = results['elasticity']
                method_display = method_name.replace('_', ' ').title()
                
                table.add_row(
                    method_display,
                    f"{elasticity:.3f}",
                    insights.get(method_display, "Analysis completed"),
                    impacts.get(method_display, "Business insights available")
                )
        
        self.console.print(table)
        
        # Business recommendations
        recommendations = Panel(
            "[bold green]💡 IMMEDIATE ACTION ITEMS:[/bold green]\n\n"
            "[yellow]🎯 Pricing Strategy:[/yellow]\n"
            "• Use fixed effects to set store-specific prices\n"
            "• Apply IV results for most reliable pricing decisions\n"
            "• Use random effects for company-wide pricing policies\n\n"
            "[blue]📊 Store Management:[/blue]\n"
            "• Identify high-sensitivity stores for promotional focus\n"
            "• Adjust pricing by season based on time effects\n"
            "• Monitor store performance against elasticity predictions\n\n"
            "[magenta]📈 Time-Based Strategy:[/magenta]\n"
            "• Plan promotions during high-sensitivity periods\n"
            "• Use seasonal pricing adjustments\n"
            "• Monitor elasticity changes over time",
            title="🎯 Your Next Steps",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(recommendations)
        
    def run_interactive_analysis(self):
        """Run interactive analysis with user choices."""
        self.print_banner()
        
        # Initialize estimator
        self.console.print("\n[bold blue]🚀 Initializing panel data analysis...[/bold blue]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Loading data and models...", total=None)
            self.estimator = PanelElasticityEstimator()
            progress.update(task, description="✅ Ready for analysis!")
        
        # Analysis menu
        while True:
            self.console.print("\n[bold cyan]📋 PANEL DATA ANALYSIS MENU[/bold cyan]")
            self.console.print("1. 📈 Fixed Effects (Store & Time)")
            self.console.print("2. 🎯 IV/2SLS (Instrumental Variables)")
            self.console.print("3. 🎲 Random Effects (Average)")
            self.console.print("4. 📊 View Summary Dashboard")
            self.console.print("5. 🚪 Exit")
            
            choice = input("\n[bold yellow]Select analysis (1-5): [/bold yellow]").strip()
            
            if choice == '1':
                self.run_fixed_effects_analysis()
            elif choice == '2':
                self.run_iv_analysis()
            elif choice == '3':
                self.run_random_effects_analysis()
            elif choice == '4':
                self.create_summary_dashboard()
            elif choice == '5':
                self.console.print("\n[bold green]👋 Thank you for using the Panel Data Analysis Tool![/bold green]")
                break
            else:
                self.console.print("[red]❌ Invalid choice. Please select 1-5.[/red]")
            
            input("\n[bold blue]Press Enter to continue...[/bold blue]")

def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(description='Panel Data Analysis CLI')
    parser.add_argument('--method', type=int, choices=[1,2,3,4], 
                       help='Run specific method: 1=Fixed Effects, 2=IV/2SLS, 3=Random Effects, 4=Dashboard')
    parser.add_argument('--auto-exit', action='store_true', 
                       help='Exit automatically after running (no interactive menu)')
    
    args = parser.parse_args()
    
    try:
        cli = LinearModelsCLI()
        
        # If specific method requested, run it and exit
        if args.method:
            cli.print_banner()
            cli.console.print("\n[bold blue]🚀 Initializing panel data analysis...[/bold blue]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=cli.console,
            ) as progress:
                task = progress.add_task("Loading data and models...", total=None)
                cli.estimator = PanelElasticityEstimator()
                progress.update(task, description="✅ Ready for analysis!")
            
            if args.method == 1:
                cli.run_fixed_effects_analysis()
            elif args.method == 2:
                cli.run_iv_analysis()
            elif args.method == 3:
                cli.run_random_effects_analysis()
            elif args.method == 4:
                cli.create_summary_dashboard()
            
            if args.auto_exit:
                return
        else:
            cli.run_interactive_analysis()
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Analysis interrupted by user. Goodbye!{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}[/red]")

if __name__ == "__main__":
    main()

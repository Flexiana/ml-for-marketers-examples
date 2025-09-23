#!/usr/bin/env python3
"""
Statsmodels AIDS CLI with Colorful ASCII Visualizations for Marketing Managers

This module provides an interactive command-line interface for AIDS/QUAIDS
demand system estimation using statsmodels, with colorful ASCII charts and 
marketing-friendly explanations.

Usage:
    python example_statsmodels_cli.py
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

from example_statsmodels_aids import AIDSEstimator

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

class StatsmodelsCLI:
    """Interactive CLI for Statsmodels AIDS analysis with marketing focus."""
    
    def __init__(self):
        self.console = Console()
        self.estimator = None
        self.results = {}
        
    def print_banner(self):
        """Print colorful banner."""
        banner = f"""
{Colors.GREEN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🔄 AIDS/QUAIDS DEMAND SYSTEM FOR MARKETING MANAGERS 🔄                     ║
║                                                                              ║
║  📊 Complete Product Portfolio Analysis                                     ║
║  🚀 Powered by Statsmodels Library                                          ║
║  💡 Understand how all products interact in your portfolio                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}
        """
        self.console.print(banner)
        
    def print_method_explanation(self, method: str, description: str, business_value: str, real_world_example: str):
        """Print method explanation in marketing terms."""
        panel = Panel(
            f"[bold green]{method}[/bold green]\n\n"
            f"[yellow]What it tells you:[/yellow] {description}\n\n"
            f"[green]Why you care:[/green] {business_value}\n\n"
            f"[blue]Real example:[/blue] {real_world_example}",
            title="🔄 What This Means for Your Business",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)
        
    def create_elasticity_matrix_chart(self, matrix: np.ndarray, products: List[str], title: str) -> str:
        """Create ASCII elasticity matrix chart."""
        if matrix is None or len(products) == 0:
            return f"{Colors.RED}No elasticity data available for {title}{Colors.END}"
        
        chart = f"\n{Colors.BOLD}{Colors.GREEN}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*80}{Colors.END}\n"
        
        # Header
        header = f"{'Product':<15}"
        for product in products:
            header += f"{product[:8]:<12}"
        chart += f"{Colors.CYAN}{header}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'-'*80}{Colors.END}\n"
        
        # Matrix rows
        for i, product in enumerate(products):
            row = f"{product[:12]:<15}"
            for j in range(len(products)):
                value = matrix[i, j]
                if i == j:  # Own-price elasticity
                    if value < -1.0:
                        color = Colors.RED
                        symbol = "🔴"
                    elif value < -0.5:
                        color = Colors.YELLOW
                        symbol = "🟡"
                    else:
                        color = Colors.GREEN
                        symbol = "🟢"
                else:  # Cross-price elasticity
                    if value > 0.1:
                        color = Colors.GREEN
                        symbol = "🔄"  # Substitutes
                    elif value < -0.1:
                        color = Colors.BLUE
                        symbol = "🔗"  # Complements
                    else:
                        color = Colors.WHITE
                        symbol = "⚪"  # Independent
                
                row += f"{color}{symbol} {value:>8.3f}{Colors.END} "
            chart += row + "\n"
        
        chart += f"{Colors.YELLOW}{'='*80}{Colors.END}\n"
        
        # Business interpretation
        chart += f"\n{Colors.BOLD}📋 How to Read This Matrix:{Colors.END}\n"
        chart += f"{Colors.RED}🔴 High own-price sensitivity: Be careful raising prices{Colors.END}\n"
        chart += f"{Colors.YELLOW}🟡 Moderate own-price sensitivity: You can adjust prices carefully{Colors.END}\n"
        chart += f"{Colors.GREEN}🟢 Low own-price sensitivity: You have pricing power{Colors.END}\n"
        chart += f"{Colors.GREEN}🔄 Substitutes: When you raise price of one, customers buy the other{Colors.END}\n"
        chart += f"{Colors.BLUE}🔗 Complements: When you raise price of one, customers buy less of both{Colors.END}\n"
        chart += f"{Colors.WHITE}⚪ Independent: Products don't affect each other's sales{Colors.END}\n\n"
        
        chart += f"{Colors.BOLD}💡 Portfolio Strategy Tips:{Colors.END}\n"
        chart += f"• Focus promotions on high-sensitivity products (🔴)\n"
        chart += f"• Use substitute products (🔄) to capture customers from competitors\n"
        chart += f"• Bundle complementary products (🔗) together for better value\n"
        chart += f"• Independent products (⚪) can be priced without worrying about cannibalization\n"
        
        return chart
        
    def create_income_effects_chart(self, results: Dict, title: str) -> str:
        """Create ASCII income effects chart."""
        if not results:
            return f"{Colors.RED}No income effects data available for {title}{Colors.END}"
            
        chart = f"\n{Colors.BOLD}{Colors.GREEN}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        # Income effects by product
        if 'income_effects' in results:
            income_effects = results['income_effects']
            chart += f"{Colors.CYAN}Income Effects by Product:{Colors.END}\n"
            
            for product, effect in income_effects.items():
                if effect > 0.1:
                    color = Colors.GREEN
                    symbol = "📈"
                    interpretation = "Luxury good - demand increases with income"
                elif effect > 0:
                    color = Colors.YELLOW
                    symbol = "📊"
                    interpretation = "Normal good - demand increases slightly with income"
                elif effect > -0.1:
                    color = Colors.BLUE
                    symbol = "📉"
                    interpretation = "Inferior good - demand decreases with income"
                else:
                    color = Colors.RED
                    symbol = "📉"
                    interpretation = "Strong inferior good - demand decreases significantly with income"
                
                chart += f"  {color}{symbol} {product}: {effect:>6.3f} - {interpretation}{Colors.END}\n"
        
        # Engel curve analysis
        if 'engel_curves' in results:
            chart += f"\n{Colors.CYAN}Engel Curve Analysis:{Colors.END}\n"
            engel_curves = results['engel_curves']
            
            for product, curve_type in engel_curves.items():
                if curve_type == 'quadratic':
                    color = Colors.MAGENTA
                    symbol = "📈"
                    interpretation = "Non-linear relationship with income"
                else:
                    color = Colors.CYAN
                    symbol = "📊"
                    interpretation = "Linear relationship with income"
                
                chart += f"  {color}{symbol} {product}: {curve_type} - {interpretation}{Colors.END}\n"
        
        chart += f"\n{Colors.BOLD}💡 Business Insights:{Colors.END}\n"
        chart += f"• Luxury goods (📈): Target high-income customers with premium pricing\n"
        chart += f"• Normal goods (📊): Standard pricing strategies work well\n"
        chart += f"• Inferior goods (📉): Focus on value pricing and cost reduction\n"
        chart += f"• Non-linear curves (📈): Income effects change as customers get richer\n"
        
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        return chart
        
    def create_welfare_analysis_chart(self, results: Dict, title: str) -> str:
        """Create ASCII welfare analysis chart."""
        if not results:
            return f"{Colors.RED}No welfare analysis data available for {title}{Colors.END}"
            
        chart = f"\n{Colors.BOLD}{Colors.GREEN}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*70}{Colors.END}\n"
        
        # Welfare effects
        if 'welfare_effects' in results:
            welfare = results['welfare_effects']
            chart += f"{Colors.CYAN}Consumer Welfare Impact:{Colors.END}\n"
            
            if 'compensating_variation' in welfare:
                cv = welfare['compensating_variation']
                chart += f"  Compensating Variation: ${cv:.2f}\n"
                chart += f"    (How much to pay customers to keep them as happy as before)\n"
            
            if 'equivalent_variation' in welfare:
                ev = welfare['equivalent_variation']
                chart += f"  Equivalent Variation: ${ev:.2f}\n"
                chart += f"    (How much customers would pay to avoid the price change)\n"
            
            if 'consumer_surplus_change' in welfare:
                cs = welfare['consumer_surplus_change']
                chart += f"  Consumer Surplus Change: ${cs:.2f}\n"
                chart += f"    (Net impact on customer satisfaction)\n"
        
        # Share changes
        if 'share_changes' in results:
            chart += f"\n{Colors.CYAN}Market Share Changes:{Colors.END}\n"
            share_changes = results['share_changes']
            
            for product, change in share_changes.items():
                if change > 0:
                    color = Colors.GREEN
                    symbol = "📈"
                elif change < 0:
                    color = Colors.RED
                    symbol = "📉"
                else:
                    color = Colors.WHITE
                    symbol = "➡️"
                
                chart += f"  {color}{symbol} {product}: {change:>+6.1f}pp{Colors.END}\n"
        
        # Substitution effects
        if 'substitution_effects' in results:
            chart += f"\n{Colors.CYAN}Substitution Patterns:{Colors.END}\n"
            substitution = results['substitution_effects']
            
            for product, effect in substitution.items():
                if effect > 0.1:
                    color = Colors.GREEN
                    symbol = "🔄"
                    interpretation = "Strong substitute"
                elif effect > 0:
                    color = Colors.YELLOW
                    symbol = "🔄"
                    interpretation = "Weak substitute"
                else:
                    color = Colors.BLUE
                    symbol = "🔗"
                    interpretation = "Complement"
                
                chart += f"  {color}{symbol} {product}: {effect:>6.3f} - {interpretation}{Colors.END}\n"
        
        chart += f"\n{Colors.BOLD}💡 Pricing Strategy Implications:{Colors.END}\n"
        chart += f"• Large welfare losses: Consider smaller price increases\n"
        chart += f"• Strong substitution: Competitors will gain market share\n"
        chart += f"• Complementary effects: Bundle products together\n"
        chart += f"• Monitor market share changes after price adjustments\n"
        
        chart += f"{Colors.YELLOW}{'='*70}{Colors.END}\n"
        return chart
        
    def run_basic_aids_analysis(self):
        """Run basic AIDS analysis with visualizations."""
        self.console.print("\n[bold green]🔄 Running Basic AIDS Analysis...[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Estimating AIDS demand system...", total=None)
            
            try:
                results = self.estimator.example_1_basic_aids()
                progress.update(task, description="✅ Basic AIDS analysis completed!")
                
                # Display results
                if 'elasticity_matrix' in results and 'products' in results:
                    matrix_chart = self.create_elasticity_matrix_chart(
                        results['elasticity_matrix'], 
                        results['products'],
                        "🔄 BASIC AIDS ELASTICITY MATRIX"
                    )
                    self.console.print(matrix_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Basic AIDS (Almost Ideal Demand System)",
                    "Shows you how all products in your portfolio interact with each other. It reveals which products compete, which complement each other, and how price changes affect the entire portfolio.",
                    "Gives you a complete picture of your product portfolio dynamics. You can see the full impact of pricing decisions across all products, not just individual ones.",
                    "The analysis might reveal: 'When you raise cola prices 10%, cola sales drop 12%, but water sales increase 8% (customers switch), and chips sales drop 2% (they're complements).' Now you understand the full portfolio impact."
                )
                
                self.results['basic_aids'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Basic AIDS analysis failed: {e}[/red]")
                
    def run_quaids_analysis(self):
        """Run QUAIDS analysis with visualizations."""
        self.console.print("\n[bold green]📈 Running QUAIDS Analysis...[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Estimating QUAIDS demand system...", total=None)
            
            try:
                results = self.estimator.example_2_quaids()
                progress.update(task, description="✅ QUAIDS analysis completed!")
                
                # Display results
                if 'elasticity_matrix' in results and 'products' in results:
                    matrix_chart = self.create_elasticity_matrix_chart(
                        results['elasticity_matrix'], 
                        results['products'],
                        "📈 QUAIDS ELASTICITY MATRIX"
                    )
                    self.console.print(matrix_chart)
                
                # Income effects
                income_chart = self.create_income_effects_chart(results, "💰 INCOME EFFECTS ANALYSIS")
                self.console.print(income_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "QUAIDS (Quadratic AIDS)",
                    "Like AIDS but captures how customer preferences change as they get richer or poorer. Some products become more attractive as income increases, others become less attractive.",
                    "Helps you understand how different income groups respond to your products and how to adjust your strategy as customer incomes change over time.",
                    "The analysis might reveal: 'High-income customers buy 20% more premium cola when their income increases 10%, but low-income customers buy 15% less when their income decreases 10%.' Now you can target different income groups differently."
                )
                
                self.results['quaids'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ QUAIDS analysis failed: {e}[/red]")
                
    def run_demographic_analysis(self):
        """Run demographic analysis with visualizations."""
        self.console.print("\n[bold green]👥 Running Demographic Analysis...[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Analyzing demographic effects...", total=None)
            
            try:
                results = self.estimator.example_4_demographics()
                progress.update(task, description="✅ Demographic analysis completed!")
                
                # Display results
                if 'elasticity_by_demographics' in results:
                    chart = f"\n{Colors.BOLD}{Colors.GREEN}👥 DEMOGRAPHIC ELASTICITY ANALYSIS{Colors.END}\n"
                    chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
                    
                    demo_elasticities = results['elasticity_by_demographics']
                    for demo_group, elasticities in demo_elasticities.items():
                        chart += f"\n{Colors.CYAN}{demo_group}:{Colors.END}\n"
                        for product, elasticity in elasticities.items():
                            if elasticity < -1.0:
                                color = Colors.RED
                                symbol = "🔴"
                            elif elasticity < -0.5:
                                color = Colors.YELLOW
                                symbol = "🟡"
                            else:
                                color = Colors.GREEN
                                symbol = "🟢"
                            
                            chart += f"  {color}{symbol} {product}: {elasticity:>6.3f}{Colors.END}\n"
                    
                    chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
                    self.console.print(chart)
                
                # Method explanation
                self.print_method_explanation(
                    "AIDS with Demographics",
                    "Shows you how different customer groups (by income, age, location) respond to price changes across your entire product portfolio.",
                    "Lets you create targeted pricing and marketing strategies for different customer segments. You can price differently for different demographic groups.",
                    "The analysis might reveal: 'High-income urban customers barely notice price increases on premium products (3% sales drop), but low-income rural customers are very sensitive (18% sales drop).' Now you can segment your pricing strategy."
                )
                
                self.results['demographics'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Demographic analysis failed: {e}[/red]")
                
    def run_welfare_analysis(self):
        """Run welfare analysis with visualizations."""
        self.console.print("\n[bold green]💰 Running Welfare Analysis...[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Analyzing consumer welfare impact...", total=None)
            
            try:
                results = self.estimator.example_5_welfare()
                progress.update(task, description="✅ Welfare analysis completed!")
                
                # Display results
                welfare_chart = self.create_welfare_analysis_chart(results, "💰 CONSUMER WELFARE ANALYSIS")
                self.console.print(welfare_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Welfare Analysis with AIDS",
                    "Measures the impact of price changes on customer satisfaction and well-being. It shows how much customers lose or gain from your pricing decisions.",
                    "Helps you understand the customer impact of pricing decisions and balance profitability with customer satisfaction. Large welfare losses might indicate pricing that's too aggressive.",
                    "The analysis might reveal: 'A 10% price increase causes customers to lose $25 in satisfaction (welfare loss), but also causes 15% of customers to switch to competitors.' This helps you balance profit and customer retention."
                )
                
                self.results['welfare'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Welfare analysis failed: {e}[/red]")
                
    def create_summary_dashboard(self):
        """Create a summary dashboard of all results."""
        if not self.results:
            self.console.print("[red]No results to display in summary dashboard.[/red]")
            return
            
        # Create summary table
        table = Table(title="🔄 AIDS/QUAIDS ANALYSIS SUMMARY", box=box.ROUNDED)
        table.add_column("Analysis", style="cyan", no_wrap=True)
        table.add_column("Key Insight", style="magenta")
        table.add_column("Business Impact", style="green")
        
        insights = {
            "Basic AIDS": "Complete portfolio elasticity matrix",
            "QUAIDS": "Income effects and non-linear preferences",
            "Demographics": "Segmented elasticity by customer groups",
            "Welfare": "Consumer impact of pricing decisions"
        }
        
        impacts = {
            "Basic AIDS": "Portfolio pricing optimization",
            "QUAIDS": "Income-based pricing strategies",
            "Demographics": "Segmented marketing and pricing",
            "Welfare": "Customer satisfaction optimization"
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
            "[yellow]🎯 Portfolio Pricing:[/yellow]\n"
            "• Use elasticity matrix to optimize all product prices together\n"
            "• Focus promotions on high-sensitivity products\n"
            "• Bundle complementary products for better value\n\n"
            "[blue]📊 Customer Segmentation:[/blue]\n"
            "• Price differently for different income groups\n"
            "• Adjust pricing by demographic characteristics\n"
            "• Monitor how preferences change with income\n\n"
            "[magenta]💰 Customer Satisfaction:[/magenta]\n"
            "• Balance profitability with customer welfare\n"
            "• Monitor substitution patterns after price changes\n"
            "• Use welfare analysis to guide pricing decisions",
            title="🎯 Your Next Steps",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(recommendations)
        
    def run_interactive_analysis(self):
        """Run interactive analysis with user choices."""
        self.print_banner()
        
        # Initialize estimator
        self.console.print("\n[bold blue]🚀 Initializing AIDS/QUAIDS analysis...[/bold blue]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Loading data and models...", total=None)
            self.estimator = AIDSEstimator()
            progress.update(task, description="✅ Ready for analysis!")
        
        # Analysis menu
        while True:
            self.console.print("\n[bold cyan]📋 AIDS/QUAIDS ANALYSIS MENU[/bold cyan]")
            self.console.print("1. 🔄 Basic AIDS (Portfolio Analysis)")
            self.console.print("2. 📈 QUAIDS (Income Effects)")
            self.console.print("3. 👥 Demographics (Customer Segments)")
            self.console.print("4. 💰 Welfare Analysis (Customer Impact)")
            self.console.print("5. 📊 View Summary Dashboard")
            self.console.print("6. 🚪 Exit")
            
            choice = input("\n[bold yellow]Select analysis (1-6): [/bold yellow]").strip()
            
            if choice == '1':
                self.run_basic_aids_analysis()
            elif choice == '2':
                self.run_quaids_analysis()
            elif choice == '3':
                self.run_demographic_analysis()
            elif choice == '4':
                self.run_welfare_analysis()
            elif choice == '5':
                self.create_summary_dashboard()
            elif choice == '6':
                self.console.print("\n[bold green]👋 Thank you for using the AIDS/QUAIDS Analysis Tool![/bold green]")
                break
            else:
                self.console.print("[red]❌ Invalid choice. Please select 1-6.[/red]")
            
            input("\n[bold blue]Press Enter to continue...[/bold blue]")

def main():
    """Main function to run the CLI."""
    try:
        cli = StatsmodelsCLI()
        cli.run_interactive_analysis()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Analysis interrupted by user. Goodbye!{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}[/red]")

if __name__ == "__main__":
    main()

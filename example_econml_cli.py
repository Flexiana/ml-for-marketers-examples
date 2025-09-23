#!/usr/bin/env python3
"""
EconML CLI with Colorful ASCII Visualizations for Marketing Managers

This module provides an interactive command-line interface for cross-price elasticity
estimation using EconML methods, with colorful ASCII charts and marketing-friendly
explanations.

Usage:
    python example_econml_cli.py
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

from example_econml import EconMLElasticityEstimator

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

@dataclass
class ElasticityResult:
    method: str
    elasticity: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    business_impact: str

class EconMLCLI:
    """Interactive CLI for EconML elasticity estimation with marketing focus."""
    
    def __init__(self):
        self.console = Console()
        self.estimator = None
        self.results = {}
        
    def print_banner(self):
        """Print colorful banner."""
        banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🎯 CROSS-PRICE ELASTICITY ANALYSIS FOR MARKETING MANAGERS 🎯                ║
║                                                                              ║
║  📊 Advanced AI-Powered Price Sensitivity Analysis                           ║
║  🚀 Powered by Microsoft's EconML Library                                    ║
║  💡 Understand how price changes affect demand across your product portfolio ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}
        """
        self.console.print(banner)
        
    def print_method_explanation(self, method: str, description: str, business_value: str, real_world_example: str):
        """Print method explanation in marketing terms."""
        panel = Panel(
            f"[bold cyan]{method}[/bold cyan]\n\n"
            f"[yellow]What it tells you:[/yellow] {description}\n\n"
            f"[green]Why you care:[/green] {business_value}\n\n"
            f"[blue]Real example:[/blue] {real_world_example}",
            title="📈 What This Means for Your Business",
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(panel)
        
    def create_elasticity_chart(self, results: Dict, title: str) -> str:
        """Create ASCII elasticity chart."""
        if not results:
            return f"{Colors.RED}No data available for {title}{Colors.END}"
            
        # Extract elasticities
        methods = []
        values = []
        for method, data in results.items():
            if isinstance(data, dict) and 'elasticity' in data and data['elasticity'] is not None:
                methods.append(method.replace('_', ' ').title())
                values.append(data['elasticity'])
        
        if not values:
            return f"{Colors.RED}No valid elasticity data for {title}{Colors.END}"
        
        # Create ASCII bar chart
        max_val = max(abs(v) for v in values)
        min_val = min(values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        chart = f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        for i, (method, value) in enumerate(zip(methods, values)):
            # Color based on elasticity value
            if value < -1.5:
                color = Colors.RED
                bar_char = "█"
            elif value < -1.0:
                color = Colors.YELLOW
                bar_char = "▓"
            elif value < -0.5:
                color = Colors.GREEN
                bar_char = "▒"
            else:
                color = Colors.BLUE
                bar_char = "░"
            
            # Calculate bar length
            bar_length = int((abs(value) / max_val) * 40) if max_val > 0 else 0
            bar = bar_char * bar_length
            
            # Format the line
            chart += f"{color}{method:<20} {bar:<40} {value:>6.3f}{Colors.END}\n"
        
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        # Add interpretation
        avg_elasticity = np.mean(values)
        if avg_elasticity < -1.0:
            interpretation = f"{Colors.RED}🚨 HIGH PRICE SENSITIVITY: Customers will buy significantly less when you raise prices{Colors.END}\n"
            interpretation += f"{Colors.RED}   → Use promotions and discounts to drive volume{Colors.END}\n"
            interpretation += f"{Colors.RED}   → Be careful with price increases - test small changes first{Colors.END}"
        elif avg_elasticity < -0.5:
            interpretation = f"{Colors.YELLOW}⚠️ MODERATE PRICE SENSITIVITY: Customers notice price changes but won't abandon you{Colors.END}\n"
            interpretation += f"{Colors.YELLOW}   → You can adjust prices but monitor sales closely{Colors.END}\n"
            interpretation += f"{Colors.YELLOW}   → Focus on value proposition to reduce sensitivity{Colors.END}"
        else:
            interpretation = f"{Colors.GREEN}✅ LOW PRICE SENSITIVITY: Customers are loyal and less affected by price changes{Colors.END}\n"
            interpretation += f"{Colors.GREEN}   → You have pricing power - consider premium pricing{Colors.END}\n"
            interpretation += f"{Colors.GREEN}   → Focus on quality and brand building{Colors.END}"
        
        chart += f"\n{Colors.BOLD}📊 What This Means for Your Business:{Colors.END}\n{interpretation}\n"
        
        return chart
        
    def create_cross_price_matrix(self, matrix: np.ndarray, products: List[str]) -> str:
        """Create ASCII cross-price elasticity matrix."""
        if matrix is None or len(products) == 0:
            return f"{Colors.RED}No cross-price data available{Colors.END}"
        
        chart = f"\n{Colors.BOLD}{Colors.MAGENTA}🔄 CROSS-PRICE ELASTICITY MATRIX{Colors.END}\n"
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
        
        # Legend and business interpretation
        chart += f"\n{Colors.BOLD}📋 How to Read This Matrix:{Colors.END}\n"
        chart += f"{Colors.RED}🔴 High own-price sensitivity: Be careful raising prices{Colors.END}\n"
        chart += f"{Colors.YELLOW}🟡 Moderate own-price sensitivity: You can adjust prices carefully{Colors.END}\n"
        chart += f"{Colors.GREEN}🟢 Low own-price sensitivity: You have pricing power{Colors.END}\n"
        chart += f"{Colors.GREEN}🔄 Substitutes: When you raise price of one, customers buy the other{Colors.END}\n"
        chart += f"{Colors.BLUE}🔗 Complements: When you raise price of one, customers buy less of both{Colors.END}\n"
        chart += f"{Colors.WHITE}⚪ Independent: Products don't affect each other's sales{Colors.END}\n\n"
        
        chart += f"{Colors.BOLD}💡 Business Strategy Tips:{Colors.END}\n"
        chart += f"• Focus promotions on high-sensitivity products (🔴)\n"
        chart += f"• Use substitute products (🔄) to capture customers from competitors\n"
        chart += f"• Bundle complementary products (🔗) together for better value\n"
        chart += f"• Independent products (⚪) can be priced without worrying about cannibalization\n"
        
        return chart
        
    def create_heterogeneity_chart(self, cate: np.ndarray, title: str) -> str:
        """Create ASCII heterogeneity distribution chart."""
        if cate is None or len(cate) == 0:
            return f"{Colors.RED}No heterogeneity data available for {title}{Colors.END}"
        
        chart = f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        # Create histogram
        hist, bins = np.histogram(cate, bins=20)
        max_count = max(hist)
        
        for i in range(len(hist)):
            bin_start = bins[i]
            bin_end = bins[i+1]
            count = hist[i]
            
            # Color based on elasticity value
            avg_elasticity = (bin_start + bin_end) / 2
            if avg_elasticity < -1.5:
                color = Colors.RED
                bar_char = "█"
            elif avg_elasticity < -1.0:
                color = Colors.YELLOW
                bar_char = "▓"
            elif avg_elasticity < -0.5:
                color = Colors.GREEN
                bar_char = "▒"
            else:
                color = Colors.BLUE
                bar_char = "░"
            
            # Create bar
            bar_length = int((count / max_count) * 40) if max_count > 0 else 0
            bar = bar_char * bar_length
            
            chart += f"{color}{bin_start:>6.2f} to {bin_end:>6.2f}: {bar:<40} {count:>4d}{Colors.END}\n"
        
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        # Statistics
        mean_elasticity = np.mean(cate)
        std_elasticity = np.std(cate)
        
        chart += f"\n{Colors.BOLD}📊 Statistics:{Colors.END}\n"
        chart += f"Mean elasticity: {mean_elasticity:.3f}\n"
        chart += f"Standard deviation: {std_elasticity:.3f}\n"
        chart += f"Range: {np.min(cate):.3f} to {np.max(cate):.3f}\n"
        
        return chart
        
    def run_dml_analysis(self):
        """Run DML analysis with visualizations."""
        self.console.print("\n[bold cyan]🔬 Running Double Machine Learning Analysis...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Analyzing price sensitivity...", total=None)
            
            try:
                results = self.estimator.example_1_double_ml()
                progress.update(task, description="✅ DML analysis completed!")
                
                # Display results
                chart = self.create_elasticity_chart(results, "🎯 DOUBLE MACHINE LEARNING RESULTS")
                self.console.print(chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Double Machine Learning (DML)",
                    "Shows you exactly how much sales will change when you change prices, after accounting for everything else that affects demand (like weather, holidays, competitor actions).",
                    "You get clean, reliable numbers for pricing decisions. No more guessing if a sales drop was due to your price increase or something else.",
                    "If you raise your cola price by 10% and see a 12% sales drop, DML tells you: '8% of that drop was due to your price increase, 4% was due to the heat wave that week.'"
                )
                
                self.results['dml'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ DML analysis failed: {e}[/red]")
                
    def run_iv_analysis(self):
        """Run IV analysis with visualizations."""
        self.console.print("\n[bold cyan]🎯 Running Instrumental Variables Analysis...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Estimating causal effects...", total=None)
            
            try:
                results = self.estimator.example_2_instrumental_variables()
                progress.update(task, description="✅ IV analysis completed!")
                
                # Display results
                chart = self.create_elasticity_chart(results, "🎯 INSTRUMENTAL VARIABLES RESULTS")
                self.console.print(chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Instrumental Variables (IV)",
                    "Uses cost changes (like supplier price increases) to measure how customers really respond to price changes, without any bias from your pricing strategy.",
                    "Gives you the most trustworthy price sensitivity numbers because it uses 'natural experiments' - when costs go up, you have to raise prices, and we can see how customers react.",
                    "When sugar costs increase 20% and you raise cola prices 15%, IV analysis shows: 'Customers will buy 18% less cola for every 10% price increase.' This is your true price sensitivity."
                )
                
                self.results['iv'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ IV analysis failed: {e}[/red]")
                
    def run_causal_forest_analysis(self):
        """Run Causal Forest analysis with visualizations."""
        self.console.print("\n[bold cyan]🌲 Running Causal Forest Analysis...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Discovering heterogeneous effects...", total=None)
            
            try:
                results = self.estimator.example_3_causal_forests()
                progress.update(task, description="✅ Causal Forest analysis completed!")
                
                # Display results
                chart = self.create_elasticity_chart(results, "🌲 CAUSAL FOREST RESULTS")
                self.console.print(chart)
                
                # Heterogeneity visualization
                if 'cate' in results and results['cate'] is not None:
                    heterogeneity_chart = self.create_heterogeneity_chart(
                        results['cate'], 
                        "🎯 PRICE SENSITIVITY ACROSS CUSTOMER SEGMENTS"
                    )
                    self.console.print(heterogeneity_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Causal Forest",
                    "Automatically discovers which types of customers are most sensitive to price changes - without you having to guess or manually segment them.",
                    "Shows you exactly which customer groups to target with different pricing strategies. High-income customers might not care about price, while budget-conscious ones do.",
                    "The analysis might reveal: 'Urban customers with high income barely notice a 10% price increase (only 3% sales drop), but rural customers with lower income are very sensitive (15% sales drop).'"
                )
                
                self.results['causal_forest'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Causal Forest analysis failed: {e}[/red]")
                
    def run_dr_analysis(self):
        """Run Doubly Robust analysis with visualizations."""
        self.console.print("\n[bold cyan]🛡️ Running Doubly Robust Analysis...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Building robust estimates...", total=None)
            
            try:
                results = self.estimator.example_4_doubly_robust_learners()
                progress.update(task, description="✅ Doubly Robust analysis completed!")
                
                # Display results
                chart = self.create_elasticity_chart(results, "🛡️ DOUBLY ROBUST RESULTS")
                self.console.print(chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Doubly Robust Learners",
                    "Uses two different AI methods to calculate price sensitivity, then combines them to get the most reliable answer. If one method fails, the other backs it up.",
                    "Gives you the most confident pricing decisions because it double-checks the results. You can trust these numbers even more than single-method estimates.",
                    "Instead of getting one answer like 'customers are 12% sensitive to price changes,' you get a robust estimate that says 'customers are 11-13% sensitive' with high confidence."
                )
                
                self.results['dr'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Doubly Robust analysis failed: {e}[/red]")
                
    def run_cross_price_analysis(self):
        """Run cross-price elasticity analysis with visualizations."""
        self.console.print("\n[bold cyan]🔄 Running Cross-Price Elasticity Analysis...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Analyzing product interactions...", total=None)
            
            try:
                results = self.estimator.example_5_cross_price_elasticity()
                progress.update(task, description="✅ Cross-price analysis completed!")
                
                # Display matrix
                if 'elasticity_matrix' in results and 'products' in results:
                    matrix_chart = self.create_cross_price_matrix(
                        results['elasticity_matrix'], 
                        results['products']
                    )
                    self.console.print(matrix_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Cross-Price Elasticity Matrix",
                    "Shows you exactly what happens to sales of other products when you change the price of one product. It reveals which products customers switch between.",
                    "Helps you optimize your entire product portfolio pricing. You can see if products compete with each other or if they're bought together, and price accordingly.",
                    "The matrix might show: 'When you raise Pepsi prices 10%, Coca-Cola sales increase 8% (customers switch), but water sales barely change (they're independent).' This tells you Pepsi and Coke compete directly."
                )
                
                self.results['cross_price'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Cross-price analysis failed: {e}[/red]")
                
    def create_summary_dashboard(self):
        """Create a summary dashboard of all results."""
        if not self.results:
            self.console.print("[red]No results to display in summary dashboard.[/red]")
            return
            
        # Create summary table
        table = Table(title="📊 ANALYSIS SUMMARY DASHBOARD", box=box.ROUNDED)
        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Elasticity", style="magenta", justify="center")
        table.add_column("Interpretation", style="green")
        table.add_column("Business Impact", style="yellow")
        
        for method_name, results in self.results.items():
            if isinstance(results, dict):
                # Extract elasticity
                elasticity = None
                for key, value in results.items():
                    if isinstance(value, dict) and 'elasticity' in value and value['elasticity'] is not None:
                        elasticity = value['elasticity']
                        break
                
                if elasticity is not None:
                    # Interpretation
                    if elasticity < -1.0:
                        interpretation = "High sensitivity"
                        impact = "Price changes have strong effect"
                    elif elasticity < -0.5:
                        interpretation = "Moderate sensitivity"
                        impact = "Price changes have moderate effect"
                    else:
                        interpretation = "Low sensitivity"
                        impact = "Price changes have weak effect"
                    
                    table.add_row(
                        method_name.replace('_', ' ').title(),
                        f"{elasticity:.3f}",
                        interpretation,
                        impact
                    )
        
        self.console.print(table)
        
        # Business recommendations
        recommendations = Panel(
            "[bold green]💡 IMMEDIATE ACTION ITEMS:[/bold green]\n\n"
            "[yellow]🎯 Pricing Strategy:[/yellow]\n"
            "• Products with elasticity < -1.0: Use promotions to drive volume\n"
            "• Products with elasticity > -0.5: You can raise prices without losing many customers\n"
            "• Test 5-10% price changes on your most elastic products first\n\n"
            "[blue]📊 Portfolio Management:[/blue]\n"
            "• When raising prices on one product, expect customers to switch to substitutes\n"
            "• Bundle complementary products together to reduce price sensitivity\n"
            "• Use cross-price data to set competitive prices against rivals\n\n"
            "[magenta]👥 Customer Targeting:[/magenta]\n"
            "• Send discount offers to price-sensitive customer segments\n"
            "• Focus premium marketing on less price-sensitive groups\n"
            "• Adjust pricing by location based on local sensitivity patterns",
            title="🎯 Your Next Steps",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(recommendations)
        
    def run_interactive_analysis(self):
        """Run interactive analysis with user choices."""
        self.print_banner()
        
        # Initialize estimator
        self.console.print("\n[bold blue]🚀 Initializing AI-powered elasticity analysis...[/bold blue]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Loading data and models...", total=None)
            self.estimator = EconMLElasticityEstimator()
            progress.update(task, description="✅ Ready for analysis!")
        
        # Analysis menu
        while True:
            self.console.print("\n[bold cyan]📋 ANALYSIS MENU[/bold cyan]")
            self.console.print("1. 🎯 Double Machine Learning (DML)")
            self.console.print("2. 🎯 Instrumental Variables (IV)")
            self.console.print("3. 🌲 Causal Forest (Heterogeneous Effects)")
            self.console.print("4. 🛡️ Doubly Robust Learners")
            self.console.print("5. 🔄 Cross-Price Elasticity Matrix")
            self.console.print("6. 📊 View Summary Dashboard")
            self.console.print("7. 🚪 Exit")
            
            choice = input("\n[bold yellow]Select analysis (1-7): [/bold yellow]").strip()
            
            if choice == '1':
                self.run_dml_analysis()
            elif choice == '2':
                self.run_iv_analysis()
            elif choice == '3':
                self.run_causal_forest_analysis()
            elif choice == '4':
                self.run_dr_analysis()
            elif choice == '5':
                self.run_cross_price_analysis()
            elif choice == '6':
                self.create_summary_dashboard()
            elif choice == '7':
                self.console.print("\n[bold green]👋 Thank you for using the Elasticity Analysis Tool![/bold green]")
                break
            else:
                self.console.print("[red]❌ Invalid choice. Please select 1-7.[/red]")
            
            input("\n[bold blue]Press Enter to continue...[/bold blue]")

def main():
    """Main function to run the CLI."""
    try:
        cli = EconMLCLI()
        cli.run_interactive_analysis()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Analysis interrupted by user. Goodbye!{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")

if __name__ == "__main__":
    main()

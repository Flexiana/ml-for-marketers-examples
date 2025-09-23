#!/usr/bin/env python3
"""
PyBLP CLI with Colorful ASCII Visualizations for Marketing Managers

This module provides an interactive command-line interface for structural demand
estimation using PyBLP, with colorful ASCII charts and marketing-friendly
explanations.

Usage:
    python example_pyblp_cli.py
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

from example_pyblp import BLPElasticityEstimator

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

class PyBLPCLI:
    """Interactive CLI for PyBLP structural demand estimation with marketing focus."""
    
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
║  🏗️ STRUCTURAL DEMAND ANALYSIS FOR MARKETING MANAGERS 🏗️                    ║
║                                                                              ║
║  📊 Advanced Structural Econometric Modeling                                 ║
║  🚀 Powered by PyBLP (Berry, Levinsohn, Pakes)                              ║
║  💡 Understand consumer choice behavior and market competition               ║
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
            title="🏗️ What This Means for Your Business",
            border_style="magenta",
            padding=(1, 2)
        )
        self.console.print(panel)
        
    def create_elasticity_heatmap(self, elasticities: np.ndarray, products: List[str], title: str) -> str:
        """Create ASCII elasticity heatmap."""
        if elasticities is None or len(products) == 0:
            return f"{Colors.RED}No elasticity data available for {title}{Colors.END}"
        
        chart = f"\n{Colors.BOLD}{Colors.MAGENTA}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*80}{Colors.END}\n"
        
        # Create heatmap
        n_products = len(products)
        max_elasticity = np.max(np.abs(elasticities))
        
        # Header
        header = f"{'Product':<15}"
        for product in products:
            header += f"{product[:8]:<12}"
        chart += f"{Colors.CYAN}{header}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'-'*80}{Colors.END}\n"
        
        # Heatmap rows
        for i, product in enumerate(products):
            row = f"{product[:12]:<15}"
            for j in range(n_products):
                value = elasticities[i, j]
                
                # Color intensity based on absolute value
                intensity = abs(value) / max_elasticity if max_elasticity > 0 else 0
                
                if i == j:  # Own-price elasticity
                    if value < -1.5:
                        color = Colors.BG_RED
                        symbol = "🔴"
                    elif value < -1.0:
                        color = Colors.RED
                        symbol = "🟠"
                    elif value < -0.5:
                        color = Colors.YELLOW
                        symbol = "🟡"
                    else:
                        color = Colors.GREEN
                        symbol = "🟢"
                else:  # Cross-price elasticity
                    if value > 0.1:
                        color = Colors.BG_GREEN
                        symbol = "🔄"  # Substitutes
                    elif value < -0.1:
                        color = Colors.BG_BLUE
                        symbol = "🔗"  # Complements
                    else:
                        color = Colors.WHITE
                        symbol = "⚪"  # Independent
                
                # Create intensity bar
                bar_length = int(intensity * 8)
                bar = "█" * bar_length + "░" * (8 - bar_length)
                
                row += f"{color}{symbol} {bar} {value:>6.3f}{Colors.END} "
            chart += row + "\n"
        
        chart += f"{Colors.YELLOW}{'='*80}{Colors.END}\n"
        
        return chart
        
    def create_market_share_chart(self, market_shares: np.ndarray, products: List[str]) -> str:
        """Create ASCII market share chart."""
        if market_shares is None or len(products) == 0:
            return f"{Colors.RED}No market share data available{Colors.END}"
        
        chart = f"\n{Colors.BOLD}{Colors.CYAN}📊 MARKET SHARE DISTRIBUTION{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        # Sort by market share
        sorted_indices = np.argsort(market_shares)[::-1]
        max_share = max(market_shares)
        
        for i, idx in enumerate(sorted_indices):
            product = products[idx]
            share = market_shares[idx]
            percentage = share * 100
            
            # Color based on market share
            if percentage > 30:
                color = Colors.GREEN
                symbol = "🥇"
            elif percentage > 15:
                color = Colors.YELLOW
                symbol = "🥈"
            elif percentage > 5:
                color = Colors.BLUE
                symbol = "🥉"
            else:
                color = Colors.WHITE
                symbol = "📊"
            
            # Create bar
            bar_length = int((share / max_share) * 40) if max_share > 0 else 0
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            chart += f"{color}{symbol} {product[:12]:<12} {bar:<40} {percentage:>6.1f}%{Colors.END}\n"
        
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        return chart
        
    def create_consumer_heterogeneity_chart(self, heterogeneity: Dict, title: str) -> str:
        """Create ASCII consumer heterogeneity chart."""
        if not heterogeneity:
            return f"{Colors.RED}No heterogeneity data available for {title}{Colors.END}"
        
        chart = f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        for segment, data in heterogeneity.items():
            chart += f"\n{Colors.BOLD}{Colors.MAGENTA}📈 {segment}:{Colors.END}\n"
            
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        # Create visual representation
                        if 'elasticity' in key.lower():
                            if value < -1.0:
                                color = Colors.RED
                                symbol = "🔴"
                            elif value < -0.5:
                                color = Colors.YELLOW
                                symbol = "🟡"
                            else:
                                color = Colors.GREEN
                                symbol = "🟢"
                        else:
                            color = Colors.CYAN
                            symbol = "📊"
                        
                        chart += f"  {color}{symbol} {key:<25}: {value:>8.3f}{Colors.END}\n"
        
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        return chart
        
    def create_competition_analysis(self, results: Dict) -> str:
        """Create ASCII competition analysis chart."""
        chart = f"\n{Colors.BOLD}{Colors.RED}⚔️ COMPETITION ANALYSIS{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        # Extract competition metrics
        if 'elasticities' in results:
            elasticities = results['elasticities']
            n_products = elasticities.shape[0]
            
            # Calculate competition intensity
            competition_scores = []
            for i in range(n_products):
                # Own-price elasticity (negative)
                own_elasticity = abs(elasticities[i, i])
                # Cross-price elasticities (positive means competition)
                cross_elasticities = elasticities[i, :]
                cross_elasticities = np.delete(cross_elasticities, i)  # Remove own-price
                avg_cross_elasticity = np.mean(cross_elasticities)
                
                competition_score = own_elasticity + avg_cross_elasticity
                competition_scores.append(competition_score)
            
            # Sort by competition intensity
            sorted_indices = np.argsort(competition_scores)[::-1]
            
            chart += f"{Colors.CYAN}Competition Intensity Ranking:{Colors.END}\n"
            for i, idx in enumerate(sorted_indices):
                score = competition_scores[idx]
                
                if score > 2.0:
                    color = Colors.RED
                    symbol = "🔥"
                    level = "High"
                elif score > 1.0:
                    color = Colors.YELLOW
                    symbol = "⚡"
                    level = "Medium"
                else:
                    color = Colors.GREEN
                    symbol = "🟢"
                    level = "Low"
                
                chart += f"  {color}{symbol} Product {idx+1:<3}: {score:>6.3f} ({level}){Colors.END}\n"
        
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        return chart
        
    def run_basic_blp_analysis(self):
        """Run basic BLP analysis with visualizations."""
        self.console.print("\n[bold magenta]🏗️ Running Basic BLP Analysis...[/bold magenta]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Estimating structural demand parameters...", total=None)
            
            try:
                results = self.estimator.example_1_basic_blp()
                progress.update(task, description="✅ Basic BLP analysis completed!")
                
                # Display results
                if 'elasticities' in results and 'products' in results:
                    heatmap = self.create_elasticity_heatmap(
                        results['elasticities'], 
                        results['products'],
                        "🏗️ BASIC BLP ELASTICITY MATRIX"
                    )
                    self.console.print(heatmap)
                
                # Method explanation
                self.print_method_explanation(
                    "Basic BLP (Berry, Levinsohn, Pakes)",
                    "Shows you exactly how customers choose between products by modeling their decision process. It reveals what drives customer preferences: price, quality, brand, or other factors.",
                    "Gives you the most accurate demand predictions because it understands WHY customers buy what they buy, not just WHAT they buy. This helps you design better products and pricing.",
                    "The analysis might reveal: 'Customers care 60% about price, 25% about brand reputation, and 15% about product features.' Now you know exactly what to focus on in your marketing."
                )
                
                self.results['basic_blp'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Basic BLP analysis failed: {e}[/red]")
                
    def run_random_coefficients_analysis(self):
        """Run random coefficients analysis with visualizations."""
        self.console.print("\n[bold magenta]🎲 Running Random Coefficients Analysis...[/bold magenta]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Estimating heterogeneous preferences...", total=None)
            
            try:
                results = self.estimator.example_2_random_coefficients()
                progress.update(task, description="✅ Random coefficients analysis completed!")
                
                # Display results
                if 'elasticities' in results and 'products' in results:
                    heatmap = self.create_elasticity_heatmap(
                        results['elasticities'], 
                        results['products'],
                        "🎲 RANDOM COEFFICIENTS ELASTICITY MATRIX"
                    )
                    self.console.print(heatmap)
                
                # Method explanation
                self.print_method_explanation(
                    "Random Coefficients BLP",
                    "Automatically discovers that different customers have different preferences. Some care more about price, others about quality, brand, or convenience.",
                    "Shows you exactly which customer groups to target with different strategies. You can price differently for budget-conscious vs. quality-focused customers.",
                    "The analysis might reveal: 'High-income customers don't care about a 10% price increase (only 2% sales drop), but budget-conscious customers are very sensitive (15% sales drop).' Now you can target each group differently."
                )
                
                self.results['random_coefficients'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Random coefficients analysis failed: {e}[/red]")
                
    def run_demographic_analysis(self):
        """Run demographic analysis with visualizations."""
        self.console.print("\n[bold magenta]👥 Running Demographic Analysis...[/bold magenta]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Analyzing demographic effects...", total=None)
            
            try:
                results = self.estimator.example_3_demographic_interactions()
                progress.update(task, description="✅ Demographic analysis completed!")
                
                # Display heterogeneity
                if 'heterogeneity' in results:
                    heterogeneity_chart = self.create_consumer_heterogeneity_chart(
                        results['heterogeneity'],
                        "👥 CONSUMER HETEROGENEITY BY DEMOGRAPHICS"
                    )
                    self.console.print(heterogeneity_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Demographic Interactions",
                    "Shows you exactly how different types of customers (by income, age, location) respond to price changes. High-income customers might not care about price, while younger customers might be more sensitive.",
                    "Lets you create targeted pricing strategies for different customer groups. You can charge premium prices to less price-sensitive groups and use promotions for price-sensitive ones.",
                    "The analysis might reveal: 'Customers in urban areas with high income barely notice a 15% price increase (only 3% sales drop), but rural customers with lower income are very sensitive (20% sales drop).' You can now price by location and income."
                )
                
                self.results['demographics'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Demographic analysis failed: {e}[/red]")
                
    def run_market_share_analysis(self):
        """Run market share analysis with visualizations."""
        self.console.print("\n[bold magenta]📊 Running Market Share Analysis...[/bold magenta]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Calculating market shares...", total=None)
            
            try:
                results = self.estimator.example_4_market_shares()
                progress.update(task, description="✅ Market share analysis completed!")
                
                # Display market shares
                if 'market_shares' in results and 'products' in results:
                    market_chart = self.create_market_share_chart(
                        results['market_shares'],
                        results['products']
                    )
                    self.console.print(market_chart)
                
                # Competition analysis
                competition_chart = self.create_competition_analysis(results)
                self.console.print(competition_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Market Share Analysis",
                    "Shows you exactly how much of the market each product captures and how that changes with pricing. It reveals which products are market leaders and which are struggling.",
                    "Helps you identify opportunities to gain market share through pricing strategies. You can see which products to promote and which competitors to target.",
                    "The analysis might reveal: 'Your premium cola has 15% market share but could reach 25% with a 5% price cut, while your budget cola dominates with 40% share but is vulnerable to competitor price cuts.' Now you know where to focus your pricing strategy."
                )
                
                self.results['market_shares'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Market share analysis failed: {e}[/red]")
                
    def create_summary_dashboard(self):
        """Create a summary dashboard of all results."""
        if not self.results:
            self.console.print("[red]No results to display in summary dashboard.[/red]")
            return
            
        # Create summary table
        table = Table(title="🏗️ STRUCTURAL DEMAND ANALYSIS SUMMARY", box=box.ROUNDED)
        table.add_column("Analysis", style="cyan", no_wrap=True)
        table.add_column("Key Insight", style="magenta")
        table.add_column("Business Impact", style="green")
        
        insights = {
            "Basic BLP": "Structural demand parameters estimated",
            "Random Coefficients": "Consumer heterogeneity captured",
            "Demographics": "Demographic preferences identified",
            "Market Shares": "Competitive positioning analyzed"
        }
        
        impacts = {
            "Basic BLP": "Accurate demand forecasting",
            "Random Coefficients": "Segmented pricing strategies",
            "Demographics": "Targeted marketing campaigns",
            "Market Shares": "Competitive advantage insights"
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
            "[bold green]💡 STRATEGIC INSIGHTS:[/bold green]\n\n"
            "• Use structural demand estimates for accurate sales forecasting\n"
            "• Implement segmented pricing based on consumer heterogeneity\n"
            "• Target demographic groups with tailored pricing strategies\n"
            "• Monitor market share changes to assess competitive position\n"
            "• Optimize product portfolio based on demand interactions",
            title="🎯 Strategic Recommendations",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(recommendations)
        
    def run_interactive_analysis(self):
        """Run interactive analysis with user choices."""
        self.print_banner()
        
        # Initialize estimator
        self.console.print("\n[bold blue]🚀 Initializing structural demand analysis...[/bold blue]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Loading data and models...", total=None)
            self.estimator = BLPElasticityEstimator()
            progress.update(task, description="✅ Ready for analysis!")
        
        # Analysis menu
        while True:
            self.console.print("\n[bold cyan]📋 STRUCTURAL ANALYSIS MENU[/bold cyan]")
            self.console.print("1. 🏗️ Basic BLP (Structural Demand)")
            self.console.print("2. 🎲 Random Coefficients (Heterogeneity)")
            self.console.print("3. 👥 Demographic Interactions")
            self.console.print("4. 📊 Market Share Analysis")
            self.console.print("5. 📊 View Summary Dashboard")
            self.console.print("6. 🚪 Exit")
            
            choice = input("\n[bold yellow]Select analysis (1-6): [/bold yellow]").strip()
            
            if choice == '1':
                self.run_basic_blp_analysis()
            elif choice == '2':
                self.run_random_coefficients_analysis()
            elif choice == '3':
                self.run_demographic_analysis()
            elif choice == '4':
                self.run_market_share_analysis()
            elif choice == '5':
                self.create_summary_dashboard()
            elif choice == '6':
                self.console.print("\n[bold green]👋 Thank you for using the Structural Demand Analysis Tool![/bold green]")
                break
            else:
                self.console.print("[red]❌ Invalid choice. Please select 1-6.[/red]")
            
            input("\n[bold blue]Press Enter to continue...[/bold blue]")

def main():
    """Main function to run the CLI."""
    try:
        cli = PyBLPCLI()
        cli.run_interactive_analysis()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Analysis interrupted by user. Goodbye!{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")

if __name__ == "__main__":
    main()

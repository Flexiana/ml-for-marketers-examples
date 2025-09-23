#!/usr/bin/env python3
"""
Demo script for CLI tools

This script demonstrates the CLI tools with sample data and visualizations.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def create_demo_elasticity_chart():
    """Create a demo elasticity chart."""
    chart = f"""
{Colors.BOLD}{Colors.CYAN}🎯 DEMO: ELASTICITY ANALYSIS RESULTS{Colors.END}
{Colors.YELLOW}{'='*60}{Colors.END}

{Colors.GREEN}Method                    Elasticity    Interpretation{Colors.END}
{Colors.YELLOW}{'-'*60}{Colors.END}
{Colors.CYAN}Double ML{Colors.END}              -1.234        High sensitivity
{Colors.MAGENTA}PyBLP Structural{Colors.END}      -1.156        High sensitivity  
{Colors.BLUE}Panel Data{Colors.END}             -0.987        Moderate sensitivity
{Colors.GREEN}AIDS Demand{Colors.END}           -1.089        High sensitivity
{Colors.YELLOW}Bayesian{Colors.END}              -1.201        High sensitivity
{Colors.RED}ML DML{Colors.END}                 -1.167        High sensitivity

{Colors.YELLOW}{'='*60}{Colors.END}

{Colors.BOLD}📊 Business Insights:{Colors.END}
• Average elasticity: -1.139 (High price sensitivity)
• All methods agree: customers are very responsive to price changes
• Recommendation: Use promotional pricing to drive volume
• Monitor competitor pricing closely

{Colors.BOLD}💡 Strategic Actions:{Colors.END}
• Implement dynamic pricing based on demand elasticity
• Focus on value proposition to reduce price sensitivity
• Consider bundling strategies for complementary products
• Test price changes in controlled experiments
"""
    return chart

def create_demo_cross_price_matrix():
    """Create a demo cross-price matrix."""
    matrix = f"""
{Colors.BOLD}{Colors.MAGENTA}🔄 DEMO: CROSS-PRICE ELASTICITY MATRIX{Colors.END}
{Colors.YELLOW}{'='*80}{Colors.END}

{Colors.CYAN}Product         Cola A    Cola B    Cola C    Pepsi    Sprite{Colors.END}
{Colors.YELLOW}{'-'*80}{Colors.END}
{Colors.GREEN}Cola A{Colors.END}         🔴 -1.234   🔄  0.156   🔄  0.089   🔄  0.234   🔄  0.123
{Colors.GREEN}Cola B{Colors.END}         🔄  0.145   🔴 -1.156   🔄  0.167   🔄  0.198   🔄  0.134
{Colors.GREEN}Cola C{Colors.END}         🔄  0.098   🔄  0.123   🔴 -1.089   🔄  0.156   🔄  0.145
{Colors.BLUE}Pepsi{Colors.END}          🔄  0.234   🔄  0.198   🔄  0.156   🔴 -1.201   🔄  0.167
{Colors.BLUE}Sprite{Colors.END}         🔄  0.123   🔄  0.134   🔄  0.145   🔄  0.167   🔴 -1.167

{Colors.YELLOW}{'='*80}{Colors.END}

{Colors.BOLD}📋 Legend:{Colors.END}
{Colors.RED}🔴 High own-price sensitivity{Colors.END}
{Colors.GREEN}🔄 Substitutes (positive cross-price){Colors.END}
{Colors.BLUE}🔗 Complements (negative cross-price){Colors.END}

{Colors.BOLD}💡 Portfolio Insights:{Colors.END}
• All products are substitutes (positive cross-price elasticities)
• Cola brands compete most with each other
• Pepsi and Sprite show moderate substitution
• Price changes in one product affect others significantly
"""
    return matrix

def create_demo_heterogeneity_chart():
    """Create a demo heterogeneity chart."""
    chart = f"""
{Colors.BOLD}{Colors.CYAN}👥 DEMO: CUSTOMER HETEROGENEITY ANALYSIS{Colors.END}
{Colors.YELLOW}{'='*60}{Colors.END}

{Colors.BOLD}📈 Price Sensitivity by Customer Segment:{Colors.END}

{Colors.RED}High Income (>$75k):{Colors.END}
  🔴 Elasticity: -0.856 (Low sensitivity)
  💡 Premium pricing strategies effective

{Colors.YELLOW}Medium Income ($35k-$75k):{Colors.END}
  🟡 Elasticity: -1.234 (Moderate sensitivity)
  💡 Balanced pricing approach recommended

{Colors.GREEN}Low Income (<$35k):{Colors.END}
  🔴 Elasticity: -1.567 (High sensitivity)
  💡 Promotional pricing drives volume

{Colors.BOLD}📊 Geographic Segmentation:{Colors.END}

{Colors.BLUE}Urban Areas:{Colors.END}
  🔵 Elasticity: -1.089 (Moderate sensitivity)
  💡 Focus on convenience and quality

{Colors.MAGENTA}Rural Areas:{Colors.END}
  🔴 Elasticity: -1.456 (High sensitivity)
  💡 Price-based strategies more effective

{Colors.YELLOW}{'='*60}{Colors.END}

{Colors.BOLD}🎯 Strategic Recommendations:{Colors.END}
• Implement income-based pricing tiers
• Use geographic pricing strategies
• Target high-income segments with premium products
• Focus promotions on price-sensitive segments
"""
    return chart

def run_demo():
    """Run the demo."""
    console = Console()
    
    # Banner
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🎯 ELASTICITY ANALYSIS CLI DEMO FOR MARKETING MANAGERS 🎯                   ║
║                                                                              ║
║  📊 Interactive Command-Line Interface with Colorful Visualizations         ║
║  🚀 Powered by Rich Python Library                                          ║
║  💡 Marketing-Friendly Explanations and Business Insights                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}
    """
    console.print(banner)
    
    # Demo sections
    demos = [
        ("Elasticity Analysis", create_demo_elasticity_chart),
        ("Cross-Price Matrix", create_demo_cross_price_matrix),
        ("Customer Heterogeneity", create_demo_heterogeneity_chart)
    ]
    
    for title, demo_func in demos:
        console.print(f"\n[bold cyan]📊 {title} Demo[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Generating {title.lower()}...", total=None)
            time.sleep(1)  # Simulate processing
            progress.update(task, description=f"✅ {title} demo ready!")
        
        # Display the demo
        demo_content = demo_func()
        console.print(demo_content)
        
        input(f"\n[bold blue]Press Enter to continue to next demo...[/bold blue]")
    
    # Final message
    final_panel = Panel(
        "[bold green]🎉 Demo Complete![/bold green]\n\n"
        "You've seen examples of:\n"
        "• Colorful ASCII elasticity charts\n"
        "• Cross-price substitution matrices\n"
        "• Customer heterogeneity analysis\n"
        "• Business insights and recommendations\n\n"
        "To run the full CLI tools:\n"
        "• python example_econml_cli.py\n"
        "• python example_pyblp_cli.py\n"
        "• python example_all_methods_cli.py\n\n"
        "These tools provide interactive analysis with real data!",
        title="🚀 Next Steps",
        border_style="green",
        padding=(1, 2)
    )
    console.print(final_panel)

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Demo interrupted by user. Goodbye!{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")

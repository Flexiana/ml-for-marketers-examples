#!/usr/bin/env python3
"""
Simple Demo script for CLI tools (without Rich dependency)

This script demonstrates the CLI tools with sample data and visualizations.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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

def print_banner():
    """Print colorful banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🎯 ELASTICITY ANALYSIS CLI DEMO FOR MARKETING MANAGERS 🎯                   ║
║                                                                              ║
║  📊 Interactive Command-Line Interface with Colorful Visualizations         ║
║  🚀 Powered by Python Terminal Colors                                       ║
║  💡 Marketing-Friendly Explanations and Business Insights                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}
    """
    print(banner)

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

def create_ascii_bar_chart(data, title, max_width=50):
    """Create ASCII bar chart."""
    chart = f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}\n"
    chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
    
    max_val = max(abs(v) for v in data.values())
    
    for method, value in data.items():
        # Color based on value
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
        
        # Create bar
        bar_length = int((abs(value) / max_val) * max_width) if max_val > 0 else 0
        bar = bar_char * bar_length + "░" * (max_width - bar_length)
        
        chart += f"{color}{method:<20} {bar} {value:>6.3f}{Colors.END}\n"
    
    chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
    return chart

def run_demo():
    """Run the demo."""
    print_banner()
    
    # Demo sections
    demos = [
        ("Elasticity Analysis", create_demo_elasticity_chart),
        ("Cross-Price Matrix", create_demo_cross_price_matrix),
        ("Customer Heterogeneity", create_demo_heterogeneity_chart)
    ]
    
    for title, demo_func in demos:
        print(f"\n{Colors.BOLD}{Colors.CYAN}📊 {title} Demo{Colors.END}")
        
        # Simulate processing
        print(f"{Colors.YELLOW}⏳ Generating {title.lower()}...{Colors.END}")
        time.sleep(1)
        print(f"{Colors.GREEN}✅ {title} demo ready!{Colors.END}")
        
        # Display the demo
        demo_content = demo_func()
        print(demo_content)
        
        input(f"\n{Colors.BLUE}Press Enter to continue to next demo...{Colors.END}")
    
    # ASCII bar chart demo
    print(f"\n{Colors.BOLD}{Colors.CYAN}📊 ASCII Bar Chart Demo{Colors.END}")
    sample_data = {
        "Double ML": -1.234,
        "PyBLP": -1.156,
        "Panel Data": -0.987,
        "AIDS": -1.089,
        "Bayesian": -1.201,
        "ML DML": -1.167
    }
    bar_chart = create_ascii_bar_chart(sample_data, "🎯 ELASTICITY COMPARISON")
    print(bar_chart)
    
    # Final message
    final_message = f"""
{Colors.GREEN}{Colors.BOLD}🎉 Demo Complete!{Colors.END}

{Colors.WHITE}You've seen examples of:{Colors.END}
• Colorful ASCII elasticity charts
• Cross-price substitution matrices  
• Customer heterogeneity analysis
• Business insights and recommendations
• Interactive ASCII bar charts

{Colors.YELLOW}To run the full CLI tools:{Colors.END}
• python example_econml_cli.py
• python example_pyblp_cli.py  
• python example_all_methods_cli.py

{Colors.CYAN}These tools provide interactive analysis with real data!{Colors.END}
"""
    print(final_message)

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Demo interrupted by user. Goodbye!{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")

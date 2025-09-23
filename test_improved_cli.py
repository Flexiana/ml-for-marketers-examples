#!/usr/bin/env python3
"""
Test script to demonstrate the improved CLI with better business explanations
"""

import sys
import os
import time
import numpy as np
import pandas as pd
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

def create_improved_elasticity_chart():
    """Create an improved elasticity chart with business-focused explanations."""
    chart = f"""
{Colors.BOLD}{Colors.CYAN}🎯 IMPROVED: ELASTICITY ANALYSIS RESULTS{Colors.END}
{Colors.YELLOW}{'='*70}{Colors.END}

{Colors.GREEN}Method                    Elasticity    What This Means for You{Colors.END}
{Colors.YELLOW}{'-'*70}{Colors.END}
{Colors.CYAN}Double ML{Colors.END}              -1.234        🚨 HIGH SENSITIVITY: Be very careful with price increases
{Colors.MAGENTA}PyBLP Structural{Colors.END}      -1.156        🚨 HIGH SENSITIVITY: Customers will switch if you raise prices
{Colors.BLUE}Panel Data{Colors.END}             -0.987        ⚠️ MODERATE SENSITIVITY: You can adjust prices carefully
{Colors.GREEN}AIDS Demand{Colors.END}           -1.089        🚨 HIGH SENSITIVITY: Focus on promotions to drive volume
{Colors.YELLOW}Bayesian{Colors.END}              -1.201        🚨 HIGH SENSITIVITY: Test small price changes first
{Colors.RED}ML DML{Colors.END}                 -1.167        🚨 HIGH SENSITIVITY: Use discounts strategically

{Colors.YELLOW}{'='*70}{Colors.END}

{Colors.BOLD}📊 What This Means for Your Business:{Colors.END}
{Colors.RED}🚨 HIGH PRICE SENSITIVITY: Customers will buy significantly less when you raise prices{Colors.END}
{Colors.RED}   → Use promotions and discounts to drive volume{Colors.END}
{Colors.RED}   → Be careful with price increases - test small changes first{Colors.END}

{Colors.BOLD}💡 IMMEDIATE ACTION ITEMS:{Colors.END}

{Colors.YELLOW}🎯 Pricing Strategy:{Colors.END}
• Products with elasticity < -1.0: Use promotions to drive volume
• Products with elasticity > -0.5: You can raise prices without losing many customers
• Test 5-10% price changes on your most elastic products first

{Colors.BLUE}📊 Portfolio Management:{Colors.END}
• When raising prices on one product, expect customers to switch to substitutes
• Bundle complementary products together to reduce price sensitivity
• Use cross-price data to set competitive prices against rivals

{Colors.MAGENTA}👥 Customer Targeting:{Colors.END}
• Send discount offers to price-sensitive customer segments
• Focus premium marketing on less price-sensitive groups
• Adjust pricing by location based on local sensitivity patterns
"""
    return chart

def create_improved_cross_price_matrix():
    """Create an improved cross-price matrix with business explanations."""
    matrix = f"""
{Colors.BOLD}{Colors.MAGENTA}🔄 IMPROVED: CROSS-PRICE ELASTICITY MATRIX{Colors.END}
{Colors.YELLOW}{'='*90}{Colors.END}

{Colors.CYAN}Product         Cola A    Cola B    Cola C    Pepsi    Sprite{Colors.END}
{Colors.YELLOW}{'-'*90}{Colors.END}
{Colors.GREEN}Cola A{Colors.END}         🔴 -1.234   🔄  0.156   🔄  0.089   🔄  0.234   🔄  0.123
{Colors.GREEN}Cola B{Colors.END}         🔄  0.145   🔴 -1.156   🔄  0.167   🔄  0.198   🔄  0.134
{Colors.GREEN}Cola C{Colors.END}         🔄  0.098   🔄  0.123   🔴 -1.089   🔄  0.156   🔄  0.145
{Colors.BLUE}Pepsi{Colors.END}          🔄  0.234   🔄  0.198   🔄  0.156   🔴 -1.201   🔄  0.167
{Colors.BLUE}Sprite{Colors.END}         🔄  0.123   🔄  0.134   🔄  0.145   🔄  0.167   🔴 -1.167

{Colors.YELLOW}{'='*90}{Colors.END}

{Colors.BOLD}📋 How to Read This Matrix:{Colors.END}
{Colors.RED}🔴 High own-price sensitivity: Be careful raising prices{Colors.END}
{Colors.YELLOW}🟡 Moderate own-price sensitivity: You can adjust prices carefully{Colors.END}
{Colors.GREEN}🟢 Low own-price sensitivity: You have pricing power{Colors.END}
{Colors.GREEN}🔄 Substitutes: When you raise price of one, customers buy the other{Colors.END}
{Colors.BLUE}🔗 Complements: When you raise price of one, customers buy less of both{Colors.END}
{Colors.WHITE}⚪ Independent: Products don't affect each other's sales{Colors.END}

{Colors.BOLD}💡 Business Strategy Tips:{Colors.END}
• Focus promotions on high-sensitivity products (🔴)
• Use substitute products (🔄) to capture customers from competitors
• Bundle complementary products (🔗) together for better value
• Independent products (⚪) can be priced without worrying about cannibalization

{Colors.BOLD}🎯 Real Business Example:{Colors.END}
{Colors.CYAN}When you raise Pepsi prices 10%, Coca-Cola sales increase 8% (customers switch),{Colors.END}
{Colors.CYAN}but water sales barely change (they're independent). This tells you Pepsi and Coke{Colors.END}
{Colors.CYAN}compete directly - use this to set competitive prices!{Colors.END}
"""
    return matrix

def create_method_explanation_demo():
    """Create a demo of improved method explanations."""
    explanation = f"""
{Colors.BOLD}{Colors.CYAN}📈 What This Means for Your Business{Colors.END}
{Colors.YELLOW}{'='*60}{Colors.END}

{Colors.BOLD}Double Machine Learning (DML){Colors.END}

{Colors.YELLOW}What it tells you:{Colors.END} Shows you exactly how much sales will change when you change prices, 
after accounting for everything else that affects demand (like weather, holidays, competitor actions).

{Colors.GREEN}Why you care:{Colors.END} You get clean, reliable numbers for pricing decisions. No more guessing 
if a sales drop was due to your price increase or something else.

{Colors.BLUE}Real example:{Colors.END} If you raise your cola price by 10% and see a 12% sales drop, DML tells you: 
'8% of that drop was due to your price increase, 4% was due to the heat wave that week.'

{Colors.YELLOW}{'-'*60}{Colors.END}

{Colors.BOLD}Instrumental Variables (IV){Colors.END}

{Colors.YELLOW}What it tells you:{Colors.END} Uses cost changes (like supplier price increases) to measure how 
customers really respond to price changes, without any bias from your pricing strategy.

{Colors.GREEN}Why you care:{Colors.END} Gives you the most trustworthy price sensitivity numbers because it uses 
'natural experiments' - when costs go up, you have to raise prices, and we can see how customers react.

{Colors.BLUE}Real example:{Colors.END} When sugar costs increase 20% and you raise cola prices 15%, IV analysis 
shows: 'Customers will buy 18% less cola for every 10% price increase.' This is your true price sensitivity.

{Colors.YELLOW}{'-'*60}{Colors.END}

{Colors.BOLD}Cross-Price Elasticity Matrix{Colors.END}

{Colors.YELLOW}What it tells you:{Colors.END} Shows you exactly what happens to sales of other products when you 
change the price of one product. It reveals which products customers switch between.

{Colors.GREEN}Why you care:{Colors.END} Helps you optimize your entire product portfolio pricing. You can see if 
products compete with each other or if they're bought together, and price accordingly.

{Colors.BLUE}Real example:{Colors.END} The matrix might show: 'When you raise Pepsi prices 10%, Coca-Cola sales 
increase 8% (customers switch), but water sales barely change (they're independent).' This tells you 
Pepsi and Coke compete directly.
"""
    return explanation

def run_improved_demo():
    """Run the improved demo."""
    print(f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🎯 IMPROVED CLI DEMO - BUSINESS-FOCUSED EXPLANATIONS 🎯                    ║
║                                                                              ║
║  📊 Clear, Actionable Insights for Marketing Managers                       ║
║  🚀 Real Examples and Immediate Action Items                               ║
║  💡 No Technical Jargon - Just Business Value                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}
    """)
    
    # Demo sections
    demos = [
        ("Improved Elasticity Analysis", create_improved_elasticity_chart),
        ("Improved Cross-Price Matrix", create_improved_cross_price_matrix),
        ("Method Explanations", create_method_explanation_demo)
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
    
    # Final message
    final_message = f"""
{Colors.GREEN}{Colors.BOLD}🎉 Improved Demo Complete!{Colors.END}

{Colors.WHITE}Key Improvements:{Colors.END}
• Clear business language instead of technical jargon
• Real-world examples that Marketing Managers can relate to
• Immediate action items with specific recommendations
• Visual indicators (🚨⚠️✅) for quick understanding
• Strategic tips for each analysis type

{Colors.YELLOW}The improved CLI tools now provide:{Colors.END}
• What each number actually means for your business
• Why you should care about each analysis
• Real examples you can relate to
• Specific actions you can take immediately

{Colors.CYAN}These tools now speak your language and give you actionable insights!{Colors.END}
"""
    print(final_message)

if __name__ == "__main__":
    try:
        run_improved_demo()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Demo interrupted by user. Goodbye!{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")

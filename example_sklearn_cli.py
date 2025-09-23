#!/usr/bin/env python3
"""
Scikit-learn/XGBoost CLI with Colorful ASCII Visualizations for Marketing Managers

This module provides an interactive command-line interface for ML-based
Double Machine Learning using scikit-learn and XGBoost/LightGBM, with 
colorful ASCII charts and marketing-friendly explanations.

Usage:
    python example_sklearn_cli.py
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

from example_sklearn_xgb_dml import MLPipelineElasticityEstimator

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

class SklearnCLI:
    """Interactive CLI for ML-based DML analysis with marketing focus."""
    
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
║  🤖 MACHINE LEARNING DML FOR MARKETING MANAGERS 🤖                         ║
║                                                                              ║
║  📊 AI-Powered Elasticity Estimation                                        ║
║  🚀 Powered by Scikit-learn & XGBoost                                      ║
║  💡 Advanced ML models for accurate price sensitivity prediction            ║
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
            title="🤖 What This Means for Your Business",
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(panel)
        
    def create_ml_performance_chart(self, results: Dict, title: str) -> str:
        """Create ASCII ML performance chart."""
        if not results:
            return f"{Colors.RED}No ML performance data available for {title}{Colors.END}"
            
        chart = f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*70}{Colors.END}\n"
        
        # Model performance
        if 'model_performance' in results:
            performance = results['model_performance']
            chart += f"{Colors.MAGENTA}Machine Learning Model Performance:{Colors.END}\n"
            
            for model_name, perf in performance.items():
                r2_y = perf.get('r2_y', 0)
                r2_t = perf.get('r2_t', 0)
                
                # Color based on performance
                if r2_y > 0.8 and r2_t > 0.8:
                    color = Colors.GREEN
                    symbol = "🚀"
                    quality = "Excellent"
                elif r2_y > 0.6 and r2_t > 0.6:
                    color = Colors.YELLOW
                    symbol = "⚡"
                    quality = "Good"
                else:
                    color = Colors.RED
                    symbol = "⚠️"
                    quality = "Needs improvement"
                
                chart += f"  {color}{symbol} {model_name}:{Colors.END}\n"
                chart += f"    Outcome R²: {r2_y:.3f}\n"
                chart += f"    Treatment R²: {r2_t:.3f}\n"
                chart += f"    Quality: {quality}\n\n"
        
        # Cross-validation results
        if 'cv_results' in results:
            cv_results = results['cv_results']
            chart += f"{Colors.MAGENTA}Cross-Validation Results:{Colors.END}\n"
            
            for fold, fold_results in cv_results.items():
                chart += f"  Fold {fold}: R² = {fold_results['r2']:.3f}\n"
            
            avg_r2 = np.mean([r['r2'] for r in cv_results.values()])
            chart += f"  Average R²: {avg_r2:.3f}\n"
            
            if avg_r2 > 0.7:
                chart += f"  {Colors.GREEN}✅ Good cross-validation performance{Colors.END}\n"
            elif avg_r2 > 0.5:
                chart += f"  {Colors.YELLOW}⚠️ Moderate cross-validation performance{Colors.END}\n"
            else:
                chart += f"  {Colors.RED}❌ Poor cross-validation performance{Colors.END}\n"
        
        chart += f"{Colors.YELLOW}{'='*70}{Colors.END}\n"
        return chart
        
    def create_elasticity_comparison_chart(self, results: Dict, title: str) -> str:
        """Create ASCII elasticity comparison chart."""
        if not results:
            return f"{Colors.RED}No elasticity comparison data available for {title}{Colors.END}"
            
        chart = f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        # Elasticity estimates
        if 'elasticity_estimates' in results:
            estimates = results['elasticity_estimates']
            chart += f"{Colors.MAGENTA}Elasticity Estimates by Method:{Colors.END}\n"
            
            for method, elasticity in estimates.items():
                if elasticity < -1.5:
                    color = Colors.RED
                    symbol = "🔴"
                    sensitivity = "Very High"
                elif elasticity < -1.0:
                    color = Colors.YELLOW
                    symbol = "🟡"
                    sensitivity = "High"
                elif elasticity < -0.5:
                    color = Colors.GREEN
                    symbol = "🟢"
                    sensitivity = "Moderate"
                else:
                    color = Colors.BLUE
                    symbol = "🔵"
                    sensitivity = "Low"
                
                chart += f"  {color}{symbol} {method}: {elasticity:>6.3f} ({sensitivity}){Colors.END}\n"
        
        # Method comparison
        if 'method_comparison' in results:
            comparison = results['method_comparison']
            chart += f"\n{Colors.MAGENTA}Method Comparison:{Colors.END}\n"
            
            if 'best_method' in comparison:
                best = comparison['best_method']
                chart += f"  Best performing: {best}\n"
            
            if 'consensus' in comparison:
                consensus = comparison['consensus']
                if consensus:
                    chart += f"  {Colors.GREEN}✅ Methods agree on elasticity range{Colors.END}\n"
                else:
                    chart += f"  {Colors.RED}❌ Methods disagree - need more investigation{Colors.END}\n"
        
        chart += f"\n{Colors.BOLD}💡 Business Insights:{Colors.END}\n"
        chart += f"• High sensitivity (🔴): Use promotions to drive volume\n"
        chart += f"• Moderate sensitivity (🟡): Adjust prices carefully\n"
        chart += f"• Low sensitivity (🟢): You have pricing power\n"
        chart += f"• Method agreement: More confidence in estimates\n"
        
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        return chart
        
    def create_feature_importance_chart(self, results: Dict, title: str) -> str:
        """Create ASCII feature importance chart."""
        if not results:
            return f"{Colors.RED}No feature importance data available for {title}{Colors.END}"
            
        chart = f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*50}{Colors.END}\n"
        
        # Feature importance
        if 'feature_importance' in results:
            importance = results['feature_importance']
            chart += f"{Colors.MAGENTA}Most Important Features for Price Prediction:{Colors.END}\n"
            
            # Sort by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            for i, (feature, imp) in enumerate(sorted_features[:10]):  # Top 10
                # Create bar chart
                bar_length = int(imp * 30)  # Scale to 30 chars max
                bar = "█" * bar_length + "░" * (30 - bar_length)
                
                if i < 3:
                    color = Colors.RED
                    symbol = "🥇"
                elif i < 6:
                    color = Colors.YELLOW
                    symbol = "🥈"
                else:
                    color = Colors.GREEN
                    symbol = "🥉"
                
                chart += f"  {color}{symbol} {feature:<20} {bar} {imp:.3f}{Colors.END}\n"
        
        # Model insights
        if 'model_insights' in results:
            insights = results['model_insights']
            chart += f"\n{Colors.MAGENTA}Model Insights:{Colors.END}\n"
            
            if 'top_predictors' in insights:
                chart += f"  Top predictors: {', '.join(insights['top_predictors'])}\n"
            
            if 'feature_interactions' in insights:
                chart += f"  Key interactions: {insights['feature_interactions']}\n"
        
        chart += f"\n{Colors.BOLD}💡 Business Insights:{Colors.END}\n"
        chart += f"• Focus on top features for pricing decisions\n"
        chart += f"• Monitor feature changes over time\n"
        chart += f"• Use feature importance to guide data collection\n"
        chart += f"• Consider feature interactions in pricing strategy\n"
        
        chart += f"{Colors.YELLOW}{'='*50}{Colors.END}\n"
        return chart
        
    def create_heterogeneity_analysis_chart(self, results: Dict, title: str) -> str:
        """Create ASCII heterogeneity analysis chart."""
        if not results:
            return f"{Colors.RED}No heterogeneity data available for {title}{Colors.END}"
            
        chart = f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}\n"
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        
        # Heterogeneity by segments
        if 'heterogeneity_by_segments' in results:
            segments = results['heterogeneity_by_segments']
            chart += f"{Colors.MAGENTA}Elasticity by Customer Segments:{Colors.END}\n"
            
            for segment, elasticity in segments.items():
                if elasticity < -1.5:
                    color = Colors.RED
                    symbol = "🔴"
                    strategy = "Use promotions"
                elif elasticity < -1.0:
                    color = Colors.YELLOW
                    symbol = "🟡"
                    strategy = "Adjust carefully"
                elif elasticity < -0.5:
                    color = Colors.GREEN
                    symbol = "🟢"
                    strategy = "Premium pricing"
                else:
                    color = Colors.BLUE
                    symbol = "🔵"
                    strategy = "High pricing power"
                
                chart += f"  {color}{symbol} {segment}: {elasticity:>6.3f} - {strategy}{Colors.END}\n"
        
        # Heterogeneity statistics
        if 'heterogeneity_stats' in results:
            stats = results['heterogeneity_stats']
            chart += f"\n{Colors.MAGENTA}Heterogeneity Statistics:{Colors.END}\n"
            
            if 'range' in stats:
                chart += f"  Elasticity range: {stats['range']:.3f}\n"
            
            if 'coefficient_of_variation' in stats:
                cv = stats['coefficient_of_variation']
                chart += f"  Coefficient of variation: {cv:.3f}\n"
                
                if cv > 0.5:
                    chart += f"  {Colors.RED}❌ High heterogeneity - segment pricing recommended{Colors.END}\n"
                elif cv > 0.2:
                    chart += f"  {Colors.YELLOW}⚠️ Moderate heterogeneity - consider segmentation{Colors.END}\n"
                else:
                    chart += f"  {Colors.GREEN}✅ Low heterogeneity - uniform pricing OK{Colors.END}\n"
        
        chart += f"\n{Colors.BOLD}💡 Segmentation Strategy:{Colors.END}\n"
        chart += f"• High heterogeneity: Use different pricing by segment\n"
        chart += f"• Low heterogeneity: Uniform pricing across segments\n"
        chart += f"• Monitor heterogeneity changes over time\n"
        chart += f"• Use ML to identify new customer segments\n"
        
        chart += f"{Colors.YELLOW}{'='*60}{Colors.END}\n"
        return chart
        
    def run_xgboost_dml_analysis(self):
        """Run XGBoost DML analysis with visualizations."""
        self.console.print("\n[bold cyan]🚀 Running XGBoost DML Analysis...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Training XGBoost models with cross-fitting...", total=None)
            
            try:
                results = self.estimator.example_1_xgboost_dml()
                progress.update(task, description="✅ XGBoost DML analysis completed!")
                
                # Display results
                performance_chart = self.create_ml_performance_chart(results, "🚀 XGBOOST DML PERFORMANCE")
                self.console.print(performance_chart)
                
                elasticity_chart = self.create_elasticity_comparison_chart(results, "📊 ELASTICITY ESTIMATES")
                self.console.print(elasticity_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "XGBoost Double Machine Learning",
                    "Uses advanced gradient boosting to predict both demand and price, then combines these predictions to get accurate elasticity estimates. It handles complex non-linear relationships automatically.",
                    "Gives you highly accurate elasticity estimates by using AI to understand complex patterns in your data. It's especially good when you have lots of data and complex relationships.",
                    "The analysis might reveal: 'XGBoost found that customers are 15% more price-sensitive on weekends and 20% less sensitive during promotions. The elasticity is -1.3 overall, but varies by these factors.'"
                )
                
                self.results['xgboost'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ XGBoost DML analysis failed: {e}[/red]")
                
    def run_lightgbm_dml_analysis(self):
        """Run LightGBM DML analysis with visualizations."""
        self.console.print("\n[bold cyan]⚡ Running LightGBM DML Analysis...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Training LightGBM models with cross-fitting...", total=None)
            
            try:
                results = self.estimator.example_2_lightgbm_dml()
                progress.update(task, description="✅ LightGBM DML analysis completed!")
                
                # Display results
                performance_chart = self.create_ml_performance_chart(results, "⚡ LIGHTGBM DML PERFORMANCE")
                self.console.print(performance_chart)
                
                elasticity_chart = self.create_elasticity_comparison_chart(results, "📊 ELASTICITY ESTIMATES")
                self.console.print(elasticity_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "LightGBM Double Machine Learning",
                    "Uses fast gradient boosting to predict demand and price patterns. It's faster than XGBoost and often just as accurate, making it great for large datasets.",
                    "Provides fast, accurate elasticity estimates that are easy to update as new data comes in. Perfect for real-time pricing decisions and large-scale analysis.",
                    "The analysis might reveal: 'LightGBM processed 100,000 records in 2 minutes and found that elasticity is -1.2 with 95% confidence between -1.4 and -1.0. It also identified that store size is the most important predictor.'"
                )
                
                self.results['lightgbm'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ LightGBM DML analysis failed: {e}[/red]")
                
    def run_ensemble_dml_analysis(self):
        """Run ensemble DML analysis with visualizations."""
        self.console.print("\n[bold cyan]🎯 Running Ensemble DML Analysis...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Training ensemble models with cross-fitting...", total=None)
            
            try:
                results = self.estimator.example_3_ensemble_dml()
                progress.update(task, description="✅ Ensemble DML analysis completed!")
                
                # Display results
                performance_chart = self.create_ml_performance_chart(results, "🎯 ENSEMBLE DML PERFORMANCE")
                self.console.print(performance_chart)
                
                elasticity_chart = self.create_elasticity_comparison_chart(results, "📊 ELASTICITY ESTIMATES")
                self.console.print(elasticity_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "Ensemble Double Machine Learning",
                    "Combines multiple AI models (Random Forest, Gradient Boosting, XGBoost) to get the most robust elasticity estimates. It's like having multiple experts vote on the best answer.",
                    "Gives you the most reliable elasticity estimates by combining the strengths of different AI models. If one model makes a mistake, the others correct it.",
                    "The analysis might reveal: 'The ensemble combines 5 different AI models and gives elasticity of -1.25 with high confidence. Even if one model is wrong, the ensemble is still accurate because it uses the best parts of each model.'"
                )
                
                self.results['ensemble'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Ensemble DML analysis failed: {e}[/red]")
                
    def run_heterogeneity_analysis(self):
        """Run heterogeneity analysis with visualizations."""
        self.console.print("\n[bold cyan]👥 Running Heterogeneity Analysis...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Analyzing customer heterogeneity...", total=None)
            
            try:
                results = self.estimator.example_4_heterogeneity()
                progress.update(task, description="✅ Heterogeneity analysis completed!")
                
                # Display results
                heterogeneity_chart = self.create_heterogeneity_analysis_chart(results, "👥 CUSTOMER HETEROGENEITY ANALYSIS")
                self.console.print(heterogeneity_chart)
                
                # Method explanation
                self.print_method_explanation(
                    "ML-Based Heterogeneity Analysis",
                    "Uses machine learning to automatically discover which customer groups have different price sensitivities, without you having to guess or manually segment them.",
                    "Shows you exactly which customer groups to target with different pricing strategies. High-income customers might not care about price, while budget-conscious ones do.",
                    "The analysis might reveal: 'ML discovered 3 customer segments: Premium (elasticity -0.8), Standard (elasticity -1.2), and Budget (elasticity -1.8). You should price differently for each group.'"
                )
                
                self.results['heterogeneity'] = results
                
            except Exception as e:
                self.console.print(f"[red]❌ Heterogeneity analysis failed: {e}[/red]")
                
    def create_summary_dashboard(self):
        """Create a summary dashboard of all results."""
        if not self.results:
            self.console.print("[red]No results to display in summary dashboard.[/red]")
            return
            
        # Create summary table
        table = Table(title="🤖 ML DML ANALYSIS SUMMARY", box=box.ROUNDED)
        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Elasticity", style="magenta", justify="center")
        table.add_column("Performance", style="green")
        table.add_column("Best For", style="yellow")
        
        method_info = {
            "XGBoost": ("High accuracy", "Complex patterns"),
            "LightGBM": ("Fast training", "Large datasets"),
            "Ensemble": ("Most robust", "Critical decisions"),
            "Heterogeneity": ("Segmentation", "Targeted pricing")
        }
        
        for method_name, results in self.results.items():
            if isinstance(results, dict) and 'elasticity' in results and results['elasticity'] is not None:
                elasticity = results['elasticity']
                method_display = method_name.replace('_', ' ').title()
                performance, best_for = method_info.get(method_display, ("N/A", "N/A"))
                
                table.add_row(
                    method_display,
                    f"{elasticity:.3f}",
                    performance,
                    best_for
                )
        
        self.console.print(table)
        
        # Business recommendations
        recommendations = Panel(
            "[bold green]💡 IMMEDIATE ACTION ITEMS:[/bold green]\n\n"
            "[yellow]🎯 Model Selection:[/yellow]\n"
            "• Use XGBoost for highest accuracy on complex data\n"
            "• Use LightGBM for fast analysis of large datasets\n"
            "• Use Ensemble for most critical pricing decisions\n"
            "• Monitor model performance over time\n\n"
            "[blue]📊 Feature Engineering:[/blue]\n"
            "• Focus on top features identified by ML models\n"
            "• Collect more data on important predictors\n"
            "• Monitor feature importance changes over time\n"
            "• Consider feature interactions in pricing\n\n"
            "[magenta]👥 Customer Segmentation:[/magenta]\n"
            "• Use heterogeneity analysis to identify segments\n"
            "• Price differently for different customer groups\n"
            "• Monitor segment changes over time\n"
            "• Use ML to discover new customer segments\n\n"
            "[red]⚠️ Model Maintenance:[/red]\n"
            "• Retrain models regularly with new data\n"
            "• Monitor model performance and accuracy\n"
            "• Update models when business conditions change\n"
            "• Validate models with A/B testing",
            title="🎯 Your Next Steps",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(recommendations)
        
    def run_interactive_analysis(self):
        """Run interactive analysis with user choices."""
        self.print_banner()
        
        # Initialize estimator
        self.console.print("\n[bold blue]🚀 Initializing ML DML analysis...[/bold blue]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Loading data and models...", total=None)
            self.estimator = MLPipelineElasticityEstimator()
            progress.update(task, description="✅ Ready for analysis!")
        
        # Analysis menu
        while True:
            self.console.print("\n[bold cyan]📋 ML DML ANALYSIS MENU[/bold cyan]")
            self.console.print("1. 🚀 XGBoost DML (High Accuracy)")
            self.console.print("2. ⚡ LightGBM DML (Fast Training)")
            self.console.print("3. 🎯 Ensemble DML (Most Robust)")
            self.console.print("4. 👥 Heterogeneity Analysis (Segmentation)")
            self.console.print("5. 📊 View Summary Dashboard")
            self.console.print("6. 🚪 Exit")
            
            choice = input("\n[bold yellow]Select analysis (1-6): [/bold yellow]").strip()
            
            if choice == '1':
                self.run_xgboost_dml_analysis()
            elif choice == '2':
                self.run_lightgbm_dml_analysis()
            elif choice == '3':
                self.run_ensemble_dml_analysis()
            elif choice == '4':
                self.run_heterogeneity_analysis()
            elif choice == '5':
                self.create_summary_dashboard()
            elif choice == '6':
                self.console.print("\n[bold green]👋 Thank you for using the ML DML Analysis Tool![/bold green]")
                break
            else:
                self.console.print("[red]❌ Invalid choice. Please select 1-6.[/red]")
            
            input("\n[bold blue]Press Enter to continue...[/bold blue]")

def main():
    """Main function to run the CLI."""
    try:
        cli = SklearnCLI()
        cli.run_interactive_analysis()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Analysis interrupted by user. Goodbye!{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}[/red]")

if __name__ == "__main__":
    main()

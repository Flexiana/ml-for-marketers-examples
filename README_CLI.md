# 🎯 Elasticity Analysis CLI Tools

Interactive command-line interfaces for cross-price elasticity estimation with colorful ASCII visualizations and marketing-friendly explanations.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install all dependencies including CLI tools
pip install -r requirements.txt

# Or install just CLI dependencies
pip install -r requirements_cli.txt
```

### 2. Run Demo
```bash
# See a demo of the CLI tools
python demo_cli.py
```

### 3. Run Individual CLI Tools
```bash
# EconML methods (DML, IV, DR, Causal Forests, Cross-Price)
python example_econml_cli.py

# PyBLP structural demand estimation
python example_pyblp_cli.py

# All methods in one interface
python example_all_methods_cli.py
```

## 📊 Features

### 🎨 Visual Features
- **Colorful ASCII Charts**: Beautiful terminal-based visualizations
- **Interactive Menus**: Easy-to-use command-line interfaces
- **Progress Indicators**: Real-time analysis progress
- **Rich Formatting**: Professional-looking output with colors and symbols

### 📈 Analysis Types
- **Elasticity Charts**: Bar charts showing price sensitivity across methods
- **Cross-Price Matrices**: Heatmaps showing product substitution patterns
- **Heterogeneity Analysis**: Customer segmentation and preference analysis
- **Competition Analysis**: Market positioning and competitive dynamics
- **Method Comparison**: Side-by-side comparison of different approaches

### 💡 Business Insights
- **Marketing-Friendly Explanations**: Clear, non-technical descriptions
- **Strategic Recommendations**: Actionable business insights
- **Risk Assessment**: Confidence intervals and uncertainty quantification
- **Portfolio Optimization**: Cross-product pricing strategies

## 🔬 Available Methods

### 1. EconML CLI (`example_econml_cli.py`)
- **Double Machine Learning (DML)**: AI-powered causal inference
- **Instrumental Variables (IV)**: Natural experiment approach
- **Causal Forests**: Heterogeneous treatment effects
- **Doubly Robust Learners**: Robust estimation methods
- **Cross-Price Elasticity Matrix**: Product interaction analysis

### 2. PyBLP CLI (`example_pyblp_cli.py`)
- **Basic BLP**: Structural demand estimation
- **Random Coefficients**: Consumer heterogeneity modeling
- **Demographic Interactions**: Customer segmentation
- **Market Share Analysis**: Competitive positioning

### 3. Comprehensive CLI (`example_all_methods_cli.py`)
- **All Methods**: Unified interface for all estimation methods
- **Method Comparison**: Side-by-side analysis
- **Business Dashboard**: Comprehensive insights
- **Strategic Recommendations**: Actionable advice

## 🎯 Marketing Manager Benefits

### 📊 Visual Analytics
- **No Technical Knowledge Required**: Clear, intuitive interfaces
- **Real-Time Results**: Immediate insights and recommendations
- **Professional Presentations**: High-quality terminal output
- **Interactive Exploration**: Drill down into specific analyses

### 💼 Business Value
- **Pricing Strategy**: Optimize prices across product portfolio
- **Market Segmentation**: Identify customer groups with different sensitivities
- **Competitive Analysis**: Understand market dynamics and positioning
- **Risk Management**: Quantify uncertainty in pricing decisions

### 🚀 Strategic Insights
- **Cross-Price Effects**: Understand how products compete or complement
- **Customer Heterogeneity**: Tailor strategies to different segments
- **Method Validation**: Use multiple approaches for robust estimates
- **Dynamic Pricing**: Implement data-driven pricing strategies

## 📋 Usage Examples

### Basic Analysis
```bash
# Run EconML analysis
python example_econml_cli.py

# Select option 1 for Double Machine Learning
# View colorful elasticity charts and business insights
```

### Comprehensive Analysis
```bash
# Run all methods
python example_all_methods_cli.py

# Select option 7 to run all analyses
# Compare results across methods
# Get comprehensive business recommendations
```

### Demo Mode
```bash
# See sample visualizations
python demo_cli.py

# View example charts and insights
# Learn about different analysis types
```

## 🎨 Visual Examples

### Elasticity Chart
```
🎯 DOUBLE MACHINE LEARNING RESULTS
============================================================
Method                 ████████████████████████████████████████████████  -1.234
PyBLP Structural       ███████████████████████████████████████████████   -1.156
Panel Data            ██████████████████████████████████████████████    -0.987
AIDS Demand           ███████████████████████████████████████████████   -1.089
Bayesian              ████████████████████████████████████████████████  -1.201
ML DML                ███████████████████████████████████████████████   -1.167
============================================================
```

### Cross-Price Matrix
```
🔄 CROSS-PRICE ELASTICITY MATRIX
================================================================================
Product         Cola A    Cola B    Cola C    Pepsi    Sprite
--------------------------------------------------------------------------------
Cola A         🔴 -1.234   🔄  0.156   🔄  0.089   🔄  0.234   🔄  0.123
Cola B         🔄  0.145   🔴 -1.156   🔄  0.167   🔄  0.198   🔄  0.134
Cola C         🔄  0.098   🔄  0.123   🔴 -1.089   🔄  0.156   🔄  0.145
Pepsi          🔄  0.234   🔄  0.198   🔄  0.156   🔴 -1.201   🔄  0.167
Sprite         🔄  0.123   🔄  0.134   🔄  0.145   🔄  0.167   🔴 -1.167
================================================================================
```

## 🛠️ Technical Details

### Dependencies
- **Rich**: Terminal formatting and progress bars
- **Colorama**: Cross-platform color support
- **Click**: Command-line interface framework
- **All core dependencies**: From main requirements.txt

### System Requirements
- **Python 3.8+**: Required for all features
- **Terminal with Color Support**: For best visual experience
- **Minimum 4GB RAM**: For complex analyses
- **Modern Terminal**: iTerm2, Windows Terminal, or similar

### Performance
- **Fast Startup**: Optimized for quick analysis
- **Progress Indicators**: Real-time feedback
- **Memory Efficient**: Handles large datasets
- **Error Handling**: Graceful failure recovery

## 🎯 Business Use Cases

### 1. Pricing Strategy
- **Portfolio Optimization**: Set prices across product lines
- **Promotional Planning**: Identify best products for discounts
- **Competitive Response**: React to competitor price changes
- **Revenue Maximization**: Find optimal pricing points

### 2. Market Analysis
- **Customer Segmentation**: Identify price-sensitive groups
- **Product Positioning**: Understand competitive dynamics
- **Market Entry**: Assess pricing for new products
- **Brand Management**: Optimize brand portfolio pricing

### 3. Strategic Planning
- **Scenario Analysis**: Test different pricing strategies
- **Risk Assessment**: Quantify pricing uncertainty
- **Performance Monitoring**: Track elasticity changes over time
- **Decision Support**: Data-driven pricing decisions

## 🚀 Getting Started

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Run Demo**: `python demo_cli.py`
3. **Try Individual Tools**: `python example_econml_cli.py`
4. **Run Comprehensive Analysis**: `python example_all_methods_cli.py`

## 📞 Support

For questions or issues:
- Check the main README.md for technical details
- Review the individual example files for specific method documentation
- Run the demo to see expected output format

## 🎉 Enjoy!

These CLI tools make advanced econometric analysis accessible to marketing managers with beautiful visualizations and clear business insights!

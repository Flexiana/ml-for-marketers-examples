#!/bin/bash
# Activation script for the elasticity estimation environment

echo "🚀 Activating Elasticity Estimation Environment..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "elasticity_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv elasticity_env"
    echo "Then run: source elasticity_env/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate the environment
source elasticity_env/bin/activate

echo "✅ Environment activated successfully!"
echo ""
echo "Available commands:"
echo "  python test_predictions.py     # Test all working examples"
echo "  python example_statsmodels_aids.py  # Run AIDS example"
echo "  python example_sklearn_xgb_dml.py   # Run ML DML example"
echo "  python main_cross_price_elasticity.py  # Run all examples"
echo ""
echo "To deactivate: deactivate"
echo "=================================================="

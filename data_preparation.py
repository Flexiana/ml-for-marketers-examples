"""
Data preparation module for cross-price elasticity examples.

This module generates and prepares realistic retail scanner data with:
- Multiple products (with substitutes and complements)
- Multiple stores and markets  
- Time periods with seasonality
- Price variations (promotions, regular prices)
- Instrumental variables (cost shifters, competitor prices)
- Consumer demographics for heterogeneous effects

The data is suitable for demonstrating various econometric methods for
estimating own-price and cross-price elasticities.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class RetailDataGenerator:
    """Generate realistic retail scanner data for demand estimation."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed."""
        np.random.seed(seed)
        self.seed = seed
        
        # Product categories and relationships
        self.products = {
            'cola_brand_A': {'category': 'cola', 'brand': 'A', 'quality': 'premium'},
            'cola_brand_B': {'category': 'cola', 'brand': 'B', 'quality': 'regular'},
            'cola_brand_C': {'category': 'cola', 'brand': 'C', 'quality': 'value'},
            'chips_brand_A': {'category': 'chips', 'brand': 'A', 'quality': 'premium'},
            'chips_brand_B': {'category': 'chips', 'brand': 'B', 'quality': 'regular'},
            'chocolate_A': {'category': 'chocolate', 'brand': 'A', 'quality': 'premium'},
            'chocolate_B': {'category': 'chocolate', 'brand': 'B', 'quality': 'regular'},
            'water_brand_A': {'category': 'water', 'brand': 'A', 'quality': 'premium'},
        }
        
        # True elasticities for simulation (for validation)
        self.true_elasticities = {
            'own': -1.2,  # Own-price elasticity
            'within_category': 0.4,  # Cross-price within category (substitutes)
            'complement': -0.15,  # Cross-price for complements (cola-chips)
            'other': 0.02  # Near-zero for unrelated products
        }
        
    def generate_base_data(self, 
                          n_stores: int = 50,
                          n_markets: int = 10,
                          n_weeks: int = 104,
                          start_date: str = '2022-01-01') -> pd.DataFrame:
        """Generate base panel structure with stores, products, and time."""
        
        stores = []
        for store_id in range(1, n_stores + 1):
            market_id = (store_id - 1) // (n_stores // n_markets) + 1
            store_type = np.random.choice(['urban', 'suburban', 'rural'])
            store_size = np.random.choice(['small', 'medium', 'large'])
            
            stores.append({
                'store_id': store_id,
                'market_id': market_id,
                'store_type': store_type,
                'store_size': store_size,
                'income_level': np.random.uniform(30000, 120000),
                'population_density': np.random.uniform(100, 5000)
            })
        
        store_df = pd.DataFrame(stores)
        
        # Generate time periods
        start = pd.to_datetime(start_date)
        dates = [start + timedelta(weeks=i) for i in range(n_weeks)]
        
        # Create panel structure
        panel_data = []
        for store in stores:
            for date in dates:
                week_of_year = date.isocalendar()[1]
                month = date.month
                quarter = (month - 1) // 3 + 1
                
                for product_id, product_info in self.products.items():
                    panel_data.append({
                        'store_id': store['store_id'],
                        'market_id': store['market_id'],
                        'store_type': store['store_type'],
                        'store_size': store['store_size'],
                        'income_level': store['income_level'],
                        'population_density': store['population_density'],
                        'date': date,
                        'week': week_of_year,
                        'month': month,
                        'quarter': quarter,
                        'product_id': product_id,
                        'category': product_info['category'],
                        'brand': product_info['brand'],
                        'quality_tier': product_info['quality']
                    })
        
        return pd.DataFrame(panel_data)
    
    def generate_instruments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate instrumental variables for identification."""
        
        # Cost shifters (Hausman instruments)
        df['wholesale_cost'] = 0.0
        df['transportation_cost'] = 0.0
        
        for product in self.products.keys():
            base_cost = {'premium': 2.0, 'regular': 1.5, 'value': 1.0}
            quality = self.products[product]['quality']
            
            # Wholesale cost varies by time and product
            product_mask = df['product_id'] == product
            df.loc[product_mask, 'wholesale_cost'] = (
                base_cost[quality] * 
                (1 + 0.2 * np.sin(2 * np.pi * df.loc[product_mask, 'week'] / 52)) +
                np.random.normal(0, 0.1, product_mask.sum())
            )
            
            # Transportation cost varies by market
            for market in df['market_id'].unique():
                market_mask = (df['product_id'] == product) & (df['market_id'] == market)
                df.loc[market_mask, 'transportation_cost'] = np.random.uniform(0.1, 0.3)
        
        # BLP-style instruments: characteristics of other products
        df['avg_rival_price_same_market'] = 0.0
        df['num_rival_products'] = 0.0
        
        for idx, row in df.iterrows():
            same_market_time = (
                (df['market_id'] == row['market_id']) & 
                (df['date'] == row['date']) &
                (df['product_id'] != row['product_id'])
            )
            if same_market_time.any():
                df.loc[idx, 'num_rival_products'] = same_market_time.sum()
        
        return df
    
    def generate_prices_and_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate prices and quantities with realistic demand patterns."""
        
        # Initialize price and quantity columns
        df['price'] = 0.0
        df['quantity'] = 0.0
        df['promotion'] = 0
        
        # Price generation with endogeneity
        quality_base_prices = {'premium': 4.0, 'regular': 3.0, 'value': 2.0}
        
        for idx, row in df.iterrows():
            # Base price from quality tier
            base_price = quality_base_prices[row['quality_tier']]
            
            # Market-level pricing (correlated with unobserved demand)
            market_effect = np.random.normal(0, 0.2)
            
            # Store-level pricing based on income
            income_effect = 0.1 * (row['income_level'] - 75000) / 45000
            
            # Seasonal pricing
            seasonal_effect = 0.1 * np.sin(2 * np.pi * row['week'] / 52)
            
            # Promotions (20% chance)
            if np.random.random() < 0.2:
                df.loc[idx, 'promotion'] = 1
                promotion_discount = np.random.uniform(0.15, 0.35)
            else:
                promotion_discount = 0
            
            # Cost-based pricing (creates exogenous variation)
            cost_markup = 1.5 + 0.3 * row['wholesale_cost'] + 0.2 * row['transportation_cost']
            
            # Final price
            df.loc[idx, 'price'] = max(0.5, 
                base_price * cost_markup * (1 + market_effect + income_effect + seasonal_effect) * (1 - promotion_discount)
            )
        
        # Calculate cross-prices for demand generation
        for idx, row in df.iterrows():
            # Get prices of other products in same store-time
            same_store_time = (
                (df['store_id'] == row['store_id']) & 
                (df['date'] == row['date'])
            )
            
            other_products = df[same_store_time & (df['product_id'] != row['product_id'])]
            
            # Calculate weighted cross-price effects
            cross_price_effect = 0
            for _, other in other_products.iterrows():
                if row['category'] == other['category']:
                    # Same category - substitutes
                    cross_price_effect += self.true_elasticities['within_category'] * np.log(other['price'])
                elif (row['category'] == 'cola' and other['category'] == 'chips') or \
                     (row['category'] == 'chips' and other['category'] == 'cola'):
                    # Complements
                    cross_price_effect += self.true_elasticities['complement'] * np.log(other['price'])
                else:
                    # Unrelated
                    cross_price_effect += self.true_elasticities['other'] * np.log(other['price'])
            
            # Demand function
            base_demand = 100 * {'premium': 1.2, 'regular': 1.0, 'value': 0.8}[row['quality_tier']]
            
            # Add various effects
            store_size_mult = {'small': 0.7, 'medium': 1.0, 'large': 1.5}[row['store_size']]
            income_effect = 0.5 * row['income_level'] / 75000 if row['quality_tier'] == 'premium' else 1.0
            
            # Unobserved product quality (creates endogeneity)
            xi = np.random.normal(0, 0.3)
            
            # Log demand with elasticities
            log_demand = (
                np.log(base_demand) +
                self.true_elasticities['own'] * np.log(row['price']) +
                cross_price_effect +
                0.3 * row['promotion'] +
                0.1 * np.sin(2 * np.pi * row['week'] / 52) +  # Seasonality
                np.log(store_size_mult) +
                np.log(income_effect) +
                xi +
                np.random.normal(0, 0.2)  # Idiosyncratic shock
            )
            
            df.loc[idx, 'quantity'] = max(0, np.exp(log_demand))
        
        # Calculate market shares and revenue
        for (store, date), group in df.groupby(['store_id', 'date']):
            total_quantity = group['quantity'].sum()
            if total_quantity > 0:
                df.loc[group.index, 'market_share'] = group['quantity'] / total_quantity
            else:
                df.loc[group.index, 'market_share'] = 0
        
        df['revenue'] = df['price'] * df['quantity']
        df['log_price'] = np.log(df['price'] + 0.01)
        df['log_quantity'] = np.log(df['quantity'] + 1)
        
        return df
    
    def add_consumer_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add consumer demographic variables for heterogeneous effects."""
        
        # Store-level demographics (already have income_level and population_density)
        # Add additional demographics
        
        for store_id in df['store_id'].unique():
            store_mask = df['store_id'] == store_id
            
            # Age distribution
            df.loc[store_mask, 'avg_age'] = np.random.uniform(25, 55)
            df.loc[store_mask, 'pct_young'] = np.random.uniform(0.2, 0.5)
            df.loc[store_mask, 'pct_families'] = np.random.uniform(0.3, 0.7)
            
            # Education
            df.loc[store_mask, 'pct_college'] = np.random.uniform(0.2, 0.7)
            
            # Urban/rural already captured in store_type
        
        return df
    
    def create_panel_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create panel data structure with appropriate indices."""
        
        # Sort and set multi-index for panel operations
        df = df.sort_values(['store_id', 'product_id', 'date'])
        
        # Create lagged variables for dynamic models
        df['lag_price'] = df.groupby(['store_id', 'product_id'])['price'].shift(1)
        df['lag_quantity'] = df.groupby(['store_id', 'product_id'])['quantity'].shift(1)
        df['lag_promotion'] = df.groupby(['store_id', 'product_id'])['promotion'].shift(1)
        
        # Create product-specific competitor price indices
        for category in df['category'].unique():
            df[f'avg_competitor_price_{category}'] = 0.0
            
            for (store, date), group in df.groupby(['store_id', 'date']):
                cat_products = group[group['category'] == category]
                for idx, row in cat_products.iterrows():
                    other_prices = cat_products[cat_products['product_id'] != row['product_id']]['price']
                    if len(other_prices) > 0:
                        df.loc[idx, f'avg_competitor_price_{category}'] = other_prices.mean()
        
        # Create time trends
        df['time_trend'] = df.groupby(['store_id', 'product_id']).cumcount() + 1
        df['time_trend_sq'] = df['time_trend'] ** 2
        
        return df
    
    def generate_complete_dataset(self,
                                 n_stores: int = 50,
                                 n_markets: int = 10,
                                 n_weeks: int = 104) -> pd.DataFrame:
        """Generate complete dataset with all features."""
        
        print("Generating retail scanner data...")
        print(f"  - {n_stores} stores across {n_markets} markets")
        print(f"  - {len(self.products)} products")
        print(f"  - {n_weeks} weeks of data")
        
        # Generate base panel
        df = self.generate_base_data(n_stores, n_markets, n_weeks)
        print(f"  - Base panel: {len(df):,} observations")
        
        # Add instruments
        df = self.generate_instruments(df)
        print("  - Added instrumental variables")
        
        # Generate prices and demand
        df = self.generate_prices_and_demand(df)
        print("  - Generated prices and quantities")
        
        # Add demographics
        df = self.add_consumer_demographics(df)
        print("  - Added consumer demographics")
        
        # Create panel structure
        df = self.create_panel_structure(df)
        print("  - Created panel structure with lags")
        
        # Add unique identifier
        df['obs_id'] = range(len(df))
        
        print(f"\nFinal dataset: {len(df):,} observations")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Products: {', '.join(df['product_id'].unique())}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 50)
        summary_vars = ['price', 'quantity', 'market_share', 'revenue']
        for var in summary_vars:
            print(f"{var:15s}: mean={df[var].mean():8.2f}, std={df[var].std():8.2f}")
        
        return df


def prepare_blp_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data specifically for BLP estimation."""
    
    # Create market definitions (market-time)
    df['market_ids'] = df['market_id'].astype(str) + '_' + df['date'].astype(str)
    
    # Product characteristics for random coefficients
    df['constant'] = 1
    quality_dummies = pd.get_dummies(df['quality_tier'], prefix='quality')
    category_dummies = pd.get_dummies(df['category'], prefix='cat')
    
    df = pd.concat([df, quality_dummies, category_dummies], axis=1)
    
    # Create outside option share
    df['inside_share'] = df.groupby('market_ids')['market_share'].transform('sum')
    df['outside_share'] = 1 - df['inside_share']
    
    # Log share ratio for linearized estimation
    df['log_share_ratio'] = np.log(df['market_share'] / df['outside_share'].clip(0.001))
    
    return df


def prepare_aids_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for AIDS/QUAIDS demand system estimation."""
    
    # Calculate expenditure shares by category
    expenditure_data = []
    
    for (store, date), group in df.groupby(['store_id', 'date']):
        total_exp = group['revenue'].sum()
        
        for category in group['category'].unique():
            cat_data = group[group['category'] == category]
            cat_exp = cat_data['revenue'].sum()
            
            expenditure_data.append({
                'store_id': store,
                'date': date,
                'category': category,
                'expenditure': cat_exp,
                'share': cat_exp / total_exp if total_exp > 0 else 0,
                'avg_price': cat_data['price'].mean(),
                'total_expenditure': total_exp,
                'income_level': group['income_level'].iloc[0],
                'store_type': group['store_type'].iloc[0]
            })
    
    aids_df = pd.DataFrame(expenditure_data)
    
    # Create price index
    aids_df['log_price'] = np.log(aids_df['avg_price'])
    aids_df['log_expenditure'] = np.log(aids_df['total_expenditure'] + 1)
    
    # Pivot for wide format needed by some implementations
    aids_wide = aids_df.pivot_table(
        index=['store_id', 'date', 'total_expenditure', 'income_level'],
        columns='category',
        values=['share', 'avg_price', 'log_price']
    )
    
    aids_wide.columns = ['_'.join(col).strip() for col in aids_wide.columns.values]
    aids_wide = aids_wide.reset_index()
    
    return aids_df, aids_wide


def save_datasets(df: pd.DataFrame, output_dir: str = 'data'):
    """Save datasets in various formats for different libraries."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main dataset
    df.to_csv(os.path.join(output_dir, 'retail_scanner_data.csv'), index=False)
    df.to_pickle(os.path.join(output_dir, 'retail_scanner_data.pkl'))
    
    # Save BLP-ready data
    blp_df = prepare_blp_data(df.copy())
    blp_df.to_csv(os.path.join(output_dir, 'blp_data.csv'), index=False)
    
    # Save AIDS data
    aids_long, aids_wide = prepare_aids_data(df.copy())
    aids_long.to_csv(os.path.join(output_dir, 'aids_data_long.csv'), index=False)
    aids_wide.to_csv(os.path.join(output_dir, 'aids_data_wide.csv'), index=False)
    
    # Save a smaller sample for quick testing
    sample_df = df[df['market_id'].isin([1, 2, 3])].copy()
    sample_df.to_csv(os.path.join(output_dir, 'retail_scanner_sample.csv'), index=False)
    
    print(f"\nDatasets saved to '{output_dir}/' directory:")
    print("  - retail_scanner_data.csv/pkl: Main dataset")
    print("  - blp_data.csv: BLP-ready format")
    print("  - aids_data_long/wide.csv: AIDS demand system format")
    print("  - retail_scanner_sample.csv: Small sample for testing")


def main():
    """Generate and save all datasets."""
    
    # Initialize generator
    generator = RetailDataGenerator(seed=42)
    
    # Generate complete dataset
    df = generator.generate_complete_dataset(
        n_stores=50,
        n_markets=10,
        n_weeks=104
    )
    
    # Save datasets
    save_datasets(df)
    
    # Print true elasticities for reference
    print("\nTrue elasticities used in data generation:")
    print("-" * 50)
    for key, value in generator.true_elasticities.items():
        print(f"{key:20s}: {value:6.3f}")
    
    return df


if __name__ == "__main__":
    df = main()

# Technical Interview Task: Elasticity Estimation Web Application

## Project Overview

You will build a web application that exposes cross-price elasticity estimation functionality from our [ml-for-marketers-examples](https://github.com/Flexiana/ml-for-marketers-examples) repository. The application should allow users to upload retail data, estimate price elasticities using machine learning methods, and visualize results through an intuitive interface.

## Repository Context

Our repository contains state-of-the-art methods for cross-price elasticity estimation using:
- **EconML**: Causal machine learning (Double ML, IV methods)
- **PyBLP**: Structural demand estimation (BLP models)
- **LinearModels**: Panel data methods
- **Statsmodels**: AIDS/QUAIDS demand systems
- **PyMC**: Bayesian hierarchical models
- **Scikit-learn/XGBoost**: ML-based DML pipelines

## Task Requirements

### 1. Backend Development (Python)

**Extract and Refactor:**
- Choose **one method** from the examples (recommended: Statsmodels AIDS or LinearModels Panel)
- Extract the core estimation logic into a clean, reusable module
- Create a trained model that can be serialized/deserialized
- Design a REST API using **FastAPI** or **Flask**

**API Endpoints Required:**
```
POST /api/upload-data
- Upload CSV file with retail data
- Validate data format and structure
- Return data summary and validation results

POST /api/estimate-elasticities
- Accept data parameters (product categories, time periods)
- Run elasticity estimation
- Return elasticity matrix with confidence intervals

GET /api/elasticity-matrix/{estimation_id}
- Retrieve previously computed elasticity matrix
- Include metadata (method used, parameters, timestamp)

GET /api/health
- Health check endpoint
```

**Data Format:**
The API should accept CSV files with columns:
```csv
date,store_id,product_id,product_category,price,quantity,market_share,income_level,promotion
2023-01-01,store_1,prod_a,food,2.50,100,0.15,high,0
2023-01-01,store_1,prod_b,food,3.20,80,0.12,high,1
...
```

### 2. Frontend Development (HTML/CSS/JavaScript)

**Create a responsive web interface with:**

**Main Dashboard:**
- File upload area for CSV data
- Data preview table (first 10 rows)
- Parameter configuration panel
- Results visualization area

**Key Features:**
- **Data Upload**: Drag-and-drop CSV upload with validation
- **Parameter Selection**: 
  - Choose product categories
  - Select time period range
  - Set confidence level (90%, 95%, 99%)
- **Results Display**:
  - Elasticity matrix heatmap
  - Confidence intervals table
  - Download results as CSV/JSON
- **Visualizations**:
  - Price vs Quantity scatter plots
  - Elasticity bar charts
  - Cross-price elasticity network diagram

**UI Requirements:**
- Clean, professional design
- Mobile-responsive layout
- Loading states and error handling
- Real-time validation feedback

### 3. Technical Implementation

**Backend Stack:**
- **Framework**: Any Python framework, for example FastAPI
- **Data Processing**: Pandas, NumPy
- **ML Libraries**: Scikit-learn, Statsmodels, or chosen method
- **Model Persistence**: Pickle or Joblib
- **API Documentation**: OpenAPI/Swagger

**Frontend Stack:**
- **Core**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: CSS Grid/Flexbox or Bootstrap
- **Charts**: Chart.js, D3.js, or Plotly.js
- **HTTP Client**: Fetch API or Axios

**Optional Enhancements:**
- **Database**: SQLite for storing estimation results
- **Authentication**: Basic user management
- **Caching**: Redis for model caching
- **Docker**: Containerization

### 4. Deliverables

**Code Structure:**
```
elasticity-app/
├── backend/
│   ├── app.py                 # Main FastAPI application
│   ├── models/
│   │   ├── elasticity_estimator.py
│   │   └── data_validator.py
│   ├── api/
│   │   ├── endpoints.py
│   │   └── schemas.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── styles/
│   │   └── main.css
│   ├── scripts/
│   │   ├── app.js
│   │   └── visualizations.js
│   └── assets/
└── README.md
```


### 5. Reference Implementation

**Start with this example:**
```python
# From example_statsmodels_aids.py
from statsmodels.api import OLS, add_constant

def estimate_aids_elasticities(data):
    """Simple AIDS elasticity estimation"""
    # Budget share equation: w_i = α_i + Σ_j γ_ij log(p_j) + β_i log(x/P)
    # Price elasticity: ε_ij = -δ_ij + (γ_ij - β_i*w_j)/w_i
    
    # Your implementation here
    pass
```

**Expected Output:**
```json
{
  "elasticity_matrix": {
    "food": {"food": -1.2, "clothing": 0.3, "transport": 0.1},
    "clothing": {"food": 0.2, "clothing": -0.8, "transport": 0.1},
    "transport": {"food": 0.1, "clothing": 0.1, "transport": -1.5}
  },
  "confidence_intervals": {
    "food": {"food": [-1.4, -1.0], "clothing": [0.1, 0.5]},
    // ... more intervals
  },
  "method": "AIDS",
  "timestamp": "2024-01-15T10:30:00Z"
}
```


### 6. Getting Started

1. **Fork the repository**: [ml-for-marketers-examples](https://github.com/Flexiana/ml-for-marketers-examples)
2. **Study the examples**: Focus on `example_statsmodels_aids.py` or `example_linearmodels.py`
3. **Run the examples**: Ensure you understand the data flow and outputs
4. **Extract core logic**: Create your estimation module
5. **Build the API**: Implement the REST endpoints
6. **Create the frontend**: Build the user interface
7. **Test thoroughly**: Use the provided sample data


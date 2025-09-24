# Technical Interview Task: Elasticity Estimation Web Application

## Project Overview

You will build a web application that exposes cross-price elasticity estimation functionality from our [ml-for-marketers-examples](https://github.com/Flexiana/ml-for-marketers-examples) repository. The application should allow users to upload retail data, estimate price elasticities using machine learning methods, and visualize results through an intuitive interface.

## What You're Building: A Quick Introduction

**What is Price Elasticity?** Price elasticity measures how much consumer demand changes when prices change. For example, if the price of coffee increases by 10% and demand drops by 15%, the price elasticity is -1.5. Cross-price elasticity measures how demand for one product (like tea) changes when the price of another product (like coffee) changes. This is crucial for businesses to understand substitution patterns, set optimal pricing strategies, and predict market responses to price changes.

**What is the AIDS Model?** The Almost Ideal Demand System (AIDS) is an econometric model that estimates complete demand relationships between multiple products simultaneously. Instead of analyzing products one-by-one, AIDS models the entire market as a system where all products interact with each other. It provides three types of elasticities: Marshallian (how demand changes with price, holding income constant), Hicksian (how demand changes with price, holding utility constant), and expenditure elasticities (how demand changes with income). The model also checks theoretical restrictions like "adding-up" (all expenditure shares must sum to 100%) and "homogeneity" (doubling all prices should halve real demand).

**Why This Matters for Business** Understanding price elasticities helps companies make data-driven decisions about pricing, product positioning, and market strategy. For example, if two products have high positive cross-price elasticity, they're substitutes - raising the price of one will increase demand for the other. If they have negative cross-price elasticity, they're complements - raising the price of one will decrease demand for both. This web application makes these complex econometric analyses accessible to business users through an intuitive interface.

**What Happens in Example 1 (Linear AIDS)?** The Linear Approximate AIDS model works in several key steps that you'll implement in your backend:

1. **Data Preparation**: The model takes retail data with expenditure shares (what percentage of total spending goes to each product), prices, and total expenditure across different stores and time periods. It calculates log prices and creates matrices for all products.

2. **Stone's Price Index**: Instead of using complex price indices, it uses Stone's approximation: log(P) = Σ(share_i × log(price_i)) for all products. This creates a "basket price" that represents the overall cost of living.

3. **Real Expenditure**: It calculates log(real_expenditure) = log(total_expenditure) - log(price_index), which represents purchasing power after accounting for price changes.

4. **Share Equations**: For each product (except one), it runs a regression: share_i = α_i + Σ(γ_ij × log(price_j)) + β_i × log(real_expenditure). The α_i is the base share, γ_ij captures price effects, and β_i captures income effects.

5. **Adding-Up Constraint**: The last product's parameters are calculated so all shares sum to 100%: α_last = 1 - Σ(α_i), γ_last = -Σ(γ_i), β_last = -Σ(β_i).

6. **Elasticity Calculation**: Finally, it converts the estimated parameters into elasticities using formulas that account for the relationships between all products in the system.

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
- Upload CSV file with retail data (you can use data/aids_data_wide.csv)
- Validate data format and structure
- Return data summary and validation results

POST /api/estimate-elasticities
- Run AIDS estimation (look at example_statsmodels_aids.py Example 1)
- Return elasticity matrices (Marshallian, Hicksian, Expenditure)
- Return parameter estimates (alpha, beta, gamma coefficients)
- Return theoretical restrictions check results
- Store the results in the database

GET /api/elasticity-matrix/{estimation_id}
- Retrieve previously computed elasticity matrices and parameters
- Return complete AIDS estimation results

GET /api/health
- Health check endpoint
```

**Data Format:**
The API should accept CSV files with columns:
```csv
store_id,date,total_expenditure,income_level,avg_price_chips,avg_price_chocolate,avg_price_cola,avg_price_water,log_price_chips,log_price_chocolate,log_price_cola,log_price_water,share_chips,share_chocolate,share_cola,share_water
1,2022-01-01,751.4026124878749,46509.13108795474,7.479709045629097,6.490351547527698,5.929525202990294,6.21632208358152,2.012193893586684,1.870316696818742,1.7799441428547444,1.827178426970975,0.12176095586272351,0.39997075846212576,0.4322060325153082,0.04606225315984255
1,2022-01-08,542.5510055258694,46509.13108795474,7.736228270029356,6.3344008083224494,5.846668546329975,7.2031789127232795,2.0459142652168647,1.8459952249780895,1.7658720197690299,1.974522444238663,0.22528088432873486,0.38811380587762023,0.3301913922022653,0.05641391759137943
1,2022-01-15,720.1168707550686,46509.13108795474,6.914679922595452,6.062693802918846,5.084876008875473,8.288450526980478,1.933646676640575,1.8021542232444208,1.626270645501264,2.11486304299008,0.09645993932288419,0.4510118778117115,0.42290539053932297,0.02962279232608132
1,2022-01-22,591.8638411847196,46509.13108795474,6.138360275355659,7.383856872041978,4.336197566472589,4.885313983085205,1.8145576503622123,1.999296113486412,1.4669978273227104,1.5862335583698772,0.20559339635904592,0.41254214644111087,0.31594228584161305,0.06592217135823004
1,2022-01-29,619.6074126962969,46509.13108795474,5.523147985285121,6.1249872890241015,5.083346768494208,5.9829576842562044,1.7089479849263962,1.8123766811672661,1.6259698573642856,1.788915041719705,0.2028687876615671,0.22961736750108072,0.5479817716100804,0.019532073227271685
1,2022-02-05,788.2755044102296,46509.13108795474,8.950048891271882,7.292128193383766,5.991844083771177,5.528980361108945,2.1916589949837855,1.9867954366651492,1.7903992251439262,1.710003415370639,0.09856686493747058,0.3175736088410996,0.5550040943619337,0.0288554318594963
1,2022-02-12,759.901605502152,46509.13108795474,5.633725400716508,6.853916772387019,5.925056574159232,6.205505689237187,1.7287709285994333,1.924820280452957,1.7791902353569142,1.8254369123926244,0.18606828111637833,0.3196201047566323,0.4716646472349451,0.022646966892044474
1,2022-02-19,656.0255174441118,46509.13108795474,8.086836681425043,8.709019864424146,6.971858258982411,11.94563280307777,2.0902376387106796,2.1643592546155848,1.9418817974141702,2.4803657557728793,0.14733160261108375,0.28570039542704007,0.531813539579686,0.035154462382190055
...
```

### 2. Frontend Development (HTML/CSS/JavaScript)

**Create a responsive web interface with:**

**Main Dashboard:**
- File upload area for CSV data
- Data preview table (first 10 rows)
- Results visualization area

**Key Features:**
- **Data Upload**: Drag-and-drop CSV upload with validation
- **Results Display**:
  - Elasticity matrix heatmap (Marshallian, Hicksian, Expenditure)
  - Parameter estimates table (alpha, beta, gamma coefficients)
  - Theoretical restrictions check (adding-up, homogeneity, symmetry)
  - Download results as CSV/JSON
- **Visualizations**:
  - Expenditure shares pie chart
  - Elasticity matrix heatmap with color coding
  - Expenditure elasticities bar chart
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



### 5. Getting Started

1. **Fork the repository**: [ml-for-marketers-examples](https://github.com/Flexiana/ml-for-marketers-examples)
2. **Study the examples**: Focus on `example_statsmodels_aids.py`
3. **Run the examples**: Ensure you understand the data flow and outputs
4. **Extract core logic**: Create your estimation module
5. **Build the API**: Implement the REST endpoints
6. **Create the frontend**: Build the user interface
7. **Test thoroughly**: Use the provided sample data


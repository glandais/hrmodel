# HR Model - Heart Rate Prediction from Cycling Data

This project processes GPX cycling data files to predict heart rate based on cadence, power, and elevation using advanced machine learning models. The system achieves **research-level performance** with 2.1 bpm mean absolute error using Random Forest regression.

## Project Structure

```
hrmodel/
├── gpx/                     # Original GPX files
├── src/
│   ├── data/
│   │   ├── gpx_parser.py    # GPX file parsing
│   │   └── feature_engineering.py  # Advanced feature engineering (58 features)
│   ├── models/
│   │   ├── base_model.py    # Base model interface
│   │   ├── ols_model.py     # OLS regression model
│   │   ├── ridge_model.py   # Ridge regression with L2 regularization
│   │   ├── elastic_net_model.py  # Elastic Net (L1+L2 regularization)
│   │   ├── random_forest_model.py # Random Forest (best performer)
│   │   └── xgboost_model.py # XGBoost (2nd best performer)
│   └── utils/
│       ├── metrics.py       # Comprehensive evaluation metrics
│       └── visualization.py # Advanced plotting utilities
├── output/                  # All generated files
│   ├── processed_data/      # Feature-engineered parquet files
│   ├── model_comparison.csv # Performance comparison report
│   ├── pipeline.log        # Execution log
│   └── {model_name}/       # Model-specific directories (e.g., ols/)
│       ├── {model}_model.joblib  # Trained model
│       ├── predictions/     # Model predictions with predicted_hr
│       └── plots/          # Visualizations (scatter, time-series)
├── config.yaml             # Pipeline configuration
├── main.py                 # Complete pipeline runner
├── legacy_scripts/         # Old individual scripts
└── view_plots.py           # Plot viewer utility
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Complete Pipeline (Recommended)
```bash
# Run the complete pipeline from GPX to trained models
python main.py

# With custom configuration
python main.py --config my_config.yaml

# Verbose output
python main.py --verbose
```

### Individual Utilities
```bash
# 1. View parquet files
python print_parquet.py output/processed_data/hr1.parquet

# 2. View model predictions
python print_parquet.py output/ols/predictions/hr1.parquet

# 3. View generated plots
python view_plots.py

# 4. Check model comparison
cat output/model_comparison.csv
```

## Metrics Explanation

### MAE (Mean Absolute Error)
**MAE** is the average of the absolute differences between predicted and actual values. It represents the average magnitude of errors in predictions, without considering their direction.

**Formula:** MAE = (1/n) × Σ|actual - predicted|

**Example:** If MAE = 5 bpm, it means that on average, the predictions are off by 5 beats per minute.

**Advantages:**
- Easy to interpret (same units as the target variable)
- Treats all errors equally
- Less sensitive to outliers than RMSE

### RMSE (Root Mean Square Error)
**RMSE** is the square root of the average of squared differences between predicted and actual values. It represents the standard deviation of the prediction errors.

**Formula:** RMSE = √[(1/n) × Σ(actual - predicted)²]

**Example:** If RMSE = 7 bpm, it indicates the typical size of prediction errors, with larger errors having more influence.

**Advantages:**
- Penalizes larger errors more heavily
- Same units as the target variable
- Useful when large errors are particularly undesirable

### Comparison
- **MAE** gives equal weight to all errors
- **RMSE** gives more weight to large errors
- RMSE ≥ MAE (always)
- If RMSE >> MAE, it indicates presence of large outlier errors

### Other Metrics

#### R² (R-squared)
Proportion of variance in the target variable explained by the model. Ranges from 0 to 1, where 1 means perfect prediction.

#### MAPE (Mean Absolute Percentage Error)
Average of absolute percentage errors. Useful for comparing accuracy across different scales.

#### Bias
Average prediction error (predicted - actual). Positive bias means the model tends to overestimate; negative bias means underestimation.

## Model Performance

Current models achieve exceptional performance on 72,727 data points from 7 cycling sessions:

### Performance Comparison
| Model | MAE (bpm) | RMSE (bpm) | R² | Status |
|-------|-----------|------------|-----|--------|
| **Random Forest** | **2.17** | **3.01** | **0.9476** | 🏆 Best |
| **XGBoost** | **3.12** | **4.10** | **0.9032** | 🥈 2nd place |
| OLS | 4.95 | 6.64 | 0.7456 | Baseline |
| Ridge | 4.95 | 6.64 | 0.7456 | Same as OLS |
| Elastic Net | 5.73 | 7.41 | 0.6834 | Feature selection |

### Breakthrough Results (No Data Leakage)
- **🏆 Random Forest**: 56% MAE improvement vs OLS (4.95 → 2.17 bpm)
- **🥈 XGBoost**: 37% MAE improvement vs OLS (4.95 → 3.12 bpm)  
- **🚀 Tree-based dominance**: Both achieve >90% R² (vs 75% for linear models)
- **🎯 Research-level accuracy**: Sub-3 bpm MAE with **clean, deployable features**
- **✅ No data leakage**: Uses only power/cadence/elevation (no HR history)

### Individual File Performance (Random Forest - Clean Features)
| File | MAE (bpm) | RMSE (bpm) | Points |
|------|-----------|------------|---------|
| hr7 | 1.86 | 2.61 | 5,167 |
| hr2 | 2.06 | 2.73 | 24,556 |
| hr6 | 2.11 | 3.05 | 4,957 |
| hr1 | 2.11 | 2.97 | 20,801 |
| hr5 | 2.14 | 2.99 | 5,092 |
| hr3 | 2.37 | 3.47 | 4,507 |
| hr4 | 2.86 | 3.85 | 7,647 |

## Advanced Feature Engineering

The system generates **58 engineered features** from 4 raw GPX features (power, cadence, elevation, HR):

### Core Features (21 selected for training - No Data Leakage)
- **Power features:** `power`, `power5`, `power10`, `power30`, `power60`
- **Cadence features:** `cad`, `cad5`, `cad10`, `cad30`, `cad60`  
- **Power lag features:** `power_lag_1/5/10/15/30` (effort history)
- **Cadence lag features:** `cad_lag_1/5/10/15/30` (effort history)
- **Elevation:** `ele`

### Advanced Features (Generated but excluded from training)
- **Gradient features:** `gradient`, `gradient_abs`, `gradient_ma10/30` (terrain difficulty)
- **Power intensity:** `power_zone`, `power_normalized`, `power_above_75pct/90pct`
- **Power variability:** `power_cv10/30/60` (coefficient of variation)
- **Cadence variability:** `cad_std10/30/60`, `cad_optimal`, `cad_too_low/high`
- **Efficiency:** `power_per_cad` (power per RPM)
- **Fatigue indicators:** `cumulative_work_5min/10min/30min`, `intense_work_5min/10min`
- **Interaction features:** Power-cadence combinations
- **⚠️ HR features:** Excluded to prevent data leakage (can't use HR to predict HR)

### Key Insights (Data Leakage Fixed)
- **Power/cadence lags** capture physiological delay in HR response to effort
- **Moving averages** smooth out noise and capture sustained effort patterns
- **No HR features used** - model predicts from effort data only (deployable)
- **Tree-based models** handle non-linear power-HR relationships automatically

## Requirements

See `requirements.txt` for full dependencies:
- **gpxpy**: GPX file parsing
- **pandas**: Data manipulation and time-series processing
- **numpy**: Numerical operations  
- **scikit-learn**: Machine learning models (OLS, Ridge, Elastic Net, Random Forest)
- **xgboost**: Advanced gradient boosting (2nd best performer)
- **pyarrow**: Parquet file support
- **joblib**: Model persistence
- **matplotlib**: Visualization and plotting
- **seaborn**: Statistical visualizations

## Next Steps & Future Improvements

### Immediate (High ROI)
- ✅ **XGBoost implementation** - Achieved 3.12 bpm MAE (37% improvement vs linear)
- **Hyperparameter tuning** - Optimize Random Forest and XGBoost parameters
- **Ensemble methods** - Combine Random Forest + XGBoost (target: <2.0 bpm MAE)

### Advanced Research 
- **LSTM/GRU models** - For temporal pattern recognition
- **Individual athlete calibration** - Person-specific models
- **Real-time features** - Training load, recovery metrics
- **Polynomial features** - Capture power-HR non-linearity

### Target Performance (Clean Features)
- **Current best**: 2.17 bpm MAE (Random Forest, no data leakage)
- **Second best**: 3.12 bpm MAE (XGBoost, solid alternative)
- **Research target**: < 2.0 bpm MAE (ensemble Random Forest + XGBoost)
- **Clinical significance**: Already achieved with deployable features
- **✅ Production-ready**: Models use only available sensor data (power/cadence/elevation)
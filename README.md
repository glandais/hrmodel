# HR Model - Heart Rate Prediction from Cycling Data

This project processes GPX cycling data files to predict heart rate based on cadence, power, and elevation using machine learning models.

## Project Structure

```
hrmodel/
├── gpx/                     # Original GPX files
├── src/
│   ├── data/
│   │   ├── gpx_parser.py    # GPX file parsing
│   │   └── feature_engineering.py  # Feature engineering
│   ├── models/
│   │   ├── base_model.py    # Base model interface
│   │   └── ols_model.py     # OLS regression model
│   └── utils/
│       ├── metrics.py       # Evaluation metrics
│       └── visualization.py # Plotting utilities
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

The OLS regression model achieves:
- **R² Score:** ~0.74 (explains 74% of heart rate variance)
- **MAE:** ~5 bpm (average error of 5 beats per minute)
- **RMSE:** ~7 bpm (standard deviation of errors)

These metrics indicate good predictive performance for heart rate estimation based on cycling power, cadence, and elevation data.

## Features Used

The model uses the following features:
- **Cadence:** Pedaling rate and its moving averages (5s, 10s, 30s, 60s)
- **Power:** Cycling power output and its moving averages (5s, 10s, 30s, 60s)
- **Elevation:** Terrain elevation

Moving averages help capture the delayed physiological response of heart rate to changes in effort.

## Requirements

See `requirements.txt` for full dependencies:
- gpxpy: GPX file parsing
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Machine learning models
- pyarrow: Parquet file support
- joblib: Model persistence
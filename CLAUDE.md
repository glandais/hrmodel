# HR Model Project - Comprehensive Regression Analysis

## Project Overview

This project aims to predict heart rate (HR) during cycling activities using various physiological and environmental features extracted from GPX files. The core objective is to compare multiple regression algorithms to find the most accurate model for HR prediction based on:

- **Cadence** (pedaling rate)
- **Power** (cycling power output)
- **Elevation** (terrain difficulty)
- **Moving averages** of these features (5s, 10s, 30s, 60s windows)

## Scientific Rationale

Heart rate response to exercise is a complex physiological process influenced by:
1. **Immediate effort** (current power/cadence)
2. **Delayed response** (HR lags behind effort changes by 15-30 seconds)
3. **Cumulative fatigue** (recent effort history affects current HR)
4. **Individual fitness** (same effort produces different HR for different athletes)
5. **Environmental factors** (elevation/altitude affects oxygen availability)

## Current Implementation

### Data Pipeline
1. **GPX → Parquet**: Extract time-series data from GPS cycling files
2. **Feature Engineering**: Calculate moving averages (5s, 10s, 30s, 60s)
3. **Model Training**: Train regression models on combined dataset
4. **Prediction & Evaluation**: Apply models and measure performance (MAE, RMSE, R²)

### Baseline Model
- **OLS (Ordinary Least Squares)**: Linear regression baseline
  - Current performance: R² = 0.74, MAE = 5 bpm, RMSE = 7 bpm
  - Serves as reference for comparing more complex models

## Regression Algorithms to Test

### Linear Models
1. **Ridge Regression** - L2 regularization to prevent overfitting
2. **Lasso Regression** - L1 regularization for feature selection
3. **Elastic Net** - Combined L1+L2 regularization
4. **Polynomial Regression** - Capture non-linear relationships
5. **Huber Regression** - Robust to outliers

### Tree-Based Models
6. **Random Forest** - Ensemble of decision trees
7. **Gradient Boosting (GBM)** - Sequential tree boosting
8. **XGBoost** - Optimized gradient boosting
9. **LightGBM** - Fast gradient boosting
10. **CatBoost** - Handles categorical features well
11. **Extra Trees** - Extremely randomized trees

### Support Vector Machines
12. **SVR (Support Vector Regression)** - Non-linear kernel regression
    - Linear kernel
    - RBF kernel
    - Polynomial kernel

### Neural Networks
13. **MLP (Multi-Layer Perceptron)** - Basic neural network
14. **LSTM** - Long Short-Term Memory for time-series
15. **GRU** - Gated Recurrent Unit (simpler than LSTM)
16. **1D CNN** - Convolutional network for time-series patterns
17. **Transformer** - Attention-based architecture
18. **TCN (Temporal Convolutional Network)** - Dilated convolutions

### Ensemble Methods
19. **Voting Regressor** - Average predictions from multiple models
20. **Stacking** - Meta-model trained on base model predictions
21. **Blending** - Weighted average of models

### Advanced/Specialized
22. **GAM (Generalized Additive Models)** - Smooth non-linear functions
23. **Quantile Regression** - Predict confidence intervals
24. **Bayesian Ridge** - Probabilistic predictions
25. **Gaussian Process Regression** - Non-parametric Bayesian approach
26. **KNN Regression** - K-nearest neighbors
27. **Isotonic Regression** - Monotonic relationships
28. **MARS (Multivariate Adaptive Regression Splines)** - Piecewise linear

### Time-Series Specific
29. **ARIMA with exogenous variables** - Traditional time-series
30. **Prophet** - Facebook's time-series forecasting
31. **State Space Models** - Kalman filtering approaches
32. **DeepAR** - Amazon's RNN-based forecasting

## Implementation Strategy

### Directory Structure
```
data/regressions/
├── ols/               # Linear regression (baseline)
├── ridge/             # Ridge regression
├── random_forest/     # Random Forest
├── xgboost/          # XGBoost
├── neural_net/       # Neural networks
├── ensemble/         # Ensemble methods
└── comparison/       # Comparative analysis
```

### Evaluation Metrics
- **MAE** (Mean Absolute Error) - Average prediction error
- **RMSE** (Root Mean Square Error) - Penalizes large errors
- **R²** (R-squared) - Variance explained
- **MAPE** (Mean Absolute Percentage Error) - Relative error
- **Training time** - Computational efficiency
- **Prediction latency** - Real-time applicability

### Model Selection Criteria
1. **Accuracy**: Lowest MAE/RMSE
2. **Generalization**: Performance on unseen data
3. **Interpretability**: Can we understand the model?
4. **Computational cost**: Training and inference time
5. **Robustness**: Performance across different cyclists

## Next Steps

1. **Implement Ridge/Lasso** regression as immediate next steps
2. **Test tree-based models** (Random Forest, XGBoost)
3. **Explore neural networks** for complex patterns
4. **Create ensemble** of best-performing models
5. **Develop comparison framework** to evaluate all models
6. **Consider individual athlete models** vs. universal model

## Expected Outcomes

Based on similar physiological modeling tasks, we expect:
- **Linear models**: R² = 0.70-0.75 (current baseline)
- **Tree-based models**: R² = 0.80-0.85
- **Neural networks**: R² = 0.82-0.88
- **Ensemble methods**: R² = 0.85-0.90

The goal is to achieve <4 bpm MAE (currently 5 bpm) while maintaining model interpretability and real-time prediction capability.

## Physiological Considerations

### Features to potentially add:
- **HR zones** - Previous time spent in different HR zones
- **Gradient** - Rate of elevation change
- **Accumulated work** - Total kilojoules expended
- **Recovery metrics** - Time since last high-intensity effort
- **Cadence variability** - Smoothness of pedaling
- **Power/weight ratio** - Normalized by athlete weight

### Domain-specific constraints:
- HR cannot exceed ~220-age (maximum HR)
- HR has physical lower bound (~40-60 bpm resting)
- HR response has ~15-30 second lag to effort changes
- HR drift occurs during long efforts (dehydration/fatigue)

## Code Organization Principles

1. **Modular design**: Each algorithm in separate module
2. **Common interface**: All models implement predict() method
3. **Reusable pipeline**: Shared data loading and preprocessing
4. **Automated comparison**: Script to train and evaluate all models
5. **Visualization**: Plots comparing model performances
6. **Hyperparameter tuning**: Grid/random search for each model
7. **Cross-validation**: Time-series aware splitting

This comprehensive approach will identify the optimal algorithm for HR prediction in cycling, balancing accuracy, interpretability, and computational efficiency.
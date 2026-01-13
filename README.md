# Trading Strategy Analysis with Machine Learning

## Overview

This project applies Machine Learning techniques to analyze and predict trading outcomes of a trading algorithm running in the forex market. The system evaluates historical trading records done by the trading algorithm. Utilized multiple classification algorithms to identify patterns that lead to profitable trades.

## Project Purpose

The goal of this project is to:
- Analyze historical trading performance using advanced technical indicators
- Build predictive models to classify winning vs. losing trades
- Compare multiple machine learning algorithms to find the best performing model
- Provide insights into usages of the algorithm in different market conditions.

## Technologies:

Tech stack: Python, Pandas, Numpy, SK-learn

## Dataset
Preprocessed trading record from USD/JPY from Aug 2024 to Aug 2025. Cleaned data, engineered technical indicators labling the market.
- **Raw Trading records data**: Entry/exit times, prices, profit/loss
- **Technical Indicators Engineered example**:
- Trend:
- Moving averages
- Moving averages crossovers
- MACD
-Oscilators:
- ATR (Average True Range)
- RSI (Relative Strength Index)
- Volumns:
- AD (Accumulation/Distribution)
- MFI (Money Flow Index)
- **Purpose**:
- to discover the impact of trading timezone, volumn and trend reversal on the trading alogrithm performance.
- Shut down the algorithm in certain market condition.

## Features Engineering

The model uses 21+ engineered features including:
- Numerical indicators (scaled using StandardScaler)
- Categorical features (one-hot encoded)
- Interaction terms (e.g., ATR_RSI_Interaction)
- Time-based features (session, day of week, hour)
- Technical pattern sequences

## Machine Learning Models
1. **Support Vector Machine (SVM)**
   - RBF kernel
   - Balanced class weights
   - Probability estimates enabled

2. **Random Forest Classifier**
   - 100 estimators
   - Balanced class weights
   - Handles non-linear relationships

3. **Logistic Regression**
   - Linear classifier
   - Balanced class weights
   - Provides interpretable coefficients

All models use:
- Stratified train-test split (70/30)
- Class balancing to handle imbalanced datasets
- Standardized feature scaling
**Output**
  - Precision
  - Recall
  - F1-score
  - Support
**Result**
- Ramdom forest performs the best. It has a 0.74 F1 score. The model is good at perdicting failure trade and average at predicting winning trade. The model's performance skewed towards failure trade as I rise the bar of support.
- My interpretaiton is that the model is demostrating some trend of simply predicting everything is false. This is not a result of the ML process but the imperfectness from the algoritm and the fact that alpha factor of the market is constantly evolving. most indicators will contribute to loss trade because they don't hit the alpha factor.
-   Dataset Size: (286, 39)
-Class Distribution:
Losing Trade    0.622378
Winning Trade     0.377622
SVM Classification Report:
               precision    recall  f1-score   support

       False       0.60      0.61      0.61        54
        True       0.32      0.31      0.32        32

    accuracy                           0.50        86


Random Forest Classification Report:
               precision    recall  f1-score   support

       False       0.68      0.81      0.74        54
        True       0.52      0.34      0.42        32

    accuracy                           0.64        86

Logistic Regression Classification Report:
               precision    recall  f1-score   support

       False       0.62      0.57      0.60        54
        True       0.36      0.41      0.38        32

    accuracy                           0.51        86


## Key Highlights

- **Data Quality**: Implements null value detection and handling
- **Feature Engineering**: Combines multiple technical indicators and market context
- **Model Comparison**: Evaluates multiple algorithms to find optimal performance
- **Class Imbalance Handling**: Uses balanced class weights to address imbalanced datasets
- **Reproducibility**: Uses random_state for consistent results

## Future Enhancements
Potential improvements could include:
- Data augumentation
- better feature engineering with better domain knowlege



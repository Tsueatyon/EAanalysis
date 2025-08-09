import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LogisticRegression
# Load preprocessed data
trades_paired = pd.read_csv('preprocessed_trading_record.csv', parse_dates=['Entry_Time', 'Exit_Time', '4H_Candle', '1D_Candle'])

# Check for nulls
if trades_paired[['ATR', 'RSI', 'Dist_to_High', 'Trend', 'Volatility', 'Momentum', 'Session', 'Day_of_Week']].isna().any().any():
    print("Warning: Null values in features. Dropping rows with nulls.")
    trades_paired = trades_paired.dropna(subset=['ATR', 'RSI', 'Dist_to_High', 'Trend', 'Volatility', 'Momentum', 'Session', 'Day_of_Week'])
    raise ValueError("Null (missing) value encountered!")


trades_paired['Win'] = trades_paired['Profit'] > 0
# Check dataset size and class imbalance
print("Dataset Size:", trades_paired.shape)
print("Class Distribution:\n", trades_paired['Win'].value_counts(normalize=True))

features = trades_paired[['ATR', 'RSI', 'Dist_to_High', 'Trend', 'Volatility', 'Momentum', 'Session', 'Day_of_Week','SMMA_Diff','ATR_RSI_Interaction','Price_SMMAs_Intersection','AD', 'MFI', 'Alligator_Jaw', 'Alligator_Teeth', 'Alligator_Lips',
                 'Gator_Upper', 'Gator_Lower', 'AO', 'Fractal_Signal']]
features = pd.get_dummies(features, columns=['Trend', 'Volatility', 'Momentum', 'Session', 'Day_of_Week'])
target = trades_paired['Win']

# Scale numerical features

scaler = StandardScaler()
numerical_cols = ['ATR', 'RSI', 'Dist_to_High',
                  'ATR_RSI_Interaction', 'SMMA_Diff', 'AD', 'MFI', 'Alligator_Jaw', 'Alligator_Teeth',
                  'Alligator_Lips', 'Gator_Upper', 'Gator_Lower', 'AO']
features[numerical_cols] = scaler.fit_transform(features[numerical_cols])

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)

# Train SVM with RBF kernel
svm = SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True)
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
print("SVM Classification Report:\n", classification_report(y_test, y_pred))

# Permutation Importance for feature ranking
perm_importance = permutation_importance(svm, X_test, y_test, n_repeats=10, random_state=42)
feature_importance = pd.Series(perm_importance.importances_mean, index=features.columns).sort_values(ascending=False)
print("Feature Importance (Permutation):\n", feature_importance)

# Optional: Compare with Random Forest for reference
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
rf_importance = pd.Series(rf.feature_importances_, index=features.columns).sort_values(ascending=False)
print("Random Forest Feature Importance:\n", rf_importance)


# Apriori for Association Rules
# Discretize numerical features for Apriori
trades_paired['ATR_Bin'] = pd.qcut(trades_paired['ATR'], q=3, labels=['Low', 'Medium', 'High'])
trades_paired['RSI_Bin'] = pd.qcut(trades_paired['RSI'], q=3, labels=['Low', 'Medium', 'High'])
trades_paired['Dist_to_High_Bin'] = pd.qcut(trades_paired['Dist_to_High'], q=3, labels=['Close', 'Medium', 'Far'])

# Prepare data for Apriori
apriori_data = pd.get_dummies(trades_paired[['Trend', 'Volatility', 'Momentum', 'Session', 'Day_of_Week', 'ATR_Bin', 'RSI_Bin', 'Dist_to_High_Bin', 'Win']])
frequent_itemsets = apriori(apriori_data, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
print("Association Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False))

lr = LogisticRegression(random_state=42, class_weight='balanced')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))
lr_importance = pd.Series(np.abs(lr.coef_[0]), index=features.columns).sort_values(ascending=False)
print("Logistic Regression Feature Importance:\n", lr_importance)
"""
House Sale Price Prediction - Machine Learning Project
======================================================
Dataset: house_prices_dataset.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 60)
print("HOUSE SALE PRICE PREDICTION")
print("=" * 60)

df = pd.read_csv('house_prices_dataset.csv')
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nBasic Statistics:\n{df.describe()}")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Check for missing values
print(f"\nMissing Values:\n{df.isnull().sum()}")

# Target variable distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(df['SalePrice'], bins=50, color='#E94560', edgecolor='white', alpha=0.8)
plt.title('Sale Price Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Sale Price ($)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(np.log1p(df['SalePrice']), bins=50, color='#0F3460', edgecolor='white', alpha=0.8)
plt.title('Log-Transformed Sale Price', fontsize=14, fontweight='bold')
plt.xlabel('Log(Sale Price)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('price_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: price_distribution.png")

# Correlation heatmap (numeric features only)
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(10, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: correlation_heatmap.png")

# Top correlations with SalePrice
print(f"\nTop Correlations with SalePrice:")
print(corr['SalePrice'].sort_values(ascending=False)[1:8])

# ============================================================
# 3. DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Encode categorical variables
le_dict = {}
categorical_cols = ['Neighborhood', 'Condition', 'HouseStyle']
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col])
    le_dict[col] = le
    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Feature Engineering
df['HouseAge'] = 2024 - df['YearBuilt']
df['RemodAge'] = 2024 - df['YearRemodAdd']
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']
df['QualArea'] = df['OverallQual'] * df['GrLivArea']

print("\nNew Features Created: HouseAge, RemodAge, TotalSF, TotalBath, QualArea")

# Select features for modeling
feature_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
                'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'LotArea',
                'HouseAge', 'TotalSF', 'TotalBath', 'QualArea',
                'Neighborhood_Encoded', 'Condition_Encoded', 'HouseStyle_Encoded']

X = df[feature_cols]
y = df['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=4, 
                                                     learning_rate=0.1, random_state=42),
}

results = {}
for name, model in models.items():
    # Use scaled data for Linear Regression, original for tree-based
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation
    if name == 'Linear Regression':
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'CV_Mean': cv_scores.mean()}
    
    print(f"\n{name}:")
    print(f"  R² Score:    {r2:.4f}")
    print(f"  RMSE:        ${rmse:,.0f}")
    print(f"  MAE:         ${mae:,.0f}")
    print(f"  CV R² (mean): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 5. XGBOOST (Install: pip install xgboost)
# ============================================================
try:
    from xgboost import XGBRegressor
    
    xgb_model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred_xgb)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    mae = mean_absolute_error(y_test, y_pred_xgb)
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')
    
    results['XGBoost'] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'CV_Mean': cv_scores.mean()}
    
    print(f"\nXGBoost:")
    print(f"  R² Score:    {r2:.4f}")
    print(f"  RMSE:        ${rmse:,.0f}")
    print(f"  MAE:         ${mae:,.0f}")
    print(f"  CV R² (mean): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance
    importance = pd.Series(xgb_model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=True)
    
    plt.figure(figsize=(8, 6))
    importance.plot(kind='barh', color='#E94560', edgecolor='white')
    plt.title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: feature_importance.png")
    
except ImportError:
    print("\nXGBoost not installed. Run: pip install xgboost")

# ============================================================
# 6. RESULTS VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(results).T
print(f"\n{results_df.to_string()}")

# Model comparison chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['#3B82F6', '#10B981', '#F59E0B', '#E94560']

names = list(results.keys())
r2_vals = [results[n]['R2'] for n in names]
rmse_vals = [results[n]['RMSE'] for n in names]
mae_vals = [results[n]['MAE'] for n in names]

axes[0].bar(names, r2_vals, color=colors[:len(names)], edgecolor='white')
axes[0].set_title('R² Score (Higher is Better)', fontweight='bold')
axes[0].set_ylim(0.7, 1.0)
axes[0].tick_params(axis='x', rotation=20)

axes[1].bar(names, rmse_vals, color=colors[:len(names)], edgecolor='white')
axes[1].set_title('RMSE (Lower is Better)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=20)

axes[2].bar(names, mae_vals, color=colors[:len(names)], edgecolor='white')
axes[2].set_title('MAE (Lower is Better)', fontweight='bold')
axes[2].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison.png")

# Best model prediction vs actual
best_model_name = max(results, key=lambda k: results[k]['R2'])
print(f"\n🏆 Best Model: {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")

print("\n" + "=" * 60)
print("PROJECT COMPLETE!")
print("=" * 60)

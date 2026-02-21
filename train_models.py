"""
Spotify Popularity Prediction - Model Training
Approach: Regression (continuous target 0-100)
Split: 70% train / 15% validation / 15% test (per project spec)
Features: 13 audio features + explicit + genre (encoded)
Excluded: metadata (track_id, artists, album_name, track_name)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SPOTIFY POPULARITY PREDICTION — REGRESSION")
print("=" * 60)

# =============================================================================
# 1. LOAD & CLEAN
# =============================================================================
print("\n[1] Loading data...")
df = pd.read_csv('dataset.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
df = df.dropna(subset=['artists', 'album_name', 'track_name'])
print(f"  Records: {len(df):,}")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print("\n[2] Preparing features...")

audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
]

# Encode categorical features
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['track_genre'])
df['explicit_num'] = df['explicit'].astype(int)

feature_cols = audio_features + ['genre_encoded', 'explicit_num']
X = df[feature_cols]
y = df['popularity']

print(f"  Features: {len(feature_cols)} ({', '.join(feature_cols)})")
print(f"  Target: popularity (min={y.min()}, max={y.max()}, mean={y.mean():.1f})")
print(f"\n  NOTE: Metadata (artists, album_name, track_name, track_id) excluded.")
print(f"        Goal is to learn from audio characteristics, not artist identity.")

# =============================================================================
# 3. TRAIN / VALIDATION / TEST SPLIT — 70:15:15
# =============================================================================
print("\n[3] Splitting data (70/15/15)...")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
# 15% of remaining ~85% ≈ 17.6% → gives us ~15% of total
val_relative = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_relative, random_state=42
)

print(f"  Train:      {len(X_train):>7,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validation: {len(X_val):>7,} ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:       {len(X_test):>7,} ({len(X_test)/len(X)*100:.1f}%)")

# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================
def evaluate(y_true, y_pred):
    return {
        'MAE':  mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2':   r2_score(y_true, y_pred)
    }

def print_metrics(name, train_m, val_m, test_m=None):
    print(f"\n  {name}:")
    print(f"    {'Set':<12} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
    print(f"    {'-'*35}")
    print(f"    {'Train':<12} {train_m['MAE']:>7.3f} {train_m['RMSE']:>7.3f} {train_m['R2']:>7.3f}")
    print(f"    {'Validation':<12} {val_m['MAE']:>7.3f} {val_m['RMSE']:>7.3f} {val_m['R2']:>7.3f}")
    if test_m:
        print(f"    {'Test':<12} {test_m['MAE']:>7.3f} {test_m['RMSE']:>7.3f} {test_m['R2']:>7.3f}")

# =============================================================================
# 5. BASELINE — LINEAR REGRESSION
# =============================================================================
print("\n[4] Training models...")
print("\n  --- Linear Regression (baseline) ---")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_train_m = evaluate(y_train, lr.predict(X_train_scaled))
lr_val_m   = evaluate(y_val,   lr.predict(X_val_scaled))
lr_test_m  = evaluate(y_test,  lr.predict(X_test_scaled))
print_metrics("Linear Regression", lr_train_m, lr_val_m, lr_test_m)

# =============================================================================
# 6. RANDOM FOREST
# =============================================================================
print("\n  --- Random Forest ---")
rf = RandomForestRegressor(
    n_estimators=100, max_depth=20, min_samples_split=5,
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_train_m = evaluate(y_train, rf.predict(X_train))
rf_val_m   = evaluate(y_val,   rf.predict(X_val))
rf_test_m  = evaluate(y_test,  rf.predict(X_test))
print_metrics("Random Forest", rf_train_m, rf_val_m, rf_test_m)

rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# =============================================================================
# 7. XGBOOST
# =============================================================================
print("\n  --- XGBoost ---")
xgb_model = xgb.XGBRegressor(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    subsample=0.8, random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_train_m = evaluate(y_train, xgb_model.predict(X_train))
xgb_val_m   = evaluate(y_val,   xgb_model.predict(X_val))
xgb_test_m  = evaluate(y_test,  xgb_model.predict(X_test))
print_metrics("XGBoost", xgb_train_m, xgb_val_m, xgb_test_m)

xgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# =============================================================================
# 8. COMPARISON & WINNER
# =============================================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON (Validation set)")
print("=" * 60)
print(f"  {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
print(f"  {'-'*45}")
models = [
    ("Linear Regression", lr_val_m),
    ("Random Forest",     rf_val_m),
    ("XGBoost",           xgb_val_m),
]
for name, m in models:
    print(f"  {name:<22} {m['MAE']:>7.3f} {m['RMSE']:>7.3f} {m['R2']:>7.3f}")

best_name, best_val = min(models, key=lambda x: x[1]['MAE'])
print(f"\n  Best on validation (MAE): {best_name}")
print(f"\n  Literature benchmark (same dataset, RF regressor): R²≈0.61")

# =============================================================================
# 9. VISUALIZATIONS
# =============================================================================
print("\n[5] Generating plots...")

rf_pred_test   = rf.predict(X_test)
xgb_pred_test  = xgb_model.predict(X_test)
lr_pred_test   = lr.predict(X_test_scaled)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Regression Results', fontsize=15, fontweight='bold')

# --- Actual vs Predicted scatter ---
for idx, (name, pred, metrics) in enumerate([
    (f"Linear Regression\nR²={lr_test_m['R2']:.3f}",  lr_pred_test,  lr_test_m),
    (f"Random Forest\nR²={rf_test_m['R2']:.3f}",      rf_pred_test,  rf_test_m),
    (f"XGBoost\nR²={xgb_test_m['R2']:.3f}",           xgb_pred_test, xgb_test_m),
]):
    ax = axes[0, idx]
    ax.scatter(y_test, pred, alpha=0.15, s=3, color='steelblue')
    ax.plot([0, 100], [0, 100], 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual Popularity')
    ax.set_ylabel('Predicted Popularity')
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

# --- Feature Importance ---
for idx, (name, imp) in enumerate([
    ("Random Forest", rf_importance),
    ("XGBoost",       xgb_importance),
]):
    ax = axes[1, idx]
    top = imp.head(10)
    ax.barh(range(len(top)), top['importance'], color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{name} — Top 10 Features')

# --- Residuals (best model) ---
ax = axes[1, 2]
residuals = y_test.values - rf_pred_test
ax.scatter(rf_pred_test, residuals, alpha=0.1, s=3, color='steelblue')
ax.axhline(0, color='red', linestyle='--', lw=1.5)
ax.set_xlabel('Predicted Popularity')
ax.set_ylabel('Residual (Actual - Predicted)')
ax.set_title('Random Forest — Residual Plot')

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: model_results.png")

# =============================================================================
# 10. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Split: 70% train / 15% val / 15% test (per spec)")
print(f"  Features: {len(feature_cols)} (audio + genre + explicit, no metadata)")
print()
print(f"  {'Model':<22} {'Test MAE':>9} {'Test RMSE':>10} {'Test R²':>8}")
print(f"  {'-'*52}")
for name, m in [("Linear Regression", lr_test_m),
                ("Random Forest",     rf_test_m),
                ("XGBoost",           xgb_test_m)]:
    print(f"  {name:<22} {m['MAE']:>9.3f} {m['RMSE']:>10.3f} {m['R2']:>8.3f}")
print()
print(f"  Literature benchmark (RF, same dataset): R²≈0.61, MSE≈190.6")
print("=" * 60)

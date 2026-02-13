import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def clean_feature_name(name):
    return name.replace('genre_encoded', 'genre').replace('explicit_num', 'explicit')

def evaluate_model(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

def print_results(model_name, metrics, importance_df):
    print(f"\n{model_name}:")
    print(f"  MAE: {metrics['MAE']:.3f} | RMSE: {metrics['RMSE']:.3f} | R2: {metrics['R2']:.3f}")
    print(f"  Top 3 features: {', '.join([clean_feature_name(f) for f in importance_df.head(3)['feature']])}")

print("="*60)
print("SPOTIFY POPULARITY PREDICTION")
print("="*60)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('dataset.csv').dropna(subset=['artists', 'album_name', 'track_name'])
print(f"Tracks: {len(df):,}")

# Feature engineering
print("[2] Preparing features...")
audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                  'speechiness', 'acousticness', 'instrumentalness', 
                  'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['track_genre'])
df['explicit_num'] = df['explicit'].astype(int)

feature_cols = audio_features + ['genre_encoded', 'explicit_num']
X, y = df[feature_cols], df['popularity']

# Train/test split
print("[3] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# Train models
print("\n[4] Training models...")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5, 
                                 random_state=42, n_jobs=-1, verbose=0)
rf_model.fit(X_train, y_train)
rf_metrics = evaluate_model(y_test, rf_model.predict(X_test))
rf_importance = pd.DataFrame({'feature': feature_cols, 
                              'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, 
                             subsample=0.8, random_state=42, n_jobs=-1, verbosity=0)
xgb_model.fit(X_train, y_train)
xgb_metrics = evaluate_model(y_test, xgb_model.predict(X_test))
xgb_importance = pd.DataFrame({'feature': feature_cols, 
                               'importance': xgb_model.feature_importances_}).sort_values('importance', ascending=False)

# Results
print_results("Random Forest", rf_metrics, rf_importance)
print_results("XGBoost", xgb_metrics, xgb_importance)

# Winner
winner = "Random Forest" if rf_metrics['MAE'] < xgb_metrics['MAE'] else "XGBoost"
print(f"\n{'='*60}")
print(f"WINNER: {winner} (MAE: {min(rf_metrics['MAE'], xgb_metrics['MAE']):.2f})")
print(f"{'='*60}")

# Visualizations
print("\n[5] Generating plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# Scatter plots
for idx, (name, pred, metrics) in enumerate([(f"RF (R²={rf_metrics['R2']:.3f})", rf_pred, rf_metrics),
                                               (f"XGB (R²={xgb_metrics['R2']:.3f})", xgb_pred, xgb_metrics)]):
    axes[0, idx].scatter(y_test, pred, alpha=0.3, s=5)
    axes[0, idx].plot([0, 100], [0, 100], 'r--', lw=2)
    axes[0, idx].set_xlabel('Actual')
    axes[0, idx].set_ylabel('Predicted')
    axes[0, idx].set_title(name)
    axes[0, idx].grid(alpha=0.3)

# Feature importance
for idx, (name, imp) in enumerate([("Random Forest", rf_importance), ("XGBoost", xgb_importance)]):
    top = imp.head(10)
    axes[1, idx].barh(range(len(top)), top['importance'])
    axes[1, idx].set_yticks(range(len(top)))
    axes[1, idx].set_yticklabels([clean_feature_name(f) for f in top['feature']])
    axes[1, idx].set_title(f"{name} - Features")
    axes[1, idx].invert_yaxis()

plt.tight_layout()
plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: model_results.png")

"""
Spotify Popularity Prediction - Enhanced Model with Enriched Features
Run this AFTER enriching dataset with enrich_dataset.py
"""

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
print("SPOTIFY POPULARITY PREDICTION - ENHANCED MODEL")
print("="*60)

# Load enriched data
print("\n[1] Loading enriched dataset...")
try:
    df = pd.read_csv('dataset_enriched.csv')
    print(f"Enriched dataset loaded: {len(df):,} tracks")
except FileNotFoundError:
    print("ERROR: dataset_enriched.csv not found!")
    print("Please run enrich_dataset.py first to add API features.")
    exit()

df = df.dropna(subset=['artists', 'album_name', 'track_name'])
print(f"After cleaning: {len(df):,} tracks")

# Feature engineering
print("\n[2] Preparing features...")

# Original audio features
audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                  'speechiness', 'acousticness', 'instrumentalness', 
                  'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

# New features from API
new_features = ['artist_popularity', 'artist_followers', 'artist_genres',
                'days_since_release', 'available_markets', 'album_total_tracks',
                'album_type_encoded', 'artist_avg_popularity_historical',
                'artist_track_count_in_dataset']

# Encode genre
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['track_genre'])
df['explicit_num'] = df['explicit'].astype(int)

# Log transform for artist_followers (often skewed)
if 'artist_followers' in df.columns:
    df['artist_followers_log'] = np.log1p(df['artist_followers'])
    new_features.append('artist_followers_log')

# Combine all features
feature_cols = audio_features + ['genre_encoded', 'explicit_num'] + new_features

# Remove features with too many missing values
feature_cols_valid = [col for col in feature_cols if col in df.columns and df[col].notna().sum() > len(df) * 0.5]

print(f"Total features: {len(feature_cols_valid)}")
print(f"  - Audio features: {len(audio_features)}")
print(f"  - API features: {len([f for f in new_features if f in feature_cols_valid])}")
print(f"  - Other: {len(feature_cols_valid) - len(audio_features) - len([f for f in new_features if f in feature_cols_valid])}")

# Handle remaining missing values
X = df[feature_cols_valid].fillna(df[feature_cols_valid].median())
y = df['popularity']

# Train/test split
print("\n[3] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# Train models
print("\n[4] Training models...")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=150, max_depth=25, min_samples_split=5, 
                                 random_state=42, n_jobs=-1, verbose=0)
rf_model.fit(X_train, y_train)
rf_metrics = evaluate_model(y_test, rf_model.predict(X_test))
rf_importance = pd.DataFrame({'feature': feature_cols_valid, 
                              'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=150, max_depth=8, learning_rate=0.1, 
                             subsample=0.8, random_state=42, n_jobs=-1, verbosity=0)
xgb_model.fit(X_train, y_train)
xgb_metrics = evaluate_model(y_test, xgb_model.predict(X_test))
xgb_importance = pd.DataFrame({'feature': feature_cols_valid, 
                               'importance': xgb_model.feature_importances_}).sort_values('importance', ascending=False)

# Results
print_results("Random Forest", rf_metrics, rf_importance)
print_results("XGBoost", xgb_metrics, xgb_importance)

# Winner
winner = "Random Forest" if rf_metrics['MAE'] < xgb_metrics['MAE'] else "XGBoost"
best_metrics = rf_metrics if winner == "Random Forest" else xgb_metrics
best_importance = rf_importance if winner == "Random Forest" else xgb_importance

print(f"\n{'='*60}")
print(f"WINNER: {winner}")
print(f"MAE: {best_metrics['MAE']:.2f} | R²: {best_metrics['R2']:.3f}")
print(f"{'='*60}")

# Feature importance comparison
print("\n[5] Top 10 Features:")
for idx, row in best_importance.head(10).iterrows():
    feat_name = clean_feature_name(row['feature'])
    feat_type = "🎵 Audio" if row['feature'] in audio_features else "🎤 API" if row['feature'] in new_features else "📊 Meta"
    print(f"  {idx+1:2d}. {feat_name:30s} {feat_type:10s} {row['importance']:.4f}")

# Comparison with baseline (original model)
print("\n[6] Improvement vs Baseline:")
print("Baseline (audio + genre only):")
print("  MAE: 11.72 | R²: 0.479")
print(f"Enhanced (with API features):")
print(f"  MAE: {best_metrics['MAE']:.2f} | R²: {best_metrics['R2']:.3f}")
mae_improvement = ((11.72 - best_metrics['MAE']) / 11.72) * 100
r2_improvement = ((best_metrics['R2'] - 0.479) / 0.479) * 100
print(f"Improvement: MAE {mae_improvement:+.1f}% | R² {r2_improvement:+.1f}%")

# Visualizations
print("\n[7] Generating plots...")
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
    colors = ['red' if f in audio_features else 'blue' if f in new_features else 'gray' for f in top['feature']]
    axes[1, idx].barh(range(len(top)), top['importance'], color=colors)
    axes[1, idx].set_yticks(range(len(top)))
    axes[1, idx].set_yticklabels([clean_feature_name(f) for f in top['feature']])
    axes[1, idx].set_title(f"{name} - Features")
    axes[1, idx].invert_yaxis()

plt.tight_layout()
plt.savefig('model_results_enhanced.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: model_results_enhanced.png")

print("\n" + "="*60)
print("DONE!")
print("="*60)

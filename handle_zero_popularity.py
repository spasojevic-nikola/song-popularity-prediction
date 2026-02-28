"""
Analiza i resavanje sumnjivim vrednosti popularnosti (0)

Ovaj skript implementira 2 pristupa:
1. Uklanjanje pesama sa popularnost = 0
2. Imputation pomocu KNN - dodeljivanje prosecne popularnosti slicnih pesama

Autori: Isidora Pavlovic, Nikola Spasojevic
"""

# Fix encoding za Windows konzolu
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

print("="*80)
print("ANALIZA I REŠAVANJE SUMNJIVIM VREDNOSTI POPULARNOSTI")
print("="*80)

# ============================================================================
# 1. UČITAVANJE I ANALIZA
# ============================================================================
print("\n[1] Učitavanje podataka...")
df = pd.read_csv('dataset.csv')
df = df.dropna(subset=['artists', 'album_name', 'track_name'])

zero_pop_count = (df['popularity'] == 0).sum()
total_songs = len(df)
zero_pop_pct = zero_pop_count / total_songs * 100

print(f"Ukupno pesama: {total_songs:,}")
print(f"Pesama sa popularnost = 0: {zero_pop_count:,} ({zero_pop_pct:.1f}%)")
print(f"Pesama sa popularnost > 0: {total_songs - zero_pop_count:,}")

# Analiza karakteristika pesama sa popularnost 0
print("\n[2] Analiza karakteristika pesama sa popularnost = 0...")
zero_pop_songs = df[df['popularity'] == 0]
non_zero_pop_songs = df[df['popularity'] > 0]

# Poređenje osnovnih statistika
audio_features = ['danceability', 'energy', 'loudness', 'acousticness', 
                  'instrumentalness', 'valence', 'tempo', 'duration_ms']

print("\nPoređenje audio karakteristika:")
print(f"{'Feature':<20} {'Pop=0 Mean':<15} {'Pop>0 Mean':<15} {'Razlika'}")
print("-" * 70)

for feature in audio_features:
    mean_zero = zero_pop_songs[feature].mean()
    mean_nonzero = non_zero_pop_songs[feature].mean()
    diff = mean_zero - mean_nonzero
    print(f"{feature:<20} {mean_zero:<15.3f} {mean_nonzero:<15.3f} {diff:+.3f}")

# ============================================================================
# 2. PRIPREMA PODATAKA ZA OBA EKSPERIMENTA
# ============================================================================
print("\n[3] Priprema podataka...")

# Enkodovanje žanra
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['track_genre'])

# Feature lista
feature_cols = audio_features + ['genre_encoded']

# ============================================================================
# EKSPERIMENT 1: Uklanjanje pesama sa popularnost = 0
# ============================================================================
print("\n" + "="*80)
print("EKSPERIMENT 1: Uklanjanje pesama sa popularnost = 0")
print("="*80)

df_removed = df[df['popularity'] > 0].copy()
print(f"Dataset nakon uklanjanja: {len(df_removed):,} pesama")

X_removed = df_removed[feature_cols]
y_removed = df_removed['popularity']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_removed, y_removed, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train_r):,} | Test: {len(X_test_r):,}")

# Treniranje Random Forest
print("Treniranje Random Forest modela...")
rf_removed = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_removed.fit(X_train_r, y_train_r)

y_pred_r = rf_removed.predict(X_test_r)
mae_removed = mean_absolute_error(y_test_r, y_pred_r)
r2_removed = r2_score(y_test_r, y_pred_r)

print(f"\nRezultati (uklanjanje):")
print(f"  MAE: {mae_removed:.3f}")
print(f"  R²:  {r2_removed:.3f}")

# ============================================================================
# EKSPERIMENT 2: KNN Imputation - dodeljivanje popularnosti sličnim pesmama
# ============================================================================
print("\n" + "="*80)
print("EKSPERIMENT 2: KNN Imputation")
print("="*80)

df_imputed = df.copy()

# Izdvajamo pesme sa popularnost = 0 i > 0
mask_zero = df_imputed['popularity'] == 0
mask_nonzero = df_imputed['popularity'] > 0

# Normalizacija features za KNN
scaler = StandardScaler()
features_all = df_imputed[feature_cols].values
features_scaled = scaler.fit_transform(features_all)

# Pesme koje imaju popularnost > 0 (sa poznatim vrednostima)
known_features = features_scaled[mask_nonzero]
known_popularity = df_imputed.loc[mask_nonzero, 'popularity'].values

# Pesme sa popularnost = 0 (koje treba popuniti)
unknown_features = features_scaled[mask_zero]

print(f"Pesama sa poznatom popularnošću: {len(known_features):,}")
print(f"Pesama za imputation: {len(unknown_features):,}")

# KNN model - pronalazi 10 najsličnijih pesama
print("\nPrimena KNN (k=10) za pronalaženje sličnih pesama...")
knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn.fit(known_features)

# Pronalazi 10 najsličnijih pesama za svaku pesmu sa pop=0
distances, indices = knn.kneighbors(unknown_features)

# Dodeljivanje prosečne popularnosti sličnih pesama
imputed_values = []
for idx_list in indices:
    similar_popularities = known_popularity[idx_list]
    # Prosek popularnosti 10 najsličnijih pesama
    avg_popularity = similar_popularities.mean()
    imputed_values.append(avg_popularity)

imputed_values = np.array(imputed_values)

# Zamena 0 sa imputiranim vrednostima
# Najpre konvertujemo kolonu u float kako bi primila imputed vrednosti
df_imputed['popularity'] = df_imputed['popularity'].astype(float)
df_imputed.loc[mask_zero, 'popularity'] = imputed_values

print(f"\nStatistika imputiranih vrednosti:")
print(f"  Mean: {np.mean(imputed_values):.2f}")
print(f"  Median: {np.median(imputed_values):.2f}")
print(f"  Min: {np.min(imputed_values):.2f}")
print(f"  Max: {np.max(imputed_values):.2f}")

# Treniranje modela na dataset-u sa imputation
X_imputed = df_imputed[feature_cols]
y_imputed = df_imputed['popularity']

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_imputed, y_imputed, test_size=0.2, random_state=42
)

print(f"\nTrain: {len(X_train_i):,} | Test: {len(X_test_i):,}")

print("Treniranje Random Forest modela...")
rf_imputed = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_imputed.fit(X_train_i, y_train_i)

y_pred_i = rf_imputed.predict(X_test_i)
mae_imputed = mean_absolute_error(y_test_i, y_pred_i)
r2_imputed = r2_score(y_test_i, y_pred_i)

print(f"\nRezultati (imputation):")
print(f"  MAE: {mae_imputed:.3f}")
print(f"  R²:  {r2_imputed:.3f}")

# ============================================================================
# 3. POREĐENJE REZULTATA
# ============================================================================
print("\n" + "="*80)
print("POREĐENJE EKSPERIMENATA")
print("="*80)

comparison = pd.DataFrame({
    'Metoda': ['Uklanjanje pop=0', 'KNN Imputation'],
    'Dataset veličina': [len(df_removed), len(df_imputed)],
    'MAE': [mae_removed, mae_imputed],
    'R²': [r2_removed, r2_imputed]
})

print("\n" + comparison.to_string(index=False))

# Određivanje pobednika
if mae_imputed < mae_removed:
    winner = "KNN Imputation"
    improvement = ((mae_removed - mae_imputed) / mae_removed) * 100
else:
    winner = "Uklanjanje pop=0"
    improvement = ((mae_imputed - mae_removed) / mae_imputed) * 100

print(f"\nBOLJA METODA: {winner}")
print(f"Poboljšanje MAE: {improvement:.2f}%")

# ============================================================================
# 4. VIZUALIZACIJA POREĐENJA
# ============================================================================
print("\n[4] Generisanje vizualizacija...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Poređenje MAE i R²
methods = ['Uklanjanje\npop=0', 'KNN\nImputation']
mae_values = [mae_removed, mae_imputed]
r2_values = [r2_removed, r2_imputed]

axes[0, 0].bar(methods, mae_values, color=['#e74c3c', '#3498db'], edgecolor='black', linewidth=1.5)
axes[0, 0].set_ylabel('MAE', fontsize=11)
axes[0, 0].set_title('Mean Absolute Error - Poređenje', fontweight='bold', fontsize=12)
axes[0, 0].grid(alpha=0.3, axis='y')
for i, (m, v) in enumerate(zip(methods, mae_values)):
    axes[0, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

axes[0, 1].bar(methods, r2_values, color=['#e74c3c', '#3498db'], edgecolor='black', linewidth=1.5)
axes[0, 1].set_ylabel('R² Score', fontsize=11)
axes[0, 1].set_title('R² Score - Poređenje', fontweight='bold', fontsize=12)
axes[0, 1].grid(alpha=0.3, axis='y')
for i, (m, v) in enumerate(zip(methods, r2_values)):
    axes[0, 1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 2. Distribucija popularnosti - originalna vs imputed
axes[1, 0].hist(df[df['popularity'] > 0]['popularity'], bins=50, alpha=0.6, 
                label='Originalne (pop>0)', color='steelblue', edgecolor='black')
axes[1, 0].hist(imputed_values, bins=30, alpha=0.6, 
                label='Imputirane vrednosti', color='orange', edgecolor='black')
axes[1, 0].set_xlabel('Popularnost', fontsize=11)
axes[1, 0].set_ylabel('Broj pesama', fontsize=11)
axes[1, 0].set_title('Distribucija: Originalne vs Imputirane', fontweight='bold', fontsize=12)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(alpha=0.3)

# 3. Veličina dataset-a
dataset_sizes = [len(df_removed), len(df_imputed)]
axes[1, 1].bar(methods, dataset_sizes, color=['#e74c3c', '#3498db'], edgecolor='black', linewidth=1.5)
axes[1, 1].set_ylabel('Broj pesama', fontsize=11)
axes[1, 1].set_title('Veličina dataset-a', fontweight='bold', fontsize=12)
axes[1, 1].grid(alpha=0.3, axis='y')
for i, (m, v) in enumerate(zip(methods, dataset_sizes)):
    axes[1, 1].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.suptitle('Poređenje pristupa za rešavanje popularnost = 0', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('zero_popularity_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("  → Sačuvano: zero_popularity_comparison.png")

# ============================================================================
# 5. ČUVANJE REZULTATA
# ============================================================================
print("\n[5] Čuvanje dataset-a...")

# Čuvanje oba dataset-a
df_removed.to_csv('dataset_removed_zeros.csv', index=False)
df_imputed.to_csv('dataset_knn_imputed.csv', index=False)

print("  → Sačuvano: dataset_removed_zeros.csv")
print("  → Sačuvano: dataset_knn_imputed.csv")

# ============================================================================
# REZIME
# ============================================================================
print("\n" + "="*80)
print("REZIME")
print("="*80)

print(f"""
ANALIZA SUMNJIVIM VREDNOSTI (popularnost = 0):
• Broj pesama sa pop=0: {zero_pop_count:,} ({zero_pop_pct:.1f}%)

EKSPERIMENT 1 - Uklanjanje:
• Dataset veličina: {len(df_removed):,}
• MAE: {mae_removed:.3f}
• R²: {r2_removed:.3f}

EKSPERIMENT 2 - KNN Imputation (k=10):
• Dataset veličina: {len(df_imputed):,}
• Prosečna imputirana vrednost: {np.mean(imputed_values):.2f}
• MAE: {mae_imputed:.3f}
• R²: {r2_imputed:.3f}

PREPORUKA:
{winner} pokazuje bolje performanse sa poboljšanjem od {improvement:.2f}%.

Oba dataset-a su sačuvana za dalja istraživanja:
- dataset_removed_zeros.csv (bez pop=0)
- dataset_knn_imputed.csv (sa KNN imputation)
""")

print("="*80)

"""
Generisanje smislenih vizualizacija za projekat predikcije popularnosti pesama

Ovaj skript generiše 5 ključnih vizualizacija koje imaju jasan narativ:
1. Distribucija popularnosti - pokazuje problem neuravnoteženosti
2. Vizualizacija klastera - pokazuje grupisanje pesama po audio karakteristikama
3. Interakcije atributa po klasterima - pokazuje šablone
4. Feature importance - pokazuje najvažnije faktore
5. MAE po klasterima - pokazuje gde model greši
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Stilizacija grafika
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

print("="*70)
print("GENERISANJE VIZUALIZACIJA - Spotify Popularity Prediction")
print("="*70)

# ============================================================================
# 1. UČITAVANJE I PRIPREMA PODATAKA
# ============================================================================
print("\n[1/6] Učitavanje podataka...")
df = pd.read_csv('dataset.csv')
df = df.dropna(subset=['artists', 'album_name', 'track_name'])
print(f"Učitano {len(df):,} pesama")

# ============================================================================
# 2. KLASTERIZACIJA
# ============================================================================
print("\n[2/6] K-means klasterizacija...")
audio_cols = ['danceability', 'energy', 'loudness', 'acousticness', 
              'instrumentalness', 'valence', 'tempo']

# Normalizacija
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[audio_cols])

# K-means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df_scaled)

# PCA za vizualizaciju
pca = PCA(n_components=2)
coords = pca.fit_transform(df_scaled)

print(f"Pronađena su 4 klastera")

# ============================================================================
# 3. TRENIRANJE MODELA (potreban za feature importance i MAE)
# ============================================================================
print("\n[3/6] Treniranje Random Forest modela...")
audio_features = ['danceability', 'energy', 'loudness', 'acousticness', 
                  'instrumentalness', 'valence', 'tempo', 'duration_ms']
df['genre_encoded'] = LabelEncoder().fit_transform(df['track_genre'])

X = df[audio_features + ['genre_encoded']]
y = df['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, rf_pred)
r2 = r2_score(y_test, rf_pred)
print(f"Model treniran - MAE: {mae:.2f}, R²: {r2:.3f}")

# ============================================================================
# VIZUALIZACIJA 1: DISTRIBUCIJA POPULARNOSTI
# ============================================================================
print("\n[4/6] Grafik 1: Distribucija popularnosti...")

plt.figure(figsize=(10, 6))
plt.hist(df['popularity'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(df['popularity'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Srednja vrednost: {df["popularity"].mean():.1f}')
plt.axvline(df['popularity'].median(), color='orange', linestyle='--', linewidth=2,
            label=f'Medijana: {df["popularity"].median():.1f}')
plt.xlabel('Popularnost', fontsize=11)
plt.ylabel('Broj pesama', fontsize=11)
plt.title('Distribucija popularnosti pesama\n(Neuravnoteženost podataka)', 
          fontsize=13, fontweight='bold', pad=15)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('viz_1_distribucija_popularnosti.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Sačuvano: viz_1_distribucija_popularnosti.png")

# ============================================================================
# VIZUALIZACIJA 2: KLASTERI (PCA + PROSEČNA POPULARNOST)
# ============================================================================
print("\n[5/6] Grafik 2: Vizualizacija klastera...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: PCA scatter
scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=df['cluster'], 
                          cmap='viridis', alpha=0.6, s=15, edgecolors='k', linewidth=0.1)
axes[0].set_xlabel('Prva glavna komponenta', fontsize=11)
axes[0].set_ylabel('Druga glavna komponenta', fontsize=11)
axes[0].set_title('Vizualizacija klastera (PCA projekcija)', fontweight='bold', fontsize=12)
cbar = plt.colorbar(scatter, ax=axes[0])
cbar.set_label('Klaster', fontsize=10)
axes[0].grid(alpha=0.3)

# Subplot 2: Prosečna popularnost
cluster_pop = df.groupby('cluster')['popularity'].mean().sort_values(ascending=False)
bars = axes[1].bar(cluster_pop.index.astype(str), cluster_pop.values, 
                   color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'], 
                   edgecolor='black', linewidth=1.2)
axes[1].set_xlabel('Klaster', fontsize=11)
axes[1].set_ylabel('Prosečna popularnost', fontsize=11)
axes[1].set_title('Prosečna popularnost po klasterima', fontweight='bold', fontsize=12)
axes[1].grid(alpha=0.3, axis='y')

# Dodaj vrednosti na stubovima
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Analiza klastera: Grupisanje i popularnost', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('viz_2_klasteri.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Sačuvano: viz_2_klasteri.png")

# ============================================================================
# VIZUALIZACIJA 3: INTERAKCIJA ENERGY-LOUDNESS PO KLASTERIMA
# ============================================================================
print("\n[6/6] Grafik 3: Interakcija atributa po klasterima...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

cluster_names = ['Klaster 0', 'Klaster 1', 'Klaster 2', 'Klaster 3']
colors = ['viridis', 'plasma', 'coolwarm', 'RdYlGn']

for i, cluster in enumerate(range(4)):
    cluster_data = df[df['cluster'] == cluster]
    scatter = axes[i].scatter(cluster_data['energy'], cluster_data['loudness'], 
                             c=cluster_data['popularity'], cmap='RdYlGn', 
                             alpha=0.6, s=25, vmin=0, vmax=100, edgecolors='k', linewidth=0.1)
    axes[i].set_xlabel('Energy (energičnost)', fontsize=10)
    axes[i].set_ylabel('Loudness (glasnoća)', fontsize=10)
    axes[i].set_title(f'{cluster_names[i]} (n={len(cluster_data):,} pesama)', 
                     fontweight='bold', fontsize=11)
    axes[i].grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[i])
    cbar.set_label('Popularnost', fontsize=9)

plt.suptitle('Interakcija Energy-Loudness po klasterima\n(Šabloni kombinacija atributa)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('viz_3_interakcije_klastera.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Sačuvano: viz_3_interakcije_klastera.png")

# ============================================================================
# VIZUALIZACIJA 4: FEATURE IMPORTANCE
# ============================================================================
print("\n[7/6] Grafik 4: Važnost atributa...")

feature_names = audio_features + ['genre_encoded']
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(8)

# Lepša imena
importance_df['feature'] = importance_df['feature'].replace({
    'genre_encoded': 'Žanr',
    'loudness': 'Glasnoća',
    'energy': 'Energičnost',
    'danceability': 'Pogodnost za ples',
    'acousticness': 'Akustičnost',
    'instrumentalness': 'Instrumentalnost',
    'valence': 'Valentnost',
    'tempo': 'Tempo',
    'duration_ms': 'Trajanje'
})

plt.figure(figsize=(10, 6))
bars = plt.barh(importance_df['feature'], importance_df['importance'], 
                color='steelblue', edgecolor='black', linewidth=1.2)
plt.xlabel('Važnost (Feature Importance)', fontsize=11)
plt.title('Najvažniji faktori za predikciju popularnosti\n(Random Forest)', 
          fontsize=13, fontweight='bold', pad=15)
plt.gca().invert_yaxis()
plt.grid(alpha=0.3, axis='x')

# Dodaj vrednosti
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2., 
            f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('viz_4_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Sačuvano: viz_4_feature_importance.png")

# ============================================================================
# VIZUALIZACIJA 5: MAE PO KLASTERIMA
# ============================================================================
print("\n[8/6] Grafik 5: Greška modela po klasterima...")

test_indices = X_test.index
test_clusters = df.loc[test_indices, 'cluster']

cluster_errors = {}
for cluster in range(4):
    mask = test_clusters == cluster
    if mask.sum() > 0:
        cluster_mae = mean_absolute_error(y_test[mask], rf_pred[mask])
        cluster_errors[cluster] = cluster_mae

plt.figure(figsize=(10, 6))
bars = plt.bar(cluster_errors.keys(), cluster_errors.values(), 
               color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'],
               edgecolor='black', linewidth=1.5)
plt.axhline(mae, color='red', linestyle='--', linewidth=2, label=f'Ukupni MAE: {mae:.2f}')
plt.xlabel('Klaster', fontsize=11)
plt.ylabel('MAE (Mean Absolute Error)', fontsize=11)
plt.title('Greška predikcije po klasterima\n(Različita preciznost modela)', 
          fontsize=13, fontweight='bold', pad=15)
plt.legend(fontsize=10)
plt.grid(alpha=0.3, axis='y')

# Dodaj vrednosti
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('viz_5_mae_po_klasterima.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Sačuvano: viz_5_mae_po_klasterima.png")

# ============================================================================
# REZIME
# ============================================================================
print("\n" + "="*70)
print("GENERISANJE ZAVRŠENO!")
print("="*70)
print("\nKreirano 5 smislenih vizualizacija:")
print("  1. viz_1_distribucija_popularnosti.png - Pokazuje neuravnoteženost podataka")
print("  2. viz_2_klasteri.png - Grupisanje pesama i razlike u popularnosti")
print("  3. viz_3_interakcije_klastera.png - Šabloni kombinacija atributa")
print("  4. viz_4_feature_importance.png - Najvažniji faktori")
print("  5. viz_5_mae_po_klasterima.png - Gde model greši")
print("\nSve vizualizacije imaju jasan narativ i doprinose razumevanju problema.")
print("="*70)

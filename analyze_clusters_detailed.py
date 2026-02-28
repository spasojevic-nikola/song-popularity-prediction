# -*- coding: utf-8 -*-
"""
Detaljna analiza klastera - Kombinacija audio karakteristika
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("="*80)
print("DETALJNA ANALIZA KLASTERA - Kombinacije Audio Karakteristika")
print("="*80)

# Učitaj podatke
df = pd.read_csv('dataset.csv')
df = df.dropna(subset=['artists', 'album_name', 'track_name'])

print(f"\nUkupno pesama: {len(df):,}\n")

# Audio karakteristike za klasterovanje
audio_cols = ['danceability', 'energy', 'loudness', 'acousticness', 
              'instrumentalness', 'valence', 'tempo']

# Normalizacija
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[audio_cols])

# K-means klasterizacija
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Analiza svakog klastera
print("DETALJNE KARAKTERISTIKE SVAKOG KLASTERA:")
print("="*80)

for cluster_id in range(4):
    cluster_data = df[df['cluster'] == cluster_id]
    n_songs = len(cluster_data)
    avg_popularity = cluster_data['popularity'].mean()
    
    print(f"\n{'='*80}")
    print(f"KLASTER {cluster_id}")
    print(f"{'='*80}")
    print(f"Broj pesama: {n_songs:,} ({n_songs/len(df)*100:.1f}%)")
    print(f"Prosečna popularnost: {avg_popularity:.2f}")
    print(f"\nProsečne vrednosti audio karakteristika:")
    print("-"*80)
    
    for col in audio_cols:
        mean_val = cluster_data[col].mean()
        overall_mean = df[col].mean()
        diff = mean_val - overall_mean
        diff_pct = (diff / overall_mean) * 100
        
        if diff_pct > 20:
            marker = "↑↑ VISOKO"
        elif diff_pct > 5:
            marker = "↑ Iznad proseka"
        elif diff_pct < -20:
            marker = "↓↓ NISKO"
        elif diff_pct < -5:
            marker = "↓ Ispod proseka"
        else:
            marker = "≈ Prosečno"
        
        print(f"{col:20s}: {mean_val:7.3f}  (opšti prosek: {overall_mean:7.3f})  {marker}")
    
    # Top žanrovi u klasteru
    print(f"\nNajčešći žanrovi u klasteru:")
    top_genres = cluster_data['track_genre'].value_counts().head(5)
    for i, (genre, count) in enumerate(top_genres.items(), 1):
        print(f"  {i}. {genre}: {count:,} pesama ({count/n_songs*100:.1f}%)")

# Poređenje klastera međusobno
print(f"\n\n{'='*80}")
print("POREĐENJE KLASTERA - Prosečna popularnost")
print(f"{'='*80}\n")

cluster_popularity = df.groupby('cluster')['popularity'].agg(['mean', 'median', 'std', 'count'])
cluster_popularity = cluster_popularity.sort_values('mean', ascending=False)
print(cluster_popularity.to_string())

# Identifikacija ključnih kombinacija
print(f"\n\n{'='*80}")
print("KLJUČNE KOMBINACIJE KARAKTERISTIKA PO KLASTERIMA")
print(f"{'='*80}\n")

for cluster_id in range(4):
    cluster_data = df[df['cluster'] == cluster_id]
    
    print(f"\nKLASTER {cluster_id}:")
    
    # Identificiraj dominantne karakteristike (>20% iznad proseka)
    dominant_high = []
    dominant_low = []
    
    for col in audio_cols:
        mean_val = cluster_data[col].mean()
        overall_mean = df[col].mean()
        diff_pct = ((mean_val - overall_mean) / overall_mean) * 100
        
        if diff_pct > 20:
            dominant_high.append(f"{col} (+{diff_pct:.0f}%)")
        elif diff_pct < -20:
            dominant_low.append(f"{col} ({diff_pct:.0f}%)")
    
    if dominant_high:
        print(f"  VISOKE vrednosti: {', '.join(dominant_high)}")
    if dominant_low:
        print(f"  NISKE vrednosti: {', '.join(dominant_low)}")
    
    avg_pop = cluster_data['popularity'].mean()
    print(f"  → Prosečna popularnost: {avg_pop:.2f}")

# Korelacije unutar klastera
print(f"\n\n{'='*80}")
print("INTERESANTNE INTERAKCIJE UNUTAR KLASTERA")
print(f"{'='*80}\n")

for cluster_id in range(4):
    cluster_data = df[df['cluster'] == cluster_id]
    
    # Korelacija energy-loudness sa popularnošću
    corr_energy = cluster_data[['energy', 'popularity']].corr().iloc[0, 1]
    corr_loudness = cluster_data[['loudness', 'popularity']].corr().iloc[0, 1]
    corr_acousticness = cluster_data[['acousticness', 'popularity']].corr().iloc[0, 1]
    corr_danceability = cluster_data[['danceability', 'popularity']].corr().iloc[0, 1]
    
    print(f"Klaster {cluster_id}:")
    print(f"  Energy ↔ Popularnost: {corr_energy:+.3f}")
    print(f"  Loudness ↔ Popularnost: {corr_loudness:+.3f}")
    print(f"  Acousticness ↔ Popularnost: {corr_acousticness:+.3f}")
    print(f"  Danceability ↔ Popularnost: {corr_danceability:+.3f}")
    print()

print("="*80)
print("ANALIZA ZAVRŠENA")
print("="*80)

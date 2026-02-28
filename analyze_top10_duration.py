# -*- coding: utf-8 -*-
"""
Analiza TOP 10% najpopularnijih pesama - šabloni i trajanje
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("ANALIZA TOP 10% NAJPOPULARNIJIH PESAMA")
print("="*80)

# Učitaj podatke
df = pd.read_csv('dataset.csv')
df = df.dropna(subset=['artists', 'album_name', 'track_name'])

# Dodaj duration_min kolonu odmah
df['duration_min'] = df['duration_ms'] / 60000

# Dodaj i duration_category odmah
bins = [0, 2, 3, 4, 5, 100]  # minute
labels = ['<2 min', '2-3 min', '3-4 min', '4-5 min', '>5 min']
df['duration_category'] = pd.cut(df['duration_min'], bins=bins, labels=labels)

print(f"\nUkupno pesama: {len(df):,}")
print(f"Opseg popularnosti: {df['popularity'].min()} - {df['popularity'].max()}\n")

# ============================================================================
# 1. ANALIZA TOP 10% PESAMA
# ============================================================================
print("="*80)
print("1. ANALIZA TOP 10% NAJPOPULARNIJIH PESAMA")
print("="*80)

# Odredi prag za top 10%
top_10_threshold = df['popularity'].quantile(0.90)
print(f"\nPrag za top 10%: popularnost ≥ {top_10_threshold:.0f}")

top_10_pct = df[df['popularity'] >= top_10_threshold]
bottom_90_pct = df[df['popularity'] < top_10_threshold]

print(f"Top 10%: {len(top_10_pct):,} pesama (prosečna popularnost: {top_10_pct['popularity'].mean():.1f})")
print(f"Bottom 90%: {len(bottom_90_pct):,} pesama (prosečna popularnost: {bottom_90_pct['popularity'].mean():.1f})")

# Poređenje ključnih karakteristika
print(f"\n{'='*80}")
print("POREĐENJE AUDIO KARAKTERISTIKA: Top 10% vs Bottom 90%")
print(f"{'='*80}\n")

audio_features = ['loudness', 'danceability', 'energy', 'acousticness', 
                  'instrumentalness', 'valence', 'tempo', 'duration_ms']

comparison_data = []
for feature in audio_features:
    top_mean = top_10_pct[feature].mean()
    bottom_mean = bottom_90_pct[feature].mean()
    diff = top_mean - bottom_mean
    diff_pct = (diff / bottom_mean) * 100 if bottom_mean != 0 else 0
    
    comparison_data.append({
        'Feature': feature,
        'Top 10% Mean': top_mean,
        'Bottom 90% Mean': bottom_mean,
        'Razlika': diff,
        'Razlika %': diff_pct
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Identifikuj najvažnije razlike
print(f"\n{'='*80}")
print("KLJUČNE KARAKTERISTIKE TOP 10% PESAMA:")
print(f"{'='*80}\n")

significant_diffs = comparison_df[abs(comparison_df['Razlika %']) > 10].sort_values('Razlika %', ascending=False)
for _, row in significant_diffs.iterrows():
    direction = "VIŠE" if row['Razlika %'] > 0 else "MANJE"
    print(f"• {row['Feature']:20s}: {direction} za {abs(row['Razlika %']):5.1f}% (Top: {row['Top 10% Mean']:.2f} vs Bottom: {row['Bottom 90% Mean']:.2f})")

# Analiza kombinacija
print(f"\n{'='*80}")
print("NAJČEŠĆI ŠABLONI U TOP 10%:")
print(f"{'='*80}\n")

# Posmatraj raspodele ključnih karakteristika
print(f"Loudness (glasnoća) u top 10%:")
print(f"  Mean:   {top_10_pct['loudness'].mean():.2f} dB")
print(f"  Median: {top_10_pct['loudness'].median():.2f} dB")
print(f"  Std:    {top_10_pct['loudness'].std():.2f}")
print(f"  Range:  {top_10_pct['loudness'].min():.2f} do {top_10_pct['loudness'].max():.2f}")

print(f"\nDanceability (plesnost) u top 10%:")
print(f"  Mean:   {top_10_pct['danceability'].mean():.3f}")
print(f"  Median: {top_10_pct['danceability'].median():.3f}")
print(f"  Std:    {top_10_pct['danceability'].std():.3f}")
print(f"  Range:  {top_10_pct['danceability'].min():.3f} do {top_10_pct['danceability'].max():.3f}")

print(f"\nInstrumentalness u top 10%:")
print(f"  Mean:   {top_10_pct['instrumentalness'].mean():.3f}")
print(f"  Median: {top_10_pct['instrumentalness'].median():.3f}")
print(f"  > 0.5 (instrumentalne): {(top_10_pct['instrumentalness'] > 0.5).sum():,} pesama ({(top_10_pct['instrumentalness'] > 0.5).sum()/len(top_10_pct)*100:.1f}%)")

# "Zlatni šablon" - pesme sa kombinacijom visokih vrednosti
print(f"\n{'='*80}")
print("'ZLATNI ŠABLON' - Kombinacije u top 10%:")
print(f"{'='*80}\n")

# Visoka loudness (> -6 dB) + visoka danceability (> 0.6)
golden_pattern_1 = top_10_pct[(top_10_pct['loudness'] > -6) & (top_10_pct['danceability'] > 0.6)]
print(f"1. Visoka loudness (>-6 dB) + visoka danceability (>0.6):")
print(f"   {len(golden_pattern_1):,} pesama ({len(golden_pattern_1)/len(top_10_pct)*100:.1f}% od top 10%)")
print(f"   Prosečna popularnost: {golden_pattern_1['popularity'].mean():.1f}")

# Niska instrumentalness (< 0.1)
golden_pattern_2 = top_10_pct[top_10_pct['instrumentalness'] < 0.1]
print(f"\n2. Niska instrumentalness (<0.1) - sa vokalima:")
print(f"   {len(golden_pattern_2):,} pesama ({len(golden_pattern_2)/len(top_10_pct)*100:.1f}% od top 10%)")
print(f"   Prosečna popularnost: {golden_pattern_2['popularity'].mean():.1f}")

# Kombinacija svih 3
golden_triple = top_10_pct[(top_10_pct['loudness'] > -6) & 
                           (top_10_pct['danceability'] > 0.6) & 
                           (top_10_pct['instrumentalness'] < 0.1)]
print(f"\n3. TROSTRUKI ŠABLON - Sve 3 karakteristike zajedno:")
print(f"   {len(golden_triple):,} pesama ({len(golden_triple)/len(top_10_pct)*100:.1f}% od top 10%)")
print(f"   Prosečna popularnost: {golden_triple['popularity'].mean():.1f}")

# ============================================================================
# 2. ANALIZA TRAJANJA PESME (DURATION_MS)
# ============================================================================
print(f"\n\n{'='*80}")
print("2. UTICAJ TRAJANJA PESME NA POPULARNOST")
print(f"{'='*80}\n")

print(f"Statistika trajanja pesama:")
print(f"  Mean:   {df['duration_min'].mean():.2f} minuta ({df['duration_ms'].mean()/1000:.0f} sekundi)")
print(f"  Median: {df['duration_min'].median():.2f} minuta ({df['duration_ms'].median()/1000:.0f} sekundi)")
print(f"  Min:    {df['duration_min'].min():.2f} minuta")
print(f"  Max:    {df['duration_min'].max():.2f} minuta")

duration_analysis = df.groupby('duration_category', observed=True).agg({
    'popularity': ['mean', 'median', 'count']
}).round(2)

print(f"\n{'='*80}")
print("POPULARNOST PO KATEGORIJAMA TRAJANJA:")
print(f"{'='*80}\n")
print(duration_analysis.to_string())

# Pronađi optimalno trajanje
optimal_duration = df.groupby('duration_category', observed=True)['popularity'].mean().idxmax()
print(f"\nNajpopularnija kategorija: {optimal_duration}")

# Analiza top 10% po trajanju
print(f"\n{'='*80}")
print("TRAJANJE U TOP 10% PESAMA:")
print(f"{'='*80}\n")

print(f"Prosečno trajanje top 10%: {top_10_pct['duration_min'].mean():.2f} minuta")
print(f"Prosečno trajanje bottom 90%: {bottom_90_pct['duration_min'].mean():.2f} minuta")
print(f"Razlika: {top_10_pct['duration_min'].mean() - bottom_90_pct['duration_min'].mean():.2f} minuta")

# Distribucija top 10% po trajanju
top_duration_dist = top_10_pct.groupby('duration_category', observed=True).size()
print(f"\nDistribucija top 10% po kategorijama:")
for cat, count in top_duration_dist.items():
    pct = count / len(top_10_pct) * 100
    print(f"  {cat}: {count:,} pesama ({pct:.1f}%)")

# Proveri da li postoji "kritična tačka"
print(f"\n{'='*80}")
print("KRITIČNA TAČKA - Gde popularnost opada?")
print(f"{'='*80}\n")

# Napravi detaljnije intervale
duration_intervals = pd.cut(df['duration_min'], bins=20)
interval_popularity = df.groupby(duration_intervals, observed=True)['popularity'].mean()

max_pop_interval = interval_popularity.idxmax()
print(f"Interval sa maksimalnom popularnošću: {max_pop_interval}")
print(f"Maksimalna prosečna popularnost: {interval_popularity.max():.2f}")

# Pesme preko 6 minuta
long_songs = df[df['duration_min'] > 6]
print(f"\nPesme duže od 6 minuta:")
print(f"  Broj: {len(long_songs):,} ({len(long_songs)/len(df)*100:.1f}%)")
print(f"  Prosečna popularnost: {long_songs['popularity'].mean():.2f}")
print(f"  To je {long_songs['popularity'].mean() - df['popularity'].mean():.2f} manje od ukupnog proseka")

# ============================================================================
# 3. VIZUALIZACIJA
# ============================================================================
print(f"\n{'='*80}")
print("3. GENERISANJE VIZUALIZACIJA")
print(f"{'='*80}\n")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Top 10% vs Bottom 90% - ključne karakteristike
features_to_plot = ['loudness', 'danceability', 'instrumentalness', 'energy']
ax1 = axes[0, 0]
x = np.arange(len(features_to_plot))
width = 0.35

top_vals = [top_10_pct[f].mean() for f in features_to_plot]
bottom_vals = [bottom_90_pct[f].mean() for f in features_to_plot]

# Normalizuj za prikaz
top_norm = [(top_10_pct[f].mean() - df[f].min()) / (df[f].max() - df[f].min()) for f in features_to_plot]
bottom_norm = [(bottom_90_pct[f].mean() - df[f].min()) / (df[f].max() - df[f].min()) for f in features_to_plot]

ax1.bar(x - width/2, top_norm, width, label='Top 10%', color='#2ecc71', edgecolor='black', linewidth=1.5)
ax1.bar(x + width/2, bottom_norm, width, label='Bottom 90%', color='#e74c3c', edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Normalizovana vrednost', fontsize=11)
ax1.set_title('Top 10% vs Bottom 90% - Ključne karakteristike', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(features_to_plot, rotation=45, ha='right')
ax1.legend()
ax1.grid(alpha=0.3, axis='y')

# 2. Popularnost po trajanju
ax2 = axes[0, 1]
duration_means = df.groupby('duration_category', observed=True)['popularity'].mean()
colors_duration = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
bars = ax2.bar(range(len(duration_means)), duration_means.values, color=colors_duration, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Prosečna popularnost', fontsize=11)
ax2.set_title('Popularnost po trajanju pesme', fontweight='bold', fontsize=12)
ax2.set_xticks(range(len(duration_means)))
ax2.set_xticklabels(duration_means.index, rotation=45, ha='right')
ax2.axhline(df['popularity'].mean(), color='red', linestyle='--', linewidth=2, label=f'Ukupni prosek: {df["popularity"].mean():.1f}')
ax2.legend()
ax2.grid(alpha=0.3, axis='y')

# Dodaj vrednosti na stubove
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# 3. Scatter - Trajanje vs Popularnost
ax3 = axes[1, 0]
# Sample 5000 za čitljivost
sample_df = df.sample(min(5000, len(df)), random_state=42)
scatter = ax3.scatter(sample_df['duration_min'], sample_df['popularity'], 
                     alpha=0.3, s=20, c=sample_df['loudness'], cmap='viridis')
ax3.set_xlabel('Trajanje (minuta)', fontsize=11)
ax3.set_ylabel('Popularnost', fontsize=11)
ax3.set_title('Trajanje vs Popularnost (boja = loudness)', fontweight='bold', fontsize=12)
ax3.axvline(3.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimalno ~3.5 min')
ax3.set_xlim(0, 10)
plt.colorbar(scatter, ax=ax3, label='Loudness')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Box plot - loudness + danceability po top/bottom
ax4 = axes[1, 1]
data_to_plot = [
    top_10_pct['loudness'].values,
    bottom_90_pct['loudness'].values,
    top_10_pct['danceability'].values * 20 - 20,  # Skaliraj za isti opseg
    bottom_90_pct['danceability'].values * 20 - 20
]
positions = [1, 2, 4, 5]
bp = ax4.boxplot(data_to_plot, positions=positions, widths=0.6,
                 patch_artist=True, notch=True)

# Bojenje
colors_box = ['#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax4.set_xticks([1.5, 4.5])
ax4.set_xticklabels(['Loudness', 'Danceability\n(skalirana)'], fontsize=11)
ax4.set_ylabel('Vrednost', fontsize=11)
ax4.set_title('Distribucija Loudness i Danceability', fontweight='bold', fontsize=12)
ax4.grid(alpha=0.3, axis='y')

# Legenda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='Top 10%'),
                   Patch(facecolor='#e74c3c', label='Bottom 90%')]
ax4.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('top_10_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("  → Sačuvano: top_10_analysis.png")

print(f"\n{'='*80}")
print("ANALIZA ZAVRŠENA")
print(f"{'='*80}")

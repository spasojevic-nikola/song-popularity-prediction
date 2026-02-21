"""
Spotify Popularity Prediction - EDA
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

print("=" * 60)
print("SPOTIFY EDA")
print("=" * 60)

# =============================================================================
# 1. LOAD & CLEAN DATA
# =============================================================================
print("\n[1] Loading data...")
df = pd.read_csv('dataset.csv')
print(f"Shape (raw): {df.shape} | Records: {len(df):,}")

# Drop unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print(f"Missing values found:\n{missing.to_string()}")
    before = len(df)
    df = df.dropna(subset=['artists', 'album_name', 'track_name'])
    print(f"Dropped {before - len(df)} rows. Remaining: {len(df):,}")
else:
    print("No missing values.")

# Duplicate track_ids
dup_count = df['track_id'].duplicated().sum()
print(f"\nDuplicate track_id rows: {dup_count:,}")
print("  Same song appears in multiple genres (intentional dataset structure).")
print("  Decision: KEEP all rows — genre is a valid feature per instance.")

zero_pop = (df['popularity'] == 0).sum()
print(f"Zero popularity songs: {zero_pop:,} ({zero_pop/len(df)*100:.1f}%) — keeping.")

# =============================================================================
# 2. POPULARITY DISTRIBUTION
# =============================================================================
print("\n[2] Popularity distribution...")
pop = df['popularity']
print(f"  Min={pop.min()} | Max={pop.max()} | Mean={pop.mean():.2f} | "
      f"Median={pop.median():.0f} | Std={pop.std():.2f} | Skew={pop.skew():.3f}")

bins = [0, 20, 40, 60, 80, 100]
labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
pop_range = pd.cut(pop, bins=bins, labels=labels, include_lowest=True)
dist = pop_range.value_counts().sort_index()
print("\n  Distribution by range:")
for rng, cnt in dist.items():
    bar = '█' * int(cnt / len(df) * 50)
    print(f"    {rng:7s}: {cnt:6,} ({cnt/len(df)*100:5.1f}%)  {bar}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('Popularity Distribution', fontsize=14, fontweight='bold')

axes[0].hist(pop, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(pop.mean(), color='red', linestyle='--', label=f'Mean: {pop.mean():.1f}')
axes[0].axvline(pop.median(), color='orange', linestyle='--', label=f'Median: {pop.median():.0f}')
axes[0].set_xlabel('Popularity')
axes[0].set_ylabel('Count')
axes[0].set_title('Histogram')
axes[0].legend()

pop.plot(kind='kde', ax=axes[1], color='darkblue', linewidth=2)
axes[1].set_xlabel('Popularity')
axes[1].set_title('Density (KDE)')

axes[2].bar(dist.index, dist.values, color='steelblue', alpha=0.7, edgecolor='black')
for i, (rng, cnt) in enumerate(dist.items()):
    axes[2].text(i, cnt + 200, f'{cnt/len(df)*100:.1f}%', ha='center', fontsize=9)
axes[2].set_xlabel('Popularity Range')
axes[2].set_ylabel('Count')
axes[2].set_title('Distribution by Range')

plt.tight_layout()
plt.savefig('popularity_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: popularity_distribution.png")

# =============================================================================
# 3. AUDIO FEATURES — DESCRIPTIVE STATS
# =============================================================================
print("\n[3] Audio feature summary...")
audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
]

print(f"\n  {'Feature':<22} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
print(f"  {'-'*58}")
for f in audio_features:
    print(f"  {f:<22} {df[f].min():>8.3f} {df[f].max():>8.3f} "
          f"{df[f].mean():>8.3f} {df[f].std():>8.3f}")

# =============================================================================
# 4. CORRELATION ANALYSIS
# =============================================================================
print("\n[4] Correlation with popularity...")
corr = df[audio_features + ['popularity']].corr()['popularity'].drop('popularity').sort_values(ascending=False)

print("\n  Correlations with popularity (sorted):")
for feat, val in corr.items():
    bar = '█' * int(abs(val) * 40)
    direction = '+' if val > 0 else '-'
    print(f"    {feat:<22}: {val:+.4f}  {direction}{bar}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Audio Features — Correlation Analysis', fontsize=14, fontweight='bold')

full_corr = df[audio_features + ['popularity']].corr()
mask = np.zeros_like(full_corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(full_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=axes[0], mask=mask, cbar_kws={"shrink": 0.8}, linewidths=0.5)
axes[0].set_title('Feature Correlation Matrix')

corr_sorted = corr.sort_values()
colors = ['salmon' if v < 0 else 'steelblue' for v in corr_sorted]
axes[1].barh(corr_sorted.index, corr_sorted.values, color=colors, edgecolor='black', alpha=0.8)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_xlabel('Pearson Correlation with Popularity')
axes[1].set_title('Correlation with Popularity')
for feat, val in corr_sorted.items():
    axes[1].text(val + (0.001 if val >= 0 else -0.001), list(corr_sorted.index).index(feat),
                 f'{val:+.3f}', va='center',
                 ha='left' if val >= 0 else 'right', fontsize=8)

plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: correlation_matrix.png")

# =============================================================================
# 5. GENRE ANALYSIS
# =============================================================================
print("\n[5] Genre analysis...")
print(f"  Total unique genres: {df['track_genre'].nunique()}")

genre_stats = df.groupby('track_genre')['popularity'].agg(['mean', 'median', 'count', 'std'])
genre_stats = genre_stats.sort_values('mean', ascending=False)

print("\n  Top 10 genres by mean popularity:")
print(f"  {'Genre':<25} {'Mean':>6} {'Median':>7} {'Std':>6} {'Count':>7}")
print(f"  {'-'*55}")
for genre, row in genre_stats.head(10).iterrows():
    print(f"  {genre:<25} {row['mean']:>6.1f} {row['median']:>7.0f} "
          f"{row['std']:>6.1f} {row['count']:>7,}")

print("\n  Bottom 5 genres by mean popularity:")
for genre, row in genre_stats.tail(5).iterrows():
    print(f"  {genre:<25} {row['mean']:>6.1f} {row['median']:>7.0f} "
          f"{row['std']:>6.1f} {row['count']:>7,}")

# ANOVA
top_genres = df['track_genre'].value_counts().head(20).index
df_top = df[df['track_genre'].isin(top_genres)]
genre_groups = [g['popularity'].values for _, g in df_top.groupby('track_genre')]
f_stat, p_value = stats.f_oneway(*genre_groups)
print(f"\n  ANOVA (top 20 genres): F={f_stat:.2f}, p={p_value:.2e}")
print(f"  Genre {'IS' if p_value < 0.001 else 'is NOT'} statistically significant (α=0.001)")

top10 = genre_stats.head(10).reset_index()
bottom10 = genre_stats.tail(10).reset_index()
display = pd.concat([top10, bottom10])
colors = ['steelblue'] * 10 + ['salmon'] * 10

fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle('Mean Popularity by Genre — Top 10 vs Bottom 10', fontsize=14, fontweight='bold')

bars = ax.barh(display['track_genre'], display['mean'], color=colors, edgecolor='black', alpha=0.85)
ax.axvline(df['popularity'].mean(), color='black', linestyle='--', linewidth=1.2,
           label=f"Dataset mean ({df['popularity'].mean():.1f})")
ax.set_xlabel('Mean Popularity')
ax.set_ylabel('Genre')
ax.invert_yaxis()

for bar, (_, row) in zip(bars, display.iterrows()):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{row['mean']:.1f}", va='center', fontsize=9)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='steelblue', label='Top 10 (most popular)'),
    Patch(color='salmon',    label='Bottom 10 (least popular)'),
    plt.Line2D([0], [0], color='black', linestyle='--', label=f"Dataset mean ({df['popularity'].mean():.1f})")
])

plt.tight_layout()
plt.savefig('genre_popularity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: genre_popularity.png")

# =============================================================================
# 6. KEY FEATURES VS POPULARITY (scatter)
# =============================================================================
print("\n[6] Feature scatter plots vs popularity...")
key_features = ['danceability', 'energy', 'loudness', 'acousticness',
                'valence', 'instrumentalness']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Key Audio Features vs Popularity', fontsize=14, fontweight='bold')

for ax, feat in zip(axes.flat, key_features):
    ax.scatter(df[feat], df['popularity'], alpha=0.05, s=2, color='steelblue')
    z = np.polyfit(df[feat], df['popularity'], 1)
    x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r-', linewidth=2, label=f'r={corr[feat]:.3f}')
    ax.set_xlabel(feat)
    ax.set_ylabel('Popularity')
    ax.set_title(feat.capitalize())
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('features_vs_popularity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: features_vs_popularity.png")

# =============================================================================
# 7. SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"  Dataset: {len(df):,} songs | {df['track_genre'].nunique()} genres")
print(f"  Target:  mean={pop.mean():.1f}, median={pop.median():.0f}, "
      f"std={pop.std():.2f}, skew={pop.skew():.2f}")
print(f"  Low popularity (≤20): {(pop<=20).sum()/len(df)*100:.1f}% of songs")
print(f"  Strongest correlation: {corr.abs().idxmax()} ({corr.abs().max():.3f})")
print(f"  Genre effect: F={f_stat:.1f}, p={p_value:.2e} — significant")
print(f"\n  Approach: REGRESSION")
print(f"  Rationale: Target is continuous 0-100, no information lost.")
print(f"             Tree-based models handle non-linear relationships.")
print("=" * 60)

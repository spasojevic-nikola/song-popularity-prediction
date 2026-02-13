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

print("="*60)
print("SPOTIFY EDA")
print("="*60)

# 1. LOAD DATA
print("\n[1] Loading data...")
df = pd.read_csv('dataset.csv')
print(f"Shape: {df.shape} | Records: {len(df):,}")

# Handle missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"Missing values: {missing[missing > 0].to_dict()}")
    df = df.dropna(subset=['artists', 'album_name', 'track_name'])
    print(f"After cleaning: {len(df):,} records")

# 2. POPULARITY DISTRIBUTION
print("\n[2] Popularity distribution...")
print(f"Range: {df['popularity'].min()}-{df['popularity'].max()} | Mean: {df['popularity'].mean():.1f} | Median: {df['popularity'].median():.0f}")

dist = {
    '0-25': (df['popularity'] <= 25).sum(),
    '26-50': ((df['popularity'] > 25) & (df['popularity'] <= 50)).sum(),
    '51-75': ((df['popularity'] > 50) & (df['popularity'] <= 75)).sum(),
    '76-100': (df['popularity'] > 75).sum()
}

for range_name, count in dist.items():
    print(f"  {range_name}: {count:6,} ({count/len(df)*100:5.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Popularity Analysis', fontsize=14, fontweight='bold')

# Histogram
axes[0, 0].hist(df['popularity'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['popularity'].mean(), color='red', linestyle='--', label=f'Mean: {df["popularity"].mean():.1f}')
axes[0, 0].set_xlabel('Popularity')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Distribution')
axes[0, 0].legend()

# Box plot
axes[0, 1].boxplot(df['popularity'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
axes[0, 1].set_ylabel('Popularity')
axes[0, 1].set_title('Box Plot')

# KDE
df['popularity'].plot(kind='kde', ax=axes[1, 0], color='darkblue', linewidth=2)
axes[1, 0].fill_between(np.arange(0, 26), 0, axes[1, 0].get_ylim()[1], alpha=0.3, color='red', label='Low Pop')
axes[1, 0].set_xlabel('Popularity')
axes[1, 0].set_title('Density')
axes[1, 0].legend()

# Bar chart
ranges = list(dist.keys())
counts = list(dist.values())
axes[1, 1].bar(ranges, counts, color=['red', 'orange', 'green', 'blue'], alpha=0.7)
axes[1, 1].set_xlabel('Range')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Distribution by Range')

plt.tight_layout()
plt.savefig('popularity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: popularity_distribution.png")

# 3. CORRELATION ANALYSIS
print("\n[3] Correlation with popularity...")
audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                  'speechiness', 'acousticness', 'instrumentalness', 
                  'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

corr = df[audio_features + ['popularity']].corr()['popularity'].drop('popularity').sort_values(ascending=False)

print("Top 5 positive:")
for feat, val in corr.head(5).items():
    print(f"  {feat:20s}: {val:+.4f}")

print("Top 5 negative:")
for feat, val in corr.tail(5).items():
    print(f"  {feat:20s}: {val:+.4f}")

# Correlation matrix visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Audio Features Correlation', fontsize=14, fontweight='bold')

# Full matrix
full_corr = df[audio_features + ['popularity']].corr()
sns.heatmap(full_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[0], cbar_kws={"shrink": 0.8})
axes[0].set_title('Correlation Matrix')

# Popularity only
corr_df = corr.to_frame()
sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=axes[1], vmin=-0.3, vmax=0.3)
axes[1].set_title('Correlation with Popularity')

plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: correlation_matrix.png")

# 4. GENRE ANALYSIS
print("\n[4] Genre analysis...")
print(f"Total genres: {df['track_genre'].nunique()}")

genre_stats = df.groupby('track_genre')['popularity'].agg(['mean', 'count']).sort_values('mean', ascending=False)
print("\nTop 5 genres by popularity:")
for genre, row in genre_stats.head(5).iterrows():
    print(f"  {genre:20s}: {row['mean']:5.1f} (n={row['count']:4,})")

print("\nBottom 5 genres:")
for genre, row in genre_stats.tail(5).iterrows():
    print(f"  {genre:20s}: {row['mean']:5.1f} (n={row['count']:4,})")

# ANOVA test
top_genres = df['track_genre'].value_counts().head(20).index
df_top = df[df['track_genre'].isin(top_genres)]
genre_groups = [group['popularity'].values for name, group in df_top.groupby('track_genre')]
f_stat, p_value = stats.f_oneway(*genre_groups)

print(f"\nANOVA Test (top 20 genres):")
print(f"  F-statistic: {f_stat:.2f}")
print(f"  P-value: {p_value:.2e}")
print(f"  Result: {'Genre IS significant' if p_value < 0.001 else 'Genre NOT significant'}")

# Genre visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Popularity by Genre (Top 20)', fontsize=14, fontweight='bold')

sns.boxplot(data=df_top, x='track_genre', y='popularity', ax=axes[0], palette='Set2')
axes[0].set_xlabel('Genre')
axes[0].set_ylabel('Popularity')
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_title('Box Plot')

sns.violinplot(data=df_top, x='track_genre', y='popularity', ax=axes[1], palette='muted')
axes[1].set_xlabel('Genre')
axes[1].set_ylabel('Popularity')
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_title('Violin Plot')

plt.tight_layout()
plt.savefig('genre_popularity_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: genre_popularity_boxplots.png")

# 5. SUMMARY
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Dataset: {len(df):,} songs, {df['track_genre'].nunique()} genres")
print(f"Popularity: mean={df['popularity'].mean():.1f}, median={df['popularity'].median():.0f}")
print(f"Distribution: Low={dist['0-25']/len(df)*100:.1f}%, High={dist['51-75']/len(df)*100:.1f}%")
print(f"Strongest correlation: {corr.abs().idxmax()} ({corr.abs().max():.3f})")
print(f"Genre significance: F={f_stat:.1f}, p={p_value:.2e}")
print("\nRecommendation: Use REGRESSION (continuous target, weak correlations)")
print("="*60)

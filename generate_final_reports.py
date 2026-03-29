# %% [markdown]
# # Skripta za generisanje finalnih grafikona i izveštaja
# Kreira vizuelno poređenje dva pristupa i listu najvažnijih atributa.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.metrics import r2_score
import os

# Postavljanje stila za grafikone
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# %%
def generate_comparison_plot():
    print("Generišem uporedni grafik modela...")
    
    # Podaci iz tvojih poslednjih uspešnih pokretanja
    data = {
        'Pristup': ['Audio + Žanr (Naučni)', 'Audio + Žanr + Izvođači (Tržišni)'],
        'R2 Score': [0.5401, 0.6613],
        'MAE': [8.5753, 6.9654]
    }
    df_comp = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Pristup', y='R2 Score', data=df_comp, palette='Blues_d', hue='Pristup', legend=False)
    ax.set_ylim(0, 0.8)
    ax.set_title('Poređenje preciznosti predikcije (R² Score)', fontsize=16)
    
    # Dodavanje vrednosti iznad barova
    for i, v in enumerate(df_comp['R2 Score']):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', fontsize=12)

    plt.savefig('comparison_r2_score.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Sačuvan: comparison_r2_score.png")

# %%
def generate_feature_importance():
    print("Učitavam podatke za analizu važnosti atributa...")
    df = pd.read_csv('dataset.csv').dropna(subset=['artists', 'album_name', 'track_name'])
    
    # Brzo pretprocesiranje (identično kao u best_results_final.py)
    df['num_artists'] = df['artists'].apply(lambda x: len(str(x).split(',')))
    df['explicit'] = df['explicit'].astype(int)
    
    # Računamo proseke žanra za relativne vrednosti
    genre_means_loudness = df.groupby('track_genre')['loudness'].transform('mean')
    df['loudness_genre_rel'] = df['loudness'] - genre_means_loudness
    
    df['energy_x_loudness'] = df['energy'] * df['loudness']
    df['is_optimal_duration'] = df['duration_ms'].between(180000, 240000).astype(int)
    
    # Klasteri
    audio_base = ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'valence']
    df['cluster'] = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(df[audio_base]))
    df = pd.get_dummies(df, columns=['cluster'], prefix='c')

    # Brand Stats
    for col in ['artists', 'track_genre']:
        stats = df.groupby(col)['popularity'].mean().fillna(0)
        df[f'{col}_mean'] = df[col].map(stats)

    features = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness', 'valence', 'tempo', 'duration_ms', 'explicit',
        'artists_mean', 'track_genre_mean', 'loudness_genre_rel', 'energy_x_loudness',
        'c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'num_artists', 'is_optimal_duration'
    ]
    
    X = df[features]
    y = df['popularity']
    
    print("Treniram XGBoost za analizu značaja (ovo je brzo)...")
    model = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    # Izvlačenje značaja
    fi_df = pd.DataFrame({'Atribut': features, 'Značaj': model.feature_importances_})
    fi_df = fi_df.sort_values(by='Značaj', ascending=False).head(12)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Značaj', y='Atribut', data=fi_df, palette='viridis', hue='Atribut', legend=False)
    plt.title('Top 12 najvažnijih faktora za popularnost pesme', fontsize=16)
    plt.xlabel('Značaj (Importance Score)')
    
    plt.savefig('feature_importance_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Sačuvan: feature_importance_final.png")

# %%
if __name__ == "__main__":
    generate_comparison_plot()
    generate_feature_importance()
    print("\nSvi grafikoni su uspešno generisani!")

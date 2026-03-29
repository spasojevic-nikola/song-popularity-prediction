# %% [markdown]
# # PROJEKAT: Predikcija popularnosti pesama na Spotify platformi
# **Fokus:** Integrisana analiza audio karakteristika, žanra i snage brenda izvođača.
# **Autori:** Isidora Pavlović, Nikola Spasojević

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import time

# %%
def get_brand_performance_stats(df, train_source, col):
    """
    Izračunava statistiku uspešnosti brenda (izvođača ili žanra).
    Ovaj proces hvata 'društveni kapital' koji ime nosi sa sobom.
    """
    # groupby + agg: Grupiše pesme po izvođaču i računa tri ključna statistička momenta.
    # Pozadinski: 'mean' hvata prosečni nivo slave, 'count' plodnost izvođača, a 'std' konzistentnost kvaliteta.
    stats = train_source.groupby(col)['popularity'].agg(['mean', 'count', 'std']).fillna(0)
    
    # HIT RATE ANALIZA:
    # Pozadinski: Filtriramo pesme sa popularnošću > 70 (značajni hitovi).
    # Odnos (hit_counts / total_counts) daje verovatnoću da će sledeća pesma tog izvođača biti hit.
    hit_counts = train_source[train_source['popularity'] > 70].groupby(col)['popularity'].count()
    total_counts = train_source.groupby(col)['popularity'].count()
    stats['hit_rate'] = (hit_counts / total_counts).fillna(0)
    
    # GLOBALNI PROSEK (Safety Net): 
    # Ako se u test setu pojavi izvođač kojeg nema u treningu, dodeljujemo mu globalni prosek.
    global_mean = train_source['popularity'].mean()
    
    # join: Povezuje izračunatu statistiku sa originalnim redovima na osnovu zajedničkog ključa (npr. ime izvođača).
    # Pozadinski: Koristi 'left join' mehanizam baze podataka radi očuvanja broja redova.
    df = df.join(stats, on=col, rsuffix=f'_{col}')
    df['mean'] = df['mean'].fillna(global_mean)
    df['count'] = df['count'].fillna(0)
    df['std'] = df['std'].fillna(0)
    df['hit_rate'] = df['hit_rate'].fillna(0)
    
    return df.rename(columns={
        'mean': f'{col}_mean', 
        'count': f'{col}_count', 
        'std': f'{col}_std', 
        'hit_rate': f'{col}_hit_rate'
    })

def load_and_preprocess_full_market(filepath='dataset.csv'):
    """
    Vrši sveobuhvatnu pripremu podataka uključujući audio, žanr i izvođače.
    """
    # pandas.read_csv: Čita CSV fajl bajt-po-bajt i gradi strukturu u memoriji.
    df = pd.read_csv(filepath).dropna(subset=['artists', 'album_name', 'track_name'])
    
    # ENGINEERING: Razdvajanje liste izvođača i brojanje članova kolaboracije.
    df['num_artists'] = df['artists'].apply(lambda x: len(str(x).split(',')))
    df['explicit'] = df['explicit'].astype(int)
    
    # RELATIVNE VREDNOSTI (Genre Normalization):
    # Pozadinski: transform('mean') računa prosek atributa unutar svakog žanra i širi tu vrednost na sve redove tog žanra.
    # Oduzimanjem tog proseka dobijamo 'devijaciju'. 
    # Značaj: Modelu je bitno da li je pesma glasnija od ostalih pesama istog žanra, a ne apsolutna vrednost u dB.
    for feat in ['loudness', 'energy', 'tempo', 'danceability']:
        df[f'{feat}_genre_rel'] = df[feat] - df.groupby('track_genre')[feat].transform('mean')
    
    # Množenje energije i glasnoće da bi se pojačao signal 'agresivnosti' zvuka.
    df['energy_x_loudness'] = df['energy'] * df['loudness']
    
    # Fokus na pesme čije je trajanje u opsegu najveće verovatnoće za uspeh (3-4 minuta).
    df['is_optimal_duration'] = df['duration_ms'].between(180000, 240000).astype(int)
    
    # K-MEANS KLASTERIZACIJA:
    # Pozadinski: Algoritam pronalazi 5 grupa (klastera) u 6-dimenzionalnom audio prostoru.
    # fit_predict: Prvo uči pozicije težišta, a zatim dodeljuje labelu (0-4) svakoj pesmi.
    audio_base = ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'valence']
    df['cluster'] = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(df[audio_base]))
    
    # One-Hot Encoding za klastere: Svaki klaster postaje zasebna kolona (c_0, c_1...).
    # Ovo sprečava matematičku grešku gde model misli da je klaster 4 'veći' od klastera 1.
    df = pd.get_dummies(df, columns=['cluster'], prefix='c')

    # PODELA PODATAKA:
    # y: Ciljna varijabla (šta model treba da pogodi).
    # X: Atributi (podaci na osnovu kojih model pogađa).
    y = df['popularity']
    X = df.drop(columns=['popularity'])
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # TARGET ENCODING (BEZBEDAN PRISTUP):
    # Pozadinski: Statistiku računamo SAMO nad trening setom da izbegnemo Data Leakage.
    # Data Leakage se dešava ako prosek u treningu sadrži informaciju iz budućih test podataka.
    train_source = X_train_raw.copy()
    train_source['popularity'] = y_train
    
    X_train_final = get_brand_performance_stats(X_train_raw.copy(), train_source, 'artists')
    X_train_final = get_brand_performance_stats(X_train_final, train_source, 'track_genre')
    X_val_final = get_brand_performance_stats(X_val_raw.copy(), train_source, 'artists')
    X_val_final = get_brand_performance_stats(X_val_final, train_source, 'track_genre')

    # Selekcija 31 najuticajnijeg atributa.
    features = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms',
        'explicit', 'num_artists', 'is_optimal_duration', 'energy_x_loudness',
        'loudness_genre_rel', 'energy_genre_rel', 'tempo_genre_rel', 'danceability_genre_rel',
        'c_0', 'c_1', 'c_2', 'c_3', 'c_4',
        'artists_mean', 'artists_count', 'artists_std', 'artists_hit_rate',
        'track_genre_mean', 'track_genre_count', 'track_genre_std', 'track_genre_hit_rate'
    ]
    
    # STANDARD SCALER: Transformacija svih vrednosti tako da im prosek bude 0, a standardna devijacija 1.
    # Pozadinski: z = (x - mean) / std.
    # Zašto: Algoritmi kao što je Ridge ili neuronske mreže su izuzetno osetljivi na različite opsege vrednosti.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final[features])
    X_val_scaled = scaler.transform(X_val_final[features])
    
    return X_train_scaled, X_val_scaled, y_train, y_val

# Pokretanje celokupnog pipeline-a pretprocesiranja
X_train, X_val, y_train, y_val = load_and_preprocess_full_market()

# %%
# TRENIRANJE TOP 3 ALGORITMA DANAŠNJICE:
models = {
    # Random Forest: Robusan ansambl stabala. Odličan za početnu procenu.
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    
    # XGBoost: Najpopularniji algoritam za tabelarne podatke. 
    # Pozadinski: Koristi Taylor-ovu ekspanziju za optimizaciju funkcije gubitka.
    "XGBoost": xgb.XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.03, random_state=42, n_jobs=-1),
    
    # LightGBM: Microsoft-ov šampion brzine. 
    # Pozadinski: Koristi 'Leaf-wise' rast stabla (umesto 'Level-wise'), što mu omogućava brže postizanje niske greške.
    "LightGBM": lgb.LGBMRegressor(n_estimators=1000, max_depth=8, learning_rate=0.03, random_state=42, n_jobs=-1)
}

results = {}

# Merenje preciznosti (R2 i MAE) za svaki model
for name, model in models.items():
    print(f"Treniranje: {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    # Rezultati predstavljaju apsolutnu preciznost u pogađanju skale od 0 do 100.
    results[name] = {"R2": r2_score(y_val, preds), "MAE": mean_absolute_error(y_val, preds)}

# %%
# Prikaz uporedne tabele rezultata (Tržišni fokus)
print("\n" + "="*50)
print(f"{'MODEL (SVI ATRIBUTI UKLJUČENI)':<20} | {'R2 SCORE':<10} | {'MAE':<10}")
print("-" * 50)
for name, metrics in results.items():
    print(f"{name:<20} | {metrics['R2']:<10.4f} | {metrics['MAE']:<10.4f}")
print("="*50)

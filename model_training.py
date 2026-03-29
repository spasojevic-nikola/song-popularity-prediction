# %% [markdown]
# # PROJEKAT: Predikcija popularnosti pesama na Spotify platformi
# **Fokus:** Analiza uticaja isključivo audio karakteristika i žanrovske pripadnosti.
# **Autori:** Isidora Pavlović, Nikola Spasojević

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import time

# %%
def load_and_preprocess_audio_only(filepath='dataset.csv'):
    """
    Učitava dataset i vrši napredni inženjering audio atributa.
    Ovaj proces transformiše sirove podatke u formate koje algoritmi lakše 'razumeju'.
    """
    
    # pandas.read_csv: Učitava podatke u DataFrame (tabelarnu strukturu u memoriji).
    # Pozadinski: Koristi C engine za brzo čitanje i parser koji automatski prepoznaje tipove kolona.
    df = pd.read_csv(filepath)
    
    # dropna: Uklanja redove gde su bitni tekstualni podaci prazni (NaN).
    # Pozadinski: Optimizuje memoriju oslobađanjem redova koji bi mogli da izazovu greške u string operacijama.
    df = df.dropna(subset=['artists', 'album_name', 'track_name'])
    
    # Konverzija boolean (True/False) u integer (1/0). Većina algoritama ne može direktno raditi sa boolean vrednostima.
    df['explicit'] = df['explicit'].astype(int)
    
    # PSIHOAKUSTIKA: Decibeli su logaritamska mera. Ljudsko uvo jačinu zvuka doživljava linearnije.
    # Formula: p = 10^(dB/20) pretvara relativni nivo u linearni pritisak.
    df['loudness_linear'] = 10 ** (df['loudness'] / 20)
    
    # INTERAKCIJE: Množenje dva atributa pomaže stablima odlučivanja da lakše uoče zajednički uticaj (npr. brza i energična pesma).
    df['dance_energy_interaction'] = df['danceability'] * df['energy']
    df['energy_loudness_sync'] = df['energy'] * df['loudness_linear']
    
    # MOOD QUADRANTS: Kreiranje binarnih atributa (0 ili 1) na osnovu praga od 0.5.
    # Ovo simulira 'Russell-ov model afekta' u muzici (Energy vs Valence).
    df['mood_happy'] = ((df['valence'] > 0.5) & (df['energy'] > 0.5)).astype(int)
    df['mood_sad'] = ((df['valence'] <= 0.5) & (df['energy'] <= 0.5)).astype(int)
    df['mood_angry'] = ((df['valence'] <= 0.5) & (df['energy'] > 0.5)).astype(int)
    df['mood_chill'] = ((df['valence'] > 0.5) & (df['energy'] <= 0.5)).astype(int)
    
    # SWEET SPOT: Na osnovu EDA analize, pesme od 3 do 4 minuta imaju najveću verovatnoću da postanu hit.
    df['is_optimal_len'] = df['duration_ms'].between(180000, 240000).astype(int)

    # SUPER-ŽANROVI (K-Means klasterizacija žanrova):
    # Pozadinski: KMeans pronalazi 'centroide' (težišta) u audio prostoru žanrova.
    # Radi tako što minimizuje unutar-klastersku sumu kvadrata (WCSS).
    # 12 klastera grupira srodne žanrove (npr. 'death metal' i 'black metal' upadaju u istu grupu).
    genre_audio = df.groupby('track_genre')[['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'tempo']].mean()
    genre_scaler = StandardScaler()
    genre_clusters = KMeans(n_clusters=12, random_state=42, n_init=10).fit_predict(genre_scaler.fit_transform(genre_audio))
    genre_to_super = dict(zip(genre_audio.index, genre_clusters))
    df['super_genre'] = df['track_genre'].map(genre_to_super)
    
    # pd.get_dummies: One-Hot Encoding. Pretvara kategoriju (0-11) u 12 zasebnih kolona sa 0 i 1.
    # Ovo sprečava model da misli da je 'super_genre 11' numerički 'veći' ili bitniji od 'super_genre 1'.
    df = pd.get_dummies(df, columns=['super_genre'], prefix='super')

    # SOFT CLUSTERING (Rastojanja):
    # Pozadinski: Za razliku od fit_predict, fit_transform vraća Euklidsko rastojanje svake pesme od svakog od 8 centroida.
    # Rezultat: Model zna da li je pesma 'blizu' profilu akustične muzike ili 'blizu' profilu dens muzike.
    audio_base = ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'valence']
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    distances = kmeans.fit_transform(StandardScaler().fit_transform(df[audio_base]))
    for i in range(8):
        df[f'audio_prof_{i}'] = distances[:, i]

    # TARGET PREPARATION: Nule zamenjujemo sa NaN kako bi ih popunili poštenim prosekom bez 'virenja' u test set.
    y = df['popularity'].replace(0, np.nan)
    X_raw = df.drop(columns=['popularity'])
    
    # train_test_split: Nasumično meša podatke (shuffling) i deli ih u proporciji 70:30.
    # Pozadinski: Koristi random_state za reproduktibilnost (da svaki put dobijemo istu podelu).
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(X_raw, y, test_size=0.30, random_state=42)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp_raw, y_temp, test_size=0.50, random_state=42)
    
    # IMPUTACIJA: Koristimo prosek TRENING seta da popunimo praznine u VALIDACIJI.
    # Ovo je 'pošten' pristup jer model u realnosti ne bi znao prosečnu popularnost pesama koje tek treba da vidi.
    t_mean = y_train.mean()
    y_train = y_train.fillna(t_mean)
    y_val = y_val.fillna(t_mean)
    
    # TARGET ENCODING: Žanr se zamenjuje prosečnom popularnošću koju taj žanr ima u trening setu.
    genre_means = X_train_raw.assign(p=y_train).groupby('track_genre')['p'].mean()
    for d in [X_train_raw, X_val_raw, X_test_raw]:
        d['genre_encoded'] = d['track_genre'].map(genre_means).fillna(t_mean)

    # FINALNA LISTA FEATURA: Odabir 32 najinformativnija stuba podataka.
    features = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
        'valence', 'tempo', 'duration_ms', 'explicit', 'genre_encoded',
        'loudness_linear', 'dance_energy_interaction', 'energy_loudness_sync',
        'mood_happy', 'mood_sad', 'mood_angry', 'mood_chill',
        'instrumental_acoustic', 'speech_dance_ratio', 'is_optimal_len'
    ] + [f'super_{i}' for i in range(12)] + [f'audio_prof_{i}' for i in range(8)]
    
    # QUANTILE TRANSFORMER: Najmoćnija transformacija za tabelarne podatke.
    # Pozadinski: Koristi kumulativnu funkciju distribucije (CDF) da mapira bilo koji raspored podataka u Gausovu (normalnu) krivu.
    # Zašto? Algoritmi gradijentnog boostinga mnogo brže i stabilnije konvergiraju kada podaci nemaju 'duge repove' i ekstremne outliere.
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train = qt.fit_transform(X_train_raw[features])
    X_val = qt.transform(X_val_raw[features])
    
    return X_train, X_val, y_train, y_val, features

# Pokretanje procesa pripreme
X_train, X_val, y_train, y_val, feature_names = load_and_preprocess_audio_only()

# %%
# DEFINICIJA TIMA MODELA:
base_models_dict = {
    # Random Forest: Radi na principu 'Baggig-a'. Pravi 150 stabala, svako nad nasumičnim delom podataka.
    # Pozadinski: Konačni rezultat je prosek svih stabala, što drastično smanjuje varijansu (overfitting).
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=18, random_state=42, n_jobs=-1),
    
    # Extra Trees (Extremely Randomized Trees): Slično kao RF, ali pri svakom deljenju čvora bira pragove POTPUNO nasumično.
    # Pozadinski: Još robusniji na šum u podacima od običnog Random Forest-a.
    "Extra Trees": ExtraTreesRegressor(n_estimators=150, max_depth=18, random_state=42, n_jobs=-1),
    
    # XGBoost (Extreme Gradient Boosting): Radi na principu 'Boosting-a'. Stabla se grade sekvencijalno.
    # Pozadinski: Svako novo stablo pokušava da ispravi rezidualnu grešku prethodnog stabla.
    # Koristi regularizaciju (L1 i L2) da spreči kompleksnost modela.
    "XGBoost": xgb.XGBRegressor(n_estimators=1500, max_depth=9, learning_rate=0.02, subsample=0.8, random_state=42, n_jobs=-1),
    
    # HistGradientBoosting: Specijalizovana verzija koja vrši 'binning' ulaznih podataka u 255 nivoa.
    # Pozadinski: Drastično ubrzava proces deljenja čvorova jer ne mora da sortira sve vrednosti atributa.
    "HistGradient": HistGradientBoostingRegressor(max_iter=1000, max_depth=10, learning_rate=0.02, random_state=42)
}

results = {}

# Treniranje i testiranje individualnih performansi
for name, model in base_models_dict.items():
    print(f"Treniranje: {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    # r2_score: Koeficijent determinacije. Predstavlja udeo varijanse popularnosti koji je objašnjen ovim modelom.
    # mae: Prosečna apsolutna greška u poenima popularnosti (0-100).
    results[name] = {"R2": r2_score(y_val, preds), "MAE": mean_absolute_error(y_val, preds)}

# %%
# STACKING REGRESSOR (Ansambl učenje):
# Pozadinski: Koristi 'Out-of-Fold' predviđanja. Deli trening set na 3 dela (cv=3).
# Svaki bazni model se trenira na 2 dela, a predviđa za 3. deo. 
# Tim predviđanjima se 'hrani' finalni meta-model (LightGBM).
# LightGBM kao meta-model uči koji bazni model je najpouzdaniji za određene tipove pesama.
print("Treniranje STACKING ansambla...")
base_estimators = [(name.lower().replace(" ", "_"), m) for name, m in base_models_dict.items()]
stacking_model = StackingRegressor(
    estimators=base_estimators,
    final_estimator=lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
    cv=3, n_jobs=-1
)
stacking_model.fit(X_train, y_train)
s_preds = stacking_model.predict(X_val)
results["STACKING (FINAL)"] = {"R2": r2_score(y_val, s_preds), "MAE": mean_absolute_error(y_val, s_preds)}

# %%
# Finalni prikaz rezultata projekta
print("\n" + "="*60)
print(f"{'MODEL (AUDIO + ŽANR FOKUS)':<25} | {'R2 SCORE':<10} | {'MAE':<10}")
print("-" * 60)
for name, metrics in results.items():
    print(f"{name:<25} | {metrics['R2']:<10.4f} | {metrics['MAE']:<10.4f}")
print("="*60)

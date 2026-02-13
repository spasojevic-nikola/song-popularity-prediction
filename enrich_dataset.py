"""
Spotify Dataset Enrichment - Add Artist & Track Features via Spotify API
"""

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
from datetime import datetime
import time
from tqdm import tqdm

def retry_on_rate_limit(func, max_retries=3):
    """Retry function with exponential backoff on rate limit"""
    for attempt in range(max_retries):
        try:
            return func()
        except SpotifyException as e:
            if e.http_status == 429:  # Rate limit
                wait_time = int(e.headers.get('Retry-After', 60))
                print(f"\nRate limited! Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                raise
            else:
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return None

# =============================================================================
# SETUP: Get Spotify API credentials from https://developer.spotify.com/
# =============================================================================
CLIENT_ID = 'YOUR_CLIENT_ID'  # Replace with your Client ID
CLIENT_SECRET = 'YOUR_CLIENT_SECRET'  # Replace with your Client Secret

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
))

def get_artist_features(artist_id):
    """Get artist-level features from Spotify API"""
    try:
        artist = retry_on_rate_limit(lambda: sp.artist(artist_id))
        if artist:
            return {
                'artist_popularity': artist['popularity'],
                'artist_followers': artist['followers']['total'],
                'artist_genres': len(artist['genres']),  # Count of genres
            }
    except Exception as e:
        print(f"Error getting artist {artist_id}: {e}")
    
    return {
        'artist_popularity': None,
        'artist_followers': None,
        'artist_genres': None
    }

def get_track_features(track_id):
    """Get track-level features from Spotify API"""
    try:
        track = retry_on_rate_limit(lambda: sp.track(track_id))
        if not track:
            return None
            
        album = track['album']
        
        # Calculate days since release
        release_date = album['release_date']
        if len(release_date) == 4:  # Only year
            release_date += '-01-01'
        elif len(release_date) == 7:  # Year-month
            release_date += '-01'
        
        days_since_release = (datetime.now() - datetime.strptime(release_date, '%Y-%m-%d')).days
        
        # Get primary artist ID
        artist_id = track['artists'][0]['id'] if track['artists'] else None
        
        return {
            'track_id_spotify': track_id,
            'artist_id': artist_id,
            'release_date': release_date,
            'days_since_release': days_since_release,
            'available_markets': len(track['available_markets']),
            'album_total_tracks': album['total_tracks'],
            'album_type': album['album_type'],  # single/album/compilation
        }
    except Exception as e:
        print(f"Error getting track {track_id}: {e}")
        return None

def enrich_dataset(input_csv='dataset.csv', output_csv='dataset_enriched.csv', sample_size=None):
    """
    Enrich dataset with Spotify API data
    
    Args:
        input_csv: Path to original dataset
        output_csv: Path to save enriched dataset
        sample_size: If set, only process first N rows (for testing)
    """
    print("Loading dataset...")
    df = pd.read_csv(input_csv)
    
    if sample_size:
        df = df.head(sample_size)
        print(f"Processing sample of {sample_size} tracks")
    
    print(f"Total tracks: {len(df):,}")
    
    # Initialize new columns
    new_cols = ['artist_id', 'artist_popularity', 'artist_followers', 'artist_genres',
                'release_date', 'days_since_release', 'available_markets', 
                'album_total_tracks', 'album_type']
    
    for col in new_cols:
        df[col] = None
    
    # Track-level enrichment
    print("\nEnriching with track features...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        track_id = row['track_id']
        
        # Get track features
        track_features = get_track_features(track_id)
        if track_features:
            for key, value in track_features.items():
                df.at[idx, key] = value
        
        # Rate limiting (Spotify API: ~180 requests/min)
        # Adjust sleep time: 0.5s = faster but risky, 1s = safe
        if idx % 100 == 0 and idx > 0:
            time.sleep(0.5)  # 0.5s = ~120 req/min, 1s = ~60 req/min
    
    # Artist-level enrichment (deduplicated)
    print("\nEnriching with artist features...")
    unique_artists = df['artist_id'].dropna().unique()
    print(f"Unique artists: {len(unique_artists):,}")
    
    artist_features_cache = {}
    for artist_id in tqdm(unique_artists):
        artist_features_cache[artist_id] = get_artist_features(artist_id)
        time.sleep(0.01)  # Small delay to avoid rate limits
    
    # Map artist features to dataframe
    for idx, row in df.iterrows():
        artist_id = row['artist_id']
        if pd.notna(artist_id) and artist_id in artist_features_cache:
            for key, value in artist_features_cache[artist_id].items():
                df.at[idx, key] = value
    
    # Calculate derived features
    print("\nCalculating derived features...")
    
    # Artist average popularity (from existing data in df)
    artist_avg_pop = df.groupby('artist_id')['popularity'].mean().to_dict()
    df['artist_avg_popularity_historical'] = df['artist_id'].map(artist_avg_pop)
    
    # Track count per artist
    artist_track_count = df.groupby('artist_id').size().to_dict()
    df['artist_track_count_in_dataset'] = df['artist_id'].map(artist_track_count)
    
    # Convert album_type to numeric
    album_type_map = {'single': 0, 'album': 1, 'compilation': 2}
    df['album_type_encoded'] = df['album_type'].map(album_type_map)
    
    # Save enriched dataset
    print(f"\nSaving enriched dataset to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    # Summary
    print("\n" + "="*60)
    print("ENRICHMENT SUMMARY")
    print("="*60)
    print(f"Original features: {len(df.columns) - len(new_cols) - 3}")
    print(f"New features: {len(new_cols) + 3}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nNew feature stats:")
    for col in ['artist_popularity', 'artist_followers', 'days_since_release', 'available_markets']:
        if col in df.columns:
            print(f"  {col:30s}: mean={df[col].mean():.1f}, missing={df[col].isna().sum()}")
    print("="*60)
    
    return df

if __name__ == "__main__":
    # Test with small sample first!
    print("Testing with 100 tracks first...")
    df_test = enrich_dataset(sample_size=100, output_csv='dataset_enriched_test.csv')
    
    # If test works, uncomment below to process full dataset
    # WARNING: This will take ~2-3 hours for 114k tracks
    # df_full = enrich_dataset(output_csv='dataset_enriched.csv')
    
    print("\n✓ Done! Check dataset_enriched_test.csv")
    print("\nTo process full dataset, uncomment the last line in the script.")

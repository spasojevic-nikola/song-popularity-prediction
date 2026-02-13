"""
Spotify Dataset Enrichment with Resume/Checkpoint Support
Can stop anytime (Ctrl+C) and resume later!
"""

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
from datetime import datetime
import time
import json
import os
from tqdm import tqdm

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

CHECKPOINT_FILE = 'enrichment_checkpoint.json'
PROGRESS_FILE = 'enrichment_progress.csv'

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

def save_checkpoint(processed_indices, artist_cache):
    """Save progress to resume later"""
    checkpoint = {
        'processed_indices': list(processed_indices),
        'artist_cache': artist_cache,
        'timestamp': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def load_checkpoint():
    """Load previous progress"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        print(f"\n✓ Found checkpoint from {checkpoint['timestamp']}")
        print(f"  Already processed: {len(checkpoint['processed_indices']):,} tracks")
        return set(checkpoint['processed_indices']), checkpoint['artist_cache']
    return set(), {}

def get_artist_features(artist_id):
    """Get artist-level features from Spotify API"""
    try:
        artist = retry_on_rate_limit(lambda: sp.artist(artist_id))
        if artist:
            return {
                'artist_popularity': artist['popularity'],
                'artist_followers': artist['followers']['total'],
                'artist_genres': len(artist['genres']),
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
            'album_type': album['album_type'],
        }
    except Exception as e:
        print(f"Error getting track {track_id}: {e}")
        return None

def enrich_dataset(input_csv='dataset.csv', output_csv='dataset_enriched.csv', 
                   sample_size=None, checkpoint_interval=100):
    """
    Enrich dataset with Spotify API data (with resume support)
    
    Args:
        input_csv: Path to original dataset
        output_csv: Path to save enriched dataset
        sample_size: If set, only process first N rows (for testing)
        checkpoint_interval: Save progress every N tracks
    """
    print("="*70)
    print("SPOTIFY DATASET ENRICHMENT (with Resume Support)")
    print("="*70)
    print("\n💡 You can stop anytime (Ctrl+C) and resume later!")
    
    # Load checkpoint if exists
    processed_indices, artist_cache = load_checkpoint()
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv(input_csv)
    
    if sample_size:
        df = df.head(sample_size)
        print(f"Processing sample of {sample_size} tracks")
    
    print(f"Total tracks: {len(df):,}")
    
    # Load progress file if exists
    if os.path.exists(PROGRESS_FILE):
        print(f"\n✓ Found progress file: {PROGRESS_FILE}")
        df_progress = pd.read_csv(PROGRESS_FILE)
        print(f"  Loaded {len(df_progress):,} previously enriched tracks")
        
        # Merge with existing data
        df = df.merge(df_progress, on='track_id', how='left', suffixes=('', '_enriched'))
        
        # Use enriched columns if available
        enriched_cols = [c for c in df.columns if c.endswith('_enriched')]
        for col in enriched_cols:
            base_col = col.replace('_enriched', '')
            if base_col not in df.columns:
                df[base_col] = df[col]
            else:
                df[base_col] = df[col].combine_first(df[base_col])
            df.drop(columns=[col], inplace=True)
    
    # Initialize new columns if they don't exist
    new_cols = ['artist_id', 'artist_popularity', 'artist_followers', 'artist_genres',
                'release_date', 'days_since_release', 'available_markets', 
                'album_total_tracks', 'album_type']
    
    for col in new_cols:
        if col not in df.columns:
            df[col] = None
    
    # Track-level enrichment
    remaining = len(df) - len(processed_indices)
    print(f"\n{'='*70}")
    print(f"Progress: {len(processed_indices):,} / {len(df):,} tracks processed")
    print(f"Remaining: {remaining:,} tracks")
    print(f"{'='*70}")
    
    if remaining == 0:
        print("\n✓ All tracks already processed!")
        df.to_csv(output_csv, index=False)
        return df
    
    print("\nEnriching with track features...")
    print("(Press Ctrl+C to pause and save progress)\n")
    
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), initial=len(processed_indices)):
            # Skip if already processed
            if idx in processed_indices:
                continue
            
            track_id = row['track_id']
            
            # Get track features
            track_features = get_track_features(track_id)
            if track_features:
                for key, value in track_features.items():
                    df.at[idx, key] = value
                
                # Cache artist for later
                if track_features['artist_id']:
                    artist_cache[track_features['artist_id']] = None  # Placeholder
            
            # Mark as processed
            processed_indices.add(idx)
            
            # Save checkpoint periodically
            if len(processed_indices) % checkpoint_interval == 0:
                print(f"\n💾 Checkpoint: Saving progress ({len(processed_indices):,} tracks)...")
                df.to_csv(PROGRESS_FILE, index=False)
                save_checkpoint(processed_indices, artist_cache)
                print("✓ Progress saved! Safe to stop now.")
            
            # Rate limiting
            if idx % 100 == 0 and idx > 0:
                time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  INTERRUPTED by user!")
        print("💾 Saving progress...")
        df.to_csv(PROGRESS_FILE, index=False)
        save_checkpoint(processed_indices, artist_cache)
        print(f"✓ Progress saved to: {PROGRESS_FILE}")
        print(f"✓ Checkpoint saved to: {CHECKPOINT_FILE}")
        print(f"\nProcessed: {len(processed_indices):,} / {len(df):,} tracks")
        print("\n🔄 To resume, just run the script again!")
        return df
    
    # Artist-level enrichment
    print("\n\nEnriching with artist features...")
    unique_artists = df['artist_id'].dropna().unique()
    print(f"Unique artists: {len(unique_artists):,}")
    
    # Check which artists already cached
    new_artists = [a for a in unique_artists if a not in artist_cache or artist_cache[a] is None]
    print(f"New artists to fetch: {len(new_artists):,}")
    
    artist_features_cache = artist_cache.copy()
    
    try:
        for artist_id in tqdm(new_artists):
            artist_features_cache[artist_id] = get_artist_features(artist_id)
            time.sleep(0.01)
            
            # Save checkpoint for artists too
            if len(artist_features_cache) % checkpoint_interval == 0:
                save_checkpoint(processed_indices, artist_features_cache)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  INTERRUPTED by user!")
        print("💾 Saving progress...")
        save_checkpoint(processed_indices, artist_features_cache)
        df.to_csv(PROGRESS_FILE, index=False)
        print("\n🔄 To resume, just run the script again!")
        return df
    
    # Map artist features to dataframe
    print("\nMapping artist features...")
    for idx, row in df.iterrows():
        artist_id = row['artist_id']
        if pd.notna(artist_id) and artist_id in artist_features_cache:
            features = artist_features_cache[artist_id]
            if features:
                for key, value in features.items():
                    df.at[idx, key] = value
    
    # Calculate derived features
    print("Calculating derived features...")
    
    # Artist average popularity
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
    
    # Clean up checkpoint files
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("✓ Removed checkpoint file (no longer needed)")
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("✓ Removed progress file (no longer needed)")
    
    # Summary
    print("\n" + "="*70)
    print("✅ ENRICHMENT COMPLETE!")
    print("="*70)
    print(f"Total tracks: {len(df):,}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nNew feature stats:")
    for col in ['artist_popularity', 'artist_followers', 'days_since_release', 'available_markets']:
        if col in df.columns:
            print(f"  {col:30s}: mean={df[col].mean():.1f}, missing={df[col].isna().sum()}")
    print("="*70)
    
    return df

if __name__ == "__main__":
    # Test with small sample first!
    print("Testing with 100 tracks first...")
    df_test = enrich_dataset(sample_size=100, output_csv='dataset_enriched_test.csv')
    
    # If test works, uncomment below to process full dataset
    # df_full = enrich_dataset(output_csv='dataset_enriched.csv')
    
    print("\n✓ Done! Check dataset_enriched_test.csv")
    print("\n📝 To process full dataset:")
    print("   1. Uncomment the line: df_full = enrich_dataset(...)")
    print("   2. Run: python enrich_dataset.py")
    print("   3. Stop anytime (Ctrl+C) and resume later!")

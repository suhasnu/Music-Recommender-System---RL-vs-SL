import pandas as pd
import json
from collections import defaultdict

# Load the CSV file
df = pd.read_csv("data/top_10000_1960-now.csv")  # Replace with your actual filename

# Create a dictionary to store songs for each genre
genre_to_songs = defaultdict(list)

# Iterate over the DataFrame
for _, row in df.iterrows():
    track_name = str(row["Track Name"]).strip()
    artist_names = str(row["Artist Name(s)"]).strip()
    genres_str = str(row["Artist Genres"])

    # Skip if any required field is missing
    if pd.isna(track_name) or pd.isna(artist_names) or pd.isna(genres_str):
        continue

    # Combine track and artist
    song = f"{track_name} - {artist_names}"

    # Split and clean genres
    try:
        genres = eval(genres_str) if genres_str.startswith("[") else genres_str.split(",")
    except:
        genres = genres_str.split(",")

    genres = [g.strip().lower() for g in genres if g.strip()]

    # Add the song to each genre
    for genre in genres:
        genre_to_songs[genre].append(song)

# Convert defaultdict to regular dict
genre_to_songs = dict(genre_to_songs)

# Save to JSON file
with open("genre_to_songs.json", "w", encoding="utf-8") as f:
    json.dump(genre_to_songs, f, indent=2, ensure_ascii=False)

# (Optional) Print a sample
for genre, songs in list(genre_to_songs.items())[:5]:
    print(f"🎧 {genre}: {songs[:3]}")

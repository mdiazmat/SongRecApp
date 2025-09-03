import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# --- App constants---
min_results = 10
max_results = 25
# This expands on some genres
genre_synonyms = {
    "pop": ["pop", "power-pop", "synth-pop"],
    "electronic": ["electronic", "electro", "edm", "house", "techno",
                   "detroit-techno", "minimal-techno", "progressive-house",
                   "dance", "club"],
    "r-n-b": ["r-n-b", "soul", "romance"],
    "latin": ["latin", "latino", "reggaeton", "salsa", "brazil"],
}

# --- Load data ---
df = pd.read_csv("dataset.csv", quotechar='"', on_bad_lines='skip', encoding='utf-8')
df.drop_duplicates(subset=["track_name", "artists"], inplace=True)
df.reset_index(drop=True, inplace=True)

# --- Checks audio features to be numeric and drop any NaNs ---
num_cols = ['tempo', 'energy', 'valence', 'danceability', 'acousticness']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=num_cols, how='any', inplace=True)
for c in ['track_name', 'album_name', 'artists', 'track_genre']: # checks that text cols are str to prevent .str errors
    if c in df.columns:
        df[c] = df[c].astype(str)

# --- Feature selection ---
features = num_cols
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

# --- Define user playlists ---
playlists = {
    "R&B": df[(df['track_genre'] == 'r-n-b') & (df['tempo'] >= 70) & (df['tempo'] <= 110)].head(10),
    "Hype Rap": df[(df['track_genre'] == 'hip-hop') & (df['energy'] >= 0.7)].head(10),
    "Beach Day": df[(df['track_genre'].isin(['reggae', 'latin', 'chill'])) & (df['valence'] > 0.6)].head(10),
    "Party": df[(df['track_genre'].isin(['party', 'dance', 'edm', 'club'])) & (df['energy'] >=0.7) & (df['tempo'] >=110)].head(10),
    "Chill Date Night": df[(df['track_genre'].isin(['romance', 'r-n-b', 'soul'])) & (df['tempo'] <= 100) & (df['energy'] <= 0.6) & (df['valence'] >= 0.4)].head(10),
    "Latino Cleaning": df[(df['track_genre'].isin(['latino', 'salsa', 'reggaeton'])) & (df['tempo'] >= 100) & (df['valence'] >= 0.6)].head(10)
}

# --- Prepare user history ---
user_history = pd.concat(playlists.values(), ignore_index=True)
user_scaled = scaler.transform(user_history[features])

# --- Collaborative Filtering ---
knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_model.fit(df_scaled)

collab_recs = []
for song_vec in user_scaled:
    distances, indices = knn_model.kneighbors([song_vec])
    for idx in indices[0]: 
        collab_recs.append(idx)

collab_scores = pd.Series(collab_recs).value_counts().head(50)
collab_df = df.loc[collab_scores.index].copy()
collab_df['collab_score'] = collab_scores.values

# --- Content-Based Filtering ---
content_similarity = cosine_similarity(user_scaled, df_scaled)
content_scores = content_similarity.mean(axis=0)
content_df = df.copy()
content_df['content_score'] = content_scores

# --- Combine for Hybrid ---
hybrid_df = df.copy()
hybrid_df['content_score'] = content_scores
hybrid_df['collab_score'] = hybrid_df.index.map(collab_scores).fillna(0).astype(float) # makes sure collab_score is float-safe
hybrid_df['hybrid_score'] = hybrid_df['content_score'] + hybrid_df['collab_score']
hybrid_df.sort_values(by='hybrid_score', ascending=False, inplace=True)
hybrid_df.drop_duplicates(subset=['track_name', 'artists'], inplace=True)

# --- Groq prompt to features ---
def get_features_from_prompt(prompt):
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "temperature": 0.2, 
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Given the user prompt: '{prompt}', extract ideal Spotify audio features and mood keywords. 
Return ONLY a valid JSON object with two keys: 'audioFeatures' and 'keywords'. 
Do not include any explanations or markdown formatting. The structure must be:
{{
  "audioFeatures": {{
    "tempo": [min, max],
    "energy": [min, max],
    "valence": [min, max],
    "danceability": [min, max],
    "acousticness": [min, max],
    "genre": ["genre1", "genre2"] #trailing comma removed

  }},
  "keywords": ["keyword1", "keyword2"]
}}
"""
                }
            ]
        }
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()
        reply = res.json()['choices'][0]['message']['content']
        match = re.search(r'\{.*\}', reply, re.DOTALL)
        if match:
            st.code(reply)
            return json.loads(match.group())
        else:
            st.error("Could not find JSON in the response.")
            return None
    
    except Exception as e:
        st.error("Groq API error:" + str(e))
        return None

# --- Generate playlist ---
def generate_hybrid_playlist_from_prompt(prompt, df):
    prefs = get_features_from_prompt(prompt)
    if prefs is None:
        return pd.DataFrame()

    filtered = df.copy()
    audio_feats = prefs.get("audioFeatures", {}) or {}
    keywords = prefs.get("keywords", []) or []

    # ----- Step 1) Genre handling (map "genre" -> track_genre with synonyms) ------
    genres = audio_feats.pop("genre", None)
    if genres:
        if isinstance(genres, str):
            genres = [genres]
        genres = [str(g).lower().strip() for g in genres if g]

        expanded = set()
        for g in genres:
            expanded.add(g)
            for s in genre_synonyms.get(g, []):
                expanded.add(s.lower())

        if expanded:
            gcol = filtered["track_genre"].astype(str).str.lower()
            pattern = r"(" + "|".join(re.escape(x) for x in expanded) + r")"
            gmask = gcol.str.contains(pattern, na=False, regex=True)
            # apply genre filter only if it leaves enough songs
            if gmask.sum() >= min_results:   # <-- use lowercase constant
                filtered = filtered[gmask]

    # ----- Step 2) Numeric features -----
    for f in ["tempo", "energy", "valence", "danceability", "acousticness"]:
        if f in audio_feats:
            val = audio_feats[f]
            if isinstance(val, list) and len(val) == 2:
                lo, hi = val
                if f == "tempo":
                    lo, hi = float(lo), float(hi)
                else:
                    lo, hi = max(0.0, float(lo)), min(1.0, float(hi))
                filtered = filtered[filtered[f].between(lo, hi)]
            elif isinstance(val, (int, float, np.floating)):
                tol = 3.0 if f == "tempo" else 0.05
                filtered = filtered[filtered[f].between(val - tol, val + tol)]

    # ----- Step 3) Keywords (soft: filter only if generous; else boost) ------
    kw_mask = None
    if keywords:
        kw_mask = pd.Series(False, index=filtered.index)
        text_cols = ['track_name', 'album_name', 'artists', 'track_genre']
        for kw in keywords:
            kwl = str(kw).lower()
            for col in text_cols:
                kw_mask |= filtered[col].astype(str).str.lower().str.contains(kwl, na=False)

        if kw_mask.sum() >= min_results:
            filtered = filtered[kw_mask]
        else:
            # soft boost: nudge up rows that match keywords
            if "hybrid_score" in filtered.columns:
                filtered.loc[kw_mask, "hybrid_score"] = filtered.loc[kw_mask, "hybrid_score"] + 0.25

    # ----- Step 4) Guarantee at least min_results by widening once if needed -----
    def _widen_once(base_df):
        widened = base_df.copy()
        for f in ["tempo", "energy", "valence", "danceability", "acousticness"]:
            val = audio_feats.get(f)
            if isinstance(val, list) and len(val) == 2:
                lo, hi = val
                if f == "tempo":
                    lo, hi = float(lo) - 10, float(hi) + 10
                else:
                    lo, hi = max(0.0, float(lo) - 0.1), min(1.0, float(hi) + 0.1)
                widened = widened[widened[f].between(lo, hi)]
        return widened

    if len(filtered) < min_results:
        # start from original df to widen ranges afresh
        filtered_try = _widen_once(df)
        if genres:
            gcol = filtered_try["track_genre"].astype(str).str.lower()
            all_terms = set(genres)
            for g in genres:
                all_terms.update(genre_synonyms.get(g, []))
            pattern = r"(" + "|".join(re.escape(x) for x in all_terms) + r")"
            gmask2 = gcol.str.contains(pattern, na=False, regex=True)
            if gmask2.sum() >= min_results:
                filtered_try = filtered_try[gmask2]
        if len(filtered_try) > len(filtered):
            filtered = filtered_try

    # ----- Step 5) Sort and cap -----
    if "hybrid_score" in filtered.columns:
        filtered = filtered.sort_values("hybrid_score", ascending=False)
    else:
        filtered = filtered.sort_values(["energy", "danceability", "valence"], ascending=False)

    return filtered.drop_duplicates(subset=['track_name', 'artists']).head(max_results)


# --- Streamlit UI --- 
st.set_page_config(page_title="🎵 Playlist Recommender") 
st.title("🎵 Playlist Recommender")

st.markdown("### 📂 Your Playlists")

for name, playlist_df in playlists.items():
    with st.expander(f"{name} Playlist"): 
        st.dataframe(playlist_df[['track_name', 'artists', 'track_genre']], use_container_width=True)
        
st.markdown("---") 
st.subheader("✨ Generate a New Playlist") 
prompt = st.text_input("Describe the kind of playlist you want:", placeholder="e.g. sad r&b, workout mix, beach day") 

if st.button("🎧 Generate"):
    with st.spinner("Generating your playlist..."):
        playlist = generate_hybrid_playlist_from_prompt(prompt, hybrid_df) 
        if not playlist.empty:
            st.success("Here's you playlist!")
            st.dataframe(playlist[['track_name', 'artists', 'track_genre']])
        else:
            st.warning("No matching songs found.")

# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZLVSz60WXidtkmz9EM1fhTQ3i_pR0ADv
"""

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

# --- Load data ---
df = pd.read_csv("dataset.csv", quotechar='"', on_bad_lines='skip', encoding='utf-8')
df.drop_duplicates(subset=["track_name", "artists"], inplace=True)
df.reset_index(drop=True, inplace=True)

# --- Feature selection ---
features = ['tempo', 'energy', 'valence', 'danceability', 'acousticness']
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

# --- Define user playlists ---
playlists = {
    "R&B": df[(df['track_genre'] == 'r-n-b') & (df['tempo'] >= 70) & (df['tempo'] <= 110)].head(10),
    "Hype Rap": df[(df['track_genre'] == 'hip-hop') & (df['energy'] >= 0.7)].head(10),
    "Beach Day": df[(df['track_genre'].isin(['reggae', 'latin', 'chill'])) & (df['valence'] > 0.6)].head(10),
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
    for idx in indices[0][1:]:  # skip the first, it's the song itself
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
hybrid_df['collab_score'] = hybrid_df.index.map(collab_scores).fillna(0)
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
            "model": "llama3-8b-8192",
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
    "genre": ["genre1", "genre2"],

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
    audio_feats = prefs.get("audioFeatures", {})
    keywords = prefs.get("keywords", [])

    for feature, value in audio_feats.items():
        if feature in df.columns:
            if isinstance(value, list) and len(value) == 2:
                filtered = filtered[(filtered[feature] >= value[0]) & (filtered[feature] <= value[1])]
            elif isinstance(value, (int, float)):
                filtered = filtered[np.isclose(filtered[feature], value, atol=0.1)]
            elif isinstance(value, str):
                filtered = filtered[filtered[feature].str.lower().str.contains(value.lower())]

    if keywords:
        keyword_mask = pd.Series(False, index=filtered.index)
        text_columns = ['track_name', 'album_name', 'artists', 'track_genre']
        for kw in keywords:
            for col in text_columns:
                keyword_mask |= filtered[col].str.lower().str.contains(kw.lower(), na=False)
        filtered = filtered[keyword_mask]

    return filtered.head(15)

# --- Streamlit UI ---
st.set_page_config(page_title="🎵 Playlist Recommender")
st.title("🎵 Playlist Recommender")

st.markdown("### 📂 Your Playlists")

for name, playlist_df in playlists.items():
    with st.expander(f"{name} Playlist"):
         st.dataframe(playlist_df[['track_name', 'artists']])

st.markdown("---")
st.subheader("✨ Generate a New Playlist")
prompt = st.text_input("Describe the kind of playlist you want:", placeholder="e.g. sad r&b, workout mix, beach day")

if st.button("🎧 Generate"):
    with st.spinner("Generating your playlist..."):
        playlist = generate_hybrid_playlist_from_prompt(prompt, hybrid_df)
        if not playlist.empty:
            st.success("Here's your playlist!")
            st.dataframe(playlist[['track_name', 'artists', 'track_genre']])
        else:
            st.warning("No matching songs found.")
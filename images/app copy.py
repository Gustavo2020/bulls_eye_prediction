import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import unicodedata
from datetime import datetime

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="Bulls Eye: Predictive Engine")

# --- PATH LOGIC ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "..", "images")
JSON_PATH = os.path.join(BASE_DIR, "..", "model", "latest_results.json")
CSV_PATH = os.path.join(BASE_DIR, "..", "processed_data", "bulls_master_timeseries.csv")

def get_img(name):
    return os.path.join(IMAGE_PATH, name)

def normalize_str(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii').replace(' ', '_')

@st.cache_data
def load_prediction_data():
    with open(JSON_PATH, 'r') as f:
        return json.load(f)

@st.cache_data
def load_historical_data():
    return pd.read_csv(CSV_PATH)

# Load Data Sources
try:
    data_json = load_prediction_data()
    master_df = load_historical_data()
except Exception as e:
    st.error(f"Error loading data sources: {e}")
    st.stop()

# --- CUSTOM CSS (Bulls Red Theme) ---
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    
    /* Header Images Scaling */
    div.stImage > img { 
        margin-bottom: -5px; 
        max-height: 50px; 
        object-fit: cover; 
        width: 100%; 
    }
    div.stImage:first-child > img { max-height: 300px; }

    /* Cards Styling (Next Game & Players) */
    .metric-card {
        background-color: #CE1141;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 0.9rem; text-transform: uppercase; font-weight: bold; opacity: 0.9; }
    .metric-value { font-size: 2.2rem; font-weight: bold; }

    .player-card {
        background-color: #CE1141;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 10px;
        min-height: 160px;
    }
    .player-name { font-weight: bold; font-size: 1.1rem; margin-bottom: 5px; }
    .player-points { font-size: 2.2rem; font-weight: bold; }
    .player-status { font-style: italic; font-size: 0.9rem; margin-top: 5px; }
    
    hr { margin-top: 1rem; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. HEADER ---
st.image(get_img("head_banner.png"), use_container_width=True)

# --- 2. NEXT GAME PREDICTION ---
st.image(get_img("next_game_prediction.png"), use_container_width=True)

game_info = data_json['game_info']
team_pred = data_json['team_prediction']

# Heading: Increased size and bold
st.markdown(f"""
    <h1 style='color: #CE1141; text-align: center; font-size: 3rem; font-weight: bold; margin-bottom: 20px;'>
        Next Game: vs {game_info['opponent']} | {game_info['date']} ({game_info['venue']})
    </h1>
    """, unsafe_allow_html=True)

m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Team Verdict</div>
            <div class="metric-value">{team_pred['verdict']}</div>
        </div>
    """, unsafe_allow_html=True)
with m_col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Win Probability</div>
            <div class="metric-value">{int(team_pred['prob_final'] * 100)}%</div>
        </div>
    """, unsafe_allow_html=True)
with m_col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rival Strength</div>
            <div class="metric-value">{game_info['opp_strength']}</div>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# --- 3. INDIVIDUAL PERFORMANCE FORECAST ---
st.image(get_img("individual_performance.png"), use_container_width=True)

players_data = data_json['players']
p_cols = st.columns(len(players_data))

for i, p in enumerate(players_data):
    with p_cols[i]:
        st.markdown(f"""
            <div class="player-card">
                <div class="player-name">{p['name']}</div>
                <div class="player-points">{p['forecast']}</div>
                <div style="font-size: 0.8rem; opacity:0.9;">Predicted Pts</div>
                <div class="player-status">Streak: {p['streak']}</div>
            </div>
        """, unsafe_allow_html=True)

st.divider()

# --- 4 & 5. LAST 10 GAMES TREND ---
st.image(get_img("last_10_games.png"), use_container_width=True)

cols_to_plot = []
normalized_master_cols = {normalize_str(col): col for col in master_df.columns}

for p in players_data:
    search_key = f"{normalize_str(p['name'])}_PTS"
    if search_key in normalized_master_cols:
        cols_to_plot.append(normalized_master_cols[search_key])

if cols_to_plot:
    trend_df = master_df[['GAME_DATE'] + cols_to_plot].tail(10).set_index('GAME_DATE')
    trend_df.columns = [c.replace('_PTS', '').replace('_', ' ') for c in trend_df.columns]
    bulls_colors = ["#CE1141", "#578234", "#AFA627", "#30306F", "#AAAAAA", "#C12EAB"]
    st.line_chart(trend_df, color=bulls_colors[:len(trend_df.columns)])
else:
    st.warning("No matching historical data found.")

st.divider()

# --- 6. PREDICTED POINT DISTRIBUTION ---
st.image(get_img("predicted_points_distribution.png"), use_container_width=True)

p_names = [p['name'] for p in players_data]
p_forecasts = [p['forecast'] for p in players_data]

empty_left, pie_center, empty_right = st.columns([1, 2, 1])
with pie_center:
    fig, ax = plt.subplots(figsize=(5, 5))
    bulls_palette = ['#CE1141', '#8C0D2C', '#000000', '#444444', '#666666', '#999999']
    ax.pie(p_forecasts, labels=p_names, autopct='%1.1f%%', startangle=90, 
           colors=bulls_palette[:len(p_names)], textprops={'color':"w"})
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    st.pyplot(fig)

st.divider()

# --- 7 to 12. PIPELINES AND TEAM ---
st.image(get_img("data_transformation_pipeline.png"), use_container_width=True)
st.image(get_img("data_eng_pipeline.png"), use_container_width=True)
st.divider()
st.image(get_img("ml_operations.png"), use_container_width=True)
st.image(get_img("model_performance_table.png"), use_container_width=True)
st.divider()
st.image(get_img("team.png"), use_container_width=True)

t_col1, t_col2, t_col3 = st.columns(3)
with t_col1:
    st.image(get_img("team_anu_lego.png"), caption="Anu - Data Scientist")
with t_col2:
    st.image(get_img("team_gus_lego.png"), caption="Gus - Data Architect")
with t_col3:
    st.image(get_img("team_gabe_lego.png"), caption="Gabe - ML Engineer")
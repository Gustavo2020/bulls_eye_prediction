import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import webbrowser
import time
import streamlit.components.v1 as components
import unicodedata
from datetime import datetime

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="Bulls Eye: Predictive Engine")

# --- PATH LOGIC ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "images")
JSON_10_LAST_RESULTS_PATH = os.path.join(BASE_DIR, "model", "bulls_dashboard.json")
JSON_QUANTILES_PATH = os.path.join(BASE_DIR, "model", "bulls_dashboard_quantile.json")
JSON_VICTORY_PREDICTION_PATH = os.path.join(BASE_DIR, "model", "win_probability.json")
CSV_PATH = os.path.join(BASE_DIR, "model", "predictions_history_quantile.csv")

def get_img(name):
    return os.path.join(IMAGE_PATH, name)

def normalize_str(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii').replace(' ', '_')

@st.cache_data
def load_prediction_data():
    with open(JSON_10_LAST_RESULTS_PATH, 'r') as f:
        return json.load(f)

@st.cache_data
def load_victory_data():
    with open(JSON_VICTORY_PREDICTION_PATH, 'r') as f:
        return json.load(f)
    
@st.cache_data
def load_quantiles_data():

    with open(JSON_QUANTILES_PATH, 'r') as f:
        return json.load(f)

@st.cache_data
def load_historical_data():
    return pd.read_csv(CSV_PATH)

# Load Data Sources
try:
    data_json = load_prediction_data()
    win_data  = load_victory_data()
    quantiles = load_quantiles_data()
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

game_info = win_data
team_pred = win_data

# Heading: Increased size and bold
st.markdown(f"""
    <h1 style='color: #CE1141; text-align: center; font-size: 3rem; font-weight: bold; margin-bottom: 20px;'>
        Next Game: vs {game_info['opponent']} | {game_info['game_date']} ({'Home' if game_info['is_home'] == 1 else 'Away'})
    </h1>
    """, unsafe_allow_html=True)

m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Team Verdict</div>
            <div class="metric-value">{'Win' if team_pred['predicted_result'] == 'W' else 'Loss'}</div>
        </div>
    """, unsafe_allow_html=True)
with m_col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Win Probability</div>
            <div class="metric-value">{int(team_pred['win_probability'] * 100)}%</div>
        </div>
    """, unsafe_allow_html=True)
with m_col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rival Strength</div>
            <div class="metric-value">{game_info['opponent_recent_form']['last_5_record']}</div>
        </div>
    """, unsafe_allow_html=True)

# --- Last 10 team-level predictions ---
_df_all = master_df.copy()
_df_all['GAME_DATE'] = pd.to_datetime(_df_all['GAME_DATE'])

# CHI: sum predicted and actual PTS per game
_chi_agg = (
    _df_all[_df_all['TEAM_ABBREVIATION'] == 'CHI']
    .drop_duplicates(subset=['GAME_ID', 'PLAYER_NAME'])
    .groupby(['GAME_ID', 'GAME_DATE'], as_index=False)
    .agg(CHI_Pred=('PTS_Q50', 'sum'), CHI_Actual=('PTS', 'sum'))
    .sort_values('GAME_DATE', ascending=False)
    .head(10)
)

# Opponent: sum actual PTS per game (all non-CHI rows sharing same GAME_ID)
_opp_agg = (
    _df_all[_df_all['TEAM_ABBREVIATION'] != 'CHI']
    .drop_duplicates(subset=['GAME_ID', 'PLAYER_NAME'])
    .groupby('GAME_ID', as_index=False)
    .agg(OPP_Actual=('PTS', 'sum'))
)

_team_tbl = _chi_agg.merge(_opp_agg, on='GAME_ID', how='left')

# Derive Win/Lose columns
_team_tbl['Pred. W/L'] = np.where(_team_tbl['CHI_Pred'] > _team_tbl['OPP_Actual'], 'Win', 'Lose')
_team_tbl['Act. W/L']  = np.where(_team_tbl['CHI_Actual'] > _team_tbl['OPP_Actual'], 'Win', 'Lose')
_team_tbl['Predicted'] = _team_tbl['CHI_Pred'].round(1)
_team_tbl['Actual']    = _team_tbl['CHI_Actual'].round(0).astype(int)
_team_tbl['Date']      = _team_tbl['GAME_DATE'].dt.strftime('%m/%d/%y')

_, tbl_center, _ = st.columns([1, 3, 1])
with tbl_center:
    st.markdown("<p style='text-align:center;font-weight:bold;font-size:1rem;margin-top:12px;'>Last 10 Team Predictions</p>", unsafe_allow_html=True)
    st.dataframe(
        _team_tbl[['Date', 'Predicted', 'Actual', 'Pred. W/L', 'Act. W/L']].reset_index(drop=True),
        hide_index=True,
        use_container_width=True,
    )

st.divider()

# --- 3. INDIVIDUAL PERFORMANCE FORECAST ---
st.image(get_img("individual_performance.png"), use_container_width=True)

players_data = data_json['next_game']['predictions'][:6]
p_cols = st.columns(len(players_data))

for i, p in enumerate(players_data):
    with p_cols[i]:
        st.markdown(f"""
            <div class="player-card">
                <div class="player-name">{p['player_name']}</div>
                <div class="player-points-predicted">{p['pts_predicted']}</div>
                <div style="font-size: 1.0rem; opacity:0.9;">Predicted Pts</div>
            </div>
        """, unsafe_allow_html=True)

st.divider()

# --- 4 & 5. LAST 10 GAMES TREND ---
st.image(get_img("last_10_games.png"), use_container_width=True)

chi_history = master_df[master_df['TEAM_ABBREVIATION'] == 'CHI'].copy()
chi_history['GAME_DATE'] = pd.to_datetime(chi_history['GAME_DATE'])

for row_start in range(0, 6, 3):
    row_players = players_data[row_start:row_start + 3]
    cols = st.columns(len(row_players))
    for col, p in zip(cols, row_players):
        player_history = (
            chi_history[chi_history['PLAYER_NAME'] == p['player_name']]
            .drop_duplicates(subset=['GAME_ID'])
            .sort_values('GAME_DATE', ascending=False)
            .drop_duplicates(subset=['GAME_DATE'])
            .head(10)
        )
        # Hot/Cold badge: compare last-3 avg vs last-10 avg
        form_tag = ""
        if not player_history.empty:
            pts_last_3  = player_history['PTS'].head(3).mean()
            pts_last_10 = player_history['PTS'].mean()
            if pts_last_10 > 0:
                ratio = pts_last_3 / pts_last_10
                if ratio >= 1.15:
                    form_tag = "<span style='background:#E85D04;color:white;border-radius:4px;padding:2px 7px;font-size:0.75rem;margin-left:8px;font-weight:bold;'>HOT</span>"
                elif ratio <= 0.85:
                    form_tag = "<span style='background:#3A86FF;color:white;border-radius:4px;padding:2px 7px;font-size:0.75rem;margin-left:8px;font-weight:bold;'>COLD</span>"
                else:
                    form_tag = "<span style='background:#555555;color:white;border-radius:4px;padding:2px 7px;font-size:0.75rem;margin-left:8px;font-weight:bold;'>AVG</span>"

        player_df = (
            player_history[['GAME_DATE', 'PTS_Q50', 'PTS', 'PTS_ERROR']]
            .rename(columns={
                'GAME_DATE': 'Date',
                'PTS_Q50':   'Predicted',
                'PTS':       'Actual',
                'PTS_ERROR': 'Diff',
            })
            .assign(Date=lambda x: x['Date'].dt.strftime('%m/%d'))
            .reset_index(drop=True)
        )
        with col:
            st.markdown(
                f"<div style='font-weight:bold;font-size:1rem;margin-bottom:4px;'>"
                f"{p['player_name']}{form_tag}</div>",
                unsafe_allow_html=True,
            )
            if player_df.empty:
                st.warning("No data")
            else:
                st.dataframe(player_df, hide_index=True, use_container_width=True)

st.divider()

# --- 6. PREDICTED POINT DISTRIBUTION ---
st.image(get_img("predicted_points_distribution.png"), use_container_width=True)

p_names     = [p['player_name']   for p in players_data]
p_forecasts = [p['pts_predicted'] for p in players_data]

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
_pipeline_html_path   = os.path.join(BASE_DIR, "images", "bulls_pipeline.html")
_victory_html_path    = os.path.join(BASE_DIR, "images", "bulls_win_probability_pipeline.html")
_disclaimer_html_path = os.path.join(BASE_DIR, "images", "bulls_eye_disclaimer.html")
st.markdown("""
    <style>
    div[data-testid="stButton"] > button {
        font-size: 2rem !important;
        padding: 24px 0 !important;
        border-radius: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Debounce: evitar doble-open por doble rerun de Streamlit
def _open_once(key, path):
    last = st.session_state.get(f"_btn_ts_{key}", 0)
    if time.time() - last > 1.5:
        webbrowser.open(f"file://{path}")
        st.session_state[f"_btn_ts_{key}"] = time.time()

_, btn_c1, btn_c2, btn_c3, _ = st.columns([1, 2, 2, 2, 1])
with btn_c1:
    if st.button("Individual Forecast Pipeline", type="primary", use_container_width=True):
        _open_once("pipeline", _pipeline_html_path)
with btn_c2:
    if st.button("Victory Pipeline Forecast", type="secondary", use_container_width=True):
        _open_once("victory", _victory_html_path)
with btn_c3:
    if st.button("Disclaimer", type="secondary", use_container_width=True):
        _open_once("disclaimer", _disclaimer_html_path)

components.html("""
<script>
  const applyStyles = () => {
    const btns = window.parent.document.querySelectorAll('button');
    btns.forEach(btn => {
      const txt = btn.innerText.trim();
      if (txt === 'Victory Pipeline Forecast') {
        btn.style.setProperty('background-color', '#ffffff', 'important');
        btn.style.setProperty('color', '#CE1141', 'important');
        btn.style.setProperty('border', '2px solid #CE1141', 'important');
      } else if (txt === 'Disclaimer') {
        btn.style.setProperty('background-color', '#000000', 'important');
        btn.style.setProperty('color', '#ffffff', 'important');
        btn.style.setProperty('border', '1px solid #000000', 'important');
      }
    });
  };
  applyStyles();
  const observer = new MutationObserver(applyStyles);
  observer.observe(window.parent.document.body, { childList: true, subtree: true });
</script>
""", height=0)

st.divider()
st.image(get_img("team.png"), use_container_width=True)

t_col1, t_col2, t_col3 = st.columns(3)
with t_col1:
    st.image(get_img("team_anu_lego.png"), caption="Anu - Data Scientist")
with t_col2:
    st.image(get_img("team_gus_lego.png"), caption="Gus - Data Architect")
with t_col3:
    st.image(get_img("team_gabe_lego.png"), caption="Gabe - ML Engineer")
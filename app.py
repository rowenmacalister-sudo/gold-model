import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# --- NEURAL ENGINE ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input, BatchNormalization, Conv1D, MaxPooling1D

# ─────────────────────────────────────────────
#  KONFIGURASI HALAMAN & DESIGN SYSTEM
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AURUM · XAU/USD Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Palette: Deep Obsidian + Liquid Gold + Platinum Silver
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=DM+Mono:wght@300;400&family=Inter:wght@300;400;500&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #0A0A0F;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(194,153,67,0.08) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(194,153,67,0.04) 0%, transparent 60%);
    color: #E8E0D0;
    font-family: 'Inter', sans-serif;
}

/* ── Header ── */
.aurum-header {
    text-align: center;
    padding: 3rem 1rem 2rem;
    border-bottom: 1px solid rgba(194,153,67,0.15);
    margin-bottom: 2.5rem;
    position: relative;
}
.aurum-header::before {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 50%;
    transform: translateX(-50%);
    width: 200px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #C29943, transparent);
}
.aurum-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.2rem;
    font-weight: 300;
    letter-spacing: 0.25em;
    color: #C29943;
    text-transform: uppercase;
    line-height: 1;
}
.aurum-tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.4em;
    color: #7A7060;
    text-transform: uppercase;
    margin-top: 0.6rem;
}
.aurum-model-badge {
    display: inline-block;
    margin-top: 1rem;
    padding: 0.25rem 1rem;
    border: 1px solid rgba(194,153,67,0.3);
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #C29943;
    background: rgba(194,153,67,0.05);
}

/* ── Status indicator ── */
.status-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    justify-content: center;
    margin-bottom: 2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    color: #7A9E7E;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #7A9E7E;
    box-shadow: 0 0 8px #7A9E7E;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(194,153,67,0.12);
    border-radius: 4px;
    padding: 1.6rem;
    position: relative;
    transition: border-color 0.3s;
}
.card:hover { border-color: rgba(194,153,67,0.25); }
.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #7A7060;
    margin-bottom: 0.5rem;
}
.card-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 300;
    color: #E8E0D0;
    line-height: 1.1;
}
.card-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #7A7060;
    margin-top: 0.4rem;
}
.card-accent { border-left: 3px solid #C29943; }

/* ── Signal card ── */
.signal-bullish {
    background: rgba(122,158,126,0.06);
    border: 1px solid rgba(122,158,126,0.25);
    border-radius: 4px;
    padding: 1.6rem;
    text-align: center;
}
.signal-bearish {
    background: rgba(180,80,80,0.06);
    border: 1px solid rgba(180,80,80,0.25);
    border-radius: 4px;
    padding: 1.6rem;
    text-align: center;
}
.signal-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.25em;
    color: #7A7060;
    margin-bottom: 0.6rem;
}
.signal-direction {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: 0.1em;
}
.signal-bullish .signal-direction { color: #7A9E7E; }
.signal-bearish .signal-direction { color: #C46060; }
.signal-prob {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    margin-top: 0.4rem;
}
.signal-bullish .signal-prob { color: #7A9E7E; }
.signal-bearish .signal-prob { color: #C46060; }

/* ── Section titles ── */
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    font-weight: 300;
    letter-spacing: 0.1em;
    color: #C29943;
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(194,153,67,0.1);
}

/* ── Indicator row ── */
.indicator-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 0.8rem;
    margin: 1rem 0;
}
.indicator-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(194,153,67,0.1);
    border-radius: 3px;
    padding: 0.9rem 1rem;
}
.indicator-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #7A7060;
    text-transform: uppercase;
}
.indicator-val {
    font-family: 'DM Mono', monospace;
    font-size: 1rem;
    color: #E8E0D0;
    margin-top: 0.2rem;
}
.indicator-val.bullish { color: #7A9E7E; }
.indicator-val.bearish { color: #C46060; }
.indicator-val.neutral { color: #C29943; }

/* ── CTA Button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    padding: 0.9rem 2rem !important;
    background: transparent !important;
    color: #C29943 !important;
    border: 1px solid #C29943 !important;
    border-radius: 3px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.3em !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    box-shadow: none !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(194,153,67,0.1) !important;
    box-shadow: 0 0 30px rgba(194,153,67,0.12) !important;
}

/* ── Divider ── */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(194,153,67,0.3), transparent);
    margin: 2rem 0;
}

/* ── Footer ── */
.aurum-footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(194,153,67,0.1);
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    color: #4A4440;
}

/* ── Hide Streamlit defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { display: none; }
.stSelectbox label, .stSlider label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #7A7060 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="aurum-header">
    <div class="aurum-logo">AURUM</div>
    <div class="aurum-tagline">XAU / USD Predictive Intelligence Terminal</div>
    <div class="aurum-model-badge">CNN-BiLSTM · 60-STEP SEQUENCE · PRECISION 65.06%</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_ai_engine():
    model = Sequential([
        Input(shape=(60, 26)),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.load_weights('nn_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_ai_engine()
    st.markdown("""
    <div class="status-pill">
        <div class="status-dot"></div>
        NEURAL ENGINE ONLINE · CNN-BiLSTM MATRIX CONNECTED
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"[MODEL LOAD FAILURE] {e}")
    st.stop()

# ─────────────────────────────────────────────
#  SIDEBAR SETTINGS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Settings")
    lookback_period = st.selectbox(
        "Chart Period",
        ["30d", "60d", "90d", "180d"],
        index=1
    )
    chart_type = st.selectbox(
        "Chart Style",
        ["Candlestick", "Line", "Area"],
        index=0
    )
    show_indicators = st.multiselect(
        "Overlay Indicators",
        ["SMA 20", "SMA 50", "Bollinger Bands", "EMA 12", "EMA 26"],
        default=["SMA 20", "Bollinger Bands"]
    )

# ─────────────────────────────────────────────
#  HELPER: Build features
# ─────────────────────────────────────────────
def build_features(df_raw):
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df = df_raw.copy()

    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Return'].rolling(10).std()
    df['Vol_20'] = df['Log_Return'].rolling(20).std()
    df['Volatility_Ratio'] = df['Vol_10'] / df['Vol_20']

    delta = df['Close'].diff()
    gain14 = delta.where(delta > 0, 0).rolling(14).mean()
    loss14 = (-delta.where(delta < 0, 0)).rolling(14).mean()
    gain7  = delta.where(delta > 0, 0).rolling(7).mean()
    loss7  = (-delta.where(delta < 0, 0)).rolling(7).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain14 / loss14))
    df['RSI_7']  = 100 - (100 / (1 + gain7  / loss7))

    BB_Mid = df['Close'].rolling(20).mean()
    BB_Std = df['Close'].rolling(20).std()
    df['BB_Upper'] = BB_Mid + 2 * BB_Std
    df['BB_Lower'] = BB_Mid - 2 * BB_Std
    df['Pct_B']    = (df['Close'] - (BB_Mid - 2 * BB_Std)) / (4 * BB_Std)
    df['BB_Width'] = (4 * BB_Std) / BB_Mid

    EMA12 = df['Close'].ewm(span=12).mean()
    EMA26 = df['Close'].ewm(span=26).mean()
    df['EMA12'] = EMA12
    df['EMA26'] = EMA26
    df['SMA20']  = df['Close'].rolling(20).mean()
    df['SMA50']  = df['Close'].rolling(50).mean()
    df['MACD']        = (EMA12 - EMA26) / df['Close']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

    df['TR']  = np.maximum(df['High'] - df['Low'],
                np.maximum(abs(df['High'] - df['Close'].shift(1)),
                           abs(df['Low']  - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(14).mean() / df['Close']

    Low14  = df['Low'].rolling(14).min()
    High14 = df['High'].rolling(14).max()
    df['Stoch_K'] = (df['Close'] - Low14) / (High14 - Low14)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    for w in [20, 50, 200]:
        df[f'Dist_SMA{w}'] = (df['Close'] - df['Close'].rolling(w).mean()) / df['Close'].rolling(w).mean()

    df['EMA_Cross_12_26'] = df['Close'].ewm(span=12).mean() / df['Close'].ewm(span=26).mean() - 1

    for lag in [1, 2, 3, 5]:
        df[f'Return_Lag{lag}'] = df['Log_Return'].shift(lag)
        df[f'RSI14_Lag{lag}']  = df['RSI_14'].shift(lag)

    df.dropna(inplace=True)
    return df

# ─────────────────────────────────────────────
#  HELPER: Chart builder
# ─────────────────────────────────────────────
GOLD = '#C29943'
SILVER = '#A0A0A0'
GREEN = '#7A9E7E'
RED = '#C46060'
BG = '#0A0A0F'
GRID = 'rgba(194,153,67,0.06)'

def build_price_chart(df, chart_type, show_indicators):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20]
    )

    # ── Price ──
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            increasing=dict(line=dict(color=GREEN, width=1), fillcolor='rgba(122,158,126,0.6)'),
            decreasing=dict(line=dict(color=RED, width=1), fillcolor='rgba(196,96,96,0.6)'),
            name="XAU/USD"
        ), row=1, col=1)
    elif chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            line=dict(color=GOLD, width=1.5),
            name="Close"
        ), row=1, col=1)
    else:  # Area
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            line=dict(color=GOLD, width=1.5),
            fill='tozeroy',
            fillcolor='rgba(194,153,67,0.07)',
            name="Close"
        ), row=1, col=1)

    # ── Overlays ──
    if "SMA 20" in show_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'],
            line=dict(color='rgba(194,153,67,0.6)', width=1, dash='dot'),
            name="SMA 20"), row=1, col=1)

    if "SMA 50" in show_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'],
            line=dict(color='rgba(160,160,160,0.5)', width=1, dash='dot'),
            name="SMA 50"), row=1, col=1)

    if "EMA 12" in show_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA12'],
            line=dict(color='rgba(100,180,200,0.6)', width=1),
            name="EMA 12"), row=1, col=1)

    if "EMA 26" in show_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA26'],
            line=dict(color='rgba(180,130,200,0.6)', width=1),
            name="EMA 26"), row=1, col=1)

    if "Bollinger Bands" in show_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'],
            line=dict(color='rgba(194,153,67,0.25)', width=0.8),
            name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'],
            line=dict(color='rgba(194,153,67,0.25)', width=0.8),
            fill='tonexty', fillcolor='rgba(194,153,67,0.04)',
            name="BB Bands"), row=1, col=1)

    # ── RSI ──
    rsi_colors = [GREEN if v > 50 else RED for v in df['RSI_14']]
    fig.add_trace(go.Bar(x=df.index, y=df['RSI_14'],
        marker_color=rsi_colors, name="RSI 14", opacity=0.7), row=2, col=1)
    fig.add_hline(y=70, line=dict(color='rgba(196,96,96,0.4)', dash='dot', width=1), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='rgba(122,158,126,0.4)', dash='dot', width=1), row=2, col=1)
    fig.add_hline(y=50, line=dict(color='rgba(194,153,67,0.2)', width=1), row=2, col=1)

    # ── MACD ──
    macd_colors = [GREEN if v >= 0 else RED for v in df['MACD_Hist']]
    macd_raw = df['MACD'] * df['Close']  # denormalize for readability
    macd_sig_raw = df['MACD_Signal'] * df['Close']
    hist_raw = macd_raw - macd_sig_raw
    hist_colors = [GREEN if v >= 0 else RED for v in hist_raw]

    fig.add_trace(go.Bar(x=df.index, y=hist_raw,
        marker_color=hist_colors, name="MACD Hist", opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_raw,
        line=dict(color=GOLD, width=1), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_sig_raw,
        line=dict(color=SILVER, width=0.8, dash='dot'), name="Signal"), row=3, col=1)

    # ── Layout ──
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family='DM Mono, monospace', size=10, color='#7A7060'),
        margin=dict(l=10, r=10, t=20, b=10),
        height=620,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(10,10,15,0.8)',
            bordercolor='rgba(194,153,67,0.2)',
            borderwidth=1,
            font=dict(size=9),
            orientation='h',
            y=1.02, x=0
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
    )
    for i in range(1, 4):
        fig.update_xaxes(
            row=i, col=1,
            gridcolor=GRID, zerolinecolor=GRID,
            showspikes=True, spikecolor=GOLD, spikethickness=0.5
        )
        fig.update_yaxes(
            row=i, col=1,
            gridcolor=GRID, zerolinecolor=GRID
        )

    # Row labels
    fig.update_yaxes(title_text="PRICE (USD)", row=1, col=1, title_font=dict(size=9))
    fig.update_yaxes(title_text="RSI", row=2, col=1, title_font=dict(size=9))
    fig.update_yaxes(title_text="MACD", row=3, col=1, title_font=dict(size=9))

    return fig


def rsi_label(v):
    if v >= 70: return "OVERBOUGHT", "bearish"
    if v <= 30: return "OVERSOLD", "bullish"
    return f"{v:.1f}", "neutral"

def macd_label(hist):
    if hist > 0: return "BULLISH", "bullish"
    return "BEARISH", "bearish"

# ─────────────────────────────────────────────
#  MAIN ACTION BUTTON
# ─────────────────────────────────────────────
col_btn_l, col_btn, col_btn_r = st.columns([1.5, 2, 1.5])
with col_btn:
    scan_clicked = st.button("SCAN MARKET")

st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  EXECUTION
# ─────────────────────────────────────────────
if scan_clicked:
    with st.spinner("Fetching satellite data & assembling indicators…"):

        # -- Download --
        df_raw = yf.download('GC=F', period='350d', progress=False)
        df = build_features(df_raw)

        if len(df) < 60:
            st.error("Insufficient data for 60-step sequence. Please try again shortly.")
            st.stop()

        harga_sekarang = float(df['Close'].iloc[-1])
        harga_kemarin  = float(df['Close'].iloc[-2])
        perubahan      = harga_sekarang - harga_kemarin
        pct_change     = (perubahan / harga_kemarin) * 100
        high_hari_ini  = float(df['High'].iloc[-1])
        low_hari_ini   = float(df['Low'].iloc[-1])
        atr_val        = float(df['ATR'].iloc[-1]) * harga_sekarang
        rsi14_val      = float(df['RSI_14'].iloc[-1])
        rsi7_val       = float(df['RSI_7'].iloc[-1])
        stoch_k        = float(df['Stoch_K'].iloc[-1]) * 100
        macd_hist_val  = float(df['MACD_Hist'].iloc[-1])
        pct_b          = float(df['Pct_B'].iloc[-1])
        vol_ratio      = float(df['Volatility_Ratio'].iloc[-1])

        # -- Predict --
        EXCLUDE = {'Open', 'High', 'Low', 'Close', 'Volume', 'TR',
                   'BB_Upper', 'BB_Lower', 'SMA20', 'SMA50', 'EMA12', 'EMA26'}
        fitur = [c for c in df.columns if c not in EXCLUDE]
        X_live   = df.iloc[-60:][fitur]
        X_scaled = scaler.transform(X_live)
        X_lstm   = np.reshape(X_scaled, (1, 60, len(fitur)))
        probabilitas = float(model.predict(X_lstm)[0][0])

        is_bullish = probabilitas > 0.5
        prob_display = probabilitas * 100 if is_bullish else (1 - probabilitas) * 100
        direction = "BULLISH" if is_bullish else "BEARISH"

        # ── SECTION 1: Key Metrics ────────────────
        st.markdown('<div class="section-title">Market Snapshot</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            change_color = "#7A9E7E" if perubahan >= 0 else "#C46060"
            change_symbol = "▲" if perubahan >= 0 else "▼"
            st.markdown(f"""
            <div class="card card-accent">
                <div class="card-label">XAU/USD · Last Close</div>
                <div class="card-value">{harga_sekarang:,.2f}</div>
                <div class="card-sub" style="color:{change_color}">
                    {change_symbol} {abs(perubahan):.2f} &nbsp; ({pct_change:+.2f}%)
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">Session High / Low</div>
                <div class="card-value" style="font-size:1.5rem">{high_hari_ini:,.2f}</div>
                <div class="card-sub" style="color:#C46060">{low_hari_ini:,.2f} &nbsp; LOW</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">ATR · 14-Day</div>
                <div class="card-value" style="font-size:1.8rem">{atr_val:.2f}</div>
                <div class="card-sub">Daily volatility estimate (USD)</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            signal_class = "signal-bullish" if is_bullish else "signal-bearish"
            st.markdown(f"""
            <div class="{signal_class}">
                <div class="signal-label">AI SIGNAL · 5-DAY HORIZON</div>
                <div class="signal-direction">{direction}</div>
                <div class="signal-prob">Probability: {prob_display:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        # ── SECTION 2: Chart ──────────────────────
        st.markdown('<div class="section-title">Price Chart</div>', unsafe_allow_html=True)

        # Filter chart period
        period_map = {"30d": 30, "60d": 60, "90d": 90, "180d": 180}
        n_days = period_map.get(lookback_period, 60)
        df_chart = df.iloc[-n_days:]

        fig = build_price_chart(df_chart, chart_type, show_indicators)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── SECTION 3: Technical Indicators ───────
        st.markdown('<div class="section-title">Technical Indicators</div>', unsafe_allow_html=True)

        rsi14_txt, rsi14_cls = rsi_label(rsi14_val)
        rsi7_txt,  rsi7_cls  = rsi_label(rsi7_val)
        macd_txt,  macd_cls  = macd_label(macd_hist_val)
        stoch_cls = "bearish" if stoch_k > 80 else ("bullish" if stoch_k < 20 else "neutral")
        pctb_cls  = "bearish" if pct_b > 0.9 else ("bullish" if pct_b < 0.1 else "neutral")
        vol_cls   = "bearish" if vol_ratio > 1.3 else ("bullish" if vol_ratio < 0.8 else "neutral")

        st.markdown(f"""
        <div class="indicator-grid">
            <div class="indicator-item">
                <div class="indicator-name">RSI · 14</div>
                <div class="indicator-val {rsi14_cls}">{rsi14_val:.1f}</div>
            </div>
            <div class="indicator-item">
                <div class="indicator-name">RSI · 7</div>
                <div class="indicator-val {rsi7_cls}">{rsi7_val:.1f}</div>
            </div>
            <div class="indicator-item">
                <div class="indicator-name">MACD Momentum</div>
                <div class="indicator-val {macd_cls}">{macd_txt}</div>
            </div>
            <div class="indicator-item">
                <div class="indicator-name">Stochastic %K</div>
                <div class="indicator-val {stoch_cls}">{stoch_k:.1f}</div>
            </div>
            <div class="indicator-item">
                <div class="indicator-name">Bollinger %B</div>
                <div class="indicator-val {pctb_cls}">{pct_b:.3f}</div>
            </div>
            <div class="indicator-item">
                <div class="indicator-name">Volatility Ratio</div>
                <div class="indicator-val {vol_cls}">{vol_ratio:.3f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── SECTION 4: Probability Gauge ──────────
        st.markdown('<div class="section-title">Model Confidence</div>', unsafe_allow_html=True)

        col_gauge, col_detail = st.columns([1, 1])

        with col_gauge:
            gauge_val = probabilitas * 100
            gauge_color = GREEN if is_bullish else RED
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_val,
                number=dict(suffix="%", font=dict(family="Cormorant Garamond", size=36, color="#E8E0D0")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1, tickcolor=GOLD,
                              tickfont=dict(family="DM Mono", size=9, color="#7A7060")),
                    bar=dict(color=gauge_color, thickness=0.25),
                    bgcolor='rgba(255,255,255,0.02)',
                    borderwidth=1,
                    bordercolor='rgba(194,153,67,0.15)',
                    steps=[
                        dict(range=[0, 35], color='rgba(196,96,96,0.1)'),
                        dict(range=[35, 65], color='rgba(194,153,67,0.05)'),
                        dict(range=[65, 100], color='rgba(122,158,126,0.1)')
                    ],
                    threshold=dict(
                        line=dict(color=GOLD, width=2),
                        thickness=0.75, value=50
                    )
                ),
                domain=dict(x=[0, 1], y=[0, 1])
            ))
            fig_gauge.update_layout(
                paper_bgcolor=BG,
                font=dict(family='DM Mono', color='#7A7060'),
                margin=dict(l=20, r=20, t=30, b=10),
                height=260
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

        with col_detail:
            # Price range estimate using ATR
            est_up   = harga_sekarang + atr_val * 1.5
            est_down = harga_sekarang - atr_val * 1.5

            st.markdown(f"""
            <div class="card" style="height:100%; margin-top:0.5rem">
                <div class="card-label">5-Day Scenario Estimate (±1.5× ATR)</div>
                <div style="margin-top:1rem">
                    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#7A7060; letter-spacing:0.2em">UPSIDE TARGET</div>
                    <div style="font-family:'Cormorant Garamond',serif; font-size:1.8rem; color:#7A9E7E">{est_up:,.2f}</div>
                </div>
                <div style="margin-top:0.8rem">
                    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#7A7060; letter-spacing:0.2em">DOWNSIDE TARGET</div>
                    <div style="font-family:'Cormorant Garamond',serif; font-size:1.8rem; color:#C46060">{est_down:,.2f}</div>
                </div>
                <div style="margin-top:0.8rem">
                    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#7A7060; letter-spacing:0.2em">RISK / REWARD</div>
                    <div style="font-family:'DM Mono',monospace; font-size:1rem; color:#C29943">{(est_up - harga_sekarang)/(harga_sekarang - est_down):.2f}x</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── SECTION 5: Return History ─────────────
        st.markdown('<div class="section-title">Rolling Return Distribution (30-Day)</div>', unsafe_allow_html=True)

        returns_30 = df['Log_Return'].iloc[-30:] * 100
        colors_ret = [GREEN if v >= 0 else RED for v in returns_30]

        fig_ret = go.Figure(go.Bar(
            x=returns_30.index,
            y=returns_30.values,
            marker_color=colors_ret,
            marker_opacity=0.75,
            name="Daily Return %"
        ))
        fig_ret.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family='DM Mono', size=9, color='#7A7060'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=180,
            showlegend=False,
            xaxis=dict(gridcolor=GRID, showgrid=True),
            yaxis=dict(gridcolor=GRID, showgrid=True, ticksuffix="%")
        )
        st.plotly_chart(fig_ret, use_container_width=True, config={"displayModeBar": False})

        # ── Disclaimer ────────────────────────────
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="
            background: rgba(194,153,67,0.03);
            border: 1px solid rgba(194,153,67,0.1);
            border-radius: 3px;
            padding: 1rem 1.4rem;
            font-family: 'DM Mono', monospace;
            font-size: 0.65rem;
            letter-spacing: 0.12em;
            color: #4A4440;
            line-height: 1.8;
        ">
        ⚠ &nbsp; DISCLAIMER &nbsp; — &nbsp; 
        Aurum Intelligence is a quantitative research tool operating at a baseline precision of 65.06%.
        All signals represent probabilistic model output and should be treated as supplementary momentum indicators only.
        This tool does not constitute financial advice. Past model performance does not guarantee future results.
        Trade responsibly and consult a licensed financial advisor before making investment decisions.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="aurum-footer">
    AURUM · XAU/USD INTELLIGENCE TERMINAL &nbsp;·&nbsp; CNN-BiLSTM HYBRID ENGINE &nbsp;·&nbsp;
    {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
</div>
""", unsafe_allow_html=True)

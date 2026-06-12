#  AURUM · XAU/USD Predictive Intelligence Terminal

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![Accuracy](https://img.shields.io/badge/Model_Precision-65.06%25-brightgreen)

**AURUM** is an advanced machine learning research project and quantitative dashboard designed to predict the 5-day directional momentum of Gold (XAU/USD). The system utilizes a hybrid Neural Network architecture, combining feature extraction with sequential memory to process raw market data into actionable probabilistic signals.

##  Neural Architecture: CNN-BiLSTM Hybrid
The core prediction engine moves beyond traditional tabular machine learning by treating financial data as complex spatial-temporal sequences. 

* **Baseline Model:** Random Forest Classifier (Accuracy: 53.31%)
* **Primary Engine:** Convolutional Neural Network coupled with Bidirectional Long Short-Term Memory (CNN-BiLSTM)
* **Final Validated Precision:** **65.06%** (Highly significant for real-world financial market volatility, preventing data leakage boundaries).
* **Memory Matrix:** 60-Step Sequence Lookback window.

##  Arsenal of Technical Indicators
The model does not rely solely on raw price action. It continuously extracts 26 dynamic technical features, including:
* **Momentum:** RSI (7 & 14), MACD Histogram, Stochastic Oscillator (%K).
* **Volatility:** Bollinger Bands (%B, Width), Average True Range (ATR), Rolling Volatility Ratio.
* **Trend & Distances:** SMA (20, 50, 200), EMA (12, 26) Crossovers.
* **Lag Features:** 1-to-5 day historical log returns.

##  Terminal Features (GUI)
The web interface features a highly optimized, custom-styled GUI engineered for analytical precision:
* **Cyberpunk / Premium Aesthetic:** Deep Obsidian & Liquid Gold palette.
* **Interactive Charting:** Powered by `plotly`, featuring Candlestick, Line, and Area charts with dynamic technical overlays.
* **Real-Time Data Injection:** Automated satellite data extraction via the Yahoo Finance (`yfinance`) API.
* **Risk/Reward Estimator:** Computes upside/downside targets based on a 1.5x daily ATR multiplier.

---

##  Deployment & Installation Setup

### Prerequisites
Ensure you have Python installed. It is highly recommended to use a Virtual Environment (e.g., Anaconda).

### 1. Clone the Repository
```bash
git clone [https://github.com/ferdinand_macalister/Gold-Predictor-AI.git](https://github.com/ferdinand_macalister/Gold-Predictor-AI.git)
cd Gold-Predictor-AI

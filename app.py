import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="CEO Language Analyzer",
    page_icon="📊",
    layout="wide"
)

# Database connection
@st.cache_resource
def get_connection():
    return sqlite3.connect("data/earnings_research.db", check_same_thread=False)

conn = get_connection()

# Load master data
@st.cache_data
def load_data():
    df = pd.read_sql("SELECT * FROM master_analysis", conn)
    return df[df["avg_compound"] > 0].copy()

df = load_data()

# Header
st.title("📊 Do CEOs Lie With Their Words?")
st.subheader("Earnings Call Sentiment vs. Actual Stock Performance in S&P 500 Companies (2020–2025)")
st.markdown("---")

# Sidebar
st.sidebar.title("🔍 Select a Company")
tickers = sorted(df["ticker"].unique().tolist())
selected_ticker = st.sidebar.selectbox("Company Ticker", tickers)

quarters = sorted(df[df["ticker"] == selected_ticker]["quarter"].unique().tolist())
selected_quarter = st.sidebar.selectbox("Quarter", quarters)

st.sidebar.markdown("---")
st.sidebar.markdown("**About this project**")
st.sidebar.markdown("This tool analyzes CEO language from earnings calls and measures whether their confidence predicted actual stock performance.")
st.sidebar.markdown("Built with Python, NLP, FinBERT, and Machine Learning.")

# Get selected row
row = df[(df["ticker"] == selected_ticker) & 
         (df["quarter"] == selected_quarter)].iloc[0]

# Section 1 - Company Overview
st.header(f"📋 {selected_ticker} — {selected_quarter}")
st.markdown(f"**Market Period:** {row['period']} | **Earnings Date:** {row['date']}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("VADER Sentiment", f"{row['avg_compound']:.3f}", 
              help="Score from -1 (negative) to +1 (positive)")
with col2:
    st.metric("FinBERT Score", f"{row['finbert_score']:.3f}",
              help="Financial-specific sentiment score")
with col3:
    st.metric("Hedge Ratio", f"{row['hedge_ratio']:.4f}",
              help="Proportion of cautious words in transcript")
with col4:
    performance = "✅ Outperformed" if row["return_90d"] > 0 else "❌ Underperformed"
    st.metric("90-Day Performance", performance)

st.markdown("---")

# Section 2 - Stock Performance
st.header("📈 Stock Performance After Earnings")

col1, col2, col3 = st.columns(3)
with col1:
    color = "green" if row["return_30d"] > 0 else "red"
    st.metric("30-Day Return", f"{row['return_30d']}%",
              delta=f"{row['return_30d']}%")
with col2:
    st.metric("60-Day Return", f"{row['return_60d']}%",
              delta=f"{row['return_60d']}%")
with col3:
    st.metric("90-Day Return", f"{row['return_90d']}%",
              delta=f"{row['return_90d']}%")

# Stock price chart
@st.cache_data
def get_stock_data(ticker, date):
    prices = pd.read_sql(f"""
        SELECT date, close FROM stock_prices
        WHERE ticker = '{ticker}'
        ORDER BY date
    """, conn)
    prices["date"] = pd.to_datetime(prices["date"].str[:10])
    earnings_date = pd.to_datetime(date)
    mask = (prices["date"] >= earnings_date - pd.Timedelta(days=30)) & \
           (prices["date"] <= earnings_date + pd.Timedelta(days=95))
    return prices[mask]

stock_data = get_stock_data(selected_ticker, row["date"])

if len(stock_data) > 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_data["date"],
        y=stock_data["close"],
        mode="lines",
        name="Stock Price",
        line=dict(color="#2196F3", width=2)
    ))
    fig.add_vline(x=row["date"], line_dash="dash", 
                  line_color="red", annotation_text="Earnings Call")
    fig.update_layout(
        title=f"{selected_ticker} Stock Price Around {selected_quarter} Earnings Call",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Section 3 - ML Prediction
st.header("🤖 ML Model Prediction")

ml_data = pd.read_sql(f"""
    SELECT * FROM ml_predictions
    WHERE ticker = '{selected_ticker}' AND quarter = '{selected_quarter}'
""", conn)

if len(ml_data) > 0:
    ml_row = ml_data.iloc[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted", "Outperform" if ml_row["predicted"] == 1 else "Underperform")
    with col2:
        st.metric("Actual", "Outperformed" if ml_row["actual"] == 1 else "Underperformed")
    with col3:
        correct = "✅ Correct" if ml_row["correct"] == 1 else "❌ Wrong"
        st.metric("Prediction", correct)
    
    st.progress(int(ml_row["prob_outperform"]))
    st.caption(f"Model confidence: {ml_row['prob_outperform']}% probability of outperformance")

st.markdown("---")

# Section 4 - AI Analyst Report
st.header("📝 AI Analyst Report")

report_path = f"reports/company_reports/{selected_ticker}_{selected_quarter}_report.md"

if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()
    st.markdown(report_content)
else:
    st.info("Generating report on demand...")
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        verdict = "OVERCONFIDENCE TRAP" if row["avg_compound"] > 0.28 and row["return_90d"] < 0 else \
                  "HONEST & ACCURATE" if row["avg_compound"] < 0.20 and row["return_90d"] > 0 else "MIXED SIGNAL"
        
        prompt = f"""You are a senior financial analyst. Write a professional 4-paragraph earnings call analysis for {selected_ticker} {selected_quarter}.
        VADER Score: {row['avg_compound']}, FinBERT: {row['finbert_score']}, Hedge Ratio: {row['hedge_ratio']}
        30d return: {row['return_30d']}%, 60d: {row['return_60d']}%, 90d: {row['return_90d']}%
        Verdict: {verdict}
        Cover: executive summary, sentiment analysis, stock performance, conclusion."""
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )
        st.markdown(response.choices[0].message.content)
    except:
        st.error("Could not generate report. Please check your API key.")

st.markdown("---")

# Section 5 - Overall Research Findings
st.header("🔬 Key Research Findings")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📊 Dataset")
    st.markdown(f"- **{len(df)} earnings calls** analyzed")
    st.markdown(f"- **{df['ticker'].nunique()} S&P 500 companies**")
    st.markdown("- **5 market periods** covered")
    st.markdown("- 2020 to 2025")

with col2:
    st.markdown("### 🎯 Core Finding")
    st.markdown("- Higher CEO confidence → **lower returns**")
    st.markdown("- Cautious CEOs → **outperformed market**")
    st.markdown("- ML model accuracy: **60%**")
    st.markdown("- Most overconfident: **AAPL Q1 2020**")
    st.markdown("- Most honest: **HAL Q1 2020**")

st.markdown("---")
st.caption("Built by Shekinah Paul | BSc Applied Statistics & Data Analytics | GitHub: paulshekinah487-lang")
import streamlit as st
import pandas as pd
import os
from transformers import pipeline
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Client Sentiment Analyzer",
    layout="wide"
)

# =========================
# SIDEBAR BRANDING
# =========================
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=150)

st.sidebar.title("Client Dashboard")
st.sidebar.markdown("---")

# =========================
# SIMPLE LOGIN SYSTEM
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    if (
        st.session_state.username == "admin"
        and st.session_state.password == "admin123"
    ):
        st.session_state.logged_in = True
    else:
        st.sidebar.error("❌ Invalid credentials")

if not st.session_state.logged_in:
    st.title("🔐 Login")
    st.text_input("Username", key="username")
    st.text_input("Password", type="password", key="password")
    st.button("Login", on_click=login)
    st.stop()

# =========================
# LOAD MODEL (HUGGINGFACE)
# =========================
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

model = load_model()

# =========================
# LABEL MAPPING
# =========================
LABEL_MAP = {
    "LABEL_0": "Negative 😡",
    "LABEL_1": "Neutral 😐",
    "LABEL_2": "Positive 😊"
}

# =========================
# MAIN UI
# =========================
st.title("🌍 Sentiment Analyzer Enterprise")
st.caption("AI-powered Sentiment Analysis (Positive | Neutral | Negative)")

tab1, tab2, tab3 = st.tabs(["📝 Text", "📂 CSV", "🐦 Twitter / X"])

# =========================
# TEXT ANALYSIS TAB
# =========================
with tab1:
    text = st.text_area("Enter text to analyze")

    if st.button("Analyze Sentiment"):
        if text.strip() == "":
            st.warning("Please enter some text.")
        else:
            result = model(text)[0]
            sentiment = LABEL_MAP.get(result["label"], result["label"])
            confidence = result["score"] * 100

            st.markdown(f"### 🧠 Result: **{sentiment}**")

            if "Negative" in sentiment:
                st.error(f"Confidence: {confidence:.2f}%")
            elif "Neutral" in sentiment:
                st.warning(f"Confidence: {confidence:.2f}%")
            else:
                st.success(f"Confidence: {confidence:.2f}%")

            st.progress(int(confidence))

# =========================
# CSV ANALYSIS TAB
# =========================
with tab2:
    file = st.file_uploader(
        "Upload CSV file (must contain a 'text' column)",
        type=["csv"]
    )

    if file:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            with st.spinner("Analyzing sentiments..."):
                results = model(df["text"].astype(str).tolist())

            df["sentiment"] = [
                LABEL_MAP.get(r["label"], r["label"])
                for r in results
            ]

            st.success("Analysis complete!")
            st.dataframe(df, use_container_width=True)

            # Chart
            fig = px.pie(
                df,
                names="sentiment",
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================
# TWITTER / X TAB (READY)
# =========================
with tab3:
    st.info("Live Twitter / X sentiment requires API keys.")
    query = st.text_input("Keyword or Hashtag")
    st.markdown("🔌 Twitter/X integration ready (API keys required)")

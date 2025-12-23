
import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px

st.set_page_config(page_title="Client Sentiment Analyzer", layout="wide")

# ---- CLIENT BRANDING ----
st.sidebar.image("logo.png", width=150)
st.sidebar.title("Client Dashboard")

# ---- SIMPLE LOGIN ----
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    if st.session_state.username == "admin" and st.session_state.password == "admin123":
        st.session_state.logged_in = True
    else:
        st.error("Invalid credentials")

if not st.session_state.logged_in:
    st.title("🔐 Login")
    st.text_input("Username", key="username")
    st.text_input("Password", type="password", key="password")
    st.button("Login", on_click=login)
    st.stop()

# ---- MODEL ----
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

model = load_model()

st.title("🌍 Sentiment Analyzer Enterprise")

tab1, tab2, tab3 = st.tabs(["📝 Text", "📂 CSV", "🐦 Twitter/X"])

# ---- TEXT ----
with tab1:
    text = st.text_area("Enter text")
    if st.button("Analyze"):
        r = model(text)[0]
        st.success(f"{r['label']} ({r['score']*100:.2f}%)")

# ---- CSV ----
with tab2:
    file = st.file_uploader("Upload CSV with text column", type=["csv"])
    if file:
        df = pd.read_csv(file)
        results = model(df['text'].tolist())
        df['sentiment'] = [r['label'] for r in results]
        st.dataframe(df)
        fig = px.bar(df, x='sentiment', title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

# ---- TWITTER/X ----
with tab3:
    st.info("Live Twitter/X sentiment requires API keys")
    query = st.text_input("Keyword / Hashtag")
    st.markdown("🔌 Twitter API integration ready (keys required)")

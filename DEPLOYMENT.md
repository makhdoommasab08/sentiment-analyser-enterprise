
# Deployment Guide

## Streamlit Cloud
1. Push project to GitHub
2. Go to streamlit.io/cloud
3. Select repo and app.py

## AWS (EC2)
1. Launch EC2 Ubuntu
2. Install Python & pip
3. pip install -r requirements.txt
4. streamlit run app.py

## FastAPI
uvicorn api:app --host 0.0.0.0 --port 8000

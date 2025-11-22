# Create a new file: debug_app.py
import streamlit as st
import sys
import os

st.title("🔧 Debug Dashboard")

# Check if all modules can be imported
st.header("Module Import Status")

try:
    from data_pipeline.twitter_stream import AdvancedTwitterStreamer
    st.success("✅ Twitter Streamer - OK")
except Exception as e:
    st.error(f"❌ Twitter Streamer - {e}")

try:
    from data_pipeline.data_processor import RealTimeDataProcessor
    st.success("✅ Data Processor - OK")
except Exception as e:
    st.error(f"❌ Data Processor - {e}")

try:
    from nlp_models.sentiment_analyzer import AdvancedSentimentAnalyzer
    st.success("✅ Sentiment Analyzer - OK")
except Exception as e:
    st.error(f"❌ Sentiment Analyzer - {e}")

# Check if data is being processed
st.header("Data Processing Status")
try:
    processor = RealTimeDataProcessor()
    data = processor.get_dashboard_data()
    if data:
        st.success(f"✅ Data available: {len(data.get('recent_posts', []))} posts")
    else:
        st.warning("⚠️ No data available yet")
except Exception as e:
    st.error(f"❌ Data processing error: {e}")
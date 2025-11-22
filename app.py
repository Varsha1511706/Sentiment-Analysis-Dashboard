import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import json
import threading
from data_pipeline.data_processor import RealTimeDataProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Real-Time Social Media Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SentimentDashboard:
    def __init__(self):
        self.processor = RealTimeDataProcessor()
        self.data = {}
        
    def run(self):
        # Start background processing in a separate thread
        if not hasattr(st.session_state, 'processor_started'):
            thread = threading.Thread(target=self.processor.start_processing, daemon=True)
            thread.start()
            st.session_state.processor_started = True
        
        # Sidebar
        st.sidebar.title("🔍 Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10)
        
        # Data source filter
        sources = st.sidebar.multiselect(
            "Data Sources:",
            ["twitter", "reddit"],
            default=["twitter", "reddit"]
        )
        
        # Time range filter
        time_range = st.sidebar.selectbox(
            "Time Range:",
            ["Last 1 hour", "Last 6 hours", "Last 24 hours", "All time"]
        )
        
        # Main content
        st.title("🚀 Real-Time Social Media Sentiment Dashboard")
        st.markdown("---")
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
        
        # Get current data
        self.data = self.processor.get_dashboard_data()
        
        # Display metrics and charts
        self.display_overview_metrics()
        self.display_sentiment_analysis()
        self.display_topic_analysis()
        self.display_recent_posts()
    
    def display_overview_metrics(self):
        """Display overview metrics at the top"""
        st.header("📈 Overview Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_posts = self.data.get('statistics', {}).get('total_posts', 0)
            st.metric("Total Posts", f"{total_posts:,}")
        
        with col2:
            posts_per_hour = self.data.get('statistics', {}).get('posts_last_hour', 0)
            st.metric("Posts/Hour", f"{posts_per_hour:,}")
        
        with col3:
            processing_rate = self.data.get('statistics', {}).get('processing_rate', 0)
            st.metric("Processing Rate", f"{processing_rate:.1f} posts/sec")
        
        with col4:
            avg_sentiment = self.data.get('statistics', {}).get('avg_sentiment', 0)
            sentiment_color = "🟢" if avg_sentiment > 0.1 else "🔴" if avg_sentiment < -0.1 else "🟡"
            st.metric("Avg Sentiment", f"{sentiment_color} {avg_sentiment:.3f}")
        
        with col5:
            last_updated = self.data.get('last_updated', '')
            if last_updated:
                last_time = datetime.fromisoformat(last_updated)
                time_diff = (datetime.now() - last_time).total_seconds()
                st.metric("Last Update", f"{time_diff:.0f}s ago")
    
    def display_sentiment_analysis(self):
        """Display sentiment analysis charts"""
        st.header("😊 Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_sentiment_timeseries()
        
        with col2:
            self.plot_sentiment_distribution()
        
        col3, col4 = st.columns(2)
        
        with col3:
            self.plot_sentiment_by_source()
        
        with col4:
            self.plot_sentiment_heatmap()
    
    def plot_sentiment_timeseries(self):
        """Plot sentiment over time"""
        time_series = self.data.get('time_series', [])
        
        if not time_series:
            st.info("No time series data available yet")
            return
        
        df = pd.DataFrame(time_series)
        df['hour'] = pd.to_datetime(df['hour'])
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Sentiment line
        fig.add_trace(
            go.Scatter(x=df['hour'], y=df['avg_sentiment'], 
                      name="Sentiment Score", line=dict(color='blue')),
            secondary_y=False,
        )
        
        # Post count bars
        fig.add_trace(
            go.Bar(x=df['hour'], y=df['post_count'], 
                  name="Post Count", opacity=0.3),
            secondary_y=True,
        )
        
        fig.update_layout(
            title="Sentiment Trend Over Time",
            xaxis_title="Time",
            height=400
        )
        
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
        fig.update_yaxes(title_text="Post Count", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_sentiment_distribution(self):
        """Plot sentiment distribution pie chart"""
        recent_posts = self.data.get('recent_posts', [])
        
        if not recent_posts:
            st.info("No posts available for analysis")
            return
        
        sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
        
        for post in recent_posts:
            sentiment = post.get('sentiment', 'NEUTRAL')
            sentiment_counts[sentiment] += 1
        
        # Create pie chart
        fig = px.pie(
            values=list(sentiment_counts.values()),
            names=list(sentiment_counts.keys()),
            title="Sentiment Distribution",
            color=list(sentiment_counts.keys()),
            color_discrete_map={
                'POSITIVE': 'green',
                'NEGATIVE': 'red', 
                'NEUTRAL': 'gray'
            }
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_sentiment_by_source(self):
        """Plot sentiment by data source"""
        sentiment_by_source = self.data.get('sentiment_by_source', {})
        
        if not sentiment_by_source:
            st.info("No source data available")
            return
        
        # Prepare data for stacked bar chart
        sources = list(sentiment_by_source.keys())
        sentiments = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
        
        data = []
        for sentiment in sentiments:
            counts = [sentiment_by_source[source].get(sentiment, 0) for source in sources]
            data.append(go.Bar(name=sentiment, x=sources, y=counts))
        
        fig = go.Figure(data=data)
        fig.update_layout(
            title="Sentiment by Data Source",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_sentiment_heatmap(self):
        """Plot sentiment heatmap by hour and day"""
        # This would require more historical data
        st.info("Heatmap visualization - requires more historical data")
        # Implementation would track sentiment by hour/day
    
    def display_topic_analysis(self):
        """Display topic analysis"""
        st.header("🔥 Trending Topics & Themes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_trending_topics()
        
        with col2:
            self.plot_topic_sentiment()
    
    def plot_trending_topics(self):
        """Plot trending topics word cloud (simplified as bar chart)"""
        trending_topics = self.data.get('trending_topics', [])
        
        if not trending_topics:
            st.info("No trending topics data yet")
            return
        
        topics, counts = zip(*trending_topics)
        
        fig = px.bar(
            x=counts,
            y=topics,
            orientation='h',
            title="Top Trending Topics",
            labels={'x': 'Mention Count', 'y': 'Topics'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_topic_sentiment(self):
        """Plot sentiment for top topics"""
        # This would analyze sentiment per topic
        st.info("Topic-wise sentiment analysis")
        # Implementation would group posts by topic and analyze sentiment
    
    def display_recent_posts(self):
        """Display recent posts in a table"""
        st.header("📝 Recent Social Media Posts")
        
        recent_posts = self.data.get('recent_posts', [])
        
        if not recent_posts:
            st.info("No recent posts to display")
            return
        
        # Create DataFrame for display
        display_data = []
        for post in recent_posts[-20:]:  # Last 20 posts
            display_data.append({
                'Source': post.get('source', ''),
                'Text': post.get('text', '')[:100] + '...' if len(post.get('text', '')) > 100 else post.get('text', ''),
                'Sentiment': post.get('sentiment', ''),
                'Confidence': f"{post.get('confidence', 0):.2f}",
                'Time': post.get('processed_at', '')[:19]
            })
        
        df = pd.DataFrame(display_data)
        
        # Color code sentiment
        def color_sentiment(sentiment):
            if sentiment == 'POSITIVE':
                return 'color: green'
            elif sentiment == 'NEGATIVE':
                return 'color: red'
            else:
                return 'color: gray'
        
        styled_df = df.style.applymap(
            color_sentiment, 
            subset=['Sentiment']
        )
        
        st.dataframe(styled_df, use_container_width=True, height=400)

# Run the dashboard
if __name__ == "__main__":
    dashboard = SentimentDashboard()
    dashboard.run()
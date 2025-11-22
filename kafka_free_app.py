import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import logging
from collections import deque
import re
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🚀 Advanced Real-Time Social Media Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EnhancedDataGenerator:
    """Enhanced high-volume realistic social media sentiment data generator"""
    
    def __init__(self):
        # Use deque for efficient large data storage
        self.posts = deque(maxlen=10000)  # Increased capacity
        self.sentiment_history = deque(maxlen=2000)
        self.topic_counts = {}
        self.hashtag_trends = {}
        self.user_engagement = {}
        self.alert_thresholds = {'sentiment_spike': 0.4, 'volume_surge': 3.0}
        self.alerts = deque(maxlen=50)
        self.start_time = datetime.now()
        self.post_count = 0
        
        # Enhanced realistic social media content templates
        self.social_templates = [
            # Positive sentiment posts
            "Loving the new {} update! So much better than before 😍 #{}",
            "This {} feature is absolutely amazing! Game changer! 🔥",
            "Just experienced {} and it's incredible! Highly recommend! 👍",
            "The {} community is so supportive and helpful! Thank you all! 💕",
            "Wow! {} exceeded all my expectations! Mind blown! 🤯",
            
            # Negative sentiment posts  
            "Really disappointed with {}... such a letdown 😞 #{}",
            "Why did they change {}? It was perfect before! 👎",
            "Frustrated with {} - not working as expected 😠",
            "{} needs serious improvement. Very unsatisfied 💔",
            "Warning: {} has major issues. Avoid for now! 🚨",
            
            # Neutral/Question posts
            "Has anyone tried {}? Curious about your thoughts 🤔 #{}",
            "What's everyone's opinion on {}? Looking for recommendations",
            "Trying out {} for the first time. Any tips? 💡",
            "Not sure how I feel about {} yet. Need more time with it",
            "{} seems interesting. What are the pros and cons?",
            
            # Trending topics
            "The {} trend is taking over my feed! What do you think? #{}",
            "Everyone's talking about {} today! So much engagement 📈",
            "Can't escape the {} discussions! Viral moment happening 🚀",
            "{} is dominating social media right now! Thoughts?",
            
            # Personal experiences
            "My experience with {} has been life-changing! So grateful ✨",
            "Struggling with {} - anyone else having this issue? 😅",
            "Just discovered {} and it's become my new favorite! 🎉",
            "Giving up on {} after multiple attempts. Too frustrating 💔"
        ]
        
        self.topics = [
            "AI", "MachineLearning", "Tech", "Software", "App", "Gaming", 
            "Streaming", "SocialMedia", "Photography", "Music", "Art",
            "Fitness", "Cooking", "Travel", "Books", "Movies", "TVShows",
            "Fashion", "Beauty", "Lifestyle", "Productivity", "Education"
        ]
        
        self.hashtags = [
            "Tech", "Innovation", "Digital", "Future", "TechNews",
            "AI", "MachineLearning", "DataScience", "Programming",
            "Review", "Experience", "Recommendation", "Opinion",
            "Viral", "Trending", "Popular", "Buzz",
            "Love", "Hate", "Discussion", "Community"
        ]
        
        self.platforms = ['twitter', 'reddit', 'instagram', 'tiktok', 'facebook']
        
        # Pre-generate initial data
        self._generate_initial_data()
    
    def _generate_initial_data(self):
        """Generate initial dataset to make dashboard look active immediately"""
        logger.info("📊 Generating initial dataset...")
        initial_posts = []
        
        # Generate 300 initial posts from the last 2 hours
        for i in range(300):
            post_time = datetime.now() - timedelta(minutes=random.randint(1, 120))
            post = self._create_post(post_time)
            initial_posts.append(post)
        
        # Sort by time and add to posts
        initial_posts.sort(key=lambda x: x['created_at'])
        self.posts.extend(initial_posts)
        self.post_count = len(initial_posts)
        
        # Initialize topic and hashtag counts
        for post in initial_posts:
            for topic in post['topics']:
                self.topic_counts[topic] = self.topic_counts.get(topic, 0) + 1
            for hashtag in post['hashtags']:
                self.hashtag_trends[hashtag] = self.hashtag_trends.get(hashtag, 0) + 1
    
    def _create_post(self, post_time=None):
        """Create a single realistic social media post"""
        if post_time is None:
            post_time = datetime.now()
            
        template = random.choice(self.social_templates)
        topic = random.choice(self.topics)
        hashtag = random.choice(self.hashtags)
        
        # Fill template with realistic data
        text = template.format(topic, hashtag)
        
        # Enhanced sentiment analysis using TextBlob (simulated)
        sentiment, score, confidence = self._analyze_sentiment(text)
        
        # Determine platform-specific engagement metrics
        platform = random.choice(self.platforms)
        engagement = self._calculate_engagement(text, sentiment, platform)
        
        # Generate relevant hashtags
        post_hashtags = [hashtag, topic]
        if random.random() > 0.7:  # 30% chance of additional hashtags
            post_hashtags.extend(random.sample(self.hashtags, random.randint(1, 3)))
        
        post = {
            'id': f"post_{self.post_count}",
            'text': text,
            'source': platform,
            'sentiment': sentiment,
            'sentiment_score': score,
            'confidence': confidence,
            'created_at': post_time.isoformat(),
            'user_id': f"user_{random.randint(10000, 99999)}",
            'likes': engagement['likes'],
            'shares': engagement['shares'],
            'comments': engagement['comments'],
            'topics': [topic],
            'hashtags': post_hashtags,
            'viral_score': engagement['viral_score']
        }
        
        return post
    
    def _analyze_sentiment(self, text):
        """Enhanced sentiment analysis"""
        # Simulate more sophisticated sentiment analysis
        positive_indicators = ['loving', 'amazing', 'incredible', 'awesome', 'perfect', 'love', 'great', 'best', 'excellent', 'fantastic', 'wonderful', 'brilliant']
        negative_indicators = ['disappointed', 'letdown', 'frustrated', 'unsatisfied', 'hate', 'terrible', 'awful', 'worst', 'bad', 'horrible', 'avoid', 'warning']
        intensifiers = ['so', 'very', 'extremely', 'absolutely', 'really', 'seriously']
        
        words = text.lower().split()
        positive_score = 0
        negative_score = 0
        intensity = 1.0
        
        for i, word in enumerate(words):
            if word in positive_indicators:
                positive_score += 1
                # Check for intensifiers
                if i > 0 and words[i-1] in intensifiers:
                    positive_score += 0.5
                    intensity += 0.3
            elif word in negative_indicators:
                negative_score += 1
                # Check for intensifiers
                if i > 0 and words[i-1] in intensifiers:
                    negative_score += 0.5
                    intensity += 0.3
        
        # Calculate final sentiment
        if positive_score > negative_score:
            sentiment = 'POSITIVE'
            base_score = min(0.9, positive_score * 0.2 * intensity)
        elif negative_score > positive_score:
            sentiment = 'NEGATIVE'
            base_score = max(-0.9, -negative_score * 0.2 * intensity)
        else:
            sentiment = 'NEUTRAL'
            base_score = random.uniform(-0.1, 0.1)
        
        # Add some randomness for realism
        final_score = base_score + random.uniform(-0.1, 0.1)
        final_score = max(-1.0, min(1.0, final_score))
        
        confidence = min(0.98, 0.7 + (abs(final_score) * 0.3))
        
        return sentiment, round(final_score, 3), round(confidence, 3)
    
    def _calculate_engagement(self, text, sentiment, platform):
        """Calculate realistic engagement metrics based on platform and sentiment"""
        base_metrics = {
            'twitter': {'likes': (50, 500), 'shares': (5, 100), 'comments': (2, 50)},
            'instagram': {'likes': (100, 2000), 'shares': (10, 200), 'comments': (5, 150)},
            'tiktok': {'likes': (500, 5000), 'shares': (50, 500), 'comments': (20, 300)},
            'reddit': {'likes': (100, 1000), 'shares': (5, 50), 'comments': (10, 200)},
            'facebook': {'likes': (50, 800), 'shares': (10, 150), 'comments': (5, 100)}
        }
        
        platform_metrics = base_metrics[platform]
        
        # Sentiment affects engagement
        engagement_multiplier = 1.0
        if sentiment == 'POSITIVE':
            engagement_multiplier = 1.2
        elif sentiment == 'NEGATIVE':
            engagement_multiplier = 1.4  # Negative posts often get more engagement
        
        # Calculate viral score (0-100)
        text_length = len(text)
        has_emoji = int(any(char in text for char in ['😍', '🔥', '🚀', '❤️', '🎉']))
        has_hashtags = int('#' in text)
        
        viral_score = min(100, (
            (engagement_multiplier * 25) +
            (min(text_length, 280) / 280 * 25) +
            (has_emoji * 25) +
            (has_hashtags * 25)
        ))
        
        return {
            'likes': int(random.randint(*platform_metrics['likes']) * engagement_multiplier),
            'shares': int(random.randint(*platform_metrics['shares']) * engagement_multiplier),
            'comments': int(random.randint(*platform_metrics['comments']) * engagement_multiplier),
            'viral_score': round(viral_score)
        }
    
    def start_stream(self):
        """Start high-volume data generation"""
        def generate_continuously():
            batch_sizes = [3, 5, 8, 10, 15]  # Variable batch sizes for realism
            batch_intervals = [1, 2, 3]  # Variable intervals
            
            while True:
                try:
                    # Variable batch size for more realistic data flow
                    batch_size = random.choice(batch_sizes)
                    batch_interval = random.choice(batch_intervals)
                    
                    # Generate a batch of posts
                    new_posts = []
                    for _ in range(batch_size):
                        new_post = self._create_post()
                        new_posts.append(new_post)
                        self.post_count += 1
                    
                    # Add to collections
                    self.posts.extend(new_posts)
                    
                    # Update sentiment history with timestamp
                    current_time = datetime.now()
                    avg_batch_sentiment = sum(p['sentiment_score'] for p in new_posts) / len(new_posts)
                    self.sentiment_history.append({
                        'timestamp': current_time,
                        'score': avg_batch_sentiment,
                        'post_count': len(new_posts),
                        'avg_viral_score': sum(p['viral_score'] for p in new_posts) / len(new_posts)
                    })
                    
                    # Update topic and hashtag counts
                    for post in new_posts:
                        for topic in post['topics']:
                            self.topic_counts[topic] = self.topic_counts.get(topic, 0) + 1
                        for hashtag in post['hashtags']:
                            self.hashtag_trends[hashtag] = self.hashtag_trends.get(hashtag, 0) + 1
                    
                    # Check for alerts (sentiment spikes, viral posts)
                    self._check_alerts(new_posts, avg_batch_sentiment)
                    
                    # Log progress occasionally
                    if self.post_count % 200 == 0:
                        logger.info(f"📈 Generated {self.post_count} posts total")
                    
                    time.sleep(batch_interval)
                    
                except Exception as e:
                    logger.error(f"Data generation error: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=generate_continuously, daemon=True)
        thread.start()
        return thread
    
    def _check_alerts(self, new_posts, batch_sentiment):
        """Check for alert conditions"""
        current_time = datetime.now()
        
        # Check for sentiment spike
        if len(self.sentiment_history) > 10:
            recent_scores = [h['score'] for h in list(self.sentiment_history)[-10:]]
            avg_recent = sum(recent_scores) / len(recent_scores)
            
            if abs(batch_sentiment - avg_recent) > self.alert_thresholds['sentiment_spike']:
                direction = "positive" if batch_sentiment > avg_recent else "negative"
                self.alerts.append({
                    'type': 'sentiment_spike',
                    'message': f"🚨 Significant {direction} sentiment spike detected!",
                    'timestamp': current_time,
                    'severity': 'high',
                    'details': f"Sentiment changed from {avg_recent:.3f} to {batch_sentiment:.3f}"
                })
        
        # Check for viral posts
        viral_posts = [p for p in new_posts if p['viral_score'] > 80]
        for post in viral_posts:
            self.alerts.append({
                'type': 'viral_post',
                'message': f"🔥 Viral post detected! Score: {post['viral_score']}",
                'timestamp': current_time,
                'severity': 'medium',
                'details': f"Post: '{post['text'][:50]}...'"
            })
    
    def get_dashboard_data(self):
        """Get formatted data for dashboard"""
        recent_posts = list(self.posts)[-250:]  # Last 250 posts
        
        # Calculate real-time statistics
        total_posts = len(self.posts)
        
        # Posts in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        posts_last_hour = sum(1 for p in self.posts 
                             if datetime.fromisoformat(p['created_at']) > one_hour_ago)
        
        # Sentiment metrics
        if recent_posts:
            avg_sentiment = sum(p['sentiment_score'] for p in recent_posts) / len(recent_posts)
            positive_pct = sum(1 for p in recent_posts if p['sentiment'] == 'POSITIVE') / len(recent_posts) * 100
            negative_pct = sum(1 for p in recent_posts if p['sentiment'] == 'NEGATIVE') / len(recent_posts) * 100
            avg_viral_score = sum(p['viral_score'] for p in recent_posts) / len(recent_posts)
        else:
            avg_sentiment = 0
            positive_pct = 0
            negative_pct = 0
            avg_viral_score = 0
        
        # Calculate real processing rate
        time_running = (datetime.now() - self.start_time).total_seconds()
        processing_rate = total_posts / max(1, time_running)
        
        # Generate realistic time series data
        time_series = self._generate_time_series_data()
        
        # Trending topics and hashtags
        trending_topics = sorted(self.topic_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        trending_hashtags = sorted(self.hashtag_trends.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # Platform distribution
        platform_dist = {}
        for post in recent_posts:
            platform = post['source']
            platform_dist[platform] = platform_dist.get(platform, 0) + 1
        
        return {
            'recent_posts': recent_posts,
            'time_series': time_series,
            'trending_topics': trending_topics,
            'trending_hashtags': trending_hashtags,
            'platform_distribution': platform_dist,
            'alerts': list(self.alerts)[-10:],  # Last 10 alerts
            'statistics': {
                'total_posts': total_posts,
                'posts_last_hour': posts_last_hour,
                'avg_sentiment': avg_sentiment,
                'positive_pct': positive_pct,
                'negative_pct': negative_pct,
                'processing_rate': round(processing_rate, 2),
                'unique_topics': len(self.topic_counts),
                'avg_viral_score': avg_viral_score
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _generate_time_series_data(self):
        """Generate realistic time series data for charts"""
        time_series = []
        current_time = datetime.now() - timedelta(hours=6)  # 6 hours of history
        
        # Create data points (10-minute intervals for 6 hours)
        for i in range(36):
            interval_time = current_time + timedelta(minutes=i * 10)
            
            # Simulate daily patterns (more activity during peak hours)
            hour = interval_time.hour
            if 8 <= hour <= 23:  # Active hours
                base_volume = random.randint(15, 40)
                sentiment_variance = 0.4
            else:
                base_volume = random.randint(5, 12)
                sentiment_variance = 0.2
            
            # Simulate realistic sentiment patterns with some trends
            base_sentiment = 0.15  # Slightly positive bias for social media
            time_factor = np.sin(i * 0.3) * 0.2  # Oscillating pattern
            sentiment_noise = random.uniform(-sentiment_variance, sentiment_variance)
            avg_sentiment = base_sentiment + time_factor + sentiment_noise
            
            time_series.append({
                'timestamp': interval_time,
                'hour': interval_time.strftime('%H:%M'),
                'avg_sentiment': round(avg_sentiment, 3),
                'post_count': base_volume + random.randint(-8, 15),
                'viral_score': random.randint(30, 70)
            })
        
        return time_series

class EnhancedDashboard:
    def __init__(self):
        self.data_generator = EnhancedDataGenerator()
        self.data_generator.start_stream()
        self.data = {}
        st.success("🚀 Enhanced social media sentiment dashboard started!")
    
    def run(self):
        # Sidebar with enhanced controls
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Auto-refresh with more options
        col1, col2 = st.sidebar.columns(2)
        with col1:
            auto_refresh = st.checkbox("Auto-refresh", value=True)
        with col2:
            refresh_interval = st.slider("Seconds", 2, 30, 5, key="refresh")
        
        # Enhanced filters
        st.sidebar.subheader("🎯 Data Filters")
        selected_sources = st.sidebar.multiselect(
            "Platforms:",
            ["twitter", "reddit", "instagram", "tiktok", "facebook"],
            default=["twitter", "reddit", "instagram", "tiktok", "facebook"]
        )
        
        selected_sentiments = st.sidebar.multiselect(
            "Sentiments:",
            ["POSITIVE", "NEUTRAL", "NEGATIVE"],
            default=["POSITIVE", "NEUTRAL", "NEGATIVE"]
        )
        
        viral_threshold = st.sidebar.slider("Viral Score Threshold", 0, 100, 70)
        
        # Main content
        st.title("🚀 Advanced Social Media Sentiment Dashboard")
        st.markdown("**Real-time sentiment analysis across multiple platforms**")
        st.markdown("---")
        
        # Get latest data
        self.data = self.data_generator.get_dashboard_data()
        
        # Display all enhanced analysis sections
        self.display_alerts()
        self.display_overview_metrics()
        self.display_sentiment_analysis() 
        self.display_topic_analysis()
        self.display_platform_analysis()
        self.display_engagement_metrics()
        self.display_recent_posts(viral_threshold)
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
    
    def display_alerts(self):
        """Display real-time alerts"""
        alerts = self.data.get('alerts', [])
        if alerts:
            st.header("🚨 Real-time Alerts")
            for alert in reversed(alerts):
                emoji = "🔴" if alert['severity'] == 'high' else "🟡" if alert['severity'] == 'medium' else "🔵"
                with st.container():
                    st.warning(f"{emoji} **{alert['message']}** - *{alert['timestamp'].strftime('%H:%M:%S')}*")
                    with st.expander("Details"):
                        st.write(alert['details'])
            st.markdown("---")
    
    def display_overview_metrics(self):
        """Display enhanced overview metrics"""
        st.header("📈 Live Social Media Overview")
        
        stats = self.data.get('statistics', {})
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            delta = stats.get('posts_last_hour', 0) - (stats.get('total_posts', 0) // 10)
            st.metric(
                "Total Posts", 
                f"{stats.get('total_posts', 0):,}",
                delta=f"{delta:,}/hr"
            )
        
        with col2:
            st.metric("Posts/Hour", f"{stats.get('posts_last_hour', 0):,}")
        
        with col3:
            st.metric("Processing Rate", f"{stats.get('processing_rate', 0):.1f}/sec")
        
        with col4:
            avg_sentiment = stats.get('avg_sentiment', 0)
            sentiment_icon = "🟢" if avg_sentiment > 0.1 else "🔴" if avg_sentiment < -0.1 else "🟡"
            st.metric(
                "Avg Sentiment", 
                f"{sentiment_icon} {avg_sentiment:.3f}",
                f"↑{stats.get('positive_pct', 0):.1f}%"
            )
        
        with col5:
            st.metric("Unique Topics", f"{stats.get('unique_topics', 0):,}")
        
        with col6:
            viral_score = stats.get('avg_viral_score', 0)
            viral_icon = "🔥" if viral_score > 70 else "📈" if viral_score > 50 else "📊"
            st.metric("Avg Viral Score", f"{viral_icon} {viral_score:.0f}")
    
    def display_sentiment_analysis(self):
        """Display enhanced sentiment analysis charts"""
        st.header("📊 Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_sentiment_timeseries()
        
        with col2:
            self.plot_sentiment_distribution()
    
    def plot_sentiment_timeseries(self):
        """Plot enhanced sentiment over time"""
        time_series = self.data.get('time_series', [])
        
        if time_series:
            df = pd.DataFrame(time_series)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Sentiment Trend", "Post Volume & Viral Score"),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Sentiment line
            fig.add_trace(
                go.Scatter(
                    x=df['hour'], y=df['avg_sentiment'], 
                    name="Sentiment Score", 
                    line=dict(color='#636EFA', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(99, 110, 250, 0.1)',
                    hovertemplate="<b>%{x}</b><br>Sentiment: %{y:.3f}<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Post volume bars
            fig.add_trace(
                go.Bar(
                    x=df['hour'], y=df['post_count'], 
                    name="Post Volume", 
                    marker_color='#FFA15A',
                    opacity=0.6,
                    hovertemplate="<b>%{x}</b><br>Posts: %{y}<extra></extra>"
                ),
                row=2, col=1
            )
            
            # Viral score line
            fig.add_trace(
                go.Scatter(
                    x=df['hour'], y=df['viral_score'],
                    name="Viral Score",
                    line=dict(color='#00CC96', width=2, dash='dot'),
                    hovertemplate="<b>%{x}</b><br>Viral Score: %{y}<extra></extra>"
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Social Media Sentiment & Engagement Trends (6 Hours)",
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_yaxes(title_text="Sentiment Score", row=1, col=1, range=[-1, 1])
            fig.update_yaxes(title_text="Count / Score", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📈 Generating time series data...")
    
    def plot_sentiment_distribution(self):
        """Plot enhanced sentiment distribution"""
        recent_posts = self.data.get('recent_posts', [])
        
        if recent_posts:
            # Create detailed sentiment breakdown
            sentiment_bins = {
                'Very Positive (0.5-1.0)': 0,
                'Positive (0.1-0.5)': 0,
                'Neutral (-0.1-0.1)': 0,
                'Negative (-0.5--0.1)': 0,
                'Very Negative (-1.0--0.5)': 0
            }
            
            for post in recent_posts:
                score = post['sentiment_score']
                if score >= 0.5:
                    sentiment_bins['Very Positive (0.5-1.0)'] += 1
                elif score >= 0.1:
                    sentiment_bins['Positive (0.1-0.5)'] += 1
                elif score > -0.1:
                    sentiment_bins['Neutral (-0.1-0.1)'] += 1
                elif score > -0.5:
                    sentiment_bins['Negative (-0.5--0.1)'] += 1
                else:
                    sentiment_bins['Very Negative (-1.0--0.5)'] += 1
            
            # Create sunburst chart for hierarchical view
            fig = px.sunburst(
                names=list(sentiment_bins.keys()),
                parents=[''] * len(sentiment_bins),
                values=list(sentiment_bins.values()),
                title="Detailed Sentiment Distribution",
                color=list(sentiment_bins.values()),
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("😊 Analyzing sentiment patterns...")
    
    def display_topic_analysis(self):
        """Display enhanced topic analysis"""
        st.header("🔥 Trending Topics & Hashtags")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_trending_topics()
        
        with col2:
            self.plot_trending_hashtags()
    
    def plot_trending_topics(self):
        """Plot trending topics"""
        trending_topics = self.data.get('trending_topics', [])
        
        if trending_topics:
            topics, counts = zip(*trending_topics)
            
            fig = px.bar(
                x=counts,
                y=topics,
                orientation='h',
                title="Most Discussed Topics",
                labels={'x': 'Mention Count', 'y': ''},
                color=counts,
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(height=400, showlegend=False)
            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>Mentions: %{x}<extra></extra>",
                texttemplate='%{x}',
                textposition='inside'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔍 Identifying trending topics...")
    
    def plot_trending_hashtags(self):
        """Plot trending hashtags"""
        trending_hashtags = self.data.get('trending_hashtags', [])
        
        if trending_hashtags:
            hashtags, counts = zip(*trending_hashtags)
            
            fig = px.treemap(
                names=hashtags,
                parents=[''] * len(hashtags),
                values=counts,
                title="Trending Hashtags",
                color=counts,
                color_continuous_scale='rainbow'
            )
            
            fig.update_layout(height=400)
            fig.update_traces(
                hovertemplate="<b>%{label}</b><br>Mentions: %{value}<extra></extra>"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🏷️ Tracking hashtag trends...")
    
    def display_platform_analysis(self):
        """Display platform distribution analysis"""
        st.header("📱 Platform Distribution")
        
        platform_dist = self.data.get('platform_distribution', {})
        
        if platform_dist:
            col1, col2 = st.columns(2)
            
            with col1:
                # Platform distribution pie chart
                fig = px.pie(
                    values=list(platform_dist.values()),
                    names=list(platform_dist.keys()),
                    title="Post Distribution by Platform",
                    color=list(platform_dist.keys()),
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Platform engagement comparison
                recent_posts = self.data.get('recent_posts', [])
                platform_engagement = {}
                
                for post in recent_posts:
                    platform = post['source']
                    if platform not in platform_engagement:
                        platform_engagement[platform] = []
                    platform_engagement[platform].append(
                        post['likes'] + post['shares'] * 2 + post['comments'] * 3
                    )
                
                # Calculate average engagement per platform
                avg_engagement = {
                    platform: sum(engagements) / len(engagements) 
                    for platform, engagements in platform_engagement.items()
                }
                
                fig = px.bar(
                    x=list(avg_engagement.keys()),
                    y=list(avg_engagement.values()),
                    title="Average Engagement Score by Platform",
                    labels={'x': 'Platform', 'y': 'Avg Engagement Score'},
                    color=list(avg_engagement.values()),
                    color_continuous_scale='thermal'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📱 Analyzing platform distribution...")
    
    def display_engagement_metrics(self):
        """Display enhanced engagement metrics"""
        st.header("💬 Engagement Analytics")
        
        recent_posts = self.data.get('recent_posts', [])
        
        if recent_posts:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_likes = sum(p.get('likes', 0) for p in recent_posts)
            total_shares = sum(p.get('shares', 0) for p in recent_posts)
            total_comments = sum(p.get('comments', 0) for p in recent_posts)
            total_engagement = total_likes + total_shares * 2 + total_comments * 3
            viral_posts = sum(1 for p in recent_posts if p.get('viral_score', 0) > 70)
            
            with col1:
                st.metric("Total Likes", f"{total_likes:,}")
            
            with col2:
                st.metric("Total Shares", f"{total_shares:,}")
            
            with col3:
                st.metric("Total Comments", f"{total_comments:,}")
            
            with col4:
                avg_engagement = total_engagement / len(recent_posts)
                st.metric("Avg Engagement", f"{avg_engagement:.1f}")
            
            with col5:
                st.metric("Viral Posts", f"{viral_posts}", "🔥 Hot")
            
            # Engagement trend chart
            st.subheader("Engagement Trends")
            time_series = self.data.get('time_series', [])
            if time_series:
                df = pd.DataFrame(time_series)
                fig = px.area(
                    df, x='hour', y='post_count',
                    title="Post Volume Trend",
                    labels={'hour': 'Time', 'post_count': 'Posts'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Calculating engagement metrics...")
    
    def display_recent_posts(self, viral_threshold):
        """Display enhanced recent posts view"""
        st.header("📝 Live Social Feed")
        
        recent_posts = self.data.get('recent_posts', [])
        
        if recent_posts:
            # Filter by viral threshold
            filtered_posts = [p for p in recent_posts if p.get('viral_score', 0) >= viral_threshold]
            
            st.info(f"Showing {len(filtered_posts)} posts with viral score ≥ {viral_threshold}")
            
            # Show posts in a nice format
            for post in reversed(filtered_posts[-20:]):  # Last 20 filtered posts
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        # Sentiment and viral indicators
                        sentiment_color = {
                            'POSITIVE': '🟢',
                            'NEGATIVE': '🔴', 
                            'NEUTRAL': '⚪'
                        }.get(post.get('sentiment', 'NEUTRAL'), '⚪')
                        
                        viral_indicator = "🔥" if post.get('viral_score', 0) > 80 else "📈" if post.get('viral_score', 0) > 60 else ""
                        
                        st.write(f"{sentiment_color} **{post.get('source', '').title()}** {viral_indicator}")
                        st.write(f"*{post['text']}*")
                        
                        # Hashtags
                        hashtags = " ".join([f"#{tag}" for tag in post.get('hashtags', [])[:3]])
                        if hashtags:
                            st.caption(hashtags)
                    
                    with col2:
                        score = post.get('sentiment_score', 0)
                        sentiment_text = f"**{post.get('sentiment', '')}** ({score:.3f})"
                        st.write(sentiment_text)
                        st.write(f"Confidence: {post.get('confidence', 0):.1%}")
                    
                    with col3:
                        st.write(f"❤️ {post.get('likes', 0)}")
                        st.write(f"🔄 {post.get('shares', 0)}")
                        st.write(f"💬 {post.get('comments', 0)}")
                        st.write(f"📊 {post.get('viral_score', 0)}")
                
                st.divider()
        else:
            st.info("⏳ Waiting for incoming posts...")

# Run the dashboard
if __name__ == "__main__":
    import random  # Add this import at the top
    dashboard = EnhancedDashboard()
    dashboard.run()
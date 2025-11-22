import os
from dotenv import load_dotenv

load_dotenv()

# Twitter API Configuration
TWITTER_CONFIG = {
    'consumer_key': os.getenv('TWITTER_CONSUMER_KEY'),
    'consumer_secret': os.getenv('TWITTER_CONSUMER_SECRET'),
    'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
    'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
    'bearer_token': os.getenv('TWITTER_BEARER_TOKEN')
}

# Reddit API Configuration
REDDIT_CONFIG = {
    'client_id': os.getenv('REDDIT_CLIENT_ID'),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
    'user_agent': 'SentimentDashboard/1.0'
}

# Kafka Configuration
KAFKA_CONFIG = {
    'bootstrap_servers': ['localhost:9092'],
    'topic': 'social_media_posts',
    'group_id': 'sentiment_analyzer'
}

# NLP Model Configuration
NLP_CONFIG = {
    'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'topic_model': 'facebook/bart-large-mnli',
    'batch_size': 32,
    'max_length': 512
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'update_interval': 10,  # seconds
    'max_posts': 10000,
    'trending_topics_count': 10
}

# Keywords to track (customizable)
TRACK_KEYWORDS = [
    '#bitcoin', '#crypto', 'bitcoin', 'cryptocurrency',
    '#stocks', 'stock market', 'investing',
    '#AI', 'artificial intelligence', 'machine learning',
    '#tech', 'technology', 'innovation'
]
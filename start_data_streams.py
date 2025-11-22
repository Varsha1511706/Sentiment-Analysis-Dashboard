import threading
import time
import logging
from data_pipeline.twitter_stream import AdvancedTwitterStreamer
from data_pipeline.data_processor import RealTimeDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_twitter_stream():
    """Start Twitter data stream"""
    try:
        logger.info("🚀 Starting Twitter stream...")
        streamer = AdvancedTwitterStreamer()
        streamer.stream_tweets()
    except Exception as e:
        logger.error(f"❌ Twitter stream error: {e}")

def start_data_processor():
    """Start data processing"""
    try:
        logger.info("🚀 Starting data processor...")
        processor = RealTimeDataProcessor()
        processor.start_processing()
    except Exception as e:
        logger.error(f"❌ Data processor error: {e}")

if __name__ == "__main__":
    print("🎯 Starting Real-Time Data Pipeline...")
    print("=" * 50)
    
    # Start data processor first
    processor_thread = threading.Thread(target=start_data_processor, daemon=True)
    processor_thread.start()
    
    # Wait a moment for processor to initialize
    time.sleep(3)
    
    # Start Twitter stream
    twitter_thread = threading.Thread(target=start_twitter_stream, daemon=True)
    twitter_thread.start()
    
    print("✅ Data pipeline started successfully!")
    print("📊 Data should now be flowing to the dashboard")
    print("🛑 Press Ctrl+C to stop all streams")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down data streams...")
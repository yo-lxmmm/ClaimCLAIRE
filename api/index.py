"""Vercel serverless function wrapper for Flask app."""

import sys
import os

# Add parent directory to path so we can import from the project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Disable the background thread for serverless (it won't work)
os.environ['VERCEL_SERVERLESS'] = '1'

# Import the Flask app
from baseline_web_app import app, initialize_agent

# Initialize agent on first request (lazy initialization)
_initialized = False

@app.before_request
def ensure_initialized():
    """Initialize agent on first request."""
    global _initialized
    if not _initialized:
        try:
            initialize_agent()
            _initialized = True
        except Exception as e:
            import logging
            logging.error(f"Failed to initialize agent: {e}")

# Export the Flask app for Vercel
# Vercel's Python runtime automatically detects Flask apps


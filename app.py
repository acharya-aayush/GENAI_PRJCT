"""
Chitragupta Financial Advisors - Web Interface
Flask backend for the ultra-premium Nepal financial advisor
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import time
import logging
import os
from pathlib import Path

# Import our advisor
from src.ultra_premium_advisor import ConversationalAdvisor

# Setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'chitragupta-financial-advisors-2025'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global advisor instance
advisor = None

def initialize_advisor():
    """Initialize the advisor on first request"""
    global advisor
    if advisor is None:
        logger.info("Initializing Chitragupta Financial Advisor...")
        try:
            advisor = ConversationalAdvisor()
            logger.info("✅ Advisor initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize advisor: {e}")
            advisor = None
    return advisor

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        # Get user message
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Initialize advisor if needed
        current_advisor = initialize_advisor()
        if current_advisor is None:
            return jsonify({
                'error': 'Advisor not available',
                'response': 'I apologize, but I am currently unavailable. Please try again later.'
            }), 500
        
        # Get response from advisor
        logger.info(f"Processing message: {user_message}")
        start_time = time.time()
        
        response = current_advisor.chat(user_message)
        
        processing_time = time.time() - start_time
        logger.info(f"Response generated in {processing_time:.1f}s")
        
        return jsonify({
            'response': response,
            'processing_time': processing_time
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': str(e),
            'response': 'I encountered an error processing your request. Please try again.'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'advisor_loaded': advisor is not None,
        'timestamp': time.time()
    })

if __name__ == '__main__':
    print("🇳🇵 Starting Chitragupta Financial Advisors Web Interface")
    print("=" * 60)
    
    # Create templates directory if it doesn't exist
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Run the app with disabled reloader to prevent PyTorch conflicts
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

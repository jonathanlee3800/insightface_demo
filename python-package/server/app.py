"""
Main Flask application entry point
"""
from flask import Flask
import logging
from routes import register_routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Register routes
    register_routes(app)
    
    # Auto-initialize face analysis and database on startup
    from functions import initialize_face_analysis, initialize_database
    
    logger.info("Initializing face analysis model...")
    face_success = initialize_face_analysis()
    
    logger.info("Connecting to database...")
    db_success = initialize_database()
    
    if face_success and db_success:
        logger.info("✓ Server fully initialized and ready")
    else:
        logger.warning("⚠️ Server started with some initialization issues")
    
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /add_face - Add face to database")
    logger.info("  POST /search_face - Search for face")
    logger.info("  GET  /list_faces - List all faces")
    logger.info("  DELETE /delete_face - Delete face")
    
    return app

if __name__ == '__main__':
    app = create_app()
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)

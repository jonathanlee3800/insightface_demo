"""
API routes for the Flask face recognition server
"""
from flask import request, jsonify
import logging
from functions import (
    initialize_face_analysis,
    initialize_database,
    get_face_embedding,
    base64_to_cv2_image,
    add_face_to_database,
    search_face_in_database,
    list_all_faces,
    delete_face_from_database,
    is_database_connected,
    is_face_model_loaded
)

logger = logging.getLogger(__name__)

def register_routes(app):
    """Register all API routes with the Flask app"""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'qdrant_connected': is_database_connected(),
            'model_loaded': is_face_model_loaded()
        })

    @app.route('/add_face', methods=['POST'])
    def add_face():
        """Add a face to the database with name and personnelId"""
        try:
            # Check if database is connected
            if not is_database_connected():
                return jsonify({'error': 'Qdrant database not connected'}), 500
            
            # Get data from request
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            # Extract name, personnelId, and image
            name = data.get('name')
            personnel_id = data.get('personnelId')
            image_data = data.get('image')
            
            if not name:
                return jsonify({'error': 'Name is required'}), 400
            
            if not personnel_id:
                return jsonify({'error': 'personnelId is required'}), 400
            
            if not image_data:
                return jsonify({'error': 'Image data is required'}), 400
            
            # Convert base64 image to OpenCV format
            try:
                cv2_image = base64_to_cv2_image(image_data)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
            
            # Extract face embedding
            try:
                embedding, faces = get_face_embedding(cv2_image)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
            
            # Add face to database with both name and personnelId
            add_face_to_database(name, embedding, personnel_id=personnel_id)
            
            return jsonify({
                'success': True,
                'message': f'Face added successfully for {name}',
                'name': name,
                'personnelId': personnel_id,
                'faces_detected': len(faces)
            })
            
        except Exception as e:
            logger.error(f"Error adding face: {str(e)}")
            return jsonify({'error': f'Internal server error: {str(e)}'}), 500

    @app.route('/search_face', methods=['POST'])
    def search_face():
        """Search for a face in the database"""
        try:
            # Check if database is connected
            if not is_database_connected():
                return jsonify({'error': 'Qdrant database not connected'}), 500
            
            # Get data from request
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            # Extract image and threshold
            image_data = data.get('image')
            threshold = data.get('threshold', 0.55)  # Default threshold
            
            if not image_data:
                return jsonify({'error': 'Image data is required'}), 400
            
            # Convert base64 image to OpenCV format
            try:
                cv2_image = base64_to_cv2_image(image_data)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
            
            # Extract face embedding
            try:
                embedding, faces = get_face_embedding(cv2_image)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
            
            # Search for matching face
            match = search_face_in_database(embedding, threshold=threshold)
            
            if match:
                return jsonify({
                    'success': True,
                    'found': True,
                    'name': match['name'],
                    'score': match['score'],
                    'threshold': threshold
                })
            else:
                return jsonify({
                    'success': True,
                    'found': False,
                    'message': 'No matching face found',
                    'threshold': threshold
                })
            
        except Exception as e:
            logger.error(f"Error searching face: {str(e)}")
            return jsonify({'error': f'Internal server error: {str(e)}'}), 500

    @app.route('/list_faces', methods=['GET'])
    def list_faces():
        """List all faces in the database"""
        try:
            if not is_database_connected():
                return jsonify({'error': 'Qdrant database not connected'}), 500
            
            # Get all faces
            faces = list_all_faces()
            
            return jsonify({
                'success': True,
                'faces': faces,
                'count': len(faces)
            })
            
        except Exception as e:
            logger.error(f"Error listing faces: {str(e)}")
            return jsonify({'error': f'Internal server error: {str(e)}'}), 500

    @app.route('/delete_face', methods=['DELETE'])
    def delete_face():
        """Delete a face from the database"""
        try:
            if not is_database_connected():
                return jsonify({'error': 'Qdrant database not connected'}), 500
            
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            name = data.get('name')
            if not name:
                return jsonify({'error': 'Name is required'}), 400
            
            # Delete face from database
            success = delete_face_from_database(name)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Face deleted successfully for {name}'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': f'Face not found for {name}'
                }), 404
            
        except Exception as e:
            logger.error(f"Error deleting face: {str(e)}")
            return jsonify({'error': f'Internal server error: {str(e)}'}), 500


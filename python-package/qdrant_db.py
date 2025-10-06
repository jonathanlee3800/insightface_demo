import numpy as np
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

class FaceDB:
    def __init__(self, url=None, api_key=None, host="localhost", port=6333):
        """
        Initialize Qdrant Face Database
        
        Args:
            url: Qdrant Cloud URL (e.g., "https://xyz-example.eu-central.aws.cloud.qdrant.io:6333")
            api_key: Qdrant Cloud API key
            host: Qdrant server host (default: localhost, used if url not provided)
            port: Qdrant server port (default: 6333, used if url not provided)
        """
        if url and api_key:
            # Qdrant Cloud connection
            self.client = QdrantClient(url=url, api_key=api_key)
            print(f"Connecting to Qdrant Cloud: {url}")
        else:
            # Local Qdrant connection
            self.client = QdrantClient(host=host, port=port)
            print(f"Connecting to local Qdrant: {host}:{port}")
        self.collection_name = "faces"
        self.embedding_size = 512  # InsightFace embedding size
        
        # Create collection if it doesn't exist
        self._setup_collection()
    
    def _setup_collection(self):
        """Create the faces collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                print(f"Creating collection '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_size, 
                        distance=Distance.COSINE
                    )
                )
                print(f"✓ Collection '{self.collection_name}' created!")
            else:
                print(f"✓ Collection '{self.collection_name}' ready")
                
        except Exception as e:
            print(f"Error setting up collection: {e}")
            raise
    
    def add_face(self, name: str, embedding: np.ndarray) -> str:
        """
        Add a face embedding to the database
        
        Args:
            name: Person's name
            embedding: Face embedding vector (512-dimensional)
            
        Returns:
            point_id: Unique ID of the inserted point
        """
        if embedding.shape[0] != self.embedding_size:
            raise ValueError(f"Embedding size must be {self.embedding_size}, got {embedding.shape[0]}")
        
        # Generate unique ID
        point_id = str(uuid.uuid4())
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload={"name": name}
        )
        
        # Insert into Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            print(f"✓ Added face for '{name}'")
            return point_id
            
        except Exception as e:
            print(f"Error adding face: {e}")
            raise
    
    def search_face(self, embedding: np.ndarray, threshold: float = 0.55) -> Optional[Dict]:
        """
        Search for the best matching face in the database
        
        Args:
            embedding: Query face embedding
            threshold: Minimum similarity score (0-1)
            
        Returns:
            Best match with name and score, or None if no match
        """
        if embedding.shape[0] != self.embedding_size:
            raise ValueError(f"Embedding size must be {self.embedding_size}, got {embedding.shape[0]}")
        
        try:
            # Search for similar vectors
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding.tolist(),
                limit=1,
                score_threshold=threshold
            )
            
            if results:
                result = results[0]
                return {
                    'name': result.payload.get('name', 'Unknown'),
                    'score': result.score
                }
            return None
            
        except Exception as e:
            print(f"Error searching faces: {e}")
            return None
    
    def list_people(self) -> List[str]:
        """Get list of all people in the database"""
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            names = set()
            for point in points:
                if 'name' in point.payload:
                    names.add(point.payload['name'])
            
            return sorted(list(names))
            
        except Exception as e:
            print(f"Error listing people: {e}")
            return []
    
    def delete_person(self, name: str) -> bool:
        """Delete all faces for a specific person"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="name",
                            match=MatchValue(value=name)
                        )
                    ]
                )
            )
            print(f"✓ Deleted all faces for '{name}'")
            return True
            
        except Exception as e:
            print(f"Error deleting person: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            people = self.list_people()
            
            return {
                'total_faces': info.points_count,
                'total_people': len(people),
                'people_list': people
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {'total_faces': 0, 'total_people': 0, 'people_list': []}


# Simple functions for easy use
def create_face_db(url=None, api_key=None):
    """Create and return a FaceDB instance"""
    return FaceDB(url=url, api_key=api_key)

def test_connection(url=None, api_key=None):
    """Test if Qdrant is running and accessible"""
    try:
        if url and api_key:
            client = QdrantClient(url=url, api_key=api_key)
            print("Testing Qdrant Cloud connection...")
        else:
            client = QdrantClient("localhost", port=6333)
            print("Testing local Qdrant connection...")
            
        collections = client.get_collections()
        print("✓ Qdrant is running and accessible")
        return True
    except Exception as e:
        print(f"✗ Cannot connect to Qdrant: {e}")
        if not url:
            print("For local: Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        else:
            print("Check your Qdrant Cloud URL and API key")
        return False

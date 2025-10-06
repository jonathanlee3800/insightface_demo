# Qdrant Configuration
import os

# Your Qdrant Cloud Settings
QDRANT_URL = "https://243c191c-49c8-4b01-9be8-7095f1eab9a7.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.KdD9y5LzpO-CFK4EKot94vj8ojgmaOCa1DpihFnApco"

# Alternative: Use environment variables (uncomment to use)
# QDRANT_URL = os.getenv("QDRANT_URL", "https://your-cluster-url.qdrant.io:6333")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "your-api-key-here")

# For local development, set these to None to use localhost
# QDRANT_URL = None
# QDRANT_API_KEY = None

import os

# Base directory (root of project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Data directories
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CHUNKS_DIR = os.path.join(BASE_DIR, "data", "chunks")

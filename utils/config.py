import os
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_embedding_model(model_id):
    """
    Load an embedding model from Hugging Face.
    
    Args:
        model_id (str): Hugging Face model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    
    Returns:
        SentenceTransformer: Loaded model ready for inference
        
    Raises:
        Exception: If model fails to load due to network, memory, or other issues
    """
    try:
        # Check if model is cached locally
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_name = model_id.replace("/", "--")
        model_cache_path = Path(cache_dir) / f"models--{model_cache_name}"
        
        if model_cache_path.exists():
            logger.info(f"Model '{model_id}' found in cache at {model_cache_path}")
        else:
            logger.info(f"Model '{model_id}' not in cache. Downloading from Hugging Face...")
        
        # Load model using SentenceTransformer
        # This will automatically download if not cached
        model = SentenceTransformer(model_id)
        
        logger.info(f"Successfully loaded model '{model_id}'")
        return model
        
    except OSError as e:
        logger.error(f"Network or file system error loading '{model_id}': {str(e)}")
        raise Exception(f"Failed to download or access model '{model_id}'. Check your internet connection and model ID.") from e
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            logger.error(f"Out of memory error loading '{model_id}': {str(e)}")
            raise Exception(f"Insufficient memory to load model '{model_id}'. Try a smaller model or increase available memory.") from e
        else:
            logger.error(f"Runtime error loading '{model_id}': {str(e)}")
            raise Exception(f"Runtime error loading model '{model_id}': {str(e)}") from e
            
    except Exception as e:
        logger.error(f"Unexpected error loading '{model_id}': {str(e)}")
        raise Exception(f"Failed to load model '{model_id}': {str(e)}") from e


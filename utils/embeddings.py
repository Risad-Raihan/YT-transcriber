"""Embedding generation and similarity utilities."""

import logging
from typing import List, Dict
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """Generate embeddings for images using CLIP."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP embedder.
        
        Args:
            model_name: Name of CLIP model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.model.eval()
        logger.info("CLIP model loaded successfully")
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize embedding
            embedding = image_features.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding image {image_path}: {e}")
            return np.zeros(512)  # Return zero vector on error
    
    def embed_images_batch(self, image_paths: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for multiple images in batches.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to process at once
            
        Returns:
            Array of embeddings (n_images, embedding_dim)
        """
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            try:
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                
                # Normalize embeddings
                batch_embeddings = image_features.cpu().numpy()
                batch_embeddings = batch_embeddings / np.linalg.norm(
                    batch_embeddings, axis=1, keepdims=True
                )
                
                embeddings.append(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error embedding batch {i}-{i+batch_size}: {e}")
                # Add zero vectors for failed batch
                embeddings.append(np.zeros((len(batch_paths), 512)))
        
        return np.vstack(embeddings)


def cluster_embeddings(embeddings: np.ndarray, eps: float = 0.15, 
                      min_samples: int = 2) -> np.ndarray:
    """
    Cluster embeddings using DBSCAN.
    
    Args:
        embeddings: Array of embeddings (n_samples, embedding_dim)
        eps: Maximum distance between samples in a cluster
        min_samples: Minimum samples in a neighborhood
        
    Returns:
        Array of cluster labels (-1 for noise)
    """
    try:
        # Use cosine distance (1 - cosine similarity)
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
        
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed'
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points")
        
        return labels
        
    except Exception as e:
        logger.error(f"Error clustering embeddings: {e}")
        return np.array([-1] * len(embeddings))


def select_best_representative(image_paths: List[str], 
                               cluster_indices: List[int],
                               quality_scores: List[Dict[str, float]] = None) -> int:
    """
    Select the best representative image from a cluster.
    
    Args:
        image_paths: List of image paths in the cluster
        cluster_indices: Original indices of images
        quality_scores: Optional quality scores for each image
        
    Returns:
        Index of the best representative image
    """
    if not cluster_indices:
        return -1
    
    if len(cluster_indices) == 1:
        return cluster_indices[0]
    
    # If quality scores provided, use them
    if quality_scores:
        best_idx = 0
        best_score = -float('inf')
        
        for i, idx in enumerate(cluster_indices):
            if idx < len(quality_scores):
                score = quality_scores[idx]
                # Composite score: high blur score (sharp), good brightness
                composite = (
                    score.get('blur_score', 0) * 0.6 +
                    (100 - abs(score.get('brightness', 128) - 128)) * 0.4
                )
                
                if composite > best_score:
                    best_score = composite
                    best_idx = i
        
        return cluster_indices[best_idx]
    
    # Default: return first (or middle) image
    return cluster_indices[len(cluster_indices) // 2]


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix for embeddings.
    
    Args:
        embeddings: Array of embeddings
        
    Returns:
        Similarity matrix
    """
    return cosine_similarity(embeddings)


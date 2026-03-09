from typing import List, Optional
import numpy as np
from client.interface import SlopClient

def extract_identity_vector(
    client: SlopClient,
    identity: str,
    activities: List[str],
    model_id: str = "runwayml/stable-diffusion-v1-5",
) -> np.ndarray:
    """
    Compute an identity vector by contrasting prompts with and without an identity term.

    For each activity in `activities`, it computes:
        delta = embed("a {identity} person {activity}") - embed("a person {activity}")
    Returns the average delta over all activities.

    Returns:
        np.ndarray: Identity vector of shape (1, 77, 768)
    """
    deltas = []
    
    for activity in activities:
        # Base prompt: "a person {activity}"
        base_prompt = f"a person {activity}"
        # Identity prompt: "a {identity} person {activity}"
        identity_prompt = f"a {identity} person {activity}"
        
        # Encode both
        base_embeds, _ = client.embed(base_prompt, model_id=model_id)
        identity_embeds, _ = client.embed(identity_prompt, model_id=model_id)
        
        # Identity vector is the direction pointing from base to identity
        deltas.append(identity_embeds - base_embeds)
        
    # Average deltas to isolate the identity component from activity noise
    identity_vector = np.mean(deltas, axis=0)
    return identity_vector

def apply_identity(
    embeds: np.ndarray,
    identity_vector: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Apply an identity vector to existing prompt embeddings."""
    return embeds + scale * identity_vector

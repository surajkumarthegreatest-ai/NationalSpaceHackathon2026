import numpy as np
from scipy.spatial import cKDTree

def detect_conjunctions(r: np.ndarray, threshold_km: float) -> list[tuple[int, int]]:
    """
    Finds all pairs of objects within a collision threshold using a k-d tree.
    
    Complexity:
        - Construction: O(N log N)
        - Query: O(M log N) where M is the number of proximities found.
        - Total: O(N log N) average case, significantly outperforming O(N^2).
    
    Args:
        r: (N, 3) array of positions in kilometers.
        threshold_km: Distance in km to trigger a conjunction warning.
        
    Returns:
        List of unique tuples (i, j) where i < j, identifying objects in proximity.
    """
    # Defensive check: if fewer than 2 objects, no conjunctions possible.
    if r.shape[0] < 2:
        return []

    # Build the k-d tree. Leafsize 16-32 is optimal for 3D spatial queries 
    # to balance tree depth vs. brute-force overhead at the nodes.
    tree = cKDTree(r, leafsize=16)

    # query_pairs is a C-optimized routine specifically for finding all 
    # pairs within a fixed radius. It avoids redundant (j, i) checks.
    conjunction_set = tree.query_pairs(r=threshold_km, output_type='set')

    # Convert the set of unique pairs to a sorted list of tuples for deterministic output.
    return sorted(list(conjunction_set))
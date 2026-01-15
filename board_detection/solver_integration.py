"""
Integration between existing board detection pipeline and the new solver.

This module converts outputs from your existing classifiers/detectors
into the format expected by the BoardSolver.
"""

from typing import Dict, Set, List
import numpy as np
from board_detection.board_solver import (
    BoardSolver, 
    BoardConfiguration,
    create_default_solver
)
from board_detection.graph_clustering import angle_diff
from board_detection.board_pipeline import BoardDetectionResult


def prepare_solver_data_from_pipeline(result: BoardDetectionResult) -> Dict:
    """
    Convert BoardDetectionResult to solver data format.
    
    Args:
        result: Output from board_pipelineYolo
    
    Returns:
        Data dict for BoardSolver
    """
    data = {}
    
    # 1. White triangle detections
    detected_white_cells = set(result.white_triangles)
    white_confidences = {}
    
    if result.white_triangle_detections:
        for wt in result.white_triangle_detections:
            # You could add confidence based on detection quality
            # For now, use uniform confidence
            white_confidences[wt.cell_id] = 1.0
    
    data['detected_white_cells'] = detected_white_cells
    data['white_confidences'] = white_confidences
    
    # 2. Clustering results
    # Extract clusters from piece detections
    # Note: Your current pipeline already does clustering, so we use those results
    clusters = []
    for piece in result.pieces:
        clusters.append(set(piece.cell_ids))
    
    data['clusters'] = clusters
    
    # 3. Color values
    # Use theta_med (hue angle) as the color value
    color_values = {}
    if result.color_data:
        for cell_id, color_info in result.color_data.items():
            color_values[cell_id] = color_info.get('theta_med', 0.0)
    
    data['color_values'] = color_values
    data['color_distance_fn'] = angle_diff
    
    # 4. Empty classifier probabilities
    # TODO: Add when you have empty classifier integrated
    # For now, use heuristic: cells far from pieces are likely empty
    empty_probs = estimate_empty_probabilities_from_clustering(clusters, detected_white_cells)
    data['empty_probs'] = empty_probs
    
    return data


def estimate_empty_probabilities_from_clustering(
    clusters: List[Set[int]], 
    white_cells: Set[int]
) -> Dict[int, float]:
    """
    Estimate empty probabilities based on clustering.
    
    Cells in clusters or white detections are likely occupied.
    Other cells are likely empty.
    
    Args:
        clusters: List of cell clusters
        white_cells: Detected white triangle cells
    
    Returns:
        Dict mapping cell_id to probability of being empty
    """
    occupied = set(white_cells)
    for cluster in clusters:
        occupied.update(cluster)
    
    probs = {}
    for cell_id in range(1, 49):
        if cell_id in occupied:
            probs[cell_id] = 0.1  # Low probability of being empty
        else:
            probs[cell_id] = 0.9  # High probability of being empty
    
    return probs


def solve_board_configuration(
    result: BoardDetectionResult,
    solver: BoardSolver = None,
    search_strategy: str = 'auto',
    time_budget: float = 2.0,
    verbose: bool = True
) -> BoardConfiguration:
    """
    Find optimal board configuration from pipeline results.
    
    Args:
        result: Output from board_pipelineYolo
        solver: BoardSolver instance (creates default if None)
        search_strategy: 'greedy', 'annealing', 'beam', 'restarts', or 'auto'
        time_budget: Time budget in seconds (used for 'auto' strategy)
        verbose: Print progress
    
    Returns:
        Optimal BoardConfiguration
    """
    from board_detection.improved_search import ImprovedBoardSolver, solve_with_best_strategy
    
    if solver is None:
        solver = create_default_solver()
    
    # Upgrade to improved solver if needed
    if not isinstance(solver, ImprovedBoardSolver):
        solver = ImprovedBoardSolver(solver.scorers)
    
    # Prepare data
    data = prepare_solver_data_from_pipeline(result)
    
    if verbose:
        print(f"Solving with strategy: {search_strategy}")
    
    # Choose search strategy
    if search_strategy == 'auto':
        config, stats = solve_with_best_strategy(data, solver, time_budget, verbose)
    elif search_strategy == 'greedy':
        initial = solver.solve_from_clustering(data)
        config = solver.solve_greedy(initial, data, max_iterations=100)
    elif search_strategy == 'annealing':
        initial = solver.solve_from_clustering(data)
        config = solver.solve_simulated_annealing(initial, data, verbose=verbose)
    elif search_strategy == 'beam':
        initial = solver.solve_from_clustering(data)
        config = solver.solve_beam_search(initial, data, verbose=verbose)
    elif search_strategy == 'restarts':
        config = solver.solve_multiple_restarts(
            data, num_restarts=10, search_method='greedy', verbose=verbose
        )
    else:
        raise ValueError(f"Unknown search strategy: {search_strategy}")
    
    if verbose:
        score = solver.compute_total_score(config, data)
        print(f"\nFinal configuration:")
        print(f"  Pieces: {len(config.pieces)}")
        print(f"  White triangles: {len(config.white_triangles)}")
        print(f"  Empty cells: {len(config.empty_cells)}")
        print(f"  Valid: {config.is_valid()}")
        print(f"  Score: {score:.2f}")
    
    return config


def add_empty_classifier_to_data(
    data: Dict,
    empty_classifier_fn: callable,
    patches: np.ndarray
) -> Dict:
    """
    Add empty classifier probabilities to solver data.
    
    Args:
        data: Existing solver data dict
        empty_classifier_fn: Function that takes patches and returns probabilities
        patches: Array of image patches for each cell
    
    Returns:
        Updated data dict
    """
    # Get probabilities from classifier
    empty_probs_array = empty_classifier_fn(patches)
    
    # Convert to dict
    empty_probs = {}
    for cell_id, prob in enumerate(empty_probs_array, start=1):
        empty_probs[cell_id] = float(prob)
    
    data['empty_probs'] = empty_probs
    return data


def add_pairwise_color_classifier_to_data(
    data: Dict,
    pairwise_classifier_fn: callable,
    patches: np.ndarray
) -> Dict:
    """
    Add pairwise color similarity classifier to solver data.
    
    This allows you to train a model that takes two patches and predicts
    if they're the same color/piece.
    
    Args:
        data: Existing solver data dict
        pairwise_classifier_fn: Function(patch1, patch2) -> similarity_score
        patches: Array of image patches for each cell
    
    Returns:
        Updated data dict with 'pairwise_similarities'
    """
    n_cells = len(patches)
    similarities = {}
    
    # Compute pairwise similarities
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            cell_i = i + 1  # Cell IDs are 1-indexed
            cell_j = j + 1
            
            similarity = pairwise_classifier_fn(patches[i], patches[j])
            similarities[(cell_i, cell_j)] = float(similarity)
            similarities[(cell_j, cell_i)] = float(similarity)  # Symmetric
    
    data['pairwise_similarities'] = similarities
    return data


if __name__ == "__main__":
    # Example: integrate with existing pipeline
    from board_detection.board_pipeline import board_pipelineYolo
    
    image_path = "/path/to/test/image.jpg"
    
    # Run existing pipeline
    result = board_pipelineYolo(image_path)
    
    if result is not None:
        # Solve for optimal configuration
        config = solve_board_configuration(result)
        
        # Print results
        print("\nDetected pieces:")
        for piece in config.pieces:
            print(f"  {piece.piece_name}: cells {piece.cell_ids}")
        
        print(f"\nWhite triangles: {sorted(config.white_triangles)}")
        print(f"Empty cells: {sorted(config.empty_cells)}")

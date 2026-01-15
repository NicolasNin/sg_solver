"""
Example usage of the board solver framework.

This script shows how to:
1. Run the existing pipeline
2. Prepare data for the solver
3. Run the solver with different configurations
4. Visualize results
"""

import cv2
import numpy as np
from pathlib import Path

from board_detection.board_pipeline import board_pipelineYolo, visualize_detection
from board_detection.solver_integration import (
    solve_board_configuration,
    prepare_solver_data_from_pipeline
)
from board_detection.board_solver import (
    BoardSolver,
    EmptyClassifierScorer,
    WhiteTriangleScorer,
    ClusteringScorer,
    ShapeConstraintScorer,
    ColorConsistencyScorer
)
from board_detection.advanced_scorers import (
    PairwiseColorScorer,
    GraphClusteringScorer,
    LearnedColorAnchorScorer,
    WhiteBalanceScorer
)


def example_basic_solver(image_path: str):
    """
    Example: Use basic solver with default weights.
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Solver")
    print("=" * 60)
    
    # Run pipeline
    print(f"\nProcessing: {image_path}")
    result = board_pipelineYolo(image_path)
    
    if result is None:
        print("Pipeline failed!")
        return
    
    # Solve with default solver
    config = solve_board_configuration(result, use_greedy_optimization=True)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nPieces detected: {len(config.pieces)}")
    for piece in sorted(config.pieces, key=lambda p: p.piece_name):
        print(f"  {piece.piece_name:3s}: {len(piece.cell_ids)} cells at {piece.cell_ids}")
    
    print(f"\nWhite triangles: {len(config.white_triangles)}")
    print(f"  Cells: {sorted(config.white_triangles)}")
    
    print(f"\nEmpty cells: {len(config.empty_cells)}")
    print(f"  Cells: {sorted(config.empty_cells)}")
    
    print(f"\nConfiguration valid: {config.is_valid()}")
    
    return config, result


def example_custom_weights(image_path: str):
    """
    Example: Create solver with custom scorer weights.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Scorer Weights")
    print("=" * 60)
    
    # Run pipeline
    result = board_pipelineYolo(image_path)
    if result is None:
        return
    
    # Create custom solver
    # Increase weight on white triangles (very reliable)
    # Decrease weight on color (less reliable)
    scorers = [
        EmptyClassifierScorer(weight=1.0),
        WhiteTriangleScorer(weight=5.0),      # Increased from 3.0
        ClusteringScorer(weight=2.0),
        ShapeConstraintScorer(weight=5.0),
        ColorConsistencyScorer(weight=0.2),   # Decreased from 0.5
    ]
    
    solver = BoardSolver(scorers)
    
    # Solve
    config = solve_board_configuration(result, solver=solver)
    
    print(f"\nPieces: {len(config.pieces)}, Whites: {len(config.white_triangles)}, Valid: {config.is_valid()}")
    
    return config, result


def example_with_advanced_scorers(image_path: str):
    """
    Example: Use advanced scorers (requires additional data).
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Advanced Scorers")
    print("=" * 60)
    
    # Run pipeline
    result = board_pipelineYolo(image_path)
    if result is None:
        return
    
    # Mock color distributions (in practice, build from training data)
    color_distributions = {
        'T1': {'mean_hue': 210, 'std_hue': 15},  # Blue
        'T2': {'mean_hue': 50, 'std_hue': 10},   # Yellow
        'T3': {'mean_hue': 180, 'std_hue': 15},  # Cyan
        'T4': {'mean_hue': 330, 'std_hue': 15},  # Pink
        'T5': {'mean_hue': 0, 'std_hue': 10},    # Red
        'TF': {'mean_hue': 280, 'std_hue': 15},  # Purple
        'L4': {'mean_hue': 30, 'std_hue': 10},   # Orange
        'EL': {'mean_hue': 150, 'std_hue': 15},  # Green
        '4U': {'mean_hue': 80, 'std_hue': 15},   # Lime
        '4D': {'mean_hue': 25, 'std_hue': 15},   # Brown
    }
    
    # Create solver with advanced scorers
    scorers = [
        EmptyClassifierScorer(weight=1.0),
        WhiteTriangleScorer(weight=4.0),
        ShapeConstraintScorer(weight=5.0),
        ClusteringScorer(weight=1.5),
        ColorConsistencyScorer(weight=0.5),
        # Advanced scorers
        GraphClusteringScorer(weight=1.0),
        LearnedColorAnchorScorer(color_distributions, weight=1.0),
        WhiteBalanceScorer(weight=0.3),
    ]
    
    solver = BoardSolver(scorers)
    
    # Solve
    config = solve_board_configuration(result, solver=solver)
    
    print(f"\nPieces: {len(config.pieces)}, Whites: {len(config.white_triangles)}, Valid: {config.is_valid()}")
    
    return config, result


def compare_solvers(image_path: str):
    """
    Compare different solver configurations on the same image.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Comparing Solver Configurations")
    print("=" * 60)
    
    # Run pipeline once
    result = board_pipelineYolo(image_path)
    if result is None:
        return
    
    # Prepare data
    data = prepare_solver_data_from_pipeline(result)
    
    # Test different configurations
    configs = {
        "Default": [
            EmptyClassifierScorer(weight=1.0),
            WhiteTriangleScorer(weight=3.0),
            ClusteringScorer(weight=2.0),
            ShapeConstraintScorer(weight=5.0),
            ColorConsistencyScorer(weight=0.5),
        ],
        "High White Weight": [
            EmptyClassifierScorer(weight=1.0),
            WhiteTriangleScorer(weight=10.0),  # Very high
            ClusteringScorer(weight=2.0),
            ShapeConstraintScorer(weight=5.0),
            ColorConsistencyScorer(weight=0.5),
        ],
        "Clustering Only": [
            WhiteTriangleScorer(weight=3.0),
            ClusteringScorer(weight=10.0),  # Only clustering
            ShapeConstraintScorer(weight=5.0),
        ],
        "No Color": [
            EmptyClassifierScorer(weight=1.0),
            WhiteTriangleScorer(weight=3.0),
            ClusteringScorer(weight=2.0),
            ShapeConstraintScorer(weight=5.0),
            # No color scorer
        ],
    }
    
    results = {}
    
    for name, scorers in configs.items():
        print(f"\n{name}:")
        solver = BoardSolver(scorers)
        
        initial = solver.solve_from_clustering(data)
        final = solver.solve_greedy(initial, data, max_iterations=50)
        
        score = solver.compute_total_score(final, data)
        
        print(f"  Pieces: {len(final.pieces)}")
        print(f"  Whites: {len(final.white_triangles)}")
        print(f"  Valid: {final.is_valid()}")
        print(f"  Score: {score:.2f}")
        
        results[name] = {
            'config': final,
            'score': score,
            'valid': final.is_valid()
        }
    
    # Find best
    best_name = max(results.keys(), key=lambda k: results[k]['score'])
    print(f"\n{'='*60}")
    print(f"Best configuration: {best_name}")
    print(f"  Score: {results[best_name]['score']:.2f}")
    print(f"  Valid: {results[best_name]['valid']}")
    
    return results


def visualize_solver_results(image_path: str, output_path: str = None):
    """
    Visualize solver results on the board image.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Visualization")
    print("=" * 60)
    
    # Run pipeline and solver
    result = board_pipelineYolo(image_path)
    if result is None:
        return
    
    config = solve_board_configuration(result)
    
    # Visualize using existing function
    vis_img = visualize_detection(result, draw_piece_names=True)
    
    # Add solver-specific annotations
    # TODO: Could add more detailed visualization here
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, vis_img)
        print(f"\nVisualization saved to: {output_path}")
    else:
        # Display
        cv2.imshow("Solver Results", vis_img)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return vis_img


if __name__ == "__main__":
    # Example image path - update this to your test image
    test_image = "/home/nicolas/code/star_genius_solver/data/test_image.jpg"
    
    # Check if file exists
    if not Path(test_image).exists():
        print(f"Test image not found: {test_image}")
        print("Please update the path in this script.")
        exit(1)
    
    # Run examples
    print("Running board solver examples...\n")
    
    # Example 1: Basic solver
    config1, result1 = example_basic_solver(test_image)
    
    # Example 2: Custom weights
    # config2, result2 = example_custom_weights(test_image)
    
    # Example 3: Advanced scorers
    # config3, result3 = example_with_advanced_scorers(test_image)
    
    # Example 4: Compare solvers
    # comparison = compare_solvers(test_image)
    
    # Example 5: Visualization
    # vis = visualize_solver_results(test_image, output_path="solver_result.jpg")
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)

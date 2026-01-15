"""
Board configuration solver using constraint satisfaction and scoring.

This module reconciles multiple noisy signals (empty classifier, white detector,
clustering, color, etc.) to find the most likely board configuration.

The approach:
1. Define scorers for each signal/constraint
2. Search for configurations that maximize total weighted score
3. Enforce hard constraints (unique pieces, valid shapes)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable
import numpy as np
from collections import defaultdict
import heapq

from sg_solver import PIECE_ORIENTATIONS, ALL_PIECES
from board_detection.piece_identification import identify_piece_from_cells
from board_detection.reference_data import CELLS_GRAPH


@dataclass
class PiecePlacement:
    """Represents a piece placed on the board."""
    piece_name: str
    cell_ids: List[int]
    orientation_idx: int
    anchor_cell: int
    
    def __hash__(self):
        return hash((self.piece_name, tuple(sorted(self.cell_ids)), self.orientation_idx))
    
    def __eq__(self, other):
        return (self.piece_name == other.piece_name and 
                set(self.cell_ids) == set(other.cell_ids) and
                self.orientation_idx == other.orientation_idx)


@dataclass
class BoardConfiguration:
    """Complete board state with pieces and white triangles."""
    pieces: List[PiecePlacement] = field(default_factory=list)
    white_triangles: Set[int] = field(default_factory=set)
    empty_cells: Set[int] = field(default_factory=set)
    
    def get_occupied_cells(self) -> Set[int]:
        """Get all cells occupied by pieces or white triangles."""
        occupied = set(self.white_triangles)
        for piece in self.pieces:
            occupied.update(piece.cell_ids)
        return occupied
    
    def is_valid(self) -> bool:
        """Check if configuration is valid (no overlaps, all cells accounted for)."""
        occupied = self.get_occupied_cells()
        
        # Check no overlaps
        total_cells = len(self.white_triangles) + sum(len(p.cell_ids) for p in self.pieces)
        if total_cells != len(occupied):
            return False
        
        # Check all 48 cells accounted for
        expected_empty = set(range(1, 49)) - occupied
        if expected_empty != self.empty_cells:
            return False
        
        # Check unique pieces (except T3/3B)
        piece_names = [p.piece_name for p in self.pieces]
        # Normalize T3/3B
        normalized = []
        for name in piece_names:
            if name in ["T3", "3B"]:
                normalized.append("T3_OR_3B")
            else:
                normalized.append(name)
        
        # Should have at most 2 T3_OR_3B, and 1 of everything else
        from collections import Counter
        counts = Counter(normalized)
        for name, count in counts.items():
            if name == "T3_OR_3B" and count > 2:
                return False
            elif name != "T3_OR_3B" and count > 1:
                return False
        
        return True
    
    def copy(self):
        """Deep copy of configuration."""
        return BoardConfiguration(
            pieces=self.pieces.copy(),
            white_triangles=self.white_triangles.copy(),
            empty_cells=self.empty_cells.copy()
        )


class Scorer:
    """Base class for scoring functions."""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def score(self, config: BoardConfiguration, data: Dict) -> float:
        """Compute score for a configuration. Higher is better."""
        raise NotImplementedError
    
    def weighted_score(self, config: BoardConfiguration, data: Dict) -> float:
        """Compute weighted score."""
        return self.weight * self.score(config, data)


class EmptyClassifierScorer(Scorer):
    """Score based on empty/occupied classifier probabilities."""
    
    def score(self, config: BoardConfiguration, data: Dict) -> float:
        """
        Reward configurations where:
        - Empty cells have high empty probability
        - Occupied cells have low empty probability
        
        data should contain:
            'empty_probs': Dict[int, float] - probability that each cell is empty, keys from 1 to 48
        """
        empty_probs = data.get('empty_probs', {})
        if not empty_probs:
            return 0.0
        
        score = 0.0
        occupied = config.get_occupied_cells()
        
        for cell_id in range(1, 49):
            prob_empty = empty_probs.get(cell_id, 0.5)
            
            if cell_id in config.empty_cells:
                # Reward high empty probability
                score += prob_empty
            elif cell_id in occupied:
                # Reward low empty probability (high occupied probability)
                score += (1 - prob_empty)
        
        # Normalize by number of cells
        return score / 48.0


class WhiteTriangleScorer(Scorer):
    """Score based on white triangle detections."""
    
    def score(self, config: BoardConfiguration, data: Dict) -> float:
        """
        Reward configurations where white triangles match detections.
        
        data should contain:
            'detected_white_cells': Set[int] - cells where white triangles detected
            'white_confidences': Dict[int, float] - confidence for each detection
        """
        detected = data.get('detected_white_cells', set())
        confidences = data.get('white_confidences', {})
        
        if not detected:
            return 0.0
        
        score = 0.0
        
        # Reward matching detections
        for cell_id in config.white_triangles:
            if cell_id in detected:
                conf = confidences.get(cell_id, 1.0)
                score += conf
            else:
                # Penalty for white triangle not detected
                score -= 0.5
        
        # Penalty for detected whites not in config
        for cell_id in detected:
            if cell_id not in config.white_triangles:
                conf = confidences.get(cell_id, 1.0)
                score -= 0.3 * conf
        
        # Soft constraint: prefer 7 white triangles
        num_whites = len(config.white_triangles)
        if num_whites == 7:
            score += 2.0
        else:
            score -= abs(num_whites - 7) * 0.5
        
        return score


class ClusteringScorer(Scorer):
    """Score based on patch clustering results."""
    
    def score(self, config: BoardConfiguration, data: Dict) -> float:
        """
        Reward configurations where pieces match clustering.
        
        data should contain:
            'clusters': List[Set[int]] - clusters of similar cells
        """
        clusters = data.get('clusters', [])
        if not clusters:
            return 0.0
        
        score = 0.0
        
        # For each piece, check how well it matches a cluster
        for piece in config.pieces:
            piece_cells = set(piece.cell_ids)
            
            # Find best matching cluster
            best_match = 0.0
            for cluster in clusters:
                # Jaccard similarity
                intersection = len(piece_cells & cluster)
                union = len(piece_cells | cluster)
                if union > 0:
                    similarity = intersection / union
                    best_match = max(best_match, similarity)
            
            score += best_match
        
        # Normalize by number of pieces
        if config.pieces:
            score /= len(config.pieces)
        
        return score


class ShapeConstraintScorer(Scorer):
    """Score based on valid piece shapes and uniqueness."""
    
    def score(self, config: BoardConfiguration, data: Dict) -> float:
        """
        Hard constraint: all pieces must be valid shapes.
        Soft constraint: prefer using each piece exactly once.
        """
        score = 0.0
        
        # Check each piece is a valid shape
        for piece in config.pieces:
            piece_name, orientation, anchor = identify_piece_from_cells(piece.cell_ids)
            if piece_name is None:
                # Invalid shape - heavy penalty
                score -= 10.0
            elif piece_name != piece.piece_name:
                # Shape doesn't match claimed piece
                score -= 5.0
        
        # Check piece uniqueness
        piece_counts = defaultdict(int)
        for piece in config.pieces:
            # Normalize T3/3B
            name = piece.piece_name
            if name in ["T3", "3B"]:
                name = "T3_OR_3B"
            piece_counts[name] += 1
        
        # Reward using each piece exactly once (except T3/3B which can be 0-2)
        for piece in ALL_PIECES:
            name = piece.name
            if name in ["T3", "3B"]:
                name = "T3_OR_3B"
            
            count = piece_counts.get(name, 0)
            
            if name == "T3_OR_3B":
                # Can have 0, 1, or 2
                if count <= 2:
                    score += count * 0.5
                else:
                    score -= (count - 2) * 5.0
            else:
                # Should have exactly 1
                if count == 1:
                    score += 1.0
                elif count == 0:
                    score += 0.0  # Missing piece is ok
                else:
                    score -= (count - 1) * 5.0  # Duplicate is bad
        
        return score


class ColorConsistencyScorer(Scorer):
    """Score based on color consistency within pieces."""
    
    def score(self, config: BoardConfiguration, data: Dict) -> float:
        """
        Reward configurations where cells in same piece have similar colors.
        
        data should contain:
            'color_values': Dict[int, float] - color value (e.g., hue) for each cell
            'color_distance_fn': Callable[[float, float], float] - distance function
        """
        color_values = data.get('color_values', {})
        distance_fn = data.get('color_distance_fn', lambda a, b: abs(a - b))
        
        if not color_values:
            return 0.0
        
        score = 0.0
        
        for piece in config.pieces:
            if len(piece.cell_ids) <= 1:
                continue
            
            # Compute average pairwise distance within piece
            distances = []
            for i, cell1 in enumerate(piece.cell_ids):
                for cell2 in piece.cell_ids[i+1:]:
                    if cell1 in color_values and cell2 in color_values:
                        dist = distance_fn(color_values[cell1], color_values[cell2])
                        distances.append(dist)
            
            if distances:
                avg_dist = np.mean(distances)
                # Lower distance = higher score
                # Use exponential decay: score = exp(-dist/scale)
                score += np.exp(-avg_dist / 10.0)
        
        # Normalize by number of pieces
        if config.pieces:
            score /= len(config.pieces)
        
        return score


class BoardSolver:
    """Solver that finds optimal board configuration using scoring."""
    
    def __init__(self, scorers: List[Scorer]):
        self.scorers = scorers
    
    def compute_total_score(self, config: BoardConfiguration, data: Dict) -> float:
        """Compute total weighted score across all scorers."""
        total = 0.0
        for scorer in self.scorers:
            total += scorer.weighted_score(config, data)
        return total
    
    def solve_greedy(self, 
                     initial_config: BoardConfiguration,
                     data: Dict,
                     max_iterations: int = 100) -> BoardConfiguration:
        """
        Greedy search with local improvements.
        
        Args:
            initial_config: Starting configuration
            data: Data dict with all classifier outputs
            max_iterations: Max number of improvement iterations
        
        Returns:
            Best configuration found
        """
        current = initial_config.copy()
        current_score = self.compute_total_score(current, data)
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try local modifications
            neighbors = self._generate_neighbors(current, data)
            
            for neighbor in neighbors:
                if not neighbor.is_valid():
                    continue
                
                score = self.compute_total_score(neighbor, data)
                if score > current_score:
                    current = neighbor
                    current_score = score
                    improved = True
                    break
            
            if not improved:
                break
        
        return current
    
    def _generate_neighbors(self, config: BoardConfiguration, data: Dict) -> List[BoardConfiguration]:
        """Generate neighboring configurations by small modifications."""
        neighbors = []
        
        # Try removing a piece
        for i, piece in enumerate(config.pieces):
            neighbor = config.copy()
            removed_piece = neighbor.pieces.pop(i)
            neighbor.empty_cells.update(removed_piece.cell_ids)
            neighbors.append(neighbor)
        
        # Try adding a piece to empty cells
        clusters = data.get('clusters', [])
        for cluster in clusters:
            # Check if cluster is mostly empty
            empty_overlap = len(cluster & config.empty_cells)
            if empty_overlap >= len(cluster) * 0.5:
                # Try to identify piece
                piece_name, orientation, anchor = identify_piece_from_cells(list(cluster))
                if piece_name is not None:
                    neighbor = config.copy()
                    placement = PiecePlacement(
                        piece_name=piece_name,
                        cell_ids=list(cluster),
                        orientation_idx=orientation,
                        anchor_cell=anchor
                    )
                    neighbor.pieces.append(placement)
                    neighbor.empty_cells -= cluster
                    neighbors.append(neighbor)
        
        # Try swapping piece identities (for similar shapes)
        # TODO: implement piece swapping
        
        return neighbors
    
    def solve_from_clustering(self, data: Dict) -> BoardConfiguration:
        """
        Build initial configuration from clustering results.
        
        Args:
            data: Must contain 'clusters' and 'detected_white_cells'
        
        Returns:
            Initial configuration
        """
        clusters = data.get('clusters', [])
        detected_whites = data.get('detected_white_cells', set())
        
        config = BoardConfiguration()
        config.white_triangles = detected_whites.copy()
        
        occupied = set(detected_whites)
        
        # Try to place pieces from clusters
        for cluster in clusters:
            # Skip if overlaps with already placed
            if cluster & occupied:
                continue
            
            # Identify piece
            piece_name, orientation, anchor = identify_piece_from_cells(list(cluster))
            if piece_name is not None:
                placement = PiecePlacement(
                    piece_name=piece_name,
                    cell_ids=list(cluster),
                    orientation_idx=orientation,
                    anchor_cell=anchor
                )
                config.pieces.append(placement)
                occupied.update(cluster)
        
        # Mark remaining as empty
        config.empty_cells = set(range(1, 49)) - occupied
        
        return config


def create_default_solver() -> BoardSolver:
    """Create solver with default scorer weights."""
    scorers = [
        EmptyClassifierScorer(weight=1.0),
        WhiteTriangleScorer(weight=3.0),  # High weight - white detection is reliable
        ClusteringScorer(weight=2.0),
        ShapeConstraintScorer(weight=5.0),  # High weight - shapes must be valid
        ColorConsistencyScorer(weight=0.5),  # Lower weight - color is less reliable
    ]
    return BoardSolver(scorers)


if __name__ == "__main__":
    # Example usage
    solver = create_default_solver()
    
    # Mock data
    data = {
        'empty_probs': {i: 0.5 for i in range(1, 49)},
        'detected_white_cells': {1, 5, 10, 15, 20, 25, 30},
        'white_confidences': {i: 0.9 for i in [1, 5, 10, 15, 20, 25, 30]},
        'clusters': [
            {2, 3, 4},
            {6, 7, 8, 9},
        ],
        'color_values': {i: float(i) for i in range(1, 49)},
        'color_distance_fn': lambda a, b: abs(a - b),
    }
    
    # Build initial config from clustering
    initial = solver.solve_from_clustering(data)
    print(f"Initial config: {len(initial.pieces)} pieces, {len(initial.white_triangles)} whites")
    print(f"Initial score: {solver.compute_total_score(initial, data):.2f}")
    
    # Optimize
    final = solver.solve_greedy(initial, data)
    print(f"Final config: {len(final.pieces)} pieces, {len(final.white_triangles)} whites")
    print(f"Final score: {solver.compute_total_score(final, data):.2f}")
    print(f"Valid: {final.is_valid()}")

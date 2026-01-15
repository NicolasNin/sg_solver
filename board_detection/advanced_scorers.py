"""
Advanced scorers for board configuration optimization.

These scorers use more sophisticated signals like pairwise classifiers,
learned color models, etc.
"""

from typing import Dict, Set, List, Tuple
import numpy as np
from collections import defaultdict

from board_detection.board_solver import Scorer, BoardConfiguration
from board_detection.reference_data import CELLS_GRAPH


class PairwiseColorScorer(Scorer):
    """
    Score based on pairwise color similarity classifier.
    
    This scorer uses a trained model that predicts whether two patches
    are the same color/piece.
    """
    
    def score(self, config: BoardConfiguration, data: Dict) -> float:
        """
        Reward configurations where:
        - Cells in same piece have high pairwise similarity
        - Cells in different pieces have low pairwise similarity
        
        data should contain:
            'pairwise_similarities': Dict[(int, int), float] - similarity scores
        """
        similarities = data.get('pairwise_similarities', {})
        if not similarities:
            return 0.0
        
        score = 0.0
        n_comparisons = 0
        
        # Score within-piece similarities (should be high)
        for piece in config.pieces:
            for i, cell1 in enumerate(piece.cell_ids):
                for cell2 in piece.cell_ids[i+1:]:
                    sim = similarities.get((cell1, cell2), 0.5)
                    score += sim  # High similarity = good
                    n_comparisons += 1
        
        # Score between-piece similarities (should be low)
        # Only check neighboring pieces to avoid O(n²) comparisons
        for i, piece1 in enumerate(config.pieces):
            for piece2 in config.pieces[i+1:]:
                # Check if pieces are neighbors
                if self._are_neighbors(piece1.cell_ids, piece2.cell_ids):
                    # Sample a few cell pairs
                    for cell1 in piece1.cell_ids[:2]:  # Sample first 2 cells
                        for cell2 in piece2.cell_ids[:2]:
                            sim = similarities.get((cell1, cell2), 0.5)
                            score += (1 - sim)  # Low similarity = good
                            n_comparisons += 1
        
        # Normalize
        if n_comparisons > 0:
            score /= n_comparisons
        
        return score
    
    def _are_neighbors(self, cells1: List[int], cells2: List[int]) -> bool:
        """Check if two sets of cells are adjacent on the graph."""
        for cell1 in cells1:
            neighbors = CELLS_GRAPH.get(cell1, [])
            if any(cell2 in neighbors for cell2 in cells2):
                return True
        return False


class GraphClusteringScorer(Scorer):
    """
    Score based on graph clustering with edge weights.
    
    This uses pairwise similarities as edge weights and rewards
    configurations that respect the clustering.
    """
    
    def score(self, config: BoardConfiguration, data: Dict) -> float:
        """
        Use graph cut metric: reward high within-cluster similarity
        and low between-cluster similarity.
        
        data should contain:
            'pairwise_similarities': Dict[(int, int), float]
        """
        similarities = data.get('pairwise_similarities', {})
        if not similarities:
            return 0.0
        
        # Build cell-to-piece mapping
        cell_to_piece = {}
        for i, piece in enumerate(config.pieces):
            for cell in piece.cell_ids:
                cell_to_piece[cell] = i
        
        # White triangles and empty cells are their own "pieces"
        for cell in config.white_triangles:
            cell_to_piece[cell] = -1  # Special marker
        for cell in config.empty_cells:
            cell_to_piece[cell] = -2  # Special marker
        
        # Compute graph cut score
        within_score = 0.0
        between_score = 0.0
        
        for (cell1, cell2), sim in similarities.items():
            if cell1 not in cell_to_piece or cell2 not in cell_to_piece:
                continue
            
            # Check if cells are neighbors in graph
            if cell2 not in CELLS_GRAPH.get(cell1, []):
                continue
            
            piece1 = cell_to_piece[cell1]
            piece2 = cell_to_piece[cell2]
            
            if piece1 == piece2:
                # Same piece - want high similarity
                within_score += sim
            else:
                # Different pieces - want low similarity
                between_score += (1 - sim)
        
        return within_score + between_score


class LearnedColorAnchorScorer(Scorer):
    """
    Score based on learned color distributions for each piece type.
    
    This scorer uses a database of color distributions built from
    labeled training data.
    """
    
    def __init__(self, color_distributions: Dict[str, Dict], weight: float = 1.0):
        """
        Args:
            color_distributions: Dict mapping piece_name to color statistics
                Example: {
                    'T1': {'mean_hue': 200, 'std_hue': 10, ...},
                    'T2': {'mean_hue': 50, 'std_hue': 15, ...},
                }
            weight: Scorer weight
        """
        super().__init__(weight)
        self.color_distributions = color_distributions
    
    def score(self, config: BoardConfiguration, data: Dict) -> float:
        """
        Reward configurations where piece colors match learned distributions.
        
        data should contain:
            'color_values': Dict[int, float] - color values for each cell
        """
        color_values = data.get('color_values', {})
        if not color_values or not self.color_distributions:
            return 0.0
        
        score = 0.0
        
        for piece in config.pieces:
            # Get color distribution for this piece type
            dist = self.color_distributions.get(piece.piece_name)
            if dist is None:
                continue
            
            # Compute how well the piece's colors match the distribution
            piece_colors = [color_values[cell] for cell in piece.cell_ids 
                          if cell in color_values]
            
            if not piece_colors:
                continue
            
            # Simple Gaussian likelihood
            mean_hue = dist.get('mean_hue', 0)
            std_hue = dist.get('std_hue', 30)
            
            for color in piece_colors:
                # Circular distance for hue
                diff = abs(color - mean_hue)
                diff = min(diff, 360 - diff)
                
                # Gaussian probability
                prob = np.exp(-(diff ** 2) / (2 * std_hue ** 2))
                score += prob
        
        # Normalize by total cells
        total_cells = sum(len(p.cell_ids) for p in config.pieces)
        if total_cells > 0:
            score /= total_cells
        
        return score


class WhiteBalanceScorer(Scorer):
    """
    Score based on white balance consistency.
    
    Uses detected white triangles to estimate lighting and adjust
    color expectations accordingly.
    """
    
    def score(self, config: BoardConfiguration, data: Dict) -> float:
        """
        Estimate white balance from white triangles and check if
        other colors are consistent with this lighting.
        
        data should contain:
            'color_data': Dict[int, Dict] - full color data (L, A, B, etc.)
        """
        color_data = data.get('color_data', {})
        if not color_data:
            return 0.0
        
        # Estimate white balance from white triangles
        white_L_values = []
        white_A_values = []
        white_B_values = []
        
        for cell_id in config.white_triangles:
            if cell_id in color_data:
                cd = color_data[cell_id]
                white_L_values.append(cd.get('L_med', 100))
                white_A_values.append(cd.get('A_med', 0))
                white_B_values.append(cd.get('B_med', 0))
        
        if not white_L_values:
            return 0.0
        
        # Expected white: L=high, A=0, B=0
        # Deviation indicates color cast
        avg_L = np.mean(white_L_values)
        avg_A = np.mean(white_A_values)
        avg_B = np.mean(white_B_values)
        
        # Score: penalize if whites are not neutral
        # Good white should have L > 150 and A, B near 0
        score = 0.0
        
        if avg_L > 150:
            score += 1.0
        else:
            score += avg_L / 150.0
        
        # Penalize color cast
        color_cast = np.sqrt(avg_A**2 + avg_B**2)
        score += max(0, 1.0 - color_cast / 20.0)
        
        return score / 2.0  # Normalize to [0, 1]


def build_color_distributions_from_training_data(
    training_data: List[Dict]
) -> Dict[str, Dict]:
    """
    Build color distribution database from labeled training images.
    
    Args:
        training_data: List of dicts, each containing:
            - 'piece_name': str
            - 'color_values': List[float] - color values for this piece
    
    Returns:
        Dict mapping piece_name to color statistics
    """
    piece_colors = defaultdict(list)
    
    for sample in training_data:
        piece_name = sample['piece_name']
        colors = sample['color_values']
        piece_colors[piece_name].extend(colors)
    
    distributions = {}
    for piece_name, colors in piece_colors.items():
        colors = np.array(colors)
        
        # Compute circular mean and std for hue
        # Convert to radians
        colors_rad = colors * np.pi / 180.0
        
        # Circular mean
        mean_sin = np.mean(np.sin(colors_rad))
        mean_cos = np.mean(np.cos(colors_rad))
        mean_hue = np.arctan2(mean_sin, mean_cos) * 180.0 / np.pi
        if mean_hue < 0:
            mean_hue += 360
        
        # Circular std
        R = np.sqrt(mean_sin**2 + mean_cos**2)
        std_hue = np.sqrt(-2 * np.log(R)) * 180.0 / np.pi
        
        distributions[piece_name] = {
            'mean_hue': mean_hue,
            'std_hue': max(std_hue, 5.0),  # Minimum std
            'n_samples': len(colors)
        }
    
    return distributions


if __name__ == "__main__":
    # Example: create solver with advanced scorers
    from board_detection.board_solver import (
        BoardSolver,
        EmptyClassifierScorer,
        WhiteTriangleScorer,
        ShapeConstraintScorer
    )
    
    # Build color distributions (mock data)
    color_dists = {
        'T1': {'mean_hue': 210, 'std_hue': 15},  # Blue
        'T2': {'mean_hue': 50, 'std_hue': 10},   # Yellow
        'T5': {'mean_hue': 0, 'std_hue': 10},    # Red
        # ... etc
    }
    
    scorers = [
        EmptyClassifierScorer(weight=1.0),
        WhiteTriangleScorer(weight=3.0),
        ShapeConstraintScorer(weight=5.0),
        PairwiseColorScorer(weight=2.0),
        GraphClusteringScorer(weight=1.5),
        LearnedColorAnchorScorer(color_dists, weight=1.0),
        WhiteBalanceScorer(weight=0.5),
    ]
    
    solver = BoardSolver(scorers)
    print(f"Created solver with {len(scorers)} scorers")

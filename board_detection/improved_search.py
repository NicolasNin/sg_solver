"""
Improved search strategies for board configuration optimization.

This module provides better search algorithms than the basic greedy search:
- Simulated annealing
- Beam search  
- Multiple random restarts
- Expanded neighbor generation
"""

import random
import math
import time
import heapq
from typing import Dict, List, Tuple
from dataclasses import dataclass

from board_detection.board_solver import BoardSolver, BoardConfiguration, PiecePlacement
from board_detection.piece_identification import identify_piece_from_cells


class ImprovedBoardSolver(BoardSolver):
    """Extended BoardSolver with better search strategies."""
    
    def solve_simulated_annealing(self,
                                  initial_config: BoardConfiguration,
                                  data: Dict,
                                  T_start: float = 10.0,
                                  T_end: float = 0.01,
                                  cooling: float = 0.95,
                                  max_iterations: int = 1000,
                                  verbose: bool = False) -> BoardConfiguration:
        """
        Simulated annealing search - can escape local optima.
        
        Args:
            initial_config: Starting configuration
            data: Data dict with classifier outputs
            T_start: Initial temperature (higher = more exploration)
            T_end: Final temperature (lower = more exploitation)
            cooling: Temperature decay rate (0.9-0.99)
            max_iterations: Maximum iterations
            verbose: Print progress
        
        Returns:
            Best configuration found
        """
        current = initial_config.copy()
        current_score = self.compute_total_score(current, data)
        
        best = current.copy()
        best_score = current_score
        
        T = T_start
        iteration = 0
        accepts = 0
        rejects = 0
        
        while T > T_end and iteration < max_iterations:
            # Generate random neighbor
            neighbors = self._generate_neighbors(current, data)
            if not neighbors:
                break
            
            neighbor = random.choice(neighbors)
            
            if not neighbor.is_valid():
                iteration += 1
                continue
            
            neighbor_score = self.compute_total_score(neighbor, data)
            delta = neighbor_score - current_score
            
            # Accept if better, or with probability exp(delta/T)
            if delta > 0:
                accept = True
            else:
                accept_prob = math.exp(delta / T)
                accept = random.random() < accept_prob
            
            if accept:
                current = neighbor
                current_score = neighbor_score
                accepts += 1
                
                # Track best ever seen
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
            else:
                rejects += 1
            
            # Cool down
            T *= cooling
            iteration += 1
            
            if verbose and iteration % 100 == 0:
                accept_rate = accepts / (accepts + rejects) if (accepts + rejects) > 0 else 0
                print(f"Iter {iteration}: T={T:.3f}, current={current_score:.2f}, "
                      f"best={best_score:.2f}, accept_rate={accept_rate:.2%}")
                accepts = 0
                rejects = 0
        
        if verbose:
            print(f"Final: {iteration} iterations, best_score={best_score:.2f}")
        
        return best
    
    def solve_beam_search(self,
                         initial_config: BoardConfiguration,
                         data: Dict,
                         beam_width: int = 5,
                         max_depth: int = 20,
                         verbose: bool = False) -> BoardConfiguration:
        """
        Beam search - keeps top-k configurations at each step.
        
        Args:
            initial_config: Starting configuration
            data: Data dict with classifier outputs
            beam_width: Number of configurations to keep (k)
            max_depth: Maximum search depth
            verbose: Print progress
        
        Returns:
            Best configuration found
        """
        # Priority queue: (-score, unique_id, config)
        # Use unique_id to break ties
        initial_score = self.compute_total_score(initial_config, data)
        beam = [(-initial_score, 0, initial_config)]
        next_id = 1
        
        for depth in range(max_depth):
            # Generate all neighbors from current beam
            all_neighbors = []
            
            for neg_score, _, config in beam:
                neighbors = self._generate_neighbors(config, data)
                
                for neighbor in neighbors:
                    if not neighbor.is_valid():
                        continue
                    
                    score = self.compute_total_score(neighbor, data)
                    all_neighbors.append((-score, next_id, neighbor))
                    next_id += 1
            
            if not all_neighbors:
                if verbose:
                    print(f"Depth {depth}: No valid neighbors, stopping")
                break
            
            # Keep top beam_width
            beam = heapq.nsmallest(beam_width, all_neighbors)
            
            if verbose:
                best_score = -beam[0][0]
                avg_score = -sum(s for s, _, _ in beam) / len(beam)
                print(f"Depth {depth}: best={best_score:.2f}, avg={avg_score:.2f}, "
                      f"beam_size={len(beam)}, neighbors={len(all_neighbors)}")
        
        # Return best from final beam
        best_neg_score, _, best_config = beam[0]
        if verbose:
            print(f"Final: best_score={-best_neg_score:.2f}")
        
        return best_config
    
    def solve_multiple_restarts(self,
                                data: Dict,
                                num_restarts: int = 10,
                                search_method: str = 'greedy',
                                verbose: bool = False,
                                **search_kwargs) -> BoardConfiguration:
        """
        Run search multiple times from different random starting points.
        
        Args:
            data: Data dict with classifier outputs
            num_restarts: Number of random restarts
            search_method: 'greedy', 'annealing', or 'beam'
            verbose: Print progress
            **search_kwargs: Additional arguments for search method
        
        Returns:
            Best configuration found across all restarts
        """
        best_config = None
        best_score = -float('inf')
        
        for restart in range(num_restarts):
            # Generate random initial config by shuffling cluster order
            initial = self._random_initial_config(data)
            
            # Run search
            if search_method == 'greedy':
                final = self.solve_greedy(initial, data, **search_kwargs)
            elif search_method == 'annealing':
                final = self.solve_simulated_annealing(initial, data, **search_kwargs)
            elif search_method == 'beam':
                final = self.solve_beam_search(initial, data, **search_kwargs)
            else:
                raise ValueError(f"Unknown search method: {search_method}")
            
            score = self.compute_total_score(final, data)
            
            if verbose:
                print(f"Restart {restart+1}/{num_restarts}: score={score:.2f}, "
                      f"pieces={len(final.pieces)}, valid={final.is_valid()}")
            
            if score > best_score:
                best_config = final
                best_score = score
        
        if verbose:
            print(f"\nBest across all restarts: score={best_score:.2f}")
        
        return best_config
    
    def _random_initial_config(self, data: Dict) -> BoardConfiguration:
        """Generate random valid initial configuration by shuffling clusters."""
        clusters = list(data.get('clusters', []))
        random.shuffle(clusters)
        
        # Build config with shuffled cluster order
        data_shuffled = {**data, 'clusters': clusters}
        return self.solve_from_clustering(data_shuffled)
    
    def _generate_neighbors_extended(self, config: BoardConfiguration, data: Dict) -> List[BoardConfiguration]:
        """
        Generate neighbors with more move types.
        
        Additional moves beyond base class:
        - Swap two pieces
        - Modify piece orientation
        - Move piece to adjacent cells
        """
        neighbors = []
        
        # Base moves (remove, add)
        neighbors.extend(super()._generate_neighbors(config, data))
        
        # Move: Swap two pieces
        neighbors.extend(self._swap_piece_neighbors(config, data))
        
        # Move: Change piece orientation (for pieces that can flip/rotate)
        # TODO: Implement if needed
        
        return neighbors
    
    def _swap_piece_neighbors(self, config: BoardConfiguration, data: Dict) -> List[BoardConfiguration]:
        """
        Generate neighbors by swapping one piece with a cluster.
        
        This is useful when a piece is in the wrong place.
        """
        neighbors = []
        clusters = data.get('clusters', [])
        
        for i, piece in enumerate(config.pieces):
            piece_cells = set(piece.cell_ids)
            
            # Try swapping with each cluster
            for cluster in clusters:
                # Check if cluster is mostly empty or overlaps with this piece
                overlap_with_piece = len(cluster & piece_cells)
                overlap_with_others = len(cluster & (config.get_occupied_cells() - piece_cells))
                
                # Only swap if cluster doesn't overlap with other pieces
                if overlap_with_others == 0:
                    # Try to identify piece from cluster
                    piece_name, orientation, anchor = identify_piece_from_cells(list(cluster))
                    if piece_name is not None:
                        neighbor = config.copy()
                        
                        # Remove old piece
                        removed = neighbor.pieces.pop(i)
                        neighbor.empty_cells.update(removed.cell_ids)
                        
                        # Add new piece
                        placement = PiecePlacement(
                            piece_name=piece_name,
                            cell_ids=list(cluster),
                            orientation_idx=orientation,
                            anchor_cell=anchor
                        )
                        neighbor.pieces.append(placement)
                        neighbor.empty_cells -= cluster
                        
                        neighbors.append(neighbor)
        
        return neighbors


def solve_with_best_strategy(data: Dict,
                             solver: ImprovedBoardSolver = None,
                             time_budget: float = 5.0,
                             verbose: bool = True) -> Tuple[BoardConfiguration, Dict]:
    """
    Automatically choose best search strategy based on time budget.
    
    Args:
        data: Data dict with classifier outputs
        solver: Solver instance (creates default if None)
        time_budget: Maximum time in seconds
        verbose: Print progress
    
    Returns:
        (best_config, stats_dict)
    """
    from board_detection.board_solver import create_default_solver
    
    if solver is None:
        solver = ImprovedBoardSolver(create_default_solver().scorers)
    
    start_time = time.time()
    
    if time_budget < 1.0:
        # Fast: Single greedy search
        if verbose:
            print("Strategy: Single greedy search (fast)")
        initial = solver.solve_from_clustering(data)
        config = solver.solve_greedy(initial, data, max_iterations=50)
        method = "greedy"
    
    elif time_budget < 3.0:
        # Medium: Multiple restarts with greedy
        if verbose:
            print("Strategy: Multiple restarts with greedy (medium)")
        config = solver.solve_multiple_restarts(
            data, 
            num_restarts=5, 
            search_method='greedy',
            max_iterations=50,
            verbose=verbose
        )
        method = "multiple_restarts_greedy"
    
    else:
        # Slow: Simulated annealing with multiple restarts
        if verbose:
            print("Strategy: Simulated annealing with restarts (thorough)")
        config = solver.solve_multiple_restarts(
            data,
            num_restarts=3,
            search_method='annealing',
            T_start=10.0,
            T_end=0.01,
            cooling=0.95,
            max_iterations=500,
            verbose=verbose
        )
        method = "multiple_restarts_annealing"
    
    elapsed = time.time() - start_time
    score = solver.compute_total_score(config, data)
    
    stats = {
        'method': method,
        'time': elapsed,
        'score': score,
        'valid': config.is_valid(),
        'num_pieces': len(config.pieces),
        'num_whites': len(config.white_triangles),
    }
    
    if verbose:
        print(f"\nCompleted in {elapsed:.2f}s using {method}")
        print(f"Score: {score:.2f}, Valid: {stats['valid']}")
        print(f"Pieces: {stats['num_pieces']}, Whites: {stats['num_whites']}")
    
    return config, stats


if __name__ == "__main__":
    # Example usage
    from board_detection.board_solver import create_default_solver
    
    # Mock data
    data = {
        'empty_probs': {i: 0.5 for i in range(1, 49)},
        'detected_white_cells': {1, 5, 10, 15, 20, 25, 30},
        'white_confidences': {i: 0.9 for i in [1, 5, 10, 15, 20, 25, 30]},
        'clusters': [
            {2, 3, 4},
            {6, 7, 8, 9},
            {11, 12, 13},
        ],
        'color_values': {i: float(i) for i in range(1, 49)},
        'color_distance_fn': lambda a, b: abs(a - b),
    }
    
    # Create solver
    base_solver = create_default_solver()
    solver = ImprovedBoardSolver(base_solver.scorers)
    
    print("Testing improved search strategies...\n")
    
    # Test 1: Simulated annealing
    print("=" * 60)
    print("Test 1: Simulated Annealing")
    print("=" * 60)
    initial = solver.solve_from_clustering(data)
    result = solver.solve_simulated_annealing(initial, data, verbose=True)
    print(f"Result: {len(result.pieces)} pieces, valid={result.is_valid()}\n")
    
    # Test 2: Multiple restarts
    print("=" * 60)
    print("Test 2: Multiple Restarts")
    print("=" * 60)
    result = solver.solve_multiple_restarts(data, num_restarts=5, verbose=True)
    print(f"Result: {len(result.pieces)} pieces, valid={result.is_valid()}\n")
    
    # Test 3: Auto strategy
    print("=" * 60)
    print("Test 3: Auto Strategy Selection")
    print("=" * 60)
    result, stats = solve_with_best_strategy(data, solver, time_budget=2.0, verbose=True)
    print(f"Stats: {stats}")

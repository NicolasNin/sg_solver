"""
Piece identification from board cell IDs.

Given a set of cell IDs representing a shape on the board,
identify which puzzle piece matches that shape.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from board import Board
from pieces import PIECE_ORIENTATIONS

# Precompute board cell_id -> position mapping
_BOARD = Board.create_star()
_ID_TO_POS = {cell.cell_id: pos for pos, cell in _BOARD.cells.items()}


def identify_piece_from_cells(cell_ids: list[int]) -> str | None:
    """Identify which piece matches a shape defined by cell IDs.
    
    Args:
        cell_ids: List of cell IDs (1-48) forming a connected shape
        
    Returns:
        Piece name if a piece matches, None otherwise.
        Note: T3B is returned as T3 (same shape).
    """
    # Convert cell_ids to positions
    positions = [_ID_TO_POS[cid] for cid in cell_ids]
    
    # Create canonical shape key (same logic as Piece._canonical_key)
    min_pos = min(positions, key=lambda t: (t.y, t.x))
    shape_key = frozenset(
        (t.x - min_pos.x, t.y - min_pos.y, t.points_up)
        for t in positions
    )
    
    # Check against all piece orientations
    for piece_name, orientations in PIECE_ORIENTATIONS.items():
        if len(orientations[0].triangles) != len(cell_ids):
            continue  # Size mismatch
        for orientation in orientations:
            if orientation._canonical_key() == shape_key:
                # Normalize T3B to T3
                return "T3" if piece_name == "3B" else piece_name
    
    return None

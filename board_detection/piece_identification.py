"""
Piece identification from board cell IDs.

Given a set of cell IDs representing a shape on the board,
identify which puzzle piece matches that shape.
"""

from sg_solver import Board, PIECE_ORIENTATIONS

# Precompute board cell_id -> position mapping
_BOARD = Board.create_star()
_ID_TO_POS = {cell.cell_id: pos for pos, cell in _BOARD.cells.items()}
_POS_TO_ID = {pos: cell.cell_id for pos, cell in _BOARD.cells.items()}


def identify_piece_from_cells(cell_ids: list[int]) -> tuple[str, int, int] | tuple[None, None, None]:
    """Identify which piece matches a shape defined by cell IDs.
    
    Args:
        cell_ids: List of cell IDs (1-48) forming a connected shape
        
    Returns:
        Tuple of (piece_name, orientation_index, anchor_cell_id) if a piece matches, 
        (None, None, None) otherwise.
        anchor_cell_id is the cell where the piece's triangles[0] lands.
        Note: 3B is returned as T3 (same shape).
    """
    # Convert cell_ids to positions
    try:
        positions = [_ID_TO_POS[cid] for cid in cell_ids]
    except KeyError:
        return None, None, None
    
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
        for idx, orientation in enumerate(orientations):
            if orientation._canonical_key() == shape_key:
                # Normalize 3B to T3
                name = "T3" if piece_name == "3B" else piece_name
                
                # Find anchor: which cell corresponds to triangles[0]?
                # The piece's triangles are normalized relative to triangles[0]
                # We need to find which board position maps to triangles[0]
                piece_tri0 = orientation.triangles[0]
                
                # The shape key is relative to min_pos, so we need to find
                # the actual board position for triangles[0]
                # triangles[0] has coords (piece_tri0.x, piece_tri0.y, piece_tri0.points_up)
                # normalized key uses: (t.x - min_piece.x, t.y - min_piece.y, t.points_up)
                min_piece = min(orientation.triangles, key=lambda t: (t.y, t.x))
                tri0_rel = (piece_tri0.x - min_piece.x, piece_tri0.y - min_piece.y, piece_tri0.points_up)
                
                # Find which board position has the same relative position
                for pos in positions:
                    rel = (pos.x - min_pos.x, pos.y - min_pos.y, pos.points_up)
                    if rel == tri0_rel:
                        anchor_id = _POS_TO_ID.get(pos)
                        return name, idx, anchor_id
                
                # Fallback - no anchor found, use first cell
                return name, idx, cell_ids[0]
    
    return None, None, None

"""
Star Genius puzzle solver using backtracking.
"""

from board import Board, Piece, TrianglePos
from pieces import ALL_PIECES, PIECE_ORIENTATIONS
from viz import render_svg,display_board


def solve(board: Board, remaining_pieces: list[Piece]) -> Board | None:
    """
    Backtracking solver. Returns solved board or None if no solution.
    
    Strategy:
    - Pick most constrained empty cell (fewest empty neighbors)
    - Try each remaining piece in each orientation
    - For each placement that covers the empty cell, recurse
    """
    svg_file = render_svg(board)
    #display_board(board)
    
    if not remaining_pieces:
        return board if board.is_solved() else None
    
    # Find most constrained empty cell to fill
    empty = board.most_constrained_empty_cell()
    #empty = board.first_empty_cell()
    if empty is None:
        return board if board.is_solved() else None
    
    # Try pieces from largest to smallest (more constraining first)
    for piece in remaining_pieces:
        # Get precomputed orientations
        orientations = PIECE_ORIENTATIONS[piece.name]
        
        for oriented_piece in orientations:
            # Find anchors that would place this piece covering the empty cell
            anchors = get_anchors_covering(oriented_piece, empty.pos)
            
            for anchor in anchors:
                if board.can_place(oriented_piece, anchor):
                    # Place and recurse
                    new_board = board.copy()
                    new_board.place(oriented_piece, anchor)
                    
                    new_remaining = [p for p in remaining_pieces if p.name != piece.name]
                    print(f"cells chosen {empty.cell_id} pieces {piece.name} remaining {len(new_remaining)}")
                    result = solve(new_board, new_remaining)
                    
                    if result is not None:
                        return result
    
    return None  # No solution found


def get_anchors_covering(piece: Piece, target_pos: TrianglePos) -> list[TrianglePos]:
    """
    Return all anchor positions that would place piece covering target_pos.
    
    Since piece triangles are relative to anchor, we need to find anchors
    such that (anchor + some triangle offset) == target_pos.
    """
    anchors = []
    for triangle in piece.triangles:
        # If this triangle is at target_pos, what's the anchor?
        anchor = reverse_translate(target_pos, triangle)
        anchors.append(anchor)
    return anchors


def reverse_translate(target: TrianglePos, triangle: TrianglePos) -> TrianglePos:
    """
    Given target position and relative triangle offset, compute the anchor.
    anchor + triangle = target  =>  anchor = target - triangle
    
    anchor.points_up is determined by the grid position.
    """
    anchor_x = target.x - triangle.x
    anchor_y = target.y - triangle.y
    anchor_up = (anchor_x + anchor_y) % 2 == 0  # Grid determines orientation
    return TrianglePos(x=anchor_x, y=anchor_y, points_up=anchor_up)


def solve_puzzle(blocker_ids: list[int]) -> Board | None:
    """
    Main entry point. Takes 7 dice values (cell IDs to block).
    Returns solved board or None.
    """
    if len(blocker_ids) != 7:
        raise ValueError("Need exactly 7 blocker IDs")
    
    board = Board.create_star()
    
    # Place blockers
    for cell_id in blocker_ids:
        board.place_blocker(cell_id)
    
    # Sort pieces by size (largest first)
    pieces_by_size = sorted(ALL_PIECES, key=lambda p: len(p.triangles), reverse=True)
    
    return solve(board, pieces_by_size)


if __name__ == "__main__":
    from viz import display_board
    from dices import fixed_roll, TEST_ROLLS
    
    # Global counter for debugging
    attempts = 0
    
    def solve_with_counter(board: Board, remaining_pieces: list[Piece]) -> Board | None:
        global attempts
        attempts += 1
        if attempts % 10000 == 0:
            print(f"  {attempts} attempts, {len(remaining_pieces)} pieces left")
        return solve(board, remaining_pieces)
    
    # Use fixed roll for testing
    #blockers = fixed_roll()
    blockers = TEST_ROLLS[1]
    print(f"Solving with blockers at cells: {blockers}")
    
    result = solve_puzzle(blockers)
    
    if result:
        print(f"\nSolution found after {attempts} attempts!")
        display_board(result)
        
        # Render SVG
        svg_file = render_svg(result)
        print(f"\nSVG saved to: {svg_file}")
    else:
        print(f"\nNo solution found after {attempts} attempts")

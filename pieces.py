"""
Puzzle piece definitions for Star Genius.
"""

from board import TrianglePos, Piece


def make_piece(name: str, coords: list[tuple[int, int, bool]], can_flip: bool = True) -> Piece:
    """Helper to create a piece from coordinate tuples (x, y, points_up)."""
    triangles = tuple(TrianglePos(x, y, up) for x, y, up in coords)
    return Piece(name, triangles, can_flip)


# Trapezoid pieces: horizontal rows of triangles

TRAPEZOID_1 = make_piece("T1", [
    (0, 0, True),
], can_flip=False)

TRAPEZOID_2 = make_piece("T2", [
    (0, 0, True),
    (1, 0, False),
], can_flip=False)

TRAPEZOID_3 = make_piece("T3", [
    (0, 0, True),
    (1, 0, False),
    (2, 0, True),
])

TRAPEZOID_4 = make_piece("T4", [
    (0, 0, True),
    (1, 0, False),
    (2, 0, True),
    (3, 0, False),
])

TRAPEZOID_5 = make_piece("T5", [
    (0, 0, True),
    (1, 0, False),
    (2, 0, True),
    (3, 0, False),
    (4, 0, True),
])

# Triforce shape: 4 triangles centered on down triangle
TRIFORCE = make_piece("TF", [
    (0, 0, False),   
    (1, 0, True),  
    (-1, 0, True),  
    (0, 1, True),   
], can_flip=False)

# L-shape: trapezoid_3 + triangle on side
L_SHAPE_4 = make_piece("L4", [
    (0, 0, True),
    (1, 0, False),
    (2, 0, True),
    (0, -1, False),
], can_flip=False)

# Extended L: trapezoid_3 + two triangles
EXTENDED_L_5 = make_piece("EL", [
    (0, 0, True),
    (1, 0, False),
    (2, 0, True),
    (2, -1, False),
    (0, -1, False),
])

# Trapezoid_4 with triangle going up
TRAP4_UP = make_piece("4U", [
    (0, 0, True),
    (1, 0, False),
    (2, 0, True),
    (3, 0, False),
    (3, 1, True),
])

# Trapezoid_4 with triangle going down
TRAP4_DOWN = make_piece("4D", [
    (0, 0, True),
    (1, 0, False),
    (2, 0, True),
    (3, 0, False),
    (2, -1, False),
])

# Second copy of TRAPEZOID_3 (game has 2)
TRAPEZOID_3B = make_piece("3B", [
    (0, 0, True),
    (1, 0, False),
    (2, 0, True),
], can_flip=False)

# All pieces for the solver (11 pieces, 41 triangles + 7 blockers = 48)
ALL_PIECES = [
    TRAPEZOID_1,
    TRAPEZOID_2,
    TRAPEZOID_3,
    TRAPEZOID_3B,
    TRAPEZOID_4,
    TRAPEZOID_5,
    TRIFORCE,
    L_SHAPE_4,
    EXTENDED_L_5,
    TRAP4_UP,
    TRAP4_DOWN,
]

# Pieces with adjacency constraints (for difficulty levels)
# Level 0: No constraints
# Level 1: T1, T2 can't be adjacent
# Level 2: T1, T2, 3B can't be adjacent
# Level 3: T1, T2, 3B, TF can't be adjacent
# Level 4 (Wizard): T1, T2, 3B, TF, L4 can't be adjacent
CONSTRAINED_PIECES = [
    TRAPEZOID_1,   # 1
    TRAPEZOID_2,   # 2
    TRAPEZOID_3B,  # 3
    TRIFORCE,      # 4
    L_SHAPE_4,     # 5
]

# Precompute all orientations for each piece (for solver performance)
PIECE_ORIENTATIONS: dict[str, list] = {
    piece.name: piece.all_orientations() for piece in ALL_PIECES
}

# Colors for each piece (for visualization)
PIECE_COLORS: dict[str, str] = {
    "T1": "#00325b",  # deep blue
    "T2": "#ffc100",  # Yellow
    "T3": "#00a0de",  # Cyan
    "3B": "#00a0de",  # Cyan
    "T4": "#ee68a7",  # Rose
    "T5": "#de241b",  # Red
    "TF": "#7a2d9e",  # Purple
    "L4": "#ff8717",  # ORANGE
    "EL": "#006c43",  # teal
    "4U": "#8bd100",  # lime
    "4D": "#8a5e3c",  # BRown
}

BLOCKER_COLOR = "#FFFFFF"  # Dark gray for blockers

if __name__ == "__main__":
    from viz import display_board
    from board import Board
    
    # Show all rotations of the triforce piece
    print("=== Triforce Piece Rotations ===\n")
    
    piece = TRIFORCE
    for i in range(3):
        print(f"Rotation {i}:")
        for t in piece.triangles:
            orient = "△" if t.points_up else "▽"
            print(f"  T({t.x},{t.y}) {orient}")
        
        # Try to place on board at a central position
        board = Board.create_star()
        anchor = TrianglePos(x=6, y=3, points_up=True)
        if board.can_place(piece, anchor):
            board.place(piece, anchor)
            #display_board(board)
        else:
            print("  (Cannot place at anchor)")
        
        print()
        piece = piece.rotate_120()

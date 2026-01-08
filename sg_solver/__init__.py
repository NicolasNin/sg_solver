"""
sg_solver - Star Genius puzzle solver package

Core components:
- Board: The game board with triangle cells
- Piece: Puzzle pieces that can be placed on the board
- solve_puzzle: Main solver function
"""

from .board import Board, Piece, TrianglePos
from .pieces import ALL_PIECES, PIECE_ORIENTATIONS
from .solver import solve_puzzle
from .viz import render_svg, display_board
from .dices import roll_dice

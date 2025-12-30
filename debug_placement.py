"""
Debug script to test piece placement on the board.
Tests how EXTENDED_L_5 gets placed and translated.
"""

from board import Board, TrianglePos, Piece
from pieces import EXTENDED_L_5, PIECE_ORIENTATIONS


def show_placement(board: Board, piece: Piece, anchor: TrianglePos):
    """Show how a piece would be placed at an anchor position."""
    print(f"\nPlacing {piece.name} at anchor T({anchor.x}, {anchor.y}) {'△' if anchor.points_up else '▽'}")
    print("-" * 50)
    
    print("Piece triangles (relative):")
    for t in piece.triangles:
        print(f"  T({t.x:+d}, {t.y:+d}) {'△' if t.points_up else '▽'}")
    
    print("\nAfter _translate to board positions:")
    for t in piece.triangles:
        pos = board._translate(t, anchor)
        on_board = "✓" if pos in board.cells else "✗ OFF BOARD"
        print(f"  T({t.x:+d}, {t.y:+d}) -> T({pos.x}, {pos.y}) {'△' if pos.points_up else '▽'} {on_board}")


def find_el_in_solution():
    """Analyze where EL piece appears based on the solver output."""
    # From the solver output:
    # Row 2: ... EL EL EL
    # Row 3: ... EL EL
    # 
    # Let's figure out which cells are EL
    
    board = Board.create_star()
    
    print("=== Board cells for rows 2-3 ===")
    for pos, cell in sorted(board.cells.items(), key=lambda x: (x[0].y, x[0].x)):
        if pos.y in [2, 3]:
            print(f"  Cell {cell.cell_id:2d}: T({pos.x}, {pos.y}) {'△' if pos.points_up else '▽'}")
    
    # From the output, EL occupies cells in positions:
    # Row 2: x=8,9,10 (cells 9,10,11 roughly based on layout)
    # Row 3: x=7,8 (cells EL EL)
    # Wait, let me check the actual cell mapping
    
    print("\n=== Mapping EL from solution ===")
    print("From solution output:")
    print("Row 2: T5 T5 T5 T5 T5  X T3  X EL EL EL")
    print("Row 3:    4U 4U  X  X 4D 4D T1 EL EL")
    
    # Let's map out row 2's EL positions
    # Row 2 starts at x=0 and has 11 cells (x=0 to x=10)
    # The display shows: T5 T5 T5 T5 T5 X T3 X EL EL EL
    # That's positions:   0  1  2  3  4 5  6 7  8  9 10
    
    print("\n=== EL positions in solution ===")
    print("Row 2: x=8,9,10")
    print("Row 3: x=8,9 (based on indentation)")
    
    # Let's check if these form a valid placement
    el_positions_row2 = [(8, 2), (9, 2), (10, 2)]
    el_positions_row3 = [(8, 3), (9, 3)]
    
    print("\n=== Checking EL cells ===")
    for x, y in el_positions_row2 + el_positions_row3:
        points_up = (x + y) % 2 == 1
        pos = TrianglePos(x, y, points_up)
        if pos in board.cells:
            cell = board.cells[pos]
            print(f"  T({x}, {y}) {'△' if points_up else '▽'} = Cell {cell.cell_id}")


def test_translate():
    """Test the _translate function."""
    board = Board.create_star()
    
    print("\n=== Testing _translate function ===")
    print("The _translate function in board.py:")
    print("  return TrianglePos(")
    print("      x=anchor.x + triangle.x,")
    print("      y=anchor.y + triangle.y,") 
    print("      points_up=anchor.points_up if triangle.points_up else not anchor.points_up")
    print("  )")
    
    print("\n--- Test cases ---")
    
    # Test 1: anchor is up, triangle is up -> result should be up
    anchor = TrianglePos(5, 2, True)
    triangle = TrianglePos(0, 0, True)
    result = board._translate(triangle, anchor)
    print(f"\n  anchor=T(5,2,△), triangle=T(0,0,△)")
    print(f"  result: T({result.x},{result.y},{'△' if result.points_up else '▽'})")
    expected_up = (result.x + result.y) % 2 == 1
    print(f"  Expected orientation from (x+y)%2: {'△' if expected_up else '▽'}")
    if result.points_up != expected_up:
        print("  ⚠️  MISMATCH! points_up doesn't match expected orientation!")
    
    # Test 2: anchor is up, triangle is down -> result should be down
    anchor = TrianglePos(5, 2, True)
    triangle = TrianglePos(1, 0, False)
    result = board._translate(triangle, anchor)
    print(f"\n  anchor=T(5,2,△), triangle=T(1,0,▽)")
    print(f"  result: T({result.x},{result.y},{'△' if result.points_up else '▽'})")
    expected_up = (result.x + result.y) % 2 == 1
    print(f"  Expected orientation from (x+y)%2: {'△' if expected_up else '▽'}")
    if result.points_up != expected_up:
        print("  ⚠️  MISMATCH! points_up doesn't match expected orientation!")
    
    # Test 3: What happens with y=-1 offset?
    anchor = TrianglePos(5, 2, True)
    triangle = TrianglePos(0, -1, False)  # One of the EL piece triangles
    result = board._translate(triangle, anchor)
    print(f"\n  anchor=T(5,2,△), triangle=T(0,-1,▽)")
    print(f"  result: T({result.x},{result.y},{'△' if result.points_up else '▽'})")
    expected_up = (result.x + result.y) % 2 == 1
    print(f"  Expected orientation from (x+y)%2: {'△' if expected_up else '▽'}")
    if result.points_up != expected_up:
        print("  ⚠️  MISMATCH! points_up doesn't match expected orientation!")


def visualize_expected_vs_actual():
    """Compare expected shape vs what gets placed."""
    board = Board.create_star()
    
    print("\n=== EXTENDED_L_5 Expected Shape ===")
    print("Definition (y increases downward):")
    print("  y=-1: ▽ . ▽   (two down triangles ABOVE row 0)")
    print("  y= 0: △ ▽ △   (three triangles in a row)")
    print()
    print("Visual representation (y-axis flipped for natural view):")
    print("    ▽   ▽")  
    print("   △ ▽ △")
    print()
    
    # Check all 6 orientations
    print("=== All orientations of EL ===")
    for i, piece in enumerate(PIECE_ORIENTATIONS["EL"]):
        print(f"\nOrientation {i}:")
        
        # Create a mini-grid
        triangles = piece.triangles
        min_x = min(t.x for t in triangles)
        max_x = max(t.x for t in triangles)
        min_y = min(t.y for t in triangles)
        max_y = max(t.y for t in triangles)
        
        pos_map = {(t.x, t.y): t for t in triangles}
        
        for y in range(min_y, max_y + 1):
            row = []
            for x in range(min_x, max_x + 1):
                if (x, y) in pos_map:
                    t = pos_map[(x, y)]
                    row.append("△" if t.points_up else "▽")
                else:
                    row.append(".")
            print(f"  y={y:+d}: {' '.join(row)}")


if __name__ == "__main__":
    test_translate()
    print()
    find_el_in_solution()
    print()
    visualize_expected_vs_actual()

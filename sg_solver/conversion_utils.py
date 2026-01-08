"""
Coordinate conversion utilities for Star Genius.

Pipeline:
1. Cell ID (1-48) - the game's numbering
2. T(x, y) - Triangle grid coordinates (x=column, y=row, y increases upward)
3. Vertex coordinates (vx, vy) - Skewed grid for rotation math
4. Pixel coordinates (px, py) - For SVG rendering (py increases downward)

Let's understand each conversion step.
"""

from .board import Board, TrianglePos


def get_cell_by_id(board: Board, cell_id: int) -> tuple[TrianglePos, int]:
    """Get the position of a cell by its ID."""
    for pos, cell in board.cells.items():
        if cell.cell_id == cell_id:
            return pos, cell_id
    raise ValueError(f"No cell with id {cell_id}")


def analyze_cell(cell_id: int):
    """Analyze coordinate conversions for a specific cell."""
    board = Board.create_star()
    pos, _ = get_cell_by_id(board, cell_id)
    
    print(f"=== Cell {cell_id} Analysis ===")
    print()
    
    # Step 1: Triangle grid coordinates
    print(f"1. Triangle Grid: T({pos.x}, {pos.y}, {'△' if pos.points_up else '▽'})")
    print(f"   - x (column) = {pos.x}")
    print(f"   - y (row) = {pos.y} (y increases upward)")
    print(f"   - points_up = {pos.points_up} (from (x+y)%2==0: {(pos.x+pos.y)%2==0})")
    print()
    
    # Step 2: Vertex coordinates (skewed grid)
    verts = pos._to_vertices()
    print(f"2. Vertex Coordinates (skewed grid):")
    for i, (vx, vy) in enumerate(verts):
        print(f"   - v{i}: ({vx}, {vy})")
    print()
    
    # Explain the skew formulas
    if pos.points_up:
        expected_vx = (pos.x - pos.y) // 2
        print(f"   Formula for △: vx = (x - y) // 2 = ({pos.x} - {pos.y}) // 2 = {expected_vx}")
        print(f"   Vertices: ({expected_vx}, {pos.y}), ({expected_vx+1}, {pos.y}), ({expected_vx}, {pos.y+1})")
    else:
        expected_vx = (pos.x - pos.y + 1) // 2
        print(f"   Formula for ▽: vx = (x - y + 1) // 2 = ({pos.x} - {pos.y} + 1) // 2 = {expected_vx}")
        print(f"   Vertices: ({expected_vx}, {pos.y}), ({expected_vx-1}, {pos.y+1}), ({expected_vx}, {pos.y+1})")
    print()
    
    # Step 3: What SVG rendering needs
    print(f"3. SVG Rendering:")
    print(f"   - SVG has y increasing DOWNWARD")
    print(f"   - Our y increases UPWARD")
    print(f"   - Need to flip: svg_y = max_y - our_y")
    print()
    
    return pos, verts


def skewed_to_normal(vx: int, vy: int) -> tuple[float, float]:
    """Convert skewed grid coordinates to normal 2D coordinates.
    
    Skewed grid: e1 = (1, 0), e2 = (0.5, 1) at 60°
    Normal grid: e1 = (1, 0), e2 = (0, 1)
    
    Conversion: nx = vx + vy * 0.5, ny = vy
    """
    nx = vx + vy * 0.5
    ny = vy
    return (nx, ny)


def normal_to_svg(nx: float, ny: float, max_y: float) -> tuple[float, float]:
    """Convert normal 2D coordinates to SVG coordinates.
    
    Normal: y increases upward
    SVG: y increases downward
    
    Conversion: svg_x = nx, svg_y = max_y - ny
    """
    svg_x = nx
    svg_y = max_y - ny
    return (svg_x, svg_y)


def skewed_to_svg(vx: int, vy: int, max_y: float) -> tuple[float, float]:
    """Full pipeline: skewed → normal → SVG."""
    nx, ny = skewed_to_normal(vx, vy)
    return normal_to_svg(nx, ny, max_y)


def show_all_cells():
    """Show all cells in a table format with all coordinate systems."""
    board = Board.create_star()
    
    # Find max_y for SVG conversion
    all_verts = []
    for pos in board.cells.keys():
        all_verts.extend(pos._to_vertices())
    max_vy = max(v[1] for v in all_verts)
    
    print(f"Max vertex y = {max_vy}")
    print()
    print("Cell | T(x,y) | Or | Normal Vertices               | SVG Vertices (y flipped)")
    print("-" * 90)
    
    for cell_id in [1, 2, 48, 47]:  # Just show key cells for clarity
        pos, _ = get_cell_by_id(board, cell_id)
        skewed_verts = pos._to_vertices()
        normal_verts = [skewed_to_normal(vx, vy) for vx, vy in skewed_verts]
        svg_verts = [skewed_to_svg(vx, vy, max_vy) for vx, vy in skewed_verts]
        
        orient = "△" if pos.points_up else "▽"
        normal_str = ", ".join(f"({v[0]:.1f},{v[1]})" for v in normal_verts)
        svg_str = ", ".join(f"({v[0]:.1f},{v[1]:.1f})" for v in svg_verts)
        print(f" {cell_id:2} | T({pos.x},{pos.y}) | {orient}  | {normal_str:29} | {svg_str}")


if __name__ == "__main__":
    print("=" * 90)
    print("Coordinate Conversion Pipeline Test")
    print("=" * 90)
    print()
    show_all_cells()

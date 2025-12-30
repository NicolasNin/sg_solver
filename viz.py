"""
Visualization utilities for Star Genius board.
"""

import board as board_module


def display_board(board: "board_module.Board") -> None:
    """
    Display the board in a simple grid format.
    Shows cell IDs, 'X' for blocked cells, or piece names.
    Y increases upward, so we print from max_y down to min_y.
    """
    if not board.cells:
        print("Empty board")
        return

    # Find grid bounds
    min_y = min(pos.y for pos in board.cells.keys())
    max_y = max(pos.y for pos in board.cells.keys())
    max_x = max(pos.x for pos in board.cells.keys())

    print("\nStar Board:")
    print("=" * (max_x * 3 + 10))

    # Print from top (max_y) to bottom (min_y) since y increases upward
    for y in range(max_y, min_y - 1, -1):
        row_str = []
        for x in range(max_x + 1):
            # Orientation is determined by (x + y) % 2 == 0
            points_up = (x + y) % 2 == 0
            pos = board_module.TrianglePos(x, y, points_up)
            
            if pos in board.cells:
                cell = board.cells[pos]
                if pos in board.occupied:
                    piece_name = board.occupied[pos]
                    if piece_name is None:
                        row_str.append(" X")  # Blocker
                    else:
                        # Show first 2 chars of piece name
                        row_str.append(f"{piece_name[:2]:>2}")
                else:
                    row_str.append(f"{cell.cell_id:2d}")
            else:
                row_str.append("  ")

        print(f"y={y}: {' '.join(row_str)}")

    print("=" * (max_x * 3 + 10))
    print(f"Total cells: {len(board.cells)}")
    print(f"Empty cells: {len(board.empty_cells())}")
    print(f"Occupied: {len(board.occupied)}")


def render_svg(board: "board_module.Board", filename: str = "solution.svg") -> str:
    """
    Render board to SVG file with colored pieces.
    Returns the filename.
    """
    from pieces import PIECE_COLORS, BLOCKER_COLOR
    from conversion_utils import skewed_to_svg
    import math
    
    # SVG settings
    scale = 50  # pixels per unit
    margin = 20
    h = scale * math.sqrt(3) / 2  # height of equilateral triangle row
    
    # Find bounds in skewed coordinates
    all_verts = []
    for pos in board.cells.keys():
        all_verts.extend(pos._to_vertices())
    
    min_vx = min(v[0] for v in all_verts)
    max_vx = max(v[0] for v in all_verts)
    min_vy = min(v[1] for v in all_verts)
    max_vy = max(v[1] for v in all_verts)
    
    def vertex_to_pixel(vx: int, vy: int) -> tuple[float, float]:
        """Convert skewed vertex to SVG pixel coordinates."""
        # Pipeline: skewed -> normal -> svg (y-flipped)
        svg_x, svg_y = skewed_to_svg(vx, vy, max_vy)
        # Offset to start from 0 and scale
        px = margin + (svg_x - min_vx - min_vy * 0.5) * scale
        py = margin + svg_y * h
        return (px, py)
    
    # Calculate SVG dimensions
    # After unskew and y-flip, x ranges and y ranges
    width = margin * 2 + (max_vx - min_vx + (max_vy - min_vy) * 0.5 + 1) * scale
    height = margin * 2 + (max_vy - min_vy + 1) * h
    
    # Build SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}">',
        '<rect width="100%" height="100%" fill="#1b2856"/>',
    ]
    
    # Draw each cell
    for pos, cell in board.cells.items():
        verts = pos._to_vertices()
        points = [vertex_to_pixel(vx, vy) for vx, vy in verts]
        points_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        
        # Determine color and if piece has ring (non-flippable)
        has_ring = False
        if pos in board.occupied:
            piece_name = board.occupied[pos]
            if piece_name is None:
                color = BLOCKER_COLOR
            else:
                color = PIECE_COLORS.get(piece_name, "#CCCCCC")
                # Check if this piece is non-flippable (gets a ring)
                from pieces import ALL_PIECES
                for p in ALL_PIECES:
                    if p.name == piece_name:
                        has_ring = not p.can_flip
                        break
        else:
            color = "#1b2856"  # Empty cells dark blue
        
        # Draw outer triangle
        svg_parts.append(
            f'<polygon points="{points_str}" fill="{color}" stroke="#000" stroke-width="1"/>'
        )
        
        # Draw ring effect for non-flippable pieces
        if has_ring and piece_name is not None:
            # Compute center of triangle
            cx = sum(p[0] for p in points) / 3
            cy = sum(p[1] for p in points) / 3
            
            # Scale points toward center for inner triangles
            def scale_points(pts, factor):
                return [(cx + (px - cx) * factor, cy + (py - cy) * factor) for px, py in pts]
            
            # White ring (middle layer) - 70% size
            ring_points = scale_points(points, 0.7)
            ring_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in ring_points)
            svg_parts.append(
                f'<polygon points="{ring_str}" fill="#FFFFFF" stroke="none"/>'
            )
            
            # Inner color (top layer) - 45% size
            inner_points = scale_points(points, 0.45)
            inner_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in inner_points)
            svg_parts.append(
                f'<polygon points="{inner_str}" fill="{color}" stroke="none"/>'
            )
        
        # Add cell ID label
        cx = sum(p[0] for p in points) / 3
        cy = sum(p[1] for p in points) / 3
        svg_parts.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" '
            f'dominant-baseline="middle" font-size="10" fill="#888">{cell.cell_id}</text>'
        )
    
    svg_parts.append('</svg>')
    svg_content = "\n".join(svg_parts)
    
    with open(filename, "w") as f:
        f.write(svg_content)
    
    return filename


if __name__ == "__main__":
    board = board_module.Board.create_star()
    display_board(board)
    
    # Test blocker
    print("\nAfter placing blocker on cell 10:")
    board.place_blocker(10)
    display_board(board)


"""
Debug script to verify piece connectivity and orientations after y-axis fix.
"""

from board import TrianglePos, Piece, Board
from pieces import EXTENDED_L_5, ALL_PIECES,TRAPEZOID_4


def check_connectivity(piece: Piece) -> tuple[bool, list[str]]:
    """Check that all triangles in a piece share edges (are connected)."""
    triangles = piece.triangles
    if len(triangles) <= 1:
        return True, []
    
    def get_neighbors(t: TrianglePos) -> set[tuple[int, int, bool]]:
        """Get all neighbors of a triangle (positions that share an edge)."""
        neighbors = set()
        # Left and right neighbors
        neighbors.add((t.x - 1, t.y, not t.points_up))
        neighbors.add((t.x + 1, t.y, not t.points_up))
        # Vertical neighbor
        if t.points_up:
            neighbors.add((t.x, t.y - 1, False))  # Up triangle: bottom neighbor is at y-1 (down)
        else:
            neighbors.add((t.x, t.y + 1, True))   # Down triangle: top neighbor is at y+1 (up)
        return neighbors
    
    # Check connectivity using BFS/DFS
    pos_set = {(t.x, t.y, t.points_up) for t in triangles}
    visited = set()
    to_visit = [(triangles[0].x, triangles[0].y, triangles[0].points_up)]
    
    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
        visited.add(current)
        
        # Find this triangle
        current_tri = TrianglePos(current[0], current[1], current[2])
        neighbors = get_neighbors(current_tri)
        
        for n in neighbors:
            if n in pos_set and n not in visited:
                to_visit.append(n)
    
    # Check if all triangles were visited
    errors = []
    if len(visited) != len(triangles):
        unvisited = pos_set - visited
        for t in unvisited:
            errors.append(f"  T({t[0]}, {t[1]}) {'△' if t[2] else '▽'} not connected to rest of piece")
    
    return len(errors) == 0, errors


def ascii_grid(triangles: tuple[TrianglePos, ...]) -> str:
    """Create ASCII representation of piece triangles."""
    if not triangles:
        return "  Empty piece"
    
    min_x = min(t.x for t in triangles)
    max_x = max(t.x for t in triangles)
    min_y = min(t.y for t in triangles)
    max_y = max(t.y for t in triangles)
    
    pos_map = {(t.x, t.y): t for t in triangles}
    
    lines = []
    # Print from top (max_y) to bottom (min_y) since y increases upward
    for y in range(max_y, min_y - 1, -1):
        row = []
        for x in range(min_x, max_x + 1):
            if (x, y) in pos_map:
                t = pos_map[(x, y)]
                row.append("△" if t.points_up else "▽")
            else:
                row.append("·")
        lines.append(f"  y={y:+d}: {' '.join(row)}")
    
    return "\n".join(lines)


def main():
    print("="*60)
    print("PIECE CONNECTIVITY CHECK (after y-axis fix)")
    print("="*60)
    print("\nY now increases UPWARD:")
    print("  - Up triangle (△) at T(x,y) has neighbors:")
    print("    - Left: T(x-1, y, ▽)")
    print("    - Right: T(x+1, y, ▽)")
    print("    - Bottom: T(x, y-1, ▽)")
    print("  - Down triangle (▽) at T(x,y) has neighbors:")
    print("    - Left: T(x-1, y, △)")
    print("    - Right: T(x+1, y, △)")
    print("    - Top: T(x, y+1, △)")
    
    all_valid = True
    
    for piece in ALL_PIECES:
        print(f"\n--- {piece.name} ---")
        normalized = piece._normalize()
        print(f"Triangles:")
        for t in normalized.triangles:
            print(f"  T({t.x:+d}, {t.y:+d}) {'△' if t.points_up else '▽'}")
        
        print(ascii_grid(normalized.triangles))
        
        is_connected, errors = check_connectivity(normalized)
        if is_connected:
            print("  ✓ Connected")
        else:
            print("  ✗ NOT CONNECTED!")
            for e in errors:
                print(e)
            all_valid = False
    
    print("\n" + "="*60)
    if all_valid:
        print("✓ All pieces are valid!")
    else:
        print("✗ Some pieces have connectivity issues!")
    print("="*60)
    
    # Show EXTENDED_L_5 orientations
    print("\n\n=== EXTENDED_L_5 ALL ORIENTATIONS ===")
    for i, oriented in enumerate(EXTENDED_L_5.all_orientations()):
        is_flipped = i % 2 == 1
        rotation = i // 2
        label = f"R{rotation}{'F' if is_flipped else ''}"
        
        print(f"\n[{i}] {label}:")
        for t in oriented.triangles:
            print(f"  T({t.x:+d}, {t.y:+d}) {'△' if t.points_up else '▽'}")
        print(ascii_grid(oriented.triangles))


if __name__ == "__main__":
    main()

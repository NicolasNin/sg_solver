"""
Star Genius board model.

Coordinate system:
- x (horizontal), y (vertical) define position in a triangular grid
- y increases UPWARD (row 0 of star visual is at y=7, bottom tip at y=0)
- points_up: True for △, False for ▽
- T(x, y) where (x + y) % 2 == 0 means up triangle

Adjacency:
- All triangles have left (x-1) and right (x+1) neighbors
- △ (up) has a bottom neighbor at (x, y-1)
- ▽ (down) has a top neighbor at (x, y+1)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrianglePos:
    """A position in the triangular grid. T(x, y)."""
    x: int  # horizontal position
    y: int  # vertical position (increases downward)
    points_up: bool

    def left(self) -> "TrianglePos":
        """Left neighbor (always exists, opposite orientation)."""
        return TrianglePos(self.x - 1, self.y, not self.points_up)

    def right(self) -> "TrianglePos":
        """Right neighbor (always exists, opposite orientation)."""
        return TrianglePos(self.x + 1, self.y, not self.points_up)

    def vertical(self) -> "TrianglePos":
        """Vertical neighbor (bottom if up, top if down). Y increases upward."""
        if self.points_up:
            # Up triangle's bottom neighbor is at y-1
            return TrianglePos(self.x, self.y - 1, not self.points_up)
        else:
            # Down triangle's top neighbor is at y+1
            return TrianglePos(self.x, self.y + 1, not self.points_up)

    def _to_vertices(self) -> list[tuple[int, int]]:
        """Get the 3 vertex coordinates of this triangle.
        
        Accounts for grid skewing: vx = (x - y) // 2 for up, vx = (x - y + 1) // 2 for down.
        """
        if self.points_up:
            vx = (self.x - self.y) // 2
            vy = self.y
            return [(vx, vy), (vx + 1, vy), (vx, vy + 1)]
        else:
            vx = (self.x - self.y + 1) // 2
            vy = self.y
            return [(vx, vy), (vx - 1, vy + 1), (vx, vy + 1)]

    @staticmethod
    def _rotate_vertex_120(i: int, j: int) -> tuple[int, int]:
        """Rotate vertex 120° CW: R120(i, j) = (-i - j, i)"""
        return (-i - j, i)

    @staticmethod
    def _rotate_vertex_60(i: int, j: int) -> tuple[int, int]:
        """Rotate vertex 60° CW: R60(i, j) = (-j, i + j)"""
        return (-j, i + j)

    @classmethod
    def _from_vertices(cls, verts: list[tuple[int, int]]) -> "TrianglePos":
        """Reconstruct TrianglePos from 3 vertices.
        
        Accounts for grid skewing: x = vx*2 + vy for up, x = vx*2 + vy - 1 for down.
        """
        vs = sorted(verts, key=lambda v: (v[1], v[0]))
        vx, vy = vs[0][0], vs[0][1]
        if vs[0][1] == vs[1][1]:  # Two vertices on same y = up triangle
            x = vx * 2 + vy
            return cls(x=x, y=vy, points_up=True)
        else:  # Down triangle - single vertex at bottom
            x = vx * 2 + vy - 1
            return cls(x=x, y=vy, points_up=False)

    def rotate_120_cw(self) -> "TrianglePos":
        """Rotate this position 120° clockwise around origin."""
        verts = self._to_vertices()
        rotated = [self._rotate_vertex_120(i, j) for i, j in verts]
        return self._from_vertices(rotated)

    def rotate_60_cw(self) -> "TrianglePos":
        """Rotate this position 60° clockwise around origin.
        
        Note: This flips the triangle orientation (up <-> down).
        """
        verts = self._to_vertices()
        rotated = [self._rotate_vertex_60(i, j) for i, j in verts]
        return self._from_vertices(rotated)

    def flip_x(self) -> "TrianglePos":
        """Flip (mirror) across x=0 axis: T(x,y) -> T(-x, y)."""
        return TrianglePos(-self.x, self.y, self.points_up)


@dataclass(frozen=True)
class Piece:
    """A puzzle piece made of connected triangles."""
    name: str
    triangles: tuple[TrianglePos, ...]  # Relative positions, first is anchor
    can_flip: bool = True  # Whether this piece can be flipped

    def _normalize(self) -> "Piece":
        """Translate piece so first triangle is at origin."""
        first = self.triangles[0]
        if first.x == 0 and first.y == 0:
            return self
        normalized = tuple(
            TrianglePos(t.x - first.x, t.y - first.y, t.points_up)
            for t in self.triangles
        )
        return Piece(self.name, normalized, self.can_flip)

    def _canonical_key(self) -> frozenset:
        """Return a canonical key for shape comparison (order-independent).
        
        Normalizes by the min (y, x) position to ensure identical shapes
        produce identical keys regardless of triangle order in tuple.
        """
        min_tri = min(self.triangles, key=lambda t: (t.y, t.x))
        return frozenset(
            (t.x - min_tri.x, t.y - min_tri.y, t.points_up)
            for t in self.triangles
        )

    def rotate_60(self) -> "Piece":
        """Rotate piece 60° clockwise and normalize.
        
        Note: This changes the anchor orientation (up <-> down).
        """
        rotated = tuple(t.rotate_60_cw() for t in self.triangles)
        return Piece(self.name, rotated, self.can_flip)._normalize()

    def rotate_120(self) -> "Piece":
        """Rotate piece 120° clockwise and normalize."""
        rotated = tuple(t.rotate_120_cw() for t in self.triangles)
        return Piece(self.name, rotated, self.can_flip)._normalize()

    def flip(self) -> "Piece":
        """Flip piece (mirror across vertical axis) and normalize."""
        flipped = tuple(t.flip_x() for t in self.triangles)
        return Piece(self.name, flipped, self.can_flip)._normalize()

    def all_orientations(self, force_flip: bool = False) -> list["Piece"]:
        """Return all unique orientations.
        
        Uses 6 rotations (60° each). If can_flip or force_flip, also includes flipped versions.
        Deduplicates by canonical key (shape comparison).
        
        Args:
            force_flip: If True, always include flipped versions (ignores can_flip).
                       Used for easier difficulty modes.
        """
        include_flips = self.can_flip or force_flip
        
        seen = set()
        orientations = []
        
        p = self._normalize()
        for _ in range(6):
            # Add this rotation if shape not seen
            key = p._canonical_key()
            if key not in seen:
                seen.add(key)
                orientations.append(p)
            
            # Add flipped version if allowed
            if include_flips:
                flipped = p.flip()
                flip_key = flipped._canonical_key()
                if flip_key not in seen:
                    seen.add(flip_key)
                    orientations.append(flipped)
            
            p = p.rotate_60()
        
        return orientations


@dataclass(frozen=True)
class BoardCell:
    """A cell on the game board with a dice-matching ID."""
    pos: TrianglePos
    cell_id: int  # 1-48 for dice matching


@dataclass
class Board:
    """The star-shaped game board."""
    cells: dict[TrianglePos, BoardCell]      # Board structure (shared, immutable)
    occupied: dict[TrianglePos, str | None]  # Piece name or None=blocker
    placements: dict[str, tuple[Piece, TrianglePos]] = None  # Track placed piece objects (for orientation)

    def __post_init__(self):
        if self.placements is None:
            # Bypass frozen dataclass check if needed, but this is a standard dataclass not frozen
            self.placements = {}

    @classmethod
    def create_star(cls) -> "Board":
        """Create an empty star-shaped board with 48 cells.
        
        Y increases upward: visual row 0 (top tip) is at y=7, visual row 7 (bottom tip) is at y=0.
        """
        # Star layout: (visual_row, start_x, num_cells)
        # We convert visual_row to y = 7 - visual_row so y increases upward
        star_layout = [
            (0, 5, 1),   # Top tip
            (1, 4, 3),
            (2, 0, 11),  # Wide row with left/right points
            (3, 1, 9),
            (4, 1, 9),
            (5, 0, 11),  # Wide row with left/right points
            (6, 4, 3),
            (7, 5, 1),   # Bottom tip
        ]

        cells: dict[TrianglePos, BoardCell] = {}
        cell_id = 1

        for visual_row, start_x, num_cells in star_layout:
            y = 7 - visual_row  # Flip: visual row 0 -> y=7, visual row 7 -> y=0
            for i in range(num_cells):
                x = start_x + i
                points_up = (x + y) % 2 == 0
                pos = TrianglePos(x, y, points_up)
                cells[pos] = BoardCell(pos, cell_id)
                cell_id += 1

        return cls(cells=cells, occupied={}, placements={})

    def copy(self) -> "Board":
        """Copy for backtracking - shares cells, copies occupied."""
        return Board(
            cells=self.cells,
            occupied=self.occupied.copy(),
            placements=self.placements.copy()
        )

    def place_blocker(self, cell_id: int) -> None:
        """Block a cell by its dice number (mutates board)."""
        for pos, cell in self.cells.items():
            if cell.cell_id == cell_id:
                self.occupied[pos] = None  # None marks blocker
                return
        raise ValueError(f"No cell with id {cell_id}")

    def can_place(self, piece: Piece, anchor: TrianglePos) -> bool:
        """Check if piece can be placed with first triangle at anchor.
        
        The anchor's orientation must match the piece's first triangle orientation
        for the placement to work correctly on the grid.
        """
        # First check: anchor orientation must match piece's first triangle
        first_triangle = piece.triangles[0]
        if first_triangle.points_up != anchor.points_up:
            return False  # Orientation mismatch
        
        for triangle in piece.triangles:
            # Translate piece triangle to board position
            pos = self._translate(triangle, anchor)
            if pos is None:
                return False  # Translation failed (orientation mismatch)
            if pos not in self.cells:
                return False  # Outside board
            if pos in self.occupied:
                return False  # Already occupied
        return True

    def place(self, piece: Piece, anchor: TrianglePos) -> None:
        """Place piece on board (mutates). Call can_place first!"""
        for triangle in piece.triangles:
            pos = self._translate(triangle, anchor)
            self.occupied[pos] = piece.name
        self.placements[piece.name] = (piece, anchor)

    def remove(self, piece_name: str) -> None:
        """Remove all cells occupied by a piece."""
        to_remove = [pos for pos, name in self.occupied.items() if name == piece_name]
        for pos in to_remove:
            del self.occupied[pos]
        if piece_name in self.placements:
            del self.placements[piece_name]

    def empty_cells(self) -> list[BoardCell]:
        """Get all unoccupied cells."""
        return [cell for pos, cell in self.cells.items() if pos not in self.occupied]

    def first_empty_cell(self) -> BoardCell | None:
        """Get first unoccupied cell (for solver)."""
        for pos, cell in self.cells.items():
            if pos not in self.occupied:
                return cell
        return None

    def _count_empty_neighbors(self, pos: TrianglePos) -> int:
        """Count how many neighbors of this position are empty."""
        neighbors = [pos.left(), pos.right(), pos.vertical()]
        count = 0
        for n in neighbors:
            if n in self.cells and n not in self.occupied:
                count += 1
        return count

    def generate_cell_graph(self) -> dict[int, list[int]]:
        """Generate adjacency graph mapping cell_id to neighbor cell_ids.
        
        Returns a dict where keys are cell_ids (1-48) and values are lists
        of neighbor cell_ids that exist on the board.
        """
        graph: dict[int, list[int]] = {}
        
        for pos, cell in self.cells.items():
            neighbors = [pos.left(), pos.right(), pos.vertical()]
            neighbor_ids = []
            for n in neighbors:
                if n in self.cells:
                    neighbor_ids.append(self.cells[n].cell_id)
            graph[cell.cell_id] = neighbor_ids
        
        return graph

    def most_constrained_empty_cell(self) -> BoardCell | None:
        """Get the empty cell with fewest empty neighbors (most constrained).
        
        This heuristic helps the solver by tackling the hardest cells first,
        which leads to earlier pruning of invalid branches.
        """
        best_cell = None
        best_score = 4  # Max possible neighbors + 1
        
        for pos, cell in self.cells.items():
            if pos not in self.occupied:
                score = self._count_empty_neighbors(pos)
                if score < best_score:
                    best_score = score
                    best_cell = cell
                    if score == 0:
                        # Can't do better than 0, return immediately
                        return cell
        
        return best_cell

    def is_solved(self) -> bool:
        """True if all cells are filled."""
        return len(self.occupied) == len(self.cells)

    def _translate(self, triangle: TrianglePos, anchor: TrianglePos) -> TrianglePos | None:
        """Translate a piece triangle relative to an anchor position.
        
        Result orientation is determined by grid position.
        Checks that relative orientation (triangle vs anchor) is consistent with grid.
        """
        result_x = anchor.x + triangle.x
        result_y = anchor.y + triangle.y
        
        # Result orientation is determined by grid position
        result_up = (result_x + result_y) % 2 == 0
        
        # Relative orientation check:
        # If triangle.points_up == anchor.points_up: result should match anchor
        # If triangle.points_up != anchor.points_up: result should differ from anchor
        # (This is because pieces define triangles relative to their first triangle,
        # and can_place verifies first triangle matches anchor)
        
        if triangle.points_up == anchor.points_up:
            # Same orientation expected
            if result_up != anchor.points_up:
                return None
        else:
            # Different orientation expected  
            if result_up == anchor.points_up:
                return None
        
        return TrianglePos(x=result_x, y=result_y, points_up=result_up)

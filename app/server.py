"""
Star Genius - FastAPI Backend Server

Serves the web app and provides API endpoints for solving and board detection.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from sg_solver import Board, solve_puzzle, ALL_PIECES, PIECE_ORIENTATIONS

app = FastAPI(title="Star Genius")

# Serve static files
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")


# Request/Response models
class BoardState(BaseModel):
    blockers: list[int]  # Cell IDs of blockers
    pieces: dict[str, list[int]] = {}  # Placed pieces: name -> cell IDs


class SolveResponse(BaseModel):
    success: bool
    solution: dict[str, list[int]] | None = None  # Piece name -> cell IDs
    orientations: dict[str, int] | None = None   # Piece name -> orientation index
    anchors: dict[str, int] | None = None        # Piece name -> anchor cell ID
    error: str | None = None


class DetectResponse(BaseModel):
    success: bool
    blockers: list[int] = []
    pieces: dict[str, list[int]] = {}
    error: str | None = None


# Routes
@app.get("/")
async def index():
    """Serve the main page."""
    return FileResponse(static_path / "index.html")


@app.get("/api/test-solve", response_model=SolveResponse)
async def test_solve():
    """
    Debug endpoint: returns a fixed piece at a fixed location.
    T5 rotated (orientation 1) at cells 15,14,24,23,32.
    """
    print("test_solve - T5 orientation 1 at cells 15,14,24,23,32")
    return SolveResponse(
        success=True,
        solution={"T5": [15, 14, 24, 23, 32]},  # T5 rotated
        orientations={"T5": 1},                  # Orientation 1
        anchors={"T5": 32}                       # Anchor is cell 15 (first in list)
    )


@app.post("/api/solve", response_model=SolveResponse)
async def solve_board(state: BoardState):
    """
    Solve the puzzle given blockers and optionally pre-placed pieces.
    Returns the solution as piece name -> list of cell IDs.
    """
    try:
        # For now, only support solving from blockers (no pre-placed pieces)
        if len(state.blockers) != 7:
            return SolveResponse(
                success=False,
                error=f"Need exactly 7 blockers, got {len(state.blockers)}"
            )
        
        # Run solver
        result = solve_puzzle(state.blockers, slow=0, difficulty=0)
        
        if result is None:
            return SolveResponse(success=False, error="No solution found")
        
        # Convert result to cell IDs, orientations, and anchors
        solution = {}
        orientations = {}
        anchors = {}
        
        # Determine orientation indices and anchors
        # We need to match the placed piece shape against known orientations

        for piece_name, (placed_piece, anchor_pos) in result.placements.items():
            # Get anchor cell ID
            if anchor_pos in result.cells:
                anchors[piece_name] = result.cells[anchor_pos].cell_id
            
            known_orients = PIECE_ORIENTATIONS[piece_name]
            placed_key = placed_piece._canonical_key()
            
            found_idx = 0
            for i, p in enumerate(known_orients):
                if p._canonical_key() == placed_key:
                    found_idx = i
                    break
            orientations[piece_name] = found_idx

        for pos, piece_name in result.occupied.items():
            if piece_name is None:  # Skip blockers
                continue
            cell_id = result.cells[pos].cell_id
            if piece_name not in solution:
                solution[piece_name] = []
            solution[piece_name].append(cell_id)

        return SolveResponse(success=True, solution=solution, orientations=orientations, anchors=anchors)
        
    except Exception as e:
        return SolveResponse(success=False, error=str(e))


@app.post("/api/detect-board", response_model=DetectResponse)
async def detect_board(image: UploadFile = File(...)):
    """
    Detect board state from an uploaded image.
    Returns detected blockers and optionally pieces.
    """
    try:
        # Save uploaded image temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Import detection modules
            from board_detection.white_triangles import detect_white_triangles
            
            # Run detection
            blockers = detect_white_triangles(tmp_path)
            
            if not blockers:
                return DetectResponse(
                    success=False,
                    error="Could not detect blockers in image"
                )
            
            # Ensure we have exactly 7 blockers (or handle edge cases)
            if len(blockers) < 7:
                return DetectResponse(
                    success=True,
                    blockers=blockers,
                    error=f"Only detected {len(blockers)} blockers (expected 7)"
                )
            elif len(blockers) > 7:
                # Take first 7 (most confident)
                blockers = blockers[:7]
            
            return DetectResponse(success=True, blockers=blockers)
            
        finally:
            os.unlink(tmp_path)
            
    except ImportError as e:
        return DetectResponse(
            success=False,
            error=f"Detection module not available: {e}"
        )
    except Exception as e:
        return DetectResponse(success=False, error=str(e))


if __name__ == "__main__":
    print("Starting Star Genius server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

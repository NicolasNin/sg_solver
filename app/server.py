"""
Star Genius - FastAPI Backend Server

Serves the web app and provides API endpoints for solving and board detection.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from board import Board
from solver import solve_puzzle
from pieces import ALL_PIECES, PIECE_ORIENTATIONS

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
        
        # Convert result to cell IDs
        solution = {}
        for pos, piece_name in result.occupied.items():
            if piece_name is None:  # Skip blockers
                continue
            cell_id = result.cells[pos].cell_id
            if piece_name not in solution:
                solution[piece_name] = []
            solution[piece_name].append(cell_id)
        
        return SolveResponse(success=True, solution=solution)
        
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

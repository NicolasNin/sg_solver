"""
Star Genius - FastAPI Backend Server

Serves the web app and provides API endpoints for solving and board detection.
"""

from pathlib import Path
import sys

# Add project root to path for board_detection imports
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from sg_solver import Board, TrianglePos, solve_puzzle, solve_puzzle_from_board, ALL_PIECES, PIECE_ORIENTATIONS
from app.database import (
    init_db, save_score, get_best_time, get_leaderboard, get_global_stats,
    get_all_solved_boards, get_unsolved_boards_for_client, 
    get_unsolved_boards_for_player, get_solved_boards_for_player,
    update_score_name
)

# Initialize database on import
init_db()

app = FastAPI(title="Star Genius")
api_router = APIRouter(prefix="/api")

# Enable CORS
# Allow all origins for simplicity and robust public access
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class BoardState(BaseModel):
    blockers: list[int]  # Cell IDs of blockers
    pieces: dict[str, list[int]] = {}  # Placed pieces: name -> cell IDs
    use_full_board: bool = False  # If True, solve respecting pre-placed pieces


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
    orientations: dict[str, int] = {}  # Piece name -> orientation index
    anchors: dict[str, int] = {}  # Piece name -> anchor cell ID
    error: str | None = None


# Routes
@app.get("/")
async def index():
    """Serve the main page."""
    return FileResponse(static_path / "index.html")


@api_router.get("/test-solve", response_model=SolveResponse)
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


def board_from_state(state: BoardState) -> Board:
    """
    Reconstruct a Board object from a BoardState.
    Places blockers and any pre-placed pieces.
    """
    from sg_solver.viz import render_svg  # Debug visualization
    
    board = Board.create_star()
    
    # Build cell_id -> pos lookup
    id_to_pos = {cell.cell_id: pos for pos, cell in board.cells.items()}
    
    # Place blockers
    for cell_id in state.blockers:
        board.place_blocker(cell_id)
    
    # Place pieces (if any)
    for piece_name, cell_ids in state.pieces.items():
        # Mark cells as occupied
        for cell_id in cell_ids:
            pos = id_to_pos.get(cell_id)
            if pos:
                board.occupied[pos] = piece_name
        
        # Add to placements dict so solver knows this piece is placed
        # We use a placeholder since we don't have the exact orientation,
        # but the solver only checks placements.keys() to skip pieces
        board.placements[piece_name] = (None, None)
    
    # Debug: render the reconstructed board
    svg_path = render_svg(board)
    print(f"[DEBUG] board_from_state: rendered to {svg_path}")
    print(f"[DEBUG] Blockers: {state.blockers}")
    print(f"[DEBUG] Pieces: {state.pieces}")
    print(f"[DEBUG] Placements keys: {list(board.placements.keys())}")
    print(f"[DEBUG] Occupied count: {len(board.occupied)}")
    
    return board


@api_router.post("/solve", response_model=SolveResponse)
async def solve_board(state: BoardState):
    """
    Solve the puzzle given blockers and optionally pre-placed pieces.
    
    Args:
        state.use_full_board: If False (default), solve from just blockers.
                              If True, solve respecting pre-placed pieces.
    
    Returns the solution as piece name -> list of cell IDs.
    """
    try:
        if state.use_full_board:
            # Solve from full board state (pieces + blockers)
            board = board_from_state(state)
            result = solve_puzzle_from_board(board, slow=0, difficulty=0)
        else:
            # Original behavior: solve from blockers only
            if len(state.blockers) != 7:
                return SolveResponse(
                    success=False,
                    error=f"Need exactly 7 blockers, got {len(state.blockers)}"
                )
            result = solve_puzzle(state.blockers, slow=0, difficulty=0)
        
        if result is None:
            return SolveResponse(success=False, error="No solution found")
        
        # Convert result to cell IDs, orientations, and anchors
        solution = {}
        orientations = {}
        anchors = {}
        
        # Track which pieces were pre-placed (have None placeholders)
        pre_placed = set()
        
        for piece_name, (placed_piece, anchor_pos) in result.placements.items():
            # Skip pre-placed pieces (they have None placeholders from board_from_state)
            if placed_piece is None:
                pre_placed.add(piece_name)
                continue
                
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
            if piece_name in pre_placed:  # Skip pre-placed pieces (JS already has them)
                continue
            cell_id = result.cells[pos].cell_id
            if piece_name not in solution:
                solution[piece_name] = []
            solution[piece_name].append(cell_id)

        return SolveResponse(success=True, solution=solution, orientations=orientations, anchors=anchors)
        
    except Exception as e:
        import sys
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)        
        return SolveResponse(success=False, error=str(e))


@api_router.post("/detect-board", response_model=DetectResponse)
async def detect_board(image: UploadFile = File(...)):
    """
    Detect board state from an uploaded image.
    Returns detected blockers and pieces.
    """
    try:
        # Save uploaded image for debugging
        from datetime import datetime
        
        debug_dir = _project_root / "data" / "debug_uploads"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = debug_dir / f"capture_{timestamp}.jpg"
        
        content = await image.read()
        with open(debug_path, "wb") as f:
            f.write(content)
        
        print(f"[DEBUG] Saved uploaded image to: {debug_path}")
        
        try:
            # Import detection pipeline (new clean version)
            from board_detection.board_pipeline_clean import board_pipelineYolo2
            
            # Run detection
            result = board_pipelineYolo2(str(debug_path))
            
            if result is None:
                return DetectResponse(
                    success=False,
                    error="Could not detect board in image"
                )
            
            # Parse BoardResult.final_results
            # final_results is a list of 48 items: 'e' (empty), 'w' (white/blocker), or color name
            final_results = result.final_results
            print(final_results)
            # Extract blockers (white triangles marked as 'w')
            blockers = []
            for i, res in enumerate(final_results):
                cell_id = i + 1  # Cell IDs are 1-indexed
                if res == 'w':
                    blockers.append(cell_id)
            
            # Extract identified pieces from pipeline
            # identified_pieces is [(piece_name, orientation, anchor, cell_ids, color), ...]
            pieces = {}
            orientations = {}
            anchors = {}
            
            if result.identified_pieces:
                for piece_name, orientation, anchor, cell_ids, color in result.identified_pieces:
                    if piece_name is not None:
                        pieces[piece_name] = cell_ids
                        if orientation is not None:
                            orientations[piece_name] = orientation
                        if anchor is not None:
                            anchors[piece_name] = anchor
                    else:
                        # Piece not identified, use color as fallback key
                        pieces[f"unknown_{color}"] = cell_ids
            
            print(f"[DEBUG] Detected blockers: {blockers}")
            print(f"[DEBUG] Detected pieces: {pieces}")
            
            if len(blockers) < 7:
                return DetectResponse(
                    success=True,
                    blockers=blockers,
                    pieces=pieces,
                    orientations=orientations,
                    anchors=anchors,
                    error=f"Only detected {len(blockers)} blockers (expected 7)"
                )
            elif len(blockers) > 7:
                # Take first 7 (should not normally happen)
                blockers = blockers[:7]
            
            return DetectResponse(success=True, blockers=blockers, pieces=pieces, orientations=orientations, anchors=anchors)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return DetectResponse(success=False, error=str(e))
            
    except ImportError as e:
        print(e)
        return DetectResponse(
            success=False,
            error=f"Detection module not available: {e}"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return DetectResponse(success=False, error=str(e))


# === Score/Leaderboard Endpoints ===

class ScoreSubmission(BaseModel):
    board_code: str
    player_name: str
    time_seconds: float
    hints_used: int = 0
    pieces_placed: int = 11
    client_id: Optional[str] = None

class ScoreResponse(BaseModel):
    score_id: int
    rank: int
    is_best: bool
    best_time: float
    best_player: str

class LeaderboardEntry(BaseModel):
    rank: int
    player: str
    time: float
    hints_used: int

class LeaderboardResponse(BaseModel):
    board_code: str
    best_time: Optional[float] = None
    best_player: Optional[str] = None
    leaderboard: list[LeaderboardEntry] = []

@api_router.post("/scores", response_model=ScoreResponse)
async def submit_score(submission: ScoreSubmission):
    """Submit a solve time for a board."""
    result = save_score(
        board_code=submission.board_code,
        player_name=submission.player_name,
        time_seconds=submission.time_seconds,
        hints_used=submission.hints_used,
        pieces_placed=submission.pieces_placed,
        client_id=submission.client_id
    )
    return ScoreResponse(**result)

@api_router.get("/scores/{board_code}", response_model=LeaderboardResponse)
async def get_board_leaderboard(board_code: str, limit: int = 5):
    """Get leaderboard for a specific board."""
    best = get_best_time(board_code)
    lb = get_leaderboard(board_code, limit)
    
    return LeaderboardResponse(
        board_code=board_code,
        best_time=best["time_seconds"] if best else None,
        best_player=best["player_name"] if best else None,
        leaderboard=[LeaderboardEntry(**entry) for entry in lb]
    )

class ScoreNameUpdate(BaseModel):
    player_name: str
    client_id: str

@api_router.patch("/scores/{score_id}")
async def update_score(score_id: int, update: ScoreNameUpdate):
    """Update player name for a score (must own via client_id)."""
    success = update_score_name(score_id, update.player_name, update.client_id)
    if not success:
        raise HTTPException(status_code=404, detail="Score not found or not owned")
    return {"success": True, "score_id": score_id}

@api_router.get("/stats")
async def get_stats():
    """Get global statistics."""
    return get_global_stats()


# === Board Discovery Endpoints ===

@api_router.get("/boards")
async def get_boards():
    """Get all boards that have been solved (for suggestions)."""
    return {"boards": get_all_solved_boards()}


@api_router.get("/boards/unsolved")
async def get_unsolved_boards(client_id: Optional[str] = None, player_name: Optional[str] = None):
    """
    Get boards not yet solved by this user.
    Priority: player_name > client_id
    """
    if player_name:
        boards = get_unsolved_boards_for_player(player_name)
    elif client_id:
        boards = get_unsolved_boards_for_client(client_id)
    else:
        # No filter - return all boards
        boards = get_all_solved_boards()
    
    return {"boards": boards}


@api_router.get("/boards/player/{player_name}")
async def get_player_boards(player_name: str):
    """Get boards that this player has solved."""
    return {
        "player": player_name,
        "solved": get_solved_boards_for_player(player_name),
        "unsolved": get_unsolved_boards_for_player(player_name)
    }

# Register the API router
app.include_router(api_router)

# Serve static files - MUST be last
static_path = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    cert_path = Path(__file__).parent / "certs"
    ssl_keyfile = cert_path / "key.pem"
    ssl_certfile = cert_path / "cert.pem"
    
    if ssl_keyfile.exists() and ssl_certfile.exists():
        print("Starting Star Genius server at https://0.0.0.0:8000 (HTTPS)")
        uvicorn.run(app, host="0.0.0.0", port=8000, 
                    ssl_keyfile=str(ssl_keyfile), ssl_certfile=str(ssl_certfile))
    else:
        print("Starting Star Genius server at http://localhost:8000 (No certs found)")
        uvicorn.run(app, host="0.0.0.0", port=8000)

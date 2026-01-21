"""
Star Genius - Database module for score tracking
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

# Database file location
DB_PATH = Path(__file__).parent / "scores.db"


def get_connection():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            board_code TEXT NOT NULL,
            player_name TEXT NOT NULL,
            time_seconds REAL NOT NULL,
            difficulty TEXT DEFAULT 'normal',
            hints_used INTEGER DEFAULT 0,
            pieces_placed INTEGER DEFAULT 11,
            is_complete BOOLEAN DEFAULT 1,
            client_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index for fast leaderboard queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_board_time 
        ON scores(board_code, time_seconds)
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


def save_score(
    board_code: str,
    player_name: str,
    time_seconds: float,
    difficulty: str = "normal",
    hints_used: int = 0,
    pieces_placed: int = 11,
    is_complete: bool = True,
    client_id: Optional[str] = None
) -> dict:
    """
    Save a score to the database.
    Returns rank info and whether this is the new best time.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Insert the score
    cursor.execute("""
        INSERT INTO scores (board_code, player_name, time_seconds, difficulty, 
                           hints_used, pieces_placed, is_complete, client_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (board_code, player_name, time_seconds, difficulty, 
          hints_used, pieces_placed, is_complete, client_id))
    
    score_id = cursor.lastrowid
    
    # Get rank for this score (how many are faster)
    # Get rank for this score (how many are better)
    # Better means: fewer hints OR (same hints AND faster time)
    cursor.execute("""
        SELECT COUNT(*) + 1 as rank FROM scores 
        WHERE board_code = ? 
        AND (hints_used < ? OR (hints_used = ? AND time_seconds < ?))
    """, (board_code, hints_used, hints_used, time_seconds))
    rank = cursor.fetchone()["rank"]
    
    # Get best time for this board
    cursor.execute("""
        SELECT time_seconds, player_name FROM scores 
        WHERE board_code = ? 
        ORDER BY hints_used ASC, time_seconds ASC LIMIT 1
    """, (board_code,))
    best = cursor.fetchone()
    
    conn.commit()
    conn.close()
    
    return {
        "score_id": score_id,
        "rank": rank,
        "is_best": rank == 1,
        "best_time": best["time_seconds"] if best else time_seconds,
        "best_player": best["player_name"] if best else player_name
    }


def update_score_name(
    score_id: int,
    player_name: str,
    client_id: str
) -> bool:
    """
    Update player name for a score.
    Only allows update if client_id matches (security).
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Verify ownership and update
    cursor.execute("""
        UPDATE scores 
        SET player_name = ?
        WHERE id = ? AND client_id = ?
    """, (player_name, score_id, client_id))
    
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return updated


def get_best_time(board_code: str) -> Optional[dict]:
    """Get the best time for a specific board."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT time_seconds, player_name, created_at FROM scores 
        WHERE board_code = ? 
        ORDER BY hints_used ASC, time_seconds ASC LIMIT 1
    """, (board_code,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "time_seconds": row["time_seconds"],
            "player_name": row["player_name"],
            "created_at": row["created_at"]
        }
    return None


def get_leaderboard(board_code: str, limit: int = 5) -> list:
    """Get top scores for a specific board."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT player_name, time_seconds, hints_used, created_at 
        FROM scores 
        WHERE board_code = ? 
        ORDER BY hints_used ASC, time_seconds ASC 
        LIMIT ?
    """, (board_code, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "rank": i + 1,
            "player": row["player_name"],
            "time": row["time_seconds"],
            "hints_used": row["hints_used"],
            "date": row["created_at"]
        }
        for i, row in enumerate(rows)
    ]


def get_global_stats() -> dict:
    """Get global statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as total FROM scores")
    total = cursor.fetchone()["total"]
    
    cursor.execute("SELECT COUNT(DISTINCT board_code) as boards FROM scores")
    boards = cursor.fetchone()["boards"]
    
    cursor.execute("SELECT COUNT(DISTINCT player_name) as players FROM scores")
    players = cursor.fetchone()["players"]
    
    conn.close()
    
    return {
        "total_solves": total,
        "unique_boards": boards,
        "unique_players": players
    }


def get_all_solved_boards() -> list:
    """Get all unique board codes that have been solved."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT board_code, 
               MIN(time_seconds) as best_time,
               COUNT(*) as solve_count
        FROM scores 
        GROUP BY board_code
        ORDER BY solve_count DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "board_code": row["board_code"],
            "best_time": row["best_time"],
            "solve_count": row["solve_count"]
        }
        for row in rows
    ]


def get_unsolved_boards_for_client(client_id: str) -> list:
    """Get boards that this client_id has NOT solved."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get all boards, excluding those solved by this client
    cursor.execute("""
        SELECT DISTINCT board_code, MIN(time_seconds) as best_time
        FROM scores 
        WHERE board_code NOT IN (
            SELECT DISTINCT board_code FROM scores WHERE client_id = ?
        )
        GROUP BY board_code
    """, (client_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {"board_code": row["board_code"], "best_time": row["best_time"]}
        for row in rows
    ]


def get_unsolved_boards_for_player(player_name: str) -> list:
    """Get boards that this player has NOT solved."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT board_code, MIN(time_seconds) as best_time
        FROM scores 
        WHERE board_code NOT IN (
            SELECT DISTINCT board_code FROM scores WHERE player_name = ?
        )
        GROUP BY board_code
    """, (player_name,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {"board_code": row["board_code"], "best_time": row["best_time"]}
        for row in rows
    ]


def get_solved_boards_for_player(player_name: str) -> list:
    """Get boards that this player HAS solved."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT board_code, MIN(time_seconds) as best_time, COUNT(*) as attempts
        FROM scores 
        WHERE player_name = ?
        GROUP BY board_code
        ORDER BY best_time ASC
    """, (player_name,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "board_code": row["board_code"],
            "best_time": row["best_time"],
            "attempts": row["attempts"]
        }
        for row in rows
    ]


# Initialize DB when module is imported
if __name__ == "__main__":
    # Test the database
    init_db()
    
    # Test save
    result = save_score("test123", "TestPlayer", 45.5)
    print(f"Saved score: {result}")
    
    # Test leaderboard
    lb = get_leaderboard("test123")
    print(f"Leaderboard: {lb}")
    
    # Test stats
    stats = get_global_stats()
    print(f"Stats: {stats}")

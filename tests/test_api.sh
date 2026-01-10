#!/bin/bash
# Star Genius API Test Script
# Run from project root: ./tests/test_api.sh

BASE_URL="http://localhost:8000"

echo "=== Star Genius API Tests ==="
echo ""

# --- Score Submission ---
echo "--- POST /api/scores (Submit score) ---"
curl -s -X POST "$BASE_URL/api/scores" \
  -H "Content-Type: application/json" \
  -d '{"board_code":"testboard","player_name":"TestUser","time_seconds":42.5,"hints_used":0}'
echo -e "\n"

# --- Get Leaderboard ---
echo "--- GET /api/scores/{board_code} (Leaderboard) ---"
curl -s "$BASE_URL/api/scores/testboard"
echo -e "\n"

# --- Global Stats ---
echo "--- GET /api/stats (Global stats) ---"
curl -s "$BASE_URL/api/stats"
echo -e "\n"

# --- All Solved Boards ---
echo "--- GET /api/boards (All solved boards) ---"
curl -s "$BASE_URL/api/boards"
echo -e "\n"

# --- Unsolved Boards for Player ---
echo "--- GET /api/boards/unsolved?player_name=Alice ---"
curl -s "$BASE_URL/api/boards/unsolved?player_name=Alice"
echo -e "\n"

# --- Unsolved Boards for Client ---
echo "--- GET /api/boards/unsolved?client_id=device123 ---"
curl -s "$BASE_URL/api/boards/unsolved?client_id=device123"
echo -e "\n"

# --- Player History ---
echo "--- GET /api/boards/player/Bob ---"
curl -s "$BASE_URL/api/boards/player/Bob"
echo -e "\n"

# --- Solve Puzzle (Example) ---
echo "--- POST /api/solve (Solve puzzle - example) ---"
curl -s -X POST "$BASE_URL/api/solve" \
  -H "Content-Type: application/json" \
  -d '{"blockers":[1,5,10,15,20,25,30]}'
echo -e "\n"

echo "=== Tests Complete ==="

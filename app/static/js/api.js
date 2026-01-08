/**
 * Star Genius - API Communication
 * 
 * Backend API calls for solver and board detection.
 */

const API_BASE = '/api';

// Solve the current board state
async function solvePuzzle(boardState) {
    const response = await fetch(`${API_BASE}/solve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(boardState),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Solver failed');
    }

    return response.json();
}

// DEBUG: Test solve with fixed piece placement
async function testSolve() {
    const response = await fetch(`${API_BASE}/test-solve`, {
        method: 'GET',
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Test solver failed');
    }

    return response.json();
}

// Detect board from uploaded image
async function detectBoard(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${API_BASE}/detect-board`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Detection failed');
    }

    return response.json();
}

// Export
window.StarGenius.api = {
    solvePuzzle,
    detectBoard,
    testSolve,
};

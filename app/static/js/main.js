/**
 * Star Genius - Main Entry Point
 * 
 * Initializes the game and wires up UI controls.
 */

let board;
let game;

// === URL-based Board Sharing ===
// Encode 7 blockers (values 1-48) to a hex string using base-48
function encodeBlockers(blockers) {
    // Convert from 1-indexed (cell IDs) to 0-indexed for encoding
    let value = 0n;
    for (let i = blockers.length - 1; i >= 0; i--) {
        value = value * 48n + BigInt(blockers[i] - 1);
    }
    return value.toString(16);
}

// Decode hex string back to 7 blockers
function decodeBlockers(hex) {
    let value = BigInt('0x' + hex);
    const blockers = [];
    for (let i = 0; i < 7; i++) {
        blockers.push(Number(value % 48n) + 1);  // Convert back to 1-indexed
        value = value / 48n;
    }
    return blockers;
}

// Update URL with current board config
function updateURLWithBlockers(blockers) {
    const code = encodeBlockers(blockers);
    const url = new URL(window.location.href);
    url.searchParams.set('board', code);
    window.history.replaceState({}, '', url);
}

// Check URL for board parameter on load
function getBlockersFromURL() {
    const url = new URL(window.location.href);
    const code = url.searchParams.get('board');
    if (code) {
        try {
            return decodeBlockers(code);
        } catch (e) {
            console.error('Invalid board code:', e);
            return null;
        }
    }
    return null;
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Access StarGenius namespace after all scripts loaded
    const SG = window.StarGenius;

    initializeGame(SG);
    setupEventListeners(SG);

    // Check for board config in URL
    const urlBlockers = getBlockersFromURL();
    if (urlBlockers && urlBlockers.length === 7) {
        console.log('Starting game from URL:', urlBlockers);
        startNewGame(urlBlockers);
    }
});

function initializeGame(SG) {
    // Create board
    const svgElement = document.getElementById('board-svg');
    board = new SG.Board(svgElement);
    game = new SG.Game(board);

    // Create piece palette
    createPalette(SG);
}

function createPalette(SG) {
    const paletteEl = document.getElementById('piece-list');
    paletteEl.innerHTML = '';

    for (const name of SG.ALL_PIECES) {
        const piece = SG.PIECE_DEFINITIONS[name];
        const color = SG.PIECE_COLORS[name];

        const item = document.createElement('div');
        item.className = 'piece-item';
        item.dataset.piece = name;

        // Create mini SVG preview
        const svg = createPieceSVG(piece, color);
        item.appendChild(svg);

        // Piece name
        const label = document.createElement('span');
        label.className = 'piece-name';
        label.textContent = name;
        item.appendChild(label);

        // Click to select
        item.addEventListener('click', () => {
            game.selectPiece(name);
        });

        paletteEl.appendChild(item);
    }
}

// Create mini SVG for piece preview
function createPieceSVG(piece, color) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '-30 -30 60 50');

    const scale = 10;
    const h = scale * Math.sqrt(3) / 2;

    for (const tri of piece.triangles) {
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');

        // Simple triangle rendering for preview
        let points;
        if (tri.pointsUp) {
            const cx = tri.x * scale * 0.5;
            const cy = -tri.y * h;
            points = [
                [cx, cy - h * 0.6],
                [cx - scale * 0.5, cy + h * 0.4],
                [cx + scale * 0.5, cy + h * 0.4],
            ];
        } else {
            const cx = tri.x * scale * 0.5;
            const cy = -tri.y * h;
            points = [
                [cx, cy + h * 0.6],
                [cx - scale * 0.5, cy - h * 0.4],
                [cx + scale * 0.5, cy - h * 0.4],
            ];
        }

        polygon.setAttribute('points', points.map(p => p.join(',')).join(' '));
        polygon.setAttribute('fill', color);
        polygon.setAttribute('stroke', '#000');
        polygon.setAttribute('stroke-width', '0.5');

        svg.appendChild(polygon);
    }

    return svg;
}

function setupEventListeners(SG) {
    // Start modal buttons
    document.getElementById('btn-random').addEventListener('click', () => {
        const blockers = SG.rollDice();
        startNewGame(blockers);
    });

    document.getElementById('btn-photo').addEventListener('click', () => {
        document.getElementById('photo-input').click();
    });

    document.getElementById('photo-input').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            await loadFromPhoto(file);
        }
    });

    // Game controls
    document.getElementById('btn-new-game').addEventListener('click', () => {
        const blockers = SG.rollDice();
        startNewGame(blockers);
    });

    document.getElementById('btn-rotate').addEventListener('click', () => {
        game.rotate();
    });

    document.getElementById('btn-flip').addEventListener('click', () => {
        game.flip();
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
        game.resetToBlockers();
    });

    document.getElementById('btn-solve').addEventListener('click', async () => {
        await solveCurrentPuzzle();
    });

    // DEBUG: Listen for test-solve event from game.js
    document.addEventListener('star-genius-test-solve', async () => {
        console.log('[main.js] Received test-solve event');
        await testSolveDebug();
    });
}

function startNewGame(blockers) {
    hideModal();
    showGame();
    game.startGame(blockers);

    // Update URL for sharing
    updateURLWithBlockers(blockers);
}

async function loadFromPhoto(file) {
    showLoading('Detecting board...');

    try {
        const result = await window.StarGenius.api.detectBoard(file);
        hideLoading();

        if (result.blockers && result.blockers.length > 0) {
            startNewGame(result.blockers);
        } else {
            alert('Could not detect blockers in image');
            hideLoading();
        }
    } catch (error) {
        hideLoading();
        alert('Detection failed: ' + error.message);
    }
}

async function solveCurrentPuzzle() {
    showLoading('Solving puzzle...');

    try {
        const state = game.board.getState();
        const result = await window.StarGenius.api.solvePuzzle(state);
        hideLoading();

        if (result.solution) {
            game.applySolution(result.solution, result.orientations, result.anchors);
        } else {
            alert('No solution found');
        }
    } catch (error) {
        hideLoading();
        alert('Solver error: ' + error.message);
    }
}

// DEBUG: Test with fixed placement to debug rendering
async function testSolveDebug() {
    console.log('[DEBUG] testSolveDebug() called');
    console.log('[DEBUG] game object:', game);
    console.log('[DEBUG] api object:', window.StarGenius?.api);

    if (!game) {
        console.error('[DEBUG] No game object - start a game first!');
        return;
    }

    try {
        // Clear previous debug markers
        if (game._clearDebugMarkers) {
            game._clearDebugMarkers();
        }

        console.log('[DEBUG] About to call testSolve API...');
        const result = await window.StarGenius.api.testSolve();
        console.log('[DEBUG] Test result:', result);

        if (result.solution) {
            console.log('[DEBUG] Calling applySolution...');
            game.applySolution(result.solution, result.orientations, result.anchors);
            console.log('[DEBUG] applySolution done');
        } else {
            console.error('[DEBUG] No solution in test result');
        }
    } catch (error) {
        console.error('[DEBUG] Test solve error:', error);
    }
}

// UI helpers
function showModal() {
    document.getElementById('start-modal').classList.remove('hidden');
    document.getElementById('game-container').classList.add('hidden');
}

function hideModal() {
    document.getElementById('start-modal').classList.add('hidden');
}

function showGame() {
    document.getElementById('game-container').classList.remove('hidden');
}

function showLoading(text = 'Loading...') {
    const loading = document.getElementById('loading');
    loading.querySelector('p').textContent = text;
    loading.classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}


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

    document.getElementById('btn-choose-dice').addEventListener('click', () => {
        showDiceModal(SG);
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

    // Dice modal buttons
    document.getElementById('btn-start-dice').addEventListener('click', () => {
        const blockers = getSelectedDiceValues();
        if (blockers.length === 7) {
            hideDiceModal();
            startNewGame(blockers);
        }
    });

    document.getElementById('btn-cancel-dice').addEventListener('click', () => {
        hideDiceModal();
    });

    // Game controls
    document.getElementById('btn-new-game').addEventListener('click', () => {
        const blockers = SG.rollDice();
        startNewGame(blockers);
    });

    document.getElementById('btn-choose-dice-game').addEventListener('click', () => {
        showDiceModalFromGame(SG);
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
        await solveCurrentPuzzle(true);  // Solve from current position
    });

    document.getElementById('btn-solve-full').addEventListener('click', async () => {
        await solveCurrentPuzzle(false, true);  // Solve from scratch, reset before applying
    });

    // DEBUG: Listen for test-solve event from game.js
    document.addEventListener('star-genius-test-solve', async () => {
        console.log('[main.js] Received test-solve event');
        await testSolveDebug();
    });
}

// Dice selection state
let diceSelections = [null, null, null, null, null, null, null];

function showDiceModal(SG) {
    const modal = document.getElementById('dice-modal');
    const table = document.getElementById('dice-table');

    // Reset selections
    diceSelections = [null, null, null, null, null, null, null];

    // Populate dice values
    const rows = table.querySelectorAll('tr[data-die]');
    rows.forEach((row, dieIndex) => {
        const optionsCell = row.querySelector('.die-options');
        optionsCell.innerHTML = '';

        const dieValues = SG.ALL_DICE[dieIndex];
        dieValues.forEach(value => {
            const btn = document.createElement('button');
            btn.className = 'dice-value';
            btn.textContent = value;
            btn.dataset.value = value;
            btn.addEventListener('click', () => selectDiceValue(dieIndex, value, btn));
            optionsCell.appendChild(btn);
        });
    });

    // Disable start button initially
    document.getElementById('btn-start-dice').disabled = true;

    // Show modal
    document.getElementById('start-modal').classList.add('hidden');
    modal.classList.remove('hidden');
}

// Track where we came from (start modal or game)
let diceModalFromGame = false;

function hideDiceModal() {
    document.getElementById('dice-modal').classList.add('hidden');
    if (diceModalFromGame) {
        // Coming from game - just hide the modal
        diceModalFromGame = false;
    } else {
        // Coming from start modal - show it again
        document.getElementById('start-modal').classList.remove('hidden');
    }
}

function showDiceModalFromGame(SG) {
    diceModalFromGame = true;
    const modal = document.getElementById('dice-modal');
    const table = document.getElementById('dice-table');

    // Reset selections
    diceSelections = [null, null, null, null, null, null, null];

    // Populate dice values
    const rows = table.querySelectorAll('tr[data-die]');
    rows.forEach((row, dieIndex) => {
        const optionsCell = row.querySelector('.die-options');
        optionsCell.innerHTML = '';

        const dieValues = SG.ALL_DICE[dieIndex];
        dieValues.forEach(value => {
            const btn = document.createElement('button');
            btn.className = 'dice-value';
            btn.textContent = value;
            btn.dataset.value = value;
            btn.addEventListener('click', () => selectDiceValue(dieIndex, value, btn));
            optionsCell.appendChild(btn);
        });
    });

    // Disable start button initially
    document.getElementById('btn-start-dice').disabled = true;

    // Show modal
    modal.classList.remove('hidden');
}

function selectDiceValue(dieIndex, value, clickedBtn) {
    // Deselect previous selection in this row
    const row = clickedBtn.closest('tr');
    row.querySelectorAll('.dice-value').forEach(btn => btn.classList.remove('selected'));

    // Select new value
    clickedBtn.classList.add('selected');
    diceSelections[dieIndex] = value;

    // Update start button state
    const allSelected = diceSelections.every(v => v !== null);
    document.getElementById('btn-start-dice').disabled = !allSelected;
}

function getSelectedDiceValues() {
    return diceSelections.filter(v => v !== null);
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

            // Place detected pieces if any
            if (result.pieces && Object.keys(result.pieces).length > 0) {
                console.log('Detected pieces:', result.pieces);
                console.log('Detected orientations:', result.orientations);
                console.log('Detected anchors:', result.anchors);

                // Apply pieces with correct orientations
                for (const [pieceName, cellIds] of Object.entries(result.pieces)) {
                    // Get cell positions from cell IDs
                    const positions = cellIds.map(id => game.board.getCellById(id)).filter(p => p);
                    if (positions.length > 0) {
                        // Get orientation from detection result
                        const orientation = result.orientations?.[pieceName] || 0;

                        // Get anchor cell ID from detection result
                        const anchorId = result.anchors?.[pieceName];
                        const anchorPos = anchorId ? game.board.getCellById(anchorId) : positions[0];

                        // Place piece on board
                        game.board.placePiece(pieceName, positions);
                        game.placedPieces.set(pieceName, orientation);

                        // Get the correctly oriented piece for rendering
                        const orientedPiece = window.StarGenius.PIECE_ORIENTATIONS[pieceName]?.[orientation]
                            || window.StarGenius.PIECE_DEFINITIONS[pieceName];

                        if (orientedPiece && anchorPos) {
                            // Get snap position using correct anchor cell
                            const snapPos = game.board.getSnapPositionForPiece(orientedPiece, anchorPos);
                            if (snapPos) {
                                game.board.renderPiece(orientedPiece, snapPos[0], snapPos[1], orientation);
                            }
                        }
                    }
                }
                game._updateUI();
            }
        } else {
            alert('Could not detect blockers in image');
        }
    } catch (error) {
        hideLoading();
        alert('Detection failed: ' + error.message);
    }
}

async function solveCurrentPuzzle(useFullBoard = false, resetBeforeApply = false) {
    // Use status text instead of loading modal to avoid flicker
    const statusEl = document.getElementById('header-status');
    if (statusEl) {
        statusEl.textContent = '🔄 Solving...';
        statusEl.style.color = '#58a6ff';
    }

    try {
        const state = game.board.getState();
        state.use_full_board = useFullBoard;
        const result = await window.StarGenius.api.solvePuzzle(state);

        if (result.solution && Object.keys(result.solution).length > 0) {
            // Efficiently clear pieces without re-rendering board
            if (resetBeforeApply) {
                game.clearPiecesForSolve();
            }
            game.applySolution(result.solution, result.orientations, result.anchors);
            if (statusEl) {
                statusEl.style.color = '';
                statusEl.textContent = '✓ Solution applied';
            }
        } else {
            // Friendly message
            if (statusEl) {
                statusEl.textContent = '❌ No solution found - try a different arrangement';
                statusEl.style.color = '#ff6b6b';
                setTimeout(() => {
                    statusEl.style.color = '';
                    statusEl.textContent = 'Drag pieces onto the board';
                }, 3000);
            }
        }
    } catch (error) {
        // Show error in status
        if (statusEl) {
            statusEl.textContent = `⚠️ ${error.message}`;
            statusEl.style.color = '#ffa502';
            setTimeout(() => {
                statusEl.style.color = '';
                statusEl.textContent = 'Drag pieces onto the board';
            }, 3000);
        }
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


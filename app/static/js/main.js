/**
 * Star Genius - Main Entry Point
 * 
 * Initializes the game and wires up UI controls.
 */

// API Configuration - get base URL (api.js is loaded first)
function getApiBase() {
    const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    return isLocal ? 'http://localhost:8000' : 'https://sg.ninin.space';
}
// switch to 'https://sg.ninin.space'; for production
//http://192.168.1.11:8000 for testing local
const API_BASE_URL = getApiBase();

// Debug helper - updates visible indicator on page
function debugLog(msg) {
    console.log('[DEBUG]', msg);
    const el = document.getElementById('debug-indicator');
    if (el) el.textContent = msg;
}

let board;
let game;

// === Player Identity ===
// Device ID - generated once and persisted
// Note: crypto.randomUUID() requires HTTPS, so we use a fallback for HTTP
function generateUUID() {
    // Try native first (HTTPS only)
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    // Fallback for HTTP
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

const clientId = localStorage.getItem('sg-client-id') || (() => {
    const id = generateUUID();
    localStorage.setItem('sg-client-id', id);
    return id;
})();

// Player name - editable and persisted
let playerName = localStorage.getItem('sg-player-name') || '';

function setPlayerName(name) {
    playerName = name;
    localStorage.setItem('sg-player-name', name);
    updatePlayerNameDisplay();
}

function updatePlayerNameDisplay() {
    const el = document.getElementById('player-name');
    if (el) {
        el.textContent = playerName || 'Anonymous';
    }
}

// === Leaderboard API ===
async function fetchBestTime(boardCode) {
    try {
        const res = await fetch(`${API_BASE_URL}/api/scores/${boardCode}`);
        const data = await res.json();
        return data.best_time ? { time: data.best_time, player: data.best_player } : null;
    } catch (e) {
        console.error('Failed to fetch best time:', e);
        return null;
    }
}

async function fetchUnsolvedBoards() {
    try {
        const params = new URLSearchParams();
        if (playerName) params.set('player_name', playerName);
        params.set('client_id', clientId);
        const res = await fetch(`${API_BASE_URL}/api/boards/unsolved?${params}`);
        const data = await res.json();
        return data.boards || [];
    } catch (e) {
        console.error('Failed to fetch unsolved boards:', e);
        return [];
    }
}

async function submitScore(boardCode, timeSeconds, hintsUsed = 0) {
    try {
        const res = await fetch(`${API_BASE_URL}/api/scores`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                board_code: boardCode,
                player_name: playerName || 'Anonymous',
                time_seconds: timeSeconds,
                hints_used: hintsUsed,
                client_id: clientId
            })
        });
        return await res.json();
    } catch (e) {
        console.error('Failed to submit score:', e);
        return null;
    }
}

async function fetchLeaderboard(boardCode) {
    try {
        const res = await fetch(`${API_BASE_URL}/api/scores/${boardCode}`);
        return await res.json();
    } catch (e) {
        console.error('Failed to fetch leaderboard:', e);
        return null;
    }
}

// === Win Modal ===
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function getCurrentBoardCode() {
    // Get from URL or encode current blockers
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('board') || 'random';
}

// Track current score for name updates
let currentScoreId = null;

async function updateScoreName(scoreId, newName) {
    try {
        const res = await fetch(`${API_BASE_URL}/api/scores/${scoreId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                player_name: newName,
                client_id: clientId
            })
        });
        return res.ok;
    } catch (e) {
        console.error('Failed to update score name:', e);
        return false;
    }
}

async function showWinModal(solveTime, hintsUsed) {
    const boardCode = getCurrentBoardCode();
    const modal = document.getElementById('win-modal');

    // Display solve time
    document.getElementById('win-time').textContent = formatTime(solveTime);

    // Set name input value
    const nameInput = document.getElementById('win-name-input');
    nameInput.value = playerName || '';

    // Submit score and get rank
    const scoreResult = await submitScore(boardCode, solveTime, hintsUsed);
    if (scoreResult) {
        document.getElementById('win-rank').textContent = `#${scoreResult.rank}`;
        currentScoreId = scoreResult.score_id;  // Store for name updates
    }

    // Fetch and display leaderboard
    const leaderboardData = await fetchLeaderboard(boardCode);
    populateLeaderboard(leaderboardData, solveTime);

    // Show modal
    modal.classList.remove('hidden');
}

function populateLeaderboard(data, currentTime) {
    const container = document.getElementById('win-leaderboard');
    if (!data || !data.leaderboard || data.leaderboard.length === 0) {
        container.innerHTML = '<p class="no-scores">No other scores yet!</p>';
        return;
    }

    let html = `<table>
        <thead><tr><th>#</th><th>Player</th><th>Hints</th><th>Time</th></tr></thead>
        <tbody>`;

    for (const entry of data.leaderboard) {
        const isCurrent = Math.abs(entry.time - currentTime) < 0.1;
        const rankClass = entry.rank <= 3 ? `rank-${entry.rank}` : '';
        html += `<tr class="${isCurrent ? 'current-player' : ''} ${rankClass}">
            <td>${entry.rank}</td>
            <td>${entry.player}</td>
            <td>${entry.hints_used}</td>
            <td>${formatTime(entry.time)}</td>
        </tr>`;
    }

    html += '</tbody></table>';
    container.innerHTML = html;
}

function hideWinModal() {
    document.getElementById('win-modal').classList.add('hidden');
}

async function updateBestTimeDisplay(boardCode) {
    const bestTimeEl = document.getElementById('best-time');
    const best = await fetchBestTime(boardCode);

    if (best && best.time) {
        bestTimeEl.textContent = `🏆 ${formatTime(best.time)} (${best.player})`;
        bestTimeEl.classList.remove('hidden');
    } else {
        bestTimeEl.classList.add('hidden');
    }
}

// Game win callback
function handleGameWin(solveTime, hintsUsed) {
    showWinModal(solveTime, hintsUsed);
}

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

// Decode hex string back to 7 blockers (returns null if invalid)
function decodeBlockers(hex) {
    try {
        // Validate hex string format
        if (!hex || !/^[0-9a-fA-F]+$/.test(hex)) {
            return null;
        }
        let value = BigInt('0x' + hex);
        const blockers = [];
        for (let i = 0; i < 7; i++) {
            const blocker = Number(value % 48n) + 1;  // Convert back to 1-indexed
            if (blocker < 1 || blocker > 48) return null;
            blockers.push(blocker);
            value = value / 48n;
        }
        return blockers;
    } catch (e) {
        console.warn('Invalid board code:', hex, e);
        return null;
    }
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

// === Theme Switching ===
const THEMES = {
    dark: {
        css: 'style.css',
        svgBg: '#0d1117',
        cellFill: '#1b2856',
        cellStroke: '#1a1a2e',
        blockerColor: '#90a3ffff'
    },
    pastel: {
        css: 'style-pastel.css',
        svgBg: 'transparent',
        cellFill: '#a8d4f0',
        cellStroke: '#7fb8dc',
        blockerColor: '#FFFFFF',
        bgImage: 'bg_pastel.jpg'
    }
};

// Expose THEMES on StarGenius namespace for board.js to use
if (window.StarGenius) {
    window.StarGenius.THEMES = THEMES;
}
// Preload theme background images
function preloadThemeImages() {
    Object.values(THEMES).forEach(theme => {
        if (theme.bgImage) {
            const img = new Image();
            img.src = '/static/' + theme.bgImage;
        }
    });
}
preloadThemeImages();

function setTheme(themeName) {
    const theme = THEMES[themeName];
    if (!theme) return;

    // Set data-theme attribute on html element for CSS variable switching
    document.documentElement.setAttribute('data-theme', themeName === 'dark' ? '' : themeName);

    // Also set on body for some CSS rules
    document.body.setAttribute('data-theme', themeName === 'dark' ? '' : themeName);

    // Update SVG background if board exists
    const svgBg = document.querySelector('#board-container svg rect:first-child');
    if (svgBg) {
        svgBg.setAttribute('fill', theme.svgBg);
    }

    // Update board cell colors - only update empty cells (using data-state attribute)
    const cells = document.querySelectorAll('#board-container svg polygon[data-cell-id]');
    cells.forEach(cell => {
        const state = cell.getAttribute('data-state');
        if (state === 'empty') {
            cell.setAttribute('fill', theme.cellFill);
            cell.setAttribute('stroke', theme.cellStroke);
        } else if (state === 'blocker') {
            cell.setAttribute('fill', theme.blockerColor);
        }
    });

    // Save preference
    localStorage.setItem('sg-theme', themeName);

    // Update dropdown
    const select = document.getElementById('theme-select');
    if (select) select.value = themeName;
}

function loadSavedTheme() {
    const saved = localStorage.getItem('sg-theme') || 'dark';
    setTheme(saved);
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    debugLog('DOMContentLoaded fired');

    try {
        // Access StarGenius namespace after all scripts loaded
        const SG = window.StarGenius;

        if (!SG) {
            throw new Error('StarGenius namespace not found');
        }
        debugLog('SG namespace OK');

        // Initialize translations
        await i18n.init();
        debugLog('i18n OK');

        initializeGame(SG);
        debugLog('Game OK');

        setupEventListeners(SG);
        debugLog('Ready! API: ' + API_BASE_URL);

        // Load saved theme
        loadSavedTheme();

        // Theme switcher event
        const themeSelect = document.getElementById('theme-select');
        if (themeSelect) {
            themeSelect.addEventListener('change', (e) => {
                setTheme(e.target.value);
            });
        }

        // Check for board config in URL
        const urlBlockers = getBlockersFromURL();
        if (urlBlockers && urlBlockers.length === 7) {
            debugLog('Starting from URL...');
            startNewGame(urlBlockers);
        }
    } catch (e) {
        debugLog('ERROR: ' + e.message);
        console.error('[main.js] Initialization error:', e);
    }
});

function initializeGame(SG) {
    // Create board
    const svgElement = document.getElementById('board-svg');
    board = new SG.Board(svgElement);
    game = new SG.Game(board);

    // Connect win callback
    game.onWinCallback = handleGameWin;

    // Update player name display
    updatePlayerNameDisplay();
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
    document.getElementById('btn-random').addEventListener('click', async () => {
        debugLog('Random clicked...');

        try {
            // Try unsolved boards first, then fall back to random
            debugLog('Fetching boards...');
            const unsolved = await fetchUnsolvedBoards();
            debugLog('Got ' + unsolved.length + ' boards');

            // Try to find a valid board
            for (const board of unsolved) {
                const blockers = decodeBlockers(board.board_code);
                if (blockers) {
                    debugLog('Starting game...');
                    startNewGame(blockers);
                    return;
                }
            }
        } catch (e) {
            debugLog('API Error: ' + e.message);
            console.error('[btn-random] Error:', e);
        }

        // No valid boards found or API failed - generate truly random game
        debugLog('Using random dice...');
        const blockers = SG.rollDice();
        startNewGame(blockers);
    });

    document.getElementById('btn-choose-dice').addEventListener('click', () => {
        showDiceModal(SG);
    });

    // Photo button - use camera on mobile if supported, file picker otherwise
    document.getElementById('btn-photo').addEventListener('click', async () => {
        if (window.StarGeniusCamera && StarGeniusCamera.isMobile() && StarGeniusCamera.isSupported()) {
            // Mobile with camera support - open camera modal
            await StarGeniusCamera.open();
        } else {
            // Desktop or no camera support - use file picker
            document.getElementById('photo-input').click();
        }
    });

    document.getElementById('photo-input').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            await loadFromPhoto(file);
        }
    });

    // Camera modal handlers
    document.getElementById('btn-camera-close').addEventListener('click', () => {
        StarGeniusCamera.close();
    });

    document.getElementById('btn-camera-capture').addEventListener('click', async () => {
        const file = await StarGeniusCamera.captureAsFile();
        if (file) {
            StarGeniusCamera.close();
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

    // Mobile controls - prevent interference with dragging
    // Using both touchstart (for responsiveness) and click (for fallback)
    const btnRotate = document.getElementById('btn-mobile-rotate');
    const btnFlip = document.getElementById('btn-mobile-flip');

    // Rotate button - handle all touch events to prevent interference
    btnRotate.addEventListener('touchstart', (e) => {
        e.preventDefault();
        e.stopPropagation();
        debugLog('Rotate touchstart');
        game.rotate();
    }, { passive: false });

    btnRotate.addEventListener('touchend', (e) => {
        e.preventDefault();
        e.stopPropagation();
        debugLog('Rotate touchend (blocked)');
    }, { passive: false });

    btnRotate.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.pointerType !== 'touch') {
            game.rotate();
        }
    });

    // Flip button - handle all touch events to prevent interference
    btnFlip.addEventListener('touchstart', (e) => {
        e.preventDefault();
        e.stopPropagation();
        debugLog('Flip touchstart');
        game.flip();
    }, { passive: false });

    btnFlip.addEventListener('touchend', (e) => {
        e.preventDefault();
        e.stopPropagation();
        debugLog('Flip touchend (blocked)');
    }, { passive: false });

    btnFlip.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.pointerType !== 'touch') {
            game.flip();
        }
    });

    // Mobile Controls
    const btnMobileHelp = document.getElementById('btn-mobile-help');
    if (btnMobileHelp) {
        // Handle touch events to prevent default scrolling if needed, or just standard click
        btnMobileHelp.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            // Trigger hint mode (true)
            await solveCurrentPuzzle(true);
        });
        // Prevent double taps zooming etc
        btnMobileHelp.addEventListener('touchstart', (e) => {
            e.stopPropagation();
        }, { passive: true });
    }

    // Burger Menu Logic
    const menuBtn = document.getElementById('btn-menu');
    const closeMenuBtn = document.getElementById('btn-close-menu');
    const controlsPanel = document.getElementById('controls');

    function closeMobileMenu() {
        if (controlsPanel && controlsPanel.classList.contains('active')) {
            controlsPanel.classList.remove('active');
            controlsPanel.classList.add('hidden-mobile');
        }
    }

    if (menuBtn && controlsPanel) {
        menuBtn.addEventListener('click', () => {
            controlsPanel.classList.add('active');
            controlsPanel.classList.remove('hidden-mobile'); // Helper to ensure display: block
        });
    }

    if (closeMenuBtn && controlsPanel) {
        closeMenuBtn.addEventListener('click', () => {
            closeMobileMenu();
        });
    }

    // Helper wrapper for button actions that should close the menu
    function withMenuAutoClose(fn) {
        return async (...args) => {
            closeMobileMenu();
            if (fn) await fn(...args);
        };
    }

    // Game controls with auto-close
    document.getElementById('btn-new-game').addEventListener('click', withMenuAutoClose(() => {
        const blockers = SG.rollDice();
        startNewGame(blockers);
    }));

    document.getElementById('btn-choose-dice-game').addEventListener('click', withMenuAutoClose(() => {
        showDiceModalFromGame(SG);
    }));

    document.getElementById('btn-reset').addEventListener('click', withMenuAutoClose(() => {
        game.resetToBlockers();
    }));

    document.getElementById('btn-solve').addEventListener('click', withMenuAutoClose(async () => {
        await solveCurrentPuzzle(true);  // Solve from current position
    }));

    document.getElementById('btn-solve-full').addEventListener('click', withMenuAutoClose(async () => {
        await solveCurrentPuzzle(false, true);  // Solve from scratch, reset before applying
    }));
    // Leaderboard Modal Logic
    function showLeaderboardModal() {
        // Reuse existing Win Modal logic to populate table
        const boardCode = getCurrentBoardCode();
        fetchLeaderboard(boardCode).then(data => {
            const container = document.getElementById('leaderboard-table-content');
            // Reuse populateLeaderboard but target different container
            // We need to modify populateLeaderboard to accept container or create a new function
            // Let's create a helper since populateLeaderboard targets 'win-leaderboard' ID specifically
            if (!data || !data.leaderboard || data.leaderboard.length === 0) {
                container.innerHTML = '<p class="no-scores">No scores yet!</p>';
                return;
            }

            let html = `<table>
                <thead><tr><th>#</th><th>Player</th><th>Hints</th><th>Time</th></tr></thead>
                <tbody>`;

            for (const entry of data.leaderboard) {
                const rankClass = entry.rank <= 3 ? `rank-${entry.rank}` : '';
                html += `<tr class="${rankClass}">
                    <td>${entry.rank}</td>
                    <td>${entry.player}</td>
                    <td>${entry.hints_used}</td>
                    <td>${formatTime(entry.time)}</td>
                </tr>`;
            }

            html += '</tbody></table>';
            container.innerHTML = html;
        });
        document.getElementById('leaderboard-modal').classList.remove('hidden');
    }

    document.getElementById('btn-show-leaderboard').addEventListener('click', withMenuAutoClose(() => {
        showLeaderboardModal();
    }));

    document.getElementById('btn-close-leaderboard').addEventListener('click', () => {
        document.getElementById('leaderboard-modal').classList.add('hidden');
    });

    // Win modal handlers
    document.getElementById('btn-close-win').addEventListener('click', hideWinModal);

    document.getElementById('btn-next-game').addEventListener('click', async () => {
        hideWinModal();
        try {
            // Try to get an unsolved board, otherwise random
            const unsolved = await fetchUnsolvedBoards();

            // Try to find a valid board
            for (const board of unsolved) {
                const blockers = decodeBlockers(board.board_code);
                if (blockers) {
                    startNewGame(blockers);
                    return;
                }
            }
        } catch (e) {
            console.error('Failed to fetch unsolved boards, falling back to random:', e);
        }

        // No valid boards or API failed - truly random game
        const blockers = SG.rollDice();
        startNewGame(blockers);
    });

    // Save name when input changes in win modal - also update the submitted score
    document.getElementById('win-name-input').addEventListener('change', async (e) => {
        const newName = e.target.value;
        setPlayerName(newName);

        // Update the recently submitted score with new name
        if (currentScoreId) {
            await updateScoreName(currentScoreId, newName);
            // Refresh leaderboard to show updated name
            const boardCode = getCurrentBoardCode();
            const leaderboardData = await fetchLeaderboard(boardCode);
            // Find current time from displayed time
            const timeText = document.getElementById('win-time').textContent;
            const [mins, secs] = timeText.split(':').map(Number);
            populateLeaderboard(leaderboardData, mins * 60 + secs);
        }
    });

    // Player name click to edit
    document.getElementById('player-name').addEventListener('click', () => {
        const newName = prompt('Enter your name:', playerName);
        if (newName !== null) {
            setPlayerName(newName);
        }
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

    // Display best time for this board
    const boardCode = encodeBlockers(blockers);
    updateBestTimeDisplay(boardCode);
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
            if (resetBeforeApply) {
                // Efficiently clear pieces for solve (doesn't re-render board cells)
                game.clearPiecesForSolve();
                game.hintsUsed = 11; // Full solve penalty set BEFORE applying to catch Win callback
                game.applySolution(result.solution, result.orientations, result.anchors);
                if (statusEl) {
                    statusEl.style.color = '';
                    statusEl.textContent = '✓ Solution applied';
                }
            } else {
                // Help Me behavior (One Hint)
                const applied = game.applySinglePieceHint(result.solution, result.orientations, result.anchors);
                if (applied) {
                    if (statusEl) {
                        statusEl.style.color = '';
                        statusEl.textContent = '✓ Hint applied';
                        setTimeout(() => {
                            statusEl.textContent = 'Drag pieces or use Hint';
                        }, 2000);
                    }
                } else {
                    // Solved or no hint possible
                    showToast("Puzzle already solved!", "success");
                    if (statusEl) statusEl.textContent = 'Puzzle already solved!';
                }
            }
        } else {
            // Friendly message
            showToast("No solution found from this position!", "error");
            if (statusEl) {
                statusEl.textContent = '❌ No solution found';
                statusEl.style.color = '#ff6b6b';
                setTimeout(() => {
                    statusEl.style.color = '';
                    statusEl.textContent = 'Drag pieces onto the board';
                }, 3000);
            }
        }
    } catch (e) {
        console.error('Solver error:', e);
        if (statusEl) {
            statusEl.textContent = '❌ Solver error';
            statusEl.style.color = '#ff6b6b';
        }
        showToast("Solver error: " + e.message, "error");
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

function showToast(message, type = 'info') {
    let toast = document.getElementById('toast-notification');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'toast-notification';
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.className = `toast toast-${type} show`;
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}


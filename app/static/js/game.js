/**
 * Star Genius - Game Logic
 * 
 * Handles game state, piece selection, rotation, and interaction.
 */

// Access StarGenius namespace
const SG_GAME = window.StarGenius;

class Game {
    constructor(board) {
        this.board = board;
        this.placedPieces = new Map();  // name -> orientation index
        this.selectedPiece = null;      // Currently selected piece name
        this.currentOrientation = 0;    // Index into orientations array
        this.blockers = [];             // Cell IDs of blockers
        this.lastMousePos = null;        // Last mouse position {x, y, cell}
        this.lastMousePos = null;        // Last mouse position {x, y, cell}
        this._pendingMouseMove = null;   // Throttle flag for mousemove

        this.timerInterval = null;
        this.startTime = null;

        this._setupEventListeners();
    }

    _setupEventListeners() {
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'r' || e.key === 'R') {
                this.rotate();
            } else if (e.key === 'f' || e.key === 'F') {
                this.flip();
            } else if (e.key === 'Escape') {
                this.deselect();
            }
        });

        // Cell click handling
        this.board.svg.addEventListener('click', (e) => {
            const cell = e.target.closest('.board-cell');
            if (cell) {
                this._onCellClick(cell);
            }
        });

        // Right-click to remove
        this.board.svg.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            const cell = e.target.closest('.board-cell');
            if (cell) {
                this._onCellRightClick(cell);
            }
        });

        // Hover for floating piece preview
        this.board.svg.addEventListener('mousemove', (e) => {
            this._onMouseMove(e);
        });

        this.board.svg.addEventListener('mouseleave', () => {
            this.board.clearGhost();
        });

        // Mouse wheel for rotation
        this.board.svg.addEventListener('wheel', (e) => {
            if (this.selectedPiece) {
                e.preventDefault();
                const direction = e.deltaY > 0 ? 1 : -1;
                this.rotate(direction);
            }
        }, { passive: false });
    }

    // Initialize game with blockers
    startGame(blockerIds, resetTimer = true) {
        this.board.reset();
        this.placedPieces.clear();
        this.selectedPiece = null;
        this.currentOrientation = 0;
        this.blockers = blockerIds;

        // Place blockers
        for (const id of blockerIds) {
            this.board.placeBlocker(id);
        }

        this._updateUI();
        this._setStatus('Select a piece from the palette');

        this._startTimer(resetTimer);
    }

    // Select a piece
    selectPiece(pieceName, orientation = 0) {
        if (this.placedPieces.has(pieceName)) {
            return;  // Can't select placed piece
        }

        this.selectedPiece = pieceName;
        this.currentOrientation = orientation;
        this._updateUI();
        this._setStatus(`Selected ${pieceName} - Click a cell to place, R to rotate`);
    }

    // Deselect current piece
    deselect() {
        this.selectedPiece = null;
        this.board.clearGhost();
        this._updateUI();
        this._setStatus('Select a piece from the palette');
    }

    // Rotate selected piece
    rotate(direction = 1) {
        if (!this.selectedPiece) return;

        const orientations = SG_GAME.PIECE_ORIENTATIONS[this.selectedPiece];
        const len = orientations.length;
        this.currentOrientation = (this.currentOrientation + direction + len) % len;
        this._setStatus(`Rotated ${this.selectedPiece} (${this.currentOrientation + 1}/${len})`);
        this._updateFloatingPreview();
    }

    // Flip selected piece (cycle through orientations that are flipped)
    flip() {
        if (!this.selectedPiece) return;

        const piece = SG_GAME.PIECE_DEFINITIONS[this.selectedPiece];
        if (!piece.canFlip) {
            this._setStatus(`${this.selectedPiece} cannot be flipped`);
            return;
        }

        // Skip to next orientation (simple approach)
        this.rotate();
    }

    // Get current piece in selected orientation
    getCurrentPiece() {
        if (!this.selectedPiece) return null;
        return SG_GAME.PIECE_ORIENTATIONS[this.selectedPiece][this.currentOrientation];
    }

    // Handle cell click
    _onCellClick(cellElement) {
        const t0 = performance.now();
        console.log(`[${t0.toFixed(1)}ms] _onCellClick START`);

        // Cancel any pending mousemove to avoid interference
        if (this._pendingMouseMove) {
            cancelAnimationFrame(this._pendingMouseMove);
            this._pendingMouseMove = null;
        }

        const posKey = cellElement.getAttribute('data-pos-key');
        const pos = SG_GAME.TrianglePos.fromKey(posKey);

        // Check if cell is occupied
        const occupant = this.board.occupied.get(posKey);

        // Handle clicking on existing items
        if (occupant !== undefined) {
            if (occupant === null) {
                this._setStatus('Cannot remove blockers');
                return;
            }

            // Pick up the piece
            const orientation = this.placedPieces.get(occupant) || 0;
            this.board.removePiece(occupant);
            this.placedPieces.delete(occupant);
            this.selectPiece(occupant, orientation);
            this._setStatus(`Picked up ${occupant}`);
            this._updateUI();
            return;
        }

        const piece = this.getCurrentPiece();
        if (!piece) {
            this._setStatus('Select a piece first');
            return;
        }

        const { valid, positions } = this.board.canPlace(piece, pos);
        console.log(`[${performance.now().toFixed(1)}ms] canPlace done, valid=${valid}`);

        if (!valid) {
            this._setStatus('Cannot place here - try another cell');
            return;
        }

        // Place the piece
        console.log(`[${performance.now().toFixed(1)}ms] placePiece START`);
        this.board.placePiece(piece.name, positions);
        console.log(`[${performance.now().toFixed(1)}ms] placePiece END`);

        this.placedPieces.set(piece.name, this.currentOrientation);

        console.log(`[${performance.now().toFixed(1)}ms] clearGhost START`);
        this.board.clearGhost();
        console.log(`[${performance.now().toFixed(1)}ms] clearGhost END`);

        // Check win
        if (this.board.isSolved()) {
            this._onWin();
        } else {
            console.log(`[${performance.now().toFixed(1)}ms] deselect START`);
            this.deselect();
            console.log(`[${performance.now().toFixed(1)}ms] deselect END`);
        }

        console.log(`[${performance.now().toFixed(1)}ms] updateUI START`);
        this._updateUI();
        console.log(`[${performance.now().toFixed(1)}ms] _onCellClick END, total=${(performance.now() - t0).toFixed(1)}ms`);

        // Measure when browser actually paints
        requestAnimationFrame(() => {
            console.log(`[${performance.now().toFixed(1)}ms] RAF1 (next frame)`);
            requestAnimationFrame(() => {
                console.log(`[${performance.now().toFixed(1)}ms] RAF2 (paint complete)`);
            });
        });
    }

    // Handle right-click (remove piece)
    _onCellRightClick(cellElement) {
        const posKey = cellElement.getAttribute('data-pos-key');
        const occupant = this.board.occupied.get(posKey);

        // Can't remove blockers
        if (occupant === null) {
            this._setStatus('Cannot remove blockers');
            return;
        }

        if (occupant) {
            this.board.removePiece(occupant);
            this.placedPieces.delete(occupant);
            this._updateUI();
            this._setStatus(`Removed ${occupant}`);
        }
    }

    // Handle mouse movement - show floating piece preview (throttled with RAF)
    _onMouseMove(e) {
        // Throttle with requestAnimationFrame
        if (this._pendingMouseMove) return;

        this._pendingMouseMove = requestAnimationFrame(() => {
            this._pendingMouseMove = null;
            this._processMouseMove(e);
        });
    }

    // Actual mouse move processing
    _processMouseMove(e) {
        const piece = this.getCurrentPiece();

        // Get mouse position in SVG coordinates
        const svg = this.board.svg;
        const pt = svg.createSVGPoint();
        pt.x = e.clientX;
        pt.y = e.clientY;
        const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());

        // Check if we're over a cell to determine validity
        const cell = e.target.closest('.board-cell');
        let isValid = null;
        let cellPos = null;

        if (cell) {
            const posKey = cell.getAttribute('data-pos-key');
            cellPos = SG_GAME.TrianglePos.fromKey(posKey);
            if (piece) {
                const { valid } = this.board.canPlace(piece, cellPos);
                isValid = valid;
            }
        }

        // Save position for use by rotate/flip
        this.lastMousePos = { x: svgP.x, y: svgP.y, cellPos, isValid };

        if (!piece) {
            this.board.clearGhost();
            return;
        }

        // Show floating preview at cursor
        this.board.showFloatingPreview(piece, svgP.x, svgP.y, isValid);
    }

    // Update floating preview (called after rotation)
    _updateFloatingPreview() {
        if (!this.lastMousePos) return;

        const piece = this.getCurrentPiece();
        if (!piece) {
            this.board.clearGhost();
            return;
        }

        // Recalculate validity with new orientation
        let isValid = null;
        if (this.lastMousePos.cellPos) {
            const { valid } = this.board.canPlace(piece, this.lastMousePos.cellPos);
            isValid = valid;
        }

        this.board.showFloatingPreview(piece, this.lastMousePos.x, this.lastMousePos.y, isValid);
    }

    // Update UI (palette, buttons)
    _updateUI() {
        // Update palette items
        document.querySelectorAll('.piece-item').forEach(item => {
            const name = item.dataset.piece;
            item.classList.toggle('selected', name === this.selectedPiece);
            item.classList.toggle('placed', this.placedPieces.has(name));
        });

        // Update flip button state
        const flipBtn = document.getElementById('btn-flip');
        if (flipBtn && this.selectedPiece) {
            const piece = SG_GAME.PIECE_DEFINITIONS[this.selectedPiece];
            flipBtn.disabled = !piece.canFlip;
        }
    }

    _setStatus(text) {
        const statusEl = document.getElementById('status-text');
        if (statusEl) {
            statusEl.textContent = text;
        }
    }

    _onWin() {
        this._stopTimer();
        this._setStatus('🎉 Puzzle Solved! 🎉');
        this.board.svg.classList.add('win-animation');
        setTimeout(() => {
            this.board.svg.classList.remove('win-animation');
        }, 1500);
    }

    _startTimer(reset = true) {
        this._stopTimer();
        if (reset || !this.startTime) {
            this.startTime = Date.now();
        }
        this._updateTimerDisplay();
        this.timerInterval = setInterval(() => {
            this._updateTimerDisplay();
        }, 1000);
    }

    _stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    _updateTimerDisplay() {
        const timerEl = document.getElementById('timer');
        if (!timerEl) return;

        const delta = Math.floor((Date.now() - this.startTime) / 1000);
        const mins = Math.floor(delta / 60).toString().padStart(2, '0');
        const secs = (delta % 60).toString().padStart(2, '0');
        timerEl.textContent = `${mins}:${secs}`;
    }

    // Reset to just blockers
    resetToBlockers() {
        this.startGame(this.blockers, false);
    }

    // Apply solution from solver
    applySolution(solution) {
        // solution: { pieceName: [cellIds] }
        for (const [pieceName, cellIds] of Object.entries(solution)) {
            const positions = cellIds.map(id => this.board.getCellById(id));
            this.board.placePiece(pieceName, positions);
            this.placedPieces.set(pieceName, 0); // Default orientation
        }
        this._updateUI();

        if (this.board.isSolved()) {
            this._onWin();
        }
    }
}

// Export
window.StarGenius.Game = Game;


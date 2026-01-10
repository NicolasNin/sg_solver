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
        this.lastMousePos = null;        // Last mouse position {x, y} relative to SVG
        this._pendingMouseMove = null;   // Throttle flag for mousemove

        // DnD State
        this.isDragging = false;
        this.draggedPieceName = null;
        this.dragOffset = { x: 0, y: 0 }; // Offset from mouse to piece anchor

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
            } else if (e.key === 't' || e.key === 'T') {
                // DEBUG: Emit custom event for test-solve
                console.log('[Game] T pressed - dispatching test-solve event');
                document.dispatchEvent(new CustomEvent('star-genius-test-solve'));
            }
        });

        // Cell click handling - DISABLED for DnD to prevent conflicts
        // this.board.svg.addEventListener('click', (e) => {
        //     const cell = e.target.closest('.board-cell');
        //     if (cell) {
        //         this._onCellClick(cell);
        //     }
        // });
        // Right-click to flip piece or remove from cell
        this.board.svg.addEventListener('contextmenu', (e) => {
            e.preventDefault();

            // Check if clicking on a piece - flip it
            const pieceGroup = e.target.closest('.piece-group');
            if (pieceGroup) {
                const pieceName = pieceGroup.dataset.piece;
                // Select the piece if not already selected
                if (this.selectedPiece !== pieceName) {
                    const rotation = parseInt(pieceGroup.dataset.rotation || 0);
                    this.selectedPiece = pieceName;
                    this.currentOrientation = rotation;
                    this._updateUI();
                }
                this.flip();
                return;
            }

            // Otherwise, check if clicking on a cell
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
            if (this.selectedPiece || this.isDragging) {
                e.preventDefault();
                const direction = e.deltaY > 0 ? 1 : -1;
                this.rotate(direction);
            }
        }, { passive: false });

        // Drag and Drop Events
        this.board.svg.addEventListener('mousedown', (e) => {
            this._onMouseDown(e);
        });

        this.board.svg.addEventListener('mouseup', (e) => {
            if (e.button !== 0) return;  // Ignore right-clicks
            this._onMouseUp(e);
        });

        // Global mouse up to catch drops outside SVG
        document.addEventListener('mouseup', (e) => {
            if (e.button !== 0) return;  // Ignore right-clicks
            if (this.isDragging) {
                this._cancelDrag();
            }
        });
    }

    // Initialize game with blockers
    startGame(blockerIds, resetTimer = true, skipPieceRender = false) {
        this.board.reset();
        this.placedPieces.clear();
        this.selectedPiece = null;
        this.currentOrientation = 0;
        this.blockers = blockerIds;

        // Place blockers
        for (const id of blockerIds) {
            this.board.placeBlocker(id);
        }

        // Skip rendering pieces if we're about to apply a solution
        if (!skipPieceRender) {
            // Initialize pieces on the sidelines
            // Initialize pieces with 3-zone layout (Left, Right, Bottom)
            const INITIAL_POSITIONS = {
                // Left Zone - moved right a bit
                "4D": { x: 150, y: 100 },    // Brown
                "T4": { x: 80, y: 300 },    // Pink
                "TF": { x: 80, y: 480 },    // Purple
                "T3": { x: 80, y: 600 },    // Cyan
                "T1": { x: 100, y: 700 },   // Blue

                // Right Zone - more vertical spacing
                "EL": { x: 1000, y: 100 },   // Green
                "L4": { x: 1000, y: 280 },   // Orange
                "3B": { x: 1000, y: 480 },   // Cyan pair
                "T2": { x: 950, y: 600 },   // Yellow

                // Bottom Zone
                "T5": { x: 350, y: 700 },    // Red
                "4U": { x: 700, y: 700 },    // Lime
            };

            const allPieces = SG_GAME.ALL_PIECES; // Ensure we iterate all known pieces
            allPieces.forEach(name => {
                const piece = SG_GAME.PIECE_DEFINITIONS[name];
                const pos = INITIAL_POSITIONS[name] || { x: 0, y: 0 }; // Fallback
                this.board.renderPiece(piece, pos.x, pos.y, 0);
            });
        }

        this._updateUI();
        this._setStatus('Drag pieces onto the board');

        this._startTimer(resetTimer);
    }

    // Select a piece
    selectPiece(pieceName, orientation = 0) {
        if (this.placedPieces.has(pieceName)) {
            return;  // Can't select placed piece from palette if already placed
        }

        this.selectedPiece = pieceName;
        this.currentOrientation = orientation;

        // Spawn the piece visually at mouse position (if known) or center
        const x = this.lastMousePos ? this.lastMousePos.x : 200;
        const y = this.lastMousePos ? this.lastMousePos.y : 200;

        // Render it
        this.currentOrientation = orientation;
        const piece = this.getCurrentPiece();
        const group = this.board.renderPiece(piece, x, y, orientation);

        // If we just clicked palette, start dragging immediately?
        // User wants "snap when put on right position", implying drag behavior.
        // For palette clicks, we usually just "select" then need to click board to place (old way).
        // Best DnD UX: Click palette -> spawns attached to mouse.

        // We set dragging true, so `_onMouseMove` will move it
        this.isDragging = true;
        this.draggedPieceName = pieceName;
        this.dragOffset = { x: 0, y: 0 }; // Centered on mouse initially? Or Anchor?
        // Let's assume anchor is fine.

        this._updateUI();
        this._setStatus(`Moving ${pieceName} - R to rotate`);
    }

    // Deselect current piece
    deselect() {
        // Legacy: "If we deselect, we remove the visual element if it wasn't placed"
        // NEW: Never remove visuals. Pieces are permanent.
        // if (this.selectedPiece && !this.placedPieces.has(this.selectedPiece)) {
        //     this.board.removePieceElement(this.selectedPiece);
        // }

        this.selectedPiece = null;
        this.isDragging = false;
        this.draggedPieceName = null;
        this._updateUI();
        this._setStatus('Select a piece from the palette');
    }

    // Rotate selected piece (cycles within current side: 0-5 or 6-11)
    rotate(direction = 1) {
        if (!this.selectedPiece) return;

        // Use ordered orientations (0-5 = normal, 6-11 = flipped)
        const isFlipped = this.currentOrientation >= 6;
        const baseIndex = isFlipped ? 6 : 0;
        const rotationWithinSide = this.currentOrientation % 6;

        // Cycle within the 6 rotations of current side
        const newRotation = (rotationWithinSide + direction + 6) % 6;
        this.currentOrientation = baseIndex + newRotation;

        this._setStatus(`Rotated ${this.selectedPiece} (rot ${newRotation}${isFlipped ? ' flipped' : ''})`);

        // Re-render
        this._updateFloatingPreview();
    }

    // Flip selected piece (toggles between 0-5 and 6-11)
    flip() {
        if (!this.selectedPiece) return;

        const piece = SG_GAME.PIECE_DEFINITIONS[this.selectedPiece];
        if (!piece.canFlip) {
            this._setStatus(`${this.selectedPiece} cannot be flipped`);
            return;
        }

        // Toggle between normal (0-5) and flipped (6-11), keeping same rotation
        const rotationWithinSide = this.currentOrientation % 6;
        if (this.currentOrientation < 6) {
            this.currentOrientation = 6 + rotationWithinSide;  // Flip to 6-11
        } else {
            this.currentOrientation = rotationWithinSide;      // Flip back to 0-5
        }

        const isFlipped = this.currentOrientation >= 6;
        this._setStatus(`${this.selectedPiece} ${isFlipped ? 'flipped' : 'unflipped'}`);

        // Re-render
        this._updateFloatingPreview();
    }

    // Get current piece in selected orientation (using ordered orientations for UI)
    getCurrentPiece() {
        if (!this.selectedPiece) return null;
        return SG_GAME.PIECE_ORIENTATIONS_ORDERED[this.selectedPiece][this.currentOrientation];
    }

    // --- DnD Handlers ---

    _getMouseSVGPos(e) {
        const svg = this.board.svg;
        const pt = svg.createSVGPoint();
        pt.x = e.clientX;
        pt.y = e.clientY;
        return pt.matrixTransform(svg.getScreenCTM().inverse());
    }

    _onMouseDown(e) {
        // Ignore right-clicks - they're handled by contextmenu
        if (e.button !== 0) return;

        // Check if we clicked a piece group
        const group = e.target.closest('.piece-group');
        if (!group) {
            // If clicking empty space/cell, maybe deselect?
            // But DnD philosophy usually implies persistent selection.
            // We can keep current logic: clicking background deselects.
            if (e.target.closest('.board-cell')) {
                // Check if it's a blocker, otherwise deselect
                const posKey = e.target.closest('.board-cell').dataset.posKey;
                if (this.board.occupied.get(posKey) === null) {
                    this._setStatus('Cannot move blockers');
                } else {
                    this.deselect();
                }
            }
            return;
        }

        e.preventDefault(); // Prevent text selection

        const pieceName = group.dataset.piece;
        this.isDragging = true;
        this.draggedPieceName = pieceName;

        // Select it
        if (this.selectedPiece !== pieceName) {
            // Get current orientation
            const rotation = parseInt(group.dataset.rotation || 0);
            this.placedPieces.delete(pieceName); // Temporarily remove from logic to allow movement
            this.board.removePiece(pieceName, true); // Keep visual!

            this.selectedPiece = pieceName;
            this.currentOrientation = rotation;
            this._updateUI();
            this._setStatus(`Moving ${pieceName} - R to rotate`);
        } else {
            // Already selected
            const rotation = this.currentOrientation;
            this.placedPieces.delete(pieceName);
            this.board.removePiece(pieceName, true); // Keep visual!

            this.selectedPiece = pieceName;
            this.currentOrientation = rotation;
            this._updateUI();
        }

        // Calculate drag offset
        const mousePos = this._getMouseSVGPos(e);

        // Get piece current transform
        const transform = group.getAttribute('transform');
        let currentX = 0, currentY = 0;
        if (transform) {
            const match = transform.match(/translate\(([^,]+),\s*([^)]+)\)/);
            if (match) {
                currentX = parseFloat(match[1]);
                currentY = parseFloat(match[2]);
            }
        }

        this.dragOffset = {
            x: currentX - mousePos.x,
            y: currentY - mousePos.y
        };

        // Move to top visual layer
        this.board.pieceGroup.appendChild(group);
        group.style.cursor = 'grabbing';
    }

    _onMouseUp(e) {
        if (!this.isDragging) return;

        const pieceName = this.draggedPieceName;
        const group = this.board.pieceElements.get(pieceName);

        // Safety check - if drag was cancelled or group lost
        if (!group) {
            this.endDrag();
            return;
        }

        // Calculate drop position (centroid of piece anchor)
        // We know the anchor is at (0,0) relative to group.
        // So group position = anchor position.

        const transform = group.getAttribute('transform');
        let x = 0, y = 0;
        if (transform) {
            const match = transform.match(/translate\(([^,]+),\s*([^)]+)\)/);
            if (match) {
                x = parseFloat(match[1]);
                y = parseFloat(match[2]);
            }
        }

        // x, y is now the CENTROID position (since we use centroid-based rendering)
        const piece = this.getCurrentPiece();

        // Find nearest cell for the ANCHOR (not centroid)
        const cellPos = this.board.getNearestAnchorCell(piece, x, y);

        if (cellPos) {
            // Attempt placement (uses anchor cell for validation)
            const { valid, positions } = this.board.canPlace(piece, cellPos);

            if (valid) {
                // Success!
                this.board.placePiece(piece.name, positions);
                this.placedPieces.set(piece.name, this.currentOrientation);

                // Snap visual to correct position (centroid-based)
                const snapPos = this.board.getSnapPositionForPiece(piece, cellPos);
                if (snapPos) {
                    this.board.updatePiecePosition(piece.name, snapPos[0], snapPos[1]);
                }

                this._setStatus(`Placed ${piece.name}`);

                // Check if puzzle is solved
                if (this.board.isSolved()) {
                    this._onWin();
                }
            } else {
                // Invalid placement on board - just drop it where the mouse is (or where it was dragged)
                // If it was valid, we'd snap. If invalid, we leave it 'loose' on the board/side.

                // Remove from LOGIC grid if it was there
                if (this.placedPieces.has(pieceName)) {
                    this.placedPieces.delete(pieceName);
                    this.board.removePiece(pieceName, true); // remove from logic, keep visual
                }

                this._setStatus('Placed outside grid');
                this.deselect();
                // Do NOT call removePieceElement(pieceName)
            }
        } else {
            // Dropped outside grid (sidelines)
            // Just update visual position (already done by drag) and ensure not in logic grid
            if (this.placedPieces.has(pieceName)) {
                this.placedPieces.delete(pieceName);
                this.board.removePiece(pieceName, true);
            }

            // CRITICAL fix: ensure we don't accidentally remove visualization
            this.selectedPiece = null; // Just deselect, don't remove
            this.isDragging = false;
            this.draggedPieceName = null;
            this._updateUI();
            this._setStatus('Placed on sideline');
        }

        this.endDrag();
    }

    _cancelDrag() {
        if (!this.isDragging) return;
        // If cancelled (e.g. mouse up outside window/SVG), just leave it where it is
        const pieceName = this.draggedPieceName;
        this.deselect();
        // this.board.removePieceElement(pieceName); // CRITICAL FIX: Do NOT remove visuals!
        this.endDrag();
    }

    endDrag() {
        this.isDragging = false;
        this.draggedPieceName = null;
        // Reset cursors
        if (this.selectedPiece) {
            const g = this.board.pieceElements.get(this.selectedPiece);
            if (g) g.style.cursor = 'grab';
        }
    }

    // Handle mouse movement
    _onMouseMove(e) {
        const mousePos = this._getMouseSVGPos(e);
        this.lastMousePos = mousePos; // for rotation

        if (this.isDragging && this.draggedPieceName) {
            const x = mousePos.x + this.dragOffset.x;
            const y = mousePos.y + this.dragOffset.y;

            this.board.updatePiecePosition(this.draggedPieceName, x, y);
        }
    }

    // Actual mouse move processing (Legacy/Unused for Drag logic, but kept empty/cleaned)
    _processMouseMove(e) {
        // No longer needed for ghost preview as we drag real pieces
        // We might want to highlight the cell under the cursor though?
        // skipping for now to keep it clean DnD.
    }

    // Update floating preview (Legacy - replaced by real piece rotation)
    _updateFloatingPreview() {
        // When we rotate, we need to re-render the SVG group of the dragged/selected piece
        if (!this.selectedPiece) return;

        const piece = this.getCurrentPiece(); // Gets rotated version

        // We need to keep the screen position of the piece
        let x = 0, y = 0;

        // If piece exists on board (visual), get its position
        let existingGroup = this.board.pieceElements.get(this.selectedPiece);

        /* 
         * CRITICAL: If we are selecting a new piece from palette, it might not have a group yet.
         * But 'rotate' assumes we have something to rotate.
         * If we just clicked palette, we haven't created the visual group on board yet until we drag/place? 
         * Wait, in Lichess style, picking from palette usually attaches it to mouse immediately.
         */

        // If we are dragging, we definitely have a group.
        if (existingGroup) {
            const transform = existingGroup.getAttribute('transform');
            if (transform) {
                const match = transform.match(/translate\(([^,]+),\s*([^)]+)\)/);
                if (match) {
                    x = parseFloat(match[1]);
                    y = parseFloat(match[2]);
                }
            }

            // Re-render with new rotation
            const group = this.board.renderPiece(piece, x, y, this.currentOrientation);

            if (this.isDragging) {
                group.style.cursor = 'grabbing';
            }
        }
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

    // DEBUG: Draw a visible marker at a position
    _drawDebugMarker(x, y, label = '') {
        const svg = this.board.svg;

        // Create a red circle
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x);
        circle.setAttribute('cy', y);
        circle.setAttribute('r', 8);
        circle.setAttribute('fill', 'red');
        circle.setAttribute('stroke', 'white');
        circle.setAttribute('stroke-width', 2);
        circle.setAttribute('class', 'debug-marker');

        // Create a label
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x + 12);
        text.setAttribute('y', y + 4);
        text.setAttribute('fill', 'red');
        text.setAttribute('font-size', '14');
        text.setAttribute('font-weight', 'bold');
        text.setAttribute('class', 'debug-marker');
        text.textContent = label;

        svg.appendChild(circle);
        svg.appendChild(text);
    }

    // Clear debug markers
    _clearDebugMarkers() {
        this.board.svg.querySelectorAll('.debug-marker').forEach(el => el.remove());
    }

    _setStatus(text) {
        const statusEl = document.getElementById('header-status');
        if (statusEl) {
            statusEl.textContent = text;
        }
    }

    _onWin() {
        this._stopTimer();
        const solveTime = (Date.now() - this.startTime) / 1000;
        this._setStatus('🎉 Puzzle Solved! 🎉');
        this.board.svg.classList.add('win-animation');
        setTimeout(() => {
            this.board.svg.classList.remove('win-animation');
        }, 1500);

        // Call win callback if set (for leaderboard)
        if (this.onWinCallback) {
            this.onWinCallback(solveTime, this.hintsUsed || 0);
        }
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
    resetToBlockers(skipPieceRender = false) {
        this.startGame(this.blockers, false, skipPieceRender);
    }

    // Efficiently clear pieces for solve (doesn't re-render board cells)
    clearPiecesForSolve() {
        this.board.clearAllPieces();
        this.placedPieces.clear();
        this.selectedPiece = null;
        this.currentOrientation = 0;
    }

    // Apply solution from solver
    applySolution(solution, orientations = {}, anchors = {}) {
        // solution: { pieceName: [cellIds] }
        // orientations: { pieceName: orientationIndex }
        // anchors: { pieceName: anchorCellId }

        for (const pieceName of Object.keys(solution)) {
            // 1. Get positions for logical placement
            const positions = solution[pieceName].map(id => this.board.getCellById(id));

            // 2. Place on logical board (updates occupied map)
            this.board.placePiece(pieceName, positions);

            // 3. Update game state to track orientation
            const orientationIdx = orientations[pieceName] || 0;
            this.placedPieces.set(pieceName, orientationIdx);

            // 4. Visual Rendering
            // We need to render the piece in the correct rotation!
            // Get the specific rotated piece object (precomputed in pieces.js)
            const orientedPiece = SG_GAME.PIECE_ORIENTATIONS[pieceName][orientationIdx];

            if (!orientedPiece) {
                console.error(`Missing orientation ${orientationIdx} for ${pieceName}`);
                continue;
            }

            // Calculate visual position using anchor cell (centroid-based snapping)
            // The anchor cell is where the piece's triangles[0] lands
            let anchorCellPos = positions[0]; // Fallback
            if (anchors[pieceName]) {
                anchorCellPos = this.board.getCellById(anchors[pieceName]);
            }

            // Get the correct snap position for centroid-based rendering
            const snapPos = this.board.getSnapPositionForPiece(orientedPiece, anchorCellPos);

            // DEBUG: Log the computed position
            console.log(`[applySolution] Piece: ${pieceName}`);
            console.log(`  - orientationIdx: ${orientationIdx}`);
            console.log(`  - positions (cell IDs):`, solution[pieceName]);
            console.log(`  - anchorCellPos:`, anchorCellPos ? anchorCellPos.key() : 'null');
            console.log(`  - snapPos (px):`, snapPos);

            if (snapPos) {
                this.board.renderPiece(orientedPiece, snapPos[0], snapPos[1], orientationIdx);
            }
        }
        this._updateUI();

        if (this.board.isSolved()) {
            this._onWin();
        }
    }
}

// Export
window.StarGenius.Game = Game;


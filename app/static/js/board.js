/**
 * Star Genius - Board Rendering
 * 
 * SVG rendering of the star-shaped board with 48 triangle cells.
 * Matches coordinate conversion from viz.py and conversion_utils.py.
 */

// Access StarGenius namespace (set by pieces.js)
const SG = window.StarGenius;

// Get theme-aware board colors from main.js THEMES config
function getThemeColors() {
    const themeName = document.documentElement.getAttribute('data-theme') || 'dark';
    const themes = window.StarGenius?.THEMES || {
        dark: { cellFill: '#1b2856', cellStroke: '#1a1a2e', svgBg: '#0d1117', blockerColor: '#32408a' },
        pastel: { cellFill: '#a8d4f0', cellStroke: '#7fb8dc', svgBg: 'transparent', blockerColor: '#FFFFFF' }
    };
    const theme = themes[themeName] || themes.dark;
    return {
        emptyColor: theme.cellFill,
        cellStroke: theme.cellStroke,
        svgBg: theme.svgBg,
        blockerColor: theme.blockerColor
    };
}

// Board configuration
const BOARD_CONFIG = {
    scale: 95,          // Pixels per unit
    margin: 10,         // SVG margin (reduced to save vertical space)
    cellStrokeWidth: 1,
    labelColor: '#666',
    labelSize: 24,      // Smaller labels (was 40)
};

// Star layout: [visualRow, startX, numCells]
// From board.py Board.create_star()
const STAR_LAYOUT = [
    [0, 5, 1],    // Top tip
    [1, 4, 3],
    [2, 0, 11],   // Wide row with left/right points
    [3, 1, 9],
    [4, 1, 9],
    [5, 0, 11],   // Wide row with left/right points
    [6, 4, 3],
    [7, 5, 1],    // Bottom tip
];

// Generate all board cells
function generateBoardCells() {
    const cells = new Map();  // key -> { pos, cellId }
    let cellId = 1;

    for (const [visualRow, startX, numCells] of STAR_LAYOUT) {
        const y = 7 - visualRow;  // Flip: visual row 0 -> y=7
        for (let i = 0; i < numCells; i++) {
            const x = startX + i;
            const pointsUp = (x + y) % 2 === 0;
            const pos = new SG.TrianglePos(x, y, pointsUp);
            cells.set(pos.key(), { pos, cellId });
            cellId++;
        }
    }

    return cells;
}

// Convert skewed vertex to SVG pixel coordinates
function vertexToPixel(vx, vy, bounds) {
    const { scale, margin, minVx, maxVy } = bounds;
    const h = scale * Math.sqrt(3) / 2;  // Height of equilateral triangle row

    // Pipeline: skewed -> normal -> svg (y-flipped)
    const nx = vx + vy * 0.5;
    const ny = vy;
    const svgY = maxVy - ny;  // Flip y for SVG

    // const svgY = maxVy - ny;  // Duplicate removed

    const px = bounds.marginX + (nx - minVx) * scale;
    const py = bounds.marginY + svgY * h;

    return [px, py];
}

// Calculate SVG bounds from all cells
function calculateBounds(cells) {
    const allVerts = [];
    for (const { pos } of cells.values()) {
        allVerts.push(...pos.toVertices());
    }

    const minVx = Math.min(...allVerts.map(v => v[0]));
    const maxVx = Math.max(...allVerts.map(v => v[0]));
    const minVy = Math.min(...allVerts.map(v => v[1]));
    const maxVy = Math.max(...allVerts.map(v => v[1]));

    const h = BOARD_CONFIG.scale * Math.sqrt(3) / 2;

    const boardWidth = (maxVx - minVx + (maxVy - minVy) * 0.5 + 1) * BOARD_CONFIG.scale;
    const width = boardWidth + BOARD_CONFIG.margin * 2;
    const boardHeight = (maxVy - minVy + 1) * h;
    const height = boardHeight + BOARD_CONFIG.margin * 2;

    return {
        minVx, maxVx, minVy, maxVy,
        width, height,
        originalHeight: height, // Store original height for reset
        scale: BOARD_CONFIG.scale,
        marginX: BOARD_CONFIG.margin,
        marginY: BOARD_CONFIG.margin,
    };
}

// Create SVG polygon for a triangle cell
function createCellPolygon(pos, cellId, bounds, state = 'empty') {
    const verts = pos.toVertices();
    const pixels = verts.map(([vx, vy]) => vertexToPixel(vx, vy, bounds));
    const pointsStr = pixels.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');

    // Determine color based on state
    const themeColors = getThemeColors();
    let fill = themeColors.emptyColor;
    if (state === 'blocker') {
        fill = themeColors.blockerColor;
    } else if (state && state !== 'empty') {
        fill = SG.PIECE_COLORS[state] || '#CCCCCC';
    }

    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    polygon.setAttribute('points', pointsStr);
    polygon.setAttribute('fill', fill);
    polygon.setAttribute('stroke', themeColors.cellStroke);
    polygon.setAttribute('stroke-width', BOARD_CONFIG.cellStrokeWidth);
    polygon.setAttribute('class', 'board-cell');
    polygon.setAttribute('data-cell-id', cellId);
    polygon.setAttribute('data-pos-key', pos.key());
    polygon.setAttribute('data-state', state || 'empty');  // Track cell state!

    return { polygon, center: getCentroid(pixels) };
}

// Get centroid of polygon
function getCentroid(points) {
    const cx = points.reduce((sum, p) => sum + p[0], 0) / points.length;
    const cy = points.reduce((sum, p) => sum + p[1], 0) / points.length;
    return [cx, cy];
}

// Create cell ID label
function createCellLabel(cellId, center) {
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', center[0].toFixed(1));
    text.setAttribute('y', center[1].toFixed(1));
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('dominant-baseline', 'middle');
    text.setAttribute('font-size', BOARD_CONFIG.labelSize);
    text.setAttribute('fill', BOARD_CONFIG.labelColor);
    text.setAttribute('pointer-events', 'none');
    text.textContent = cellId;
    return text;
}

// Board class
class Board {
    constructor(svgElement) {
        this.svg = svgElement;
        this.cells = generateBoardCells();
        this.bounds = calculateBounds(this.cells);
        this.occupied = new Map();  // posKey -> pieceName or null (blocker)
        this.cellElements = new Map();  // posKey -> polygon element
        this.labelElements = new Map();

        // New: Independent piece management
        this.pieceElements = new Map(); // pieceName -> SVGGroupElement

        this._setupSVG();
        this.render();
    }

    // Get the tight pixel bounds of the actual board content
    getTightBounds() {
        let minPx = Infinity, maxPx = -Infinity;
        let minPy = Infinity, maxPy = -Infinity;

        // Iterate all cells and project
        for (const { pos } of this.cells.values()) {
            const verts = pos.toVertices();
            for (const [vx, vy] of verts) {
                const [px, py] = vertexToPixel(vx, vy, this.bounds);
                if (px < minPx) minPx = px;
                if (px > maxPx) maxPx = px;
                if (py < minPy) minPy = py;
                if (py > maxPy) maxPy = py;
            }
        }
        return { minPx, maxPx, minPy, maxPy, width: maxPx - minPx, height: maxPy - minPy };
    }

    // Set ViewBox for Mobile: Crop to tight bounds + extend height for tray
    setMobileView(trayExtraHeight) {
        const tight = this.getTightBounds();
        // Add a little padding
        const padding = 10;
        const x = tight.minPx - padding;
        const y = tight.minPy - padding;
        const w = tight.width + padding * 2;
        // Extend height downwards
        const h = tight.height + padding * 2 + trayExtraHeight;

        this.svg.setAttribute('viewBox', `${x.toFixed(1)} ${y.toFixed(1)} ${w.toFixed(1)} ${h.toFixed(1)}`);

        // Return stats logic can use to place pieces
        return {
            x, y, w, h, tight
        };
    }

    resetViewBox() {
        // Original full viewbox
        this.svg.setAttribute('viewBox', `0 0 ${this.bounds.width.toFixed(0)} ${this.bounds.originalHeight.toFixed(0)}`);
    }

    _setupSVG() {
        this.svg.setAttribute('viewBox', `0 0 ${this.bounds.width.toFixed(0)} ${this.bounds.height.toFixed(0)}`);

        // Create background
        const themeColors = getThemeColors();
        const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bg.setAttribute('width', '100%');
        bg.setAttribute('height', '100%');
        bg.setAttribute('fill', themeColors.svgBg);
        this.svg.appendChild(bg);

        // Create groups for layering
        this.cellGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.labelGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.pieceGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g'); // New container for pieces
        this.ghostGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.ghostGroup.setAttribute('class', 'ghost-piece');

        this.svg.appendChild(this.cellGroup);
        this.svg.appendChild(this.labelGroup);
        this.svg.appendChild(this.pieceGroup); // Layer pieces above cells
        this.svg.appendChild(this.ghostGroup);
    }

    render() {
        // Clear existing
        this.cellGroup.innerHTML = '';
        this.labelGroup.innerHTML = '';
        this.cellElements.clear();
        this.labelElements.clear();

        // Create cells
        for (const [key, { pos, cellId }] of this.cells) {
            const state = this.occupied.get(key) ?? 'empty';
            const { polygon, center } = createCellPolygon(pos, cellId, this.bounds, state);

            this.cellGroup.appendChild(polygon);
            this.cellElements.set(key, polygon);

            // Only show labels for empty cells
            if (state === 'empty') {
                const label = createCellLabel(cellId, center);
                this.labelGroup.appendChild(label);
                this.labelElements.set(key, label);
            }
        }
    }

    // Place blocker by cell ID
    placeBlocker(cellId) {
        for (const [key, cell] of this.cells) {
            if (cell.cellId === cellId) {
                this.occupied.set(key, null);  // null = blocker
                this._updateCell(key, 'blocker');
                return true;
            }
        }
        return false;
    }

    /* 
     * NEW PIECE RENDERING API
     * Creates independent piece groups rather than coloring cells 
     */

    // Render a piece object at a specific visual coordinate (x, y = centroid position)
    // Render a piece object at a specific visual coordinate (x, y = centroid position)
    renderPiece(piece, x, y, rotationIndex = 0) {
        // Recycle existing group if present, otherwise create new
        let group = this.pieceElements.get(piece.name);

        if (!group) {
            group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            group.setAttribute('class', 'piece-group');
            group.setAttribute('data-piece', piece.name);
            group.style.cursor = 'grab';
            this.pieceGroup.appendChild(group);
            this.pieceElements.set(piece.name, group);
        }

        // Store metadata on the DOM element for easy access
        group.dataset.rotation = rotationIndex;

        const scale = BOARD_CONFIG.scale;
        const h = scale * Math.sqrt(3) / 2;
        const color = SG.PIECE_COLORS[piece.name] || '#CCCCCC';

        // Get piece centroid (in normal coords) - this is where (x,y) will be placed
        const pieceCentroid = piece.getCentroid();

        // Get existing polygons to recycle
        // Use Array.from to get a stable list as we might append/remove
        const existingPolygons = Array.from(group.children).filter(el => el.tagName === 'polygon');

        piece.triangles.forEach((tri, i) => {
            // Recycle or create polygon
            let polygon = existingPolygons[i];
            if (!polygon) {
                polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                group.appendChild(polygon);
            }

            const verts = tri.toVertices();
            const points = verts.map(([vx, vy]) => {
                const nx = vx + vy * 0.5;
                const ny = vy;

                // Position relative to piece centroid (not anchor)
                const px = (nx - pieceCentroid[0]) * scale;
                const py = -(ny - pieceCentroid[1]) * h; // SVG y flip

                return [px, py];
            });

            polygon.setAttribute('points', points.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '));
            polygon.setAttribute('fill', color);
            polygon.setAttribute('stroke', '#fff');
            polygon.setAttribute('stroke-width', '2');
        });

        // Remove extra polygons if any (though piece triangle count is usually constant)
        while (group.children.length > piece.triangles.length) {
            // Be careful not to remove other potential children if we add them later (like debug markers)
            // For now, group only contains polygons, so lastChild is fine
            if (group.lastChild.tagName === 'polygon') {
                group.lastChild.remove();
            } else {
                break; // Should not happen based on current logic
            }
        }

        this.updatePiecePosition(piece.name, x, y);
        return group;
    }

    updatePiecePosition(name, x, y) {
        const group = this.pieceElements.get(name);
        if (group) {
            group.setAttribute('transform', `translate(${x}, ${y})`);
        }
    }

    removePieceElement(name) {
        const group = this.pieceElements.get(name);
        if (group) {
            group.remove();
            this.pieceElements.delete(name);
        }
    }

    // Convert pixel coordinate back to nearest cell
    getNearestCell(x, y) {
        let nearest = null;
        let minDistSq = Infinity;

        for (const [key, { pos, cellId }] of this.cells) {
            const center = getCentroid(pos.toVertices().map(([vx, vy]) => vertexToPixel(vx, vy, this.bounds)));
            const dx = center[0] - x;
            const dy = center[1] - y;
            const distSq = dx * dx + dy * dy;

            if (distSq < minDistSq) {
                minDistSq = distSq;
                nearest = pos;
            }
        }

        // Threshold (e.g., must be within 30px)
        if (minDistSq > 30 * 30) return null;
        return nearest;
    }

    // Get pixel center of a cell
    getCellCenter(pos) {
        if (!pos) return null;
        const verts = pos.toVertices();
        const pixels = verts.map(([vx, vy]) => vertexToPixel(vx, vy, this.bounds));
        return getCentroid(pixels);
    }

    // Get the snap position for a piece (where the centroid should be placed)
    // given the anchor cell position. This accounts for the offset from anchor to centroid.
    getSnapPositionForPiece(piece, anchorCellPos) {
        if (!anchorCellPos) return null;

        // Get anchor cell center in pixels
        const anchorCellCenter = this.getCellCenter(anchorCellPos);
        if (!anchorCellCenter) return null;

        // Get the offset from anchor to centroid in normal coords
        const centroidOffset = piece.getCentroidOffset();

        // Convert offset to pixels
        const scale = BOARD_CONFIG.scale;
        const h = scale * Math.sqrt(3) / 2;
        const offsetPx = [
            centroidOffset[0] * scale,
            -centroidOffset[1] * h  // SVG y-flip
        ];

        // Centroid position = anchor position + offset
        return [
            anchorCellCenter[0] + offsetPx[0],
            anchorCellCenter[1] + offsetPx[1]
        ];
    }

    // Find the nearest anchor cell for a piece given its centroid position
    // Returns the cell that the anchor triangle would be closest to
    getNearestAnchorCell(piece, centroidX, centroidY) {
        // Get the offset from centroid to anchor (reverse of centroidOffset)
        const centroidOffset = piece.getCentroidOffset();
        const scale = BOARD_CONFIG.scale;
        const h = scale * Math.sqrt(3) / 2;

        // Compute where the anchor would be in pixel space
        const anchorX = centroidX - centroidOffset[0] * scale;
        const anchorY = centroidY + centroidOffset[1] * h;  // Reverse SVG y-flip

        // Find nearest cell to anchor position
        return this.getNearestCell(anchorX, anchorY);
    }

    // Legacy support methods (refactored to just use logic, not rendering)
    placePiece(pieceName, positions) {
        for (const pos of positions) {
            const key = pos.key();
            this.occupied.set(key, pieceName);
            // this._updateCell(key, pieceName); // No longer updating cell colors
        }
    }

    removePiece(pieceName, keepVisual = true) {
        const toRemove = [];
        for (const [key, name] of this.occupied) {
            if (name === pieceName) {
                toRemove.push(key);
            }
        }
        for (const key of toRemove) {
            this.occupied.delete(key);
            // this._updateCell(key, 'empty'); // No longer updating cell colors
        }
        if (!keepVisual) {
            this.removePieceElement(pieceName);
        }
        return toRemove.length > 0;
    }

    // Check if piece can be placed
    canPlace(piece, anchorPos) {
        // Check anchor orientation matches piece's first triangle
        const firstTriangle = piece.triangles[0];
        if (firstTriangle.pointsUp !== anchorPos.pointsUp) {
            return { valid: false, positions: [] };
        }

        const positions = [];
        for (const triangle of piece.triangles) {
            const resultX = anchorPos.x + triangle.x;
            const resultY = anchorPos.y + triangle.y;
            const resultUp = (resultX + resultY) % 2 === 0;

            // Check orientation consistency
            if (triangle.pointsUp === anchorPos.pointsUp) {
                if (resultUp !== anchorPos.pointsUp) return { valid: false, positions: [] };
            } else {
                if (resultUp === anchorPos.pointsUp) return { valid: false, positions: [] };
            }

            const pos = new SG.TrianglePos(resultX, resultY, resultUp);
            const key = pos.key();

            // Check if position is on board
            if (!this.cells.has(key)) {
                return { valid: false, positions: [] };
            }

            // Check if position is occupied
            if (this.occupied.has(key)) {
                return { valid: false, positions: [] };
            }

            positions.push(pos);
        }

        return { valid: true, positions };
    }

    // Update single cell appearance
    _updateCell(key, state) {
        const polygon = this.cellElements.get(key);
        if (!polygon) return;

        const themeColors = getThemeColors();
        let fill = themeColors.emptyColor;
        if (state === 'blocker') {
            fill = themeColors.blockerColor;
        } else if (state && state !== 'empty') {
            fill = SG.PIECE_COLORS[state] || '#CCCCCC';
        }

        console.log(`[${performance.now().toFixed(1)}ms] _updateCell: setting fill=${fill} for key=${key}`);
        polygon.setAttribute('fill', fill);
        polygon.setAttribute('data-state', state || 'empty');  // Update state tracking!
        console.log(`[${performance.now().toFixed(1)}ms] _updateCell: fill SET, current fill=${polygon.getAttribute('fill')}`);

        // Toggle label visibility
        const label = this.labelElements.get(key);
        if (label) {
            label.style.display = state === 'empty' ? '' : 'none';
        }
    }

    // Show ghost preview snapped to a cell anchor
    showGhost(piece, anchorPos) {
        this.clearGhost();

        const { valid, positions } = this.canPlace(piece, anchorPos);
        if (!valid) return false;

        for (const pos of positions) {
            const { polygon } = createCellPolygon(pos, 0, this.bounds, piece.name);
            polygon.setAttribute('class', 'ghost-cell');
            this.ghostGroup.appendChild(polygon);
        }

        return true;
    }

    // Show floating piece preview following the mouse cursor
    showFloatingPreview(piece, mouseX, mouseY, isValid = null) {
        this.clearGhost();

        if (!piece) return;

        const scale = BOARD_CONFIG.scale;
        const h = scale * Math.sqrt(3) / 2;
        const color = SG.PIECE_COLORS[piece.name] || '#CCCCCC';

        // Determine opacity/color based on validity
        let fillColor = color;
        let opacity = 0.7;
        if (isValid === false) {
            opacity = 0.4;
        }

        // Use piece centroid as the center point (consistent with renderPiece)
        const pieceCentroid = piece.getCentroid();

        for (const tri of piece.triangles) {
            const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');

            // Use toVertices() for correct triangle shape
            const verts = tri.toVertices();
            const points = verts.map(([vx, vy]) => {
                // Convert skewed to normal
                const nx = vx + vy * 0.5;
                const ny = vy;

                // Position relative to mouse, centered on piece centroid
                const px = mouseX + (nx - pieceCentroid[0]) * scale;
                const py = mouseY - (ny - pieceCentroid[1]) * h;

                return [px, py];
            });

            polygon.setAttribute('points', points.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '));
            polygon.setAttribute('fill', fillColor);
            polygon.setAttribute('stroke', isValid === false ? '#ff4444' : '#ffffff');
            polygon.setAttribute('stroke-width', '2');
            polygon.setAttribute('opacity', opacity);
            polygon.setAttribute('class', 'floating-preview');
            polygon.style.pointerEvents = 'none';

            this.ghostGroup.appendChild(polygon);
        }
    }

    clearGhost() {
        // Use replaceChildren() - faster than innerHTML = ''
        this.ghostGroup.replaceChildren();
    }

    // Get cell position from cell ID
    getCellById(cellId) {
        for (const [key, cell] of this.cells) {
            if (cell.cellId === cellId) {
                return cell.pos;
            }
        }
        return null;
    }

    // Check if solved (all cells occupied)
    isSolved() {
        return this.occupied.size === this.cells.size;
    }

    // Clear all pieces without re-rendering board (for solve operations)
    clearAllPieces() {
        // Remove piece visuals
        for (const [name, group] of this.pieceElements) {
            group.remove();
        }
        this.pieceElements.clear();

        // Clear piece occupancy from logic (keep blockers)
        const toRemove = [];
        for (const [key, value] of this.occupied) {
            if (value !== null) {  // null = blocker, keep it
                toRemove.push(key);
            }
        }
        for (const key of toRemove) {
            this.occupied.delete(key);
        }
    }

    // Reset board
    reset() {
        this.occupied.clear();
        this.render();
    }

    // Get current state (for solver API)
    getState() {
        const blockers = [];
        const pieces = {};

        for (const [key, value] of this.occupied) {
            const cell = this.cells.get(key);
            if (value === null) {
                blockers.push(cell.cellId);
            } else {
                if (!pieces[value]) pieces[value] = [];
                pieces[value].push(cell.cellId);
            }
        }

        return { blockers, pieces };
    }
}

// Export
window.StarGenius = window.StarGenius || {};
window.StarGenius.Board = Board;

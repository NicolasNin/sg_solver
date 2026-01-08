/**
 * Star Genius - Board Rendering
 * 
 * SVG rendering of the star-shaped board with 48 triangle cells.
 * Matches coordinate conversion from viz.py and conversion_utils.py.
 */

// Access StarGenius namespace (set by pieces.js)
const SG = window.StarGenius;

// Board configuration
const BOARD_CONFIG = {
    scale: 50,          // Pixels per unit
    margin: 30,         // SVG margin
    cellStroke: '#1a1a2e',
    cellStrokeWidth: 1,
    emptyColor: '#1b2856',
    labelColor: '#666',
    labelSize: 10,
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

    const px = margin + (nx - minVx) * scale;
    const py = margin + svgY * h;

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
    const width = BOARD_CONFIG.margin * 2 + (maxVx - minVx + (maxVy - minVy) * 0.5 + 1) * BOARD_CONFIG.scale;
    const height = BOARD_CONFIG.margin * 2 + (maxVy - minVy + 1) * h;

    return {
        minVx, maxVx, minVy, maxVy,
        width, height,
        scale: BOARD_CONFIG.scale,
        margin: BOARD_CONFIG.margin,
    };
}

// Create SVG polygon for a triangle cell
function createCellPolygon(pos, cellId, bounds, state = 'empty') {
    const verts = pos.toVertices();
    const pixels = verts.map(([vx, vy]) => vertexToPixel(vx, vy, bounds));
    const pointsStr = pixels.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');

    // Determine color based on state
    let fill = BOARD_CONFIG.emptyColor;
    if (state === 'blocker') {
        fill = SG.BLOCKER_COLOR;
    } else if (state && state !== 'empty') {
        fill = SG.PIECE_COLORS[state] || '#CCCCCC';
    }

    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    polygon.setAttribute('points', pointsStr);
    polygon.setAttribute('fill', fill);
    polygon.setAttribute('stroke', BOARD_CONFIG.cellStroke);
    polygon.setAttribute('stroke-width', BOARD_CONFIG.cellStrokeWidth);
    polygon.setAttribute('class', 'board-cell');
    polygon.setAttribute('data-cell-id', cellId);
    polygon.setAttribute('data-pos-key', pos.key());

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

        this._setupSVG();
        this.render();
    }

    _setupSVG() {
        this.svg.setAttribute('viewBox', `0 0 ${this.bounds.width.toFixed(0)} ${this.bounds.height.toFixed(0)}`);

        // Create background
        const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bg.setAttribute('width', '100%');
        bg.setAttribute('height', '100%');
        bg.setAttribute('fill', '#0d1117');
        this.svg.appendChild(bg);

        // Create groups for layering
        this.cellGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.labelGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.ghostGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.ghostGroup.setAttribute('class', 'ghost-piece');

        this.svg.appendChild(this.cellGroup);
        this.svg.appendChild(this.labelGroup);
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

    // Place piece on board
    placePiece(pieceName, positions) {
        for (const pos of positions) {
            const key = pos.key();
            this.occupied.set(key, pieceName);
            this._updateCell(key, pieceName);
        }
        // Force synchronous layout to ensure immediate visual update
        this.svg.getBBox();
    }

    // Remove piece from board
    removePiece(pieceName) {
        const toRemove = [];
        for (const [key, name] of this.occupied) {
            if (name === pieceName) {
                toRemove.push(key);
            }
        }
        for (const key of toRemove) {
            this.occupied.delete(key);
            this._updateCell(key, 'empty');
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

        let fill = BOARD_CONFIG.emptyColor;
        if (state === 'blocker') {
            fill = SG.BLOCKER_COLOR;
        } else if (state && state !== 'empty') {
            fill = SG.PIECE_COLORS[state] || '#CCCCCC';
        }

        console.log(`[${performance.now().toFixed(1)}ms] _updateCell: setting fill=${fill} for key=${key}`);
        polygon.setAttribute('fill', fill);
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

        // Use FIRST triangle as anchor (this is what placement uses)
        const anchorTri = piece.triangles[0];
        const anchorVerts = anchorTri.toVertices();

        // Find centroid of anchor triangle in skewed coords
        const anchorVx = anchorVerts.reduce((s, v) => s + v[0], 0) / 3;
        const anchorVy = anchorVerts.reduce((s, v) => s + v[1], 0) / 3;

        // Convert anchor centroid to normal coords
        const anchorNx = anchorVx + anchorVy * 0.5;
        const anchorNy = anchorVy;

        for (const tri of piece.triangles) {
            const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');

            // Use toVertices() for correct triangle shape
            const verts = tri.toVertices();
            const points = verts.map(([vx, vy]) => {
                // Convert skewed to normal
                const nx = vx + vy * 0.5;
                const ny = vy;

                // Position relative to mouse (y inverted for SVG)
                const px = mouseX + (nx - anchorNx) * scale;
                const py = mouseY - (ny - anchorNy) * h;

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

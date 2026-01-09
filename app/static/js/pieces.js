/**
 * Star Genius - Piece Definitions
 * 
 * Ported from pieces.py. All piece shapes, colors, and orientation logic.
 */

// Piece colors matching pieces.py PIECE_COLORS
const PIECE_COLORS = {
    'T1': '#00325b',  // deep blue
    'T2': '#ffc100',  // Yellow
    'T3': '#00a0de',  // Cyan
    '3B': '#00a0de',  // Cyan (same as T3)
    'T4': '#ee68a7',  // Rose
    'T5': '#de241b',  // Red
    'TF': '#7a2d9e',  // Purple
    'L4': '#ff8717',  // Orange
    'EL': '#006c43',  // Teal
    '4U': '#8bd100',  // Lime
    '4D': '#8a5e3c',  // Brown
};

const BLOCKER_COLOR = '#FFFFFF';

// Triangle position class (mirrors TrianglePos from board.py)
class TrianglePos {
    constructor(x, y, pointsUp) {
        this.x = x;
        this.y = y;
        this.pointsUp = pointsUp;
    }

    // Create key for use in Maps/Sets
    key() {
        return `${this.x},${this.y},${this.pointsUp ? 1 : 0}`;
    }

    static fromKey(key) {
        const [x, y, up] = key.split(',');
        return new TrianglePos(parseInt(x), parseInt(y), up === '1');
    }

    // Get 3 vertex coordinates (skewed grid)
    toVertices() {
        if (this.pointsUp) {
            const vx = Math.floor((this.x - this.y) / 2);
            const vy = this.y;
            return [[vx, vy], [vx + 1, vy], [vx, vy + 1]];
        } else {
            const vx = Math.floor((this.x - this.y + 1) / 2);
            const vy = this.y;
            return [[vx, vy], [vx - 1, vy + 1], [vx, vy + 1]];
        }
    }

    // Rotate vertex 60° CW: R60(i, j) = (-j, i + j)
    static rotateVertex60(i, j) {
        return [-j, i + j];
    }

    // Flip vertex: (i, j) -> (-i - j, j)
    static flipVertex(i, j) {
        return [-i - j, j];
    }

    // Reconstruct TrianglePos from 3 vertices
    static fromVertices(verts) {
        const sorted = [...verts].sort((a, b) => a[1] - b[1] || a[0] - b[0]);
        const [vx, vy] = sorted[0];

        if (sorted[0][1] === sorted[1][1]) {
            // Two vertices on same y = up triangle
            const x = vx * 2 + vy;
            return new TrianglePos(x, vy, true);
        } else {
            // Down triangle
            const x = vx * 2 + vy - 1;
            return new TrianglePos(x, vy, false);
        }
    }

    // Rotate 60° CW
    rotate60() {
        const verts = this.toVertices();
        const rotated = verts.map(([i, j]) => TrianglePos.rotateVertex60(i, j));
        return TrianglePos.fromVertices(rotated);
    }

    // Flip across vertical axis
    flipX() {
        return new TrianglePos(-this.x, this.y, this.pointsUp);
    }

    // Get centroid of this triangle in normal (unskewed) coords
    getCentroid() {
        const verts = this.toVertices();
        // Convert skewed to normal coords: nx = vx + vy*0.5, ny = vy
        let sumNx = 0, sumNy = 0;
        for (const [vx, vy] of verts) {
            sumNx += vx + vy * 0.5;
            sumNy += vy;
        }
        return [sumNx / 3, sumNy / 3];
    }
}

// Piece class
class Piece {
    constructor(name, triangles, canFlip = true) {
        this.name = name;
        this.triangles = triangles;  // Array of TrianglePos
        this.canFlip = canFlip;
    }

    // Normalize so first triangle is at origin
    normalize() {
        const first = this.triangles[0];
        if (first.x === 0 && first.y === 0) {
            return this;
        }
        const normalized = this.triangles.map(t =>
            new TrianglePos(t.x - first.x, t.y - first.y, t.pointsUp)
        );
        return new Piece(this.name, normalized, this.canFlip);
    }

    // Get canonical key for deduplication
    canonicalKey() {
        const minTri = this.triangles.reduce((min, t) =>
            (t.y < min.y || (t.y === min.y && t.x < min.x)) ? t : min
        );
        const normalized = this.triangles.map(t =>
            [t.x - minTri.x, t.y - minTri.y, t.pointsUp]
        );
        normalized.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        return JSON.stringify(normalized);
    }

    // Rotate 60° CW
    rotate60() {
        const rotated = this.triangles.map(t => t.rotate60());
        return new Piece(this.name, rotated, this.canFlip).normalize();
    }

    // Flip piece
    flip() {
        const flipped = this.triangles.map(t => t.flipX());
        return new Piece(this.name, flipped, this.canFlip).normalize();
    }

    // Get centroid of entire piece in normal (unskewed) coords
    // This is the average of all triangle centroids
    getCentroid() {
        let sumNx = 0, sumNy = 0;
        for (const tri of this.triangles) {
            const [nx, ny] = tri.getCentroid();
            sumNx += nx;
            sumNy += ny;
        }
        const n = this.triangles.length;
        return [sumNx / n, sumNy / n];
    }

    // Get offset from anchor (triangles[0]) centroid to piece centroid
    // This is used to render pieces centered on their centroid instead of anchor
    getCentroidOffset() {
        const pieceCentroid = this.getCentroid();
        const anchorCentroid = this.triangles[0].getCentroid();
        return [
            pieceCentroid[0] - anchorCentroid[0],
            pieceCentroid[1] - anchorCentroid[1]
        ];
    }

    // Get all unique orientations (for solver - deduplicates)
    allOrientations(forceFlip = false) {
        const includeFlips = this.canFlip || forceFlip;
        const seen = new Set();
        const orientations = [];

        let p = this.normalize();
        for (let i = 0; i < 6; i++) {
            const key = p.canonicalKey();
            if (!seen.has(key)) {
                seen.add(key);
                orientations.push(p);
            }

            if (includeFlips) {
                const flipped = p.flip();
                const flipKey = flipped.canonicalKey();
                if (!seen.has(flipKey)) {
                    seen.add(flipKey);
                    orientations.push(flipped);
                }
            }

            p = p.rotate60();
        }

        return orientations;
    }

    // Get all 12 orientations in fixed order (for UI - predictable indexing)
    // Indices 0-5: 6 rotations of base piece (rot0, rot1, ..., rot5)
    // Indices 6-11: 6 rotations of flipped piece (flip+rot0, flip+rot1, ..., flip+rot5)
    allOrientationsOrdered() {
        const orientations = [];

        // First 6: rotations of base piece
        let p = this.normalize();
        for (let i = 0; i < 6; i++) {
            orientations.push(p);
            p = p.rotate60();
        }

        // Next 6: rotations of flipped piece
        let flipped = this.normalize().flip();
        for (let i = 0; i < 6; i++) {
            orientations.push(flipped);
            flipped = flipped.rotate60();
        }

        return orientations;
    }
}

// Helper to create piece from coordinates
function makePiece(name, coords, canFlip = true) {
    const triangles = coords.map(([x, y, up]) => new TrianglePos(x, y, up));
    return new Piece(name, triangles, canFlip);
}

// All piece definitions (from pieces.py)
const PIECE_DEFINITIONS = {
    'T1': makePiece('T1', [[0, 0, true]], false),
    'T2': makePiece('T2', [[0, 0, true], [1, 0, false]], false),
    'T3': makePiece('T3', [[0, 0, true], [1, 0, false], [2, 0, true]]),
    '3B': makePiece('3B', [[0, 0, true], [1, 0, false], [2, 0, true]], false),
    'T4': makePiece('T4', [[0, 0, true], [1, 0, false], [2, 0, true], [3, 0, false]]),
    'T5': makePiece('T5', [[0, 0, true], [1, 0, false], [2, 0, true], [3, 0, false], [4, 0, true]]),
    'TF': makePiece('TF', [[0, 0, false], [1, 0, true], [-1, 0, true], [0, 1, true]], false),
    'L4': makePiece('L4', [[0, 0, true], [1, 0, false], [2, 0, true], [0, -1, false]], false),
    'EL': makePiece('EL', [[0, 0, true], [1, 0, false], [2, 0, true], [2, -1, false], [0, -1, false]]),
    '4U': makePiece('4U', [[0, 0, true], [1, 0, false], [2, 0, true], [3, 0, false], [3, 1, true]]),
    '4D': makePiece('4D', [[0, 0, true], [1, 0, false], [2, 0, true], [3, 0, false], [2, -1, false]]),
};

// Precompute all orientations for each piece
// PIECE_ORIENTATIONS: deduplicated (for solver)
// PIECE_ORIENTATIONS_ORDERED: fixed 12 positions (for UI)
const PIECE_ORIENTATIONS = {};
const PIECE_ORIENTATIONS_ORDERED = {};
for (const [name, piece] of Object.entries(PIECE_DEFINITIONS)) {
    PIECE_ORIENTATIONS[name] = piece.allOrientations();
    PIECE_ORIENTATIONS_ORDERED[name] = piece.allOrientationsOrdered();
}

// Ordered list for display (largest to smallest for palette)
const ALL_PIECES = ['T5', '4U', '4D', 'EL', 'T4', 'L4', 'TF', 'T3', '3B', 'T2', 'T1'];

// Dice definitions (from dices.py)
// 3x 8-faced dice, 4x 6-faced dice (2 incomplete - only 3 faces known)
const ALL_DICE = [
    [25, 26, 36, 37, 38, 40, 45, 47],  // 8-faced dice 1
    [2, 4, 7, 8, 9, 17, 11, 16],       // 8-faced dice 2
    [12, 13, 23, 24, 32, 33, 41, 42],  // 8-faced dice 3
    [1, 5, 15, 34, 44, 48],            // 6-faced dice 1
    [19, 20, 21, 28, 29, 30],          // 6-faced dice 2
    [18, 22, 39],                       // 6-faced dice 3 (incomplete)
    [10, 27, 31],                       // 6-faced dice 4 (incomplete)
];

// Roll all 7 dice
function rollDice() {
    return ALL_DICE.map(die => die[Math.floor(Math.random() * die.length)]);
}

// Export for use in other modules
window.StarGenius = window.StarGenius || {};
Object.assign(window.StarGenius, {
    TrianglePos,
    Piece,
    PIECE_COLORS,
    BLOCKER_COLOR,
    PIECE_DEFINITIONS,
    PIECE_ORIENTATIONS,
    PIECE_ORIENTATIONS_ORDERED,
    ALL_PIECES,
    ALL_DICE,
    rollDice,
});

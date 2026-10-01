/**
 * The arrow-key stick's model: feel, geometry, pins and artwork.
 *
 * The scheme itself is the keyboard's — see the note at the top of input.js —
 * but every number that decides how it feels lives here, because the picture
 * and the maths have to agree: the shape the player drags across is built from
 * the same layout the input path reads its buttons from.
 *
 * Two shapes share one idea. Eight directions around a stop zone, and the
 * player decides how much room each one gets:
 *
 *   circle   three concentric zones (stop / ramp / full speed) cut by eight
 *            sectors. The four boundary angles on the right half are the
 *            parameters; the left half is their mirror, so front and back can
 *            differ but left and right cannot.
 *   square   a 3x3 grid of rectangles — eight directions and the stop zone in
 *            the middle — with no ramp: a cell is either held at full strength
 *            or not held at all. The three grid lines are the parameters, and
 *            the left/right pair is mirrored.
 *
 * Each parameter *is* one boundary, which is what makes the controls
 * independent: moving one line never rescales another. Boundaries cannot cross,
 * so a value is clamped to its neighbours rather than pushing them.
 */

const DEG = Math.PI / 180;
/** Artwork is drawn in a 200×200 box; radius 1 of a circle layout is this. */
export const PAD_UNITS = 99;
/** Degrees a finger must cross past a divider before the sector flips, capped
 *  to a quarter of the narrowest a sector can get. Without it a thumb resting
 *  on a boundary alternates between two commands every frame. */
const SECTOR_STICKY_DEG = 6;
/** Same idea for the grid, as a fraction of the pad's side. */
const SQUARE_STICKY = 0.025;
/** Smallest middle row/column the grid editor will draw, in percent. */
const SQUARE_MIN_CELL = 5;

export const PAD_SHAPES = Object.freeze(["circle", "square"]);

/**
 * Every tunable, in the units the panel and the exported JSON use.
 *
 * `pin` marks the ones a player can drag directly on the pad: `angle` pins sit
 * on a sector boundary, `column`/`row` pins on a grid line. The rest are panel
 * only. `shapes` decides which shape each field belongs to, so switching shape
 * shows the controls that shape actually has.
 */
export const PAD_TUNE_FIELDS = Object.freeze([
  {
    key: "size", unit: "px", min: 96, max: 240, step: 2, default: 148, shapes: PAD_SHAPES,
  },

  // ---- circle: the radial zones
  {
    key: "deadzone", unit: "%", min: 0, max: 60, step: 1, default: 25, shapes: ["circle"],
  },
  {
    key: "transition", unit: "%", min: 0, max: 80, step: 1, default: 35, shapes: ["circle"],
  },

  // ---- circle: the four boundary angles of the right half, from the nose
  {
    key: "forwardEdge", unit: "°", min: 0, max: 90, step: 2.5, default: 22.5, shapes: ["circle"], pin: "angle",
  },
  {
    key: "frontDiagonalEdge", unit: "°", min: 0, max: 180, step: 2.5, default: 67.5, shapes: ["circle"], pin: "angle",
  },
  {
    key: "turnEdge", unit: "°", min: 0, max: 180, step: 2.5, default: 112.5, shapes: ["circle"], pin: "angle",
  },
  {
    key: "backEdge", unit: "°", min: 0, max: 180, step: 2.5, default: 157.5, shapes: ["circle"], pin: "angle",
  },

  // ---- three-finger buttons: two paddles and a trigger, each placed by hand
  {
    key: "moveSize", unit: "px", min: 40, max: 120, step: 2, default: 64, schemes: ["pads"],
  },
  {
    key: "moveGap", unit: "px", min: 0, max: 48, step: 1, default: 8, schemes: ["pads"],
  },
  {
    key: "turnSize", unit: "px", min: 40, max: 120, step: 2, default: 64, schemes: ["pads"],
  },
  {
    key: "turnGap", unit: "px", min: 0, max: 48, step: 1, default: 8, schemes: ["pads"],
  },
  {
    key: "fireSize", unit: "px", min: 48, max: 140, step: 2, default: 88, schemes: ["pads"],
  },
  {
    key: "moveX", unit: "%", min: 5, max: 95, step: 1, default: 22, schemes: ["pads"],
  },
  {
    key: "moveY", unit: "%", min: 5, max: 95, step: 1, default: 74, schemes: ["pads"],
  },
  {
    key: "turnX", unit: "%", min: 5, max: 95, step: 1, default: 78, schemes: ["pads"],
  },
  {
    key: "turnY", unit: "%", min: 5, max: 95, step: 1, default: 74, schemes: ["pads"],
  },
  {
    key: "fireX", unit: "%", min: 5, max: 95, step: 1, default: 78, schemes: ["pads"],
  },
  {
    key: "fireY", unit: "%", min: 5, max: 95, step: 1, default: 38, schemes: ["pads"],
  },

  // ---- square: the grid lines, in percent from the pad's top-left
  {
    key: "columnEdge", unit: "%", min: SQUARE_MIN_CELL, max: 50, step: 1, default: 33, shapes: ["square"], pin: "column",
  },
  {
    key: "rowTopEdge", unit: "%", min: 0, max: 100, step: 1, default: 33, shapes: ["square"], pin: "row",
  },
  {
    key: "rowBottomEdge", unit: "%", min: 0, max: 100, step: 1, default: 67, shapes: ["square"], pin: "row",
  },
]);

export const DEFAULT_PAD_TUNE = Object.freeze({
  shape: "circle",
  ...Object.fromEntries(PAD_TUNE_FIELDS.map((field) => [field.key, field.default])),
});

/**
 * The fields the panel should show: the three-finger scheme has its own, and
 * the eight-way pad shows whichever shape is selected.
 */
export function padTuneFields(tune, scheme = "sectors") {
  if (scheme === "pads") return PAD_TUNE_FIELDS.filter((field) => field.schemes?.includes("pads"));
  return PAD_TUNE_FIELDS.filter((field) => field.shapes?.includes(tune.shape));
}

function fieldFor(key) {
  return PAD_TUNE_FIELDS.find((field) => field.key === key) ?? null;
}

function decimalsFor(step) {
  const text = String(step);
  return text.includes(".") ? text.length - text.indexOf(".") - 1 : 0;
}

function snap(field, value) {
  const clamped = Math.max(field.min, Math.min(field.max, value));
  const stepped = field.min + Math.round((clamped - field.min) / field.step) * field.step;
  return Number(stepped.toFixed(decimalsFor(field.step)));
}

/**
 * Clamp, snap and complete a feel object. Accepts anything: a stored value from
 * an older build, a hand-edited export, a field someone deleted. Unknown keys
 * are dropped and missing ones fall back to the default.
 *
 * Ordering between neighbours is enforced by clamping in one direction only —
 * the forward boundary can be pushed back by the front-diagonal one, never the
 * other way round. So `setPadTuneField()` below moves exactly the field it is
 * given and leaves every other value alone, which is what makes a slider (or a
 * pin) mean one thing.
 */
export function normalisePadTune(raw) {
  const source = raw !== null && typeof raw === "object" ? raw : {};
  const shape = PAD_SHAPES.includes(source.shape) ? source.shape : DEFAULT_PAD_TUNE.shape;
  const tune = { shape };
  for (const field of PAD_TUNE_FIELDS) {
    const value = Number(source[field.key]);
    tune[field.key] = Number.isFinite(value) ? snap(field, value) : field.default;
  }
  // A circle's four boundaries are ordered, and its two zones share the radius.
  // Each field is only ever pushed *up* against the one before it, so a value
  // that is already sensible is never rewritten by a later one.
  tune.forwardEdge = Math.min(tune.forwardEdge, 90);
  tune.frontDiagonalEdge = Math.max(tune.frontDiagonalEdge, tune.forwardEdge);
  tune.turnEdge = Math.max(tune.turnEdge, tune.frontDiagonalEdge);
  tune.backEdge = Math.max(tune.backEdge, tune.turnEdge);
  // A square's grid lines are ordered, with a middle cell left to press on.
  tune.rowTopEdge = Math.min(tune.rowTopEdge, 100 - SQUARE_MIN_CELL);
  tune.rowBottomEdge = Math.min(tune.rowBottomEdge, 100);
  tune.rowTopEdge = Math.min(tune.rowTopEdge, tune.rowBottomEdge - SQUARE_MIN_CELL);
  tune.rowBottomEdge = Math.max(tune.rowBottomEdge, tune.rowTopEdge + SQUARE_MIN_CELL);
  return tune;
}

/** The window one field may move in, given where its neighbours are. */
function fieldWindow(key, tune, spec) {
  switch (key) {
    case "forwardEdge": return [0, tune.frontDiagonalEdge];
    case "frontDiagonalEdge": return [tune.forwardEdge, tune.turnEdge];
    case "turnEdge": return [tune.frontDiagonalEdge, tune.backEdge];
    case "backEdge": return [tune.turnEdge, 180];
    case "deadzone": return [0, 60];
    case "transition": return [0, Math.min(80, 100 - tune.deadzone)];
    case "columnEdge": return [SQUARE_MIN_CELL, 50];
    case "rowTopEdge": return [0, tune.rowBottomEdge - SQUARE_MIN_CELL];
    case "rowBottomEdge": return [tune.rowTopEdge + SQUARE_MIN_CELL, 100];
    default: return [spec.min, spec.max];
  }
}

/**
 * Move one field of a feel setting. The field is clamped into the window its
 * neighbours leave it, so nothing else moves — that is what lets a slider or a
 * pin mean one number.
 */
export function setPadTuneField(tune, key, value) {
  const spec = fieldFor(key);
  if (spec === null) return normalisePadTune(tune);
  const base = normalisePadTune(tune);
  const window = fieldWindow(key, base, spec);
  const wanted = Math.max(window[0], Math.min(window[1], Number(value)));
  return normalisePadTune({ ...base, [key]: wanted });
}

const SECTOR_NAMES = ["up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left"];
const SECTOR_BUTTONS = [
  { forward: 1, backup: 0, turnLeft: 0, turnRight: 0 }, //  0 up
  { forward: 1, backup: 0, turnLeft: 0, turnRight: 1 }, //  1 up-right
  { forward: 0, backup: 0, turnLeft: 0, turnRight: 1 }, //  2 right
  { forward: 0, backup: 1, turnLeft: 0, turnRight: 1 }, //  3 down-right
  { forward: 0, backup: 1, turnLeft: 0, turnRight: 0 }, //  4 down
  { forward: 0, backup: 1, turnLeft: 1, turnRight: 0 }, //  5 down-left
  { forward: 0, backup: 0, turnLeft: 1, turnRight: 0 }, //  6 left
  { forward: 1, backup: 0, turnLeft: 1, turnRight: 0 }, //  7 up-left
];
const SECTOR_GLYPHS = ["arrow", "bend", "turn", "bend", "arrow", "bend", "turn", "bend"];

function circleLayout(tune) {
  const deadzone = tune.deadzone / 100;
  const full = Math.min(1, deadzone + tune.transition / 100);
  const ramp = Math.max(0, full - deadzone);
  const e = [tune.forwardEdge, tune.frontDiagonalEdge, tune.turnEdge, tune.backEdge];
  // One continuous chain starting at the left edge of the forward sector: the
  // four values *are* its boundaries, so they always tile the circle exactly.
  const spans = [
    [-e[0], e[0]],
    [e[0], e[1]],
    [e[1], e[2]],
    [e[2], e[3]],
    [e[3], 360 - e[3]],
    [-e[3], -e[2]],
    [-e[2], -e[1]],
    [-e[1], -e[0]],
  ];
  // Mirror-image spans come out negative; fold them into the same domain as
  // the rest of the chain so a lookup never has to guess which copy of an
  // angle it is comparing against.
  for (const span of spans) {
    while (span[0] < -e[0]) {
      span[0] += 360;
      span[1] += 360;
    }
  }
  const sectors = spans.map(([start, end], index) => ({
    index,
    name: SECTOR_NAMES[index],
    buttons: SECTOR_BUTTONS[index],
    start,
    end,
    center: (start + end) / 2,
    width: end - start,
    sticky: Math.min(SECTOR_STICKY_DEG, (end - start) / 4),
  }));
  return {
    shape: "circle",
    tune,
    deadzone,
    ramp,
    full,
    // Radius for the direction glyphs: centred on the ramp when there is one,
    // otherwise midway between the hub and the rim.
    glyphRadius: Math.max(0.42, Math.min(0.78,
      ramp > 0.02 ? deadzone + ramp * 0.6 : (deadzone + 1) / 2)),
    sectors,
  };
}

function squareLayout(tune) {
  const x0 = (tune.columnEdge / 100) * 2 - 1;
  const y0 = (tune.rowTopEdge / 100) * 2 - 1;
  const y1 = (tune.rowBottomEdge / 100) * 2 - 1;
  const columns = { left: [-1, x0], center: [x0, -x0], right: [-x0, 1] };
  const rows = { top: [-1, y0], middle: [y0, y1], bottom: [y1, 1] };
  // column:row -> direction index, or the stop cell in the middle.
  const DIRECTION = {
    "center:top": 0, "right:top": 1, "right:middle": 2, "right:bottom": 3,
    "center:bottom": 4, "left:bottom": 5, "left:middle": 6, "left:top": 7,
  };
  const cells = [];
  for (const [column, [cx0, cx1]] of Object.entries(columns)) {
    for (const [row, [cy0, cy1]] of Object.entries(rows)) {
      const direction = DIRECTION[`${column}:${row}`] ?? null;
      cells.push({
        column,
        row,
        direction,
        rect: [cx0, cy0, cx1 - cx0, cy1 - cy0],
        // Inflated by the sticky band: a finger that was in this cell keeps it
        // until it is properly out, instead of chattering on a grid line.
        sticky: [cx0 - SQUARE_STICKY, cy0 - SQUARE_STICKY, cx1 + SQUARE_STICKY, cy1 + SQUARE_STICKY],
      });
    }
  }
  return { shape: "square", tune, columns, rows, cells, x0, y0, y1 };
}

/**
 * Where the three-finger buttons sit, in percentages of the touch layer and
 * pixels for their size. Defaults are the layout a two-thumb player expects:
 * forward/reverse under the left thumb, turn under the right, and the trigger
 * clear of both in the top-right corner.
 */
export function padsLayout(rawTune = DEFAULT_PAD_TUNE) {
  const tune = normalisePadTune(rawTune);
  return {
    scheme: "pads",
    tune,
    move: { x: tune.moveX, y: tune.moveY, size: tune.moveSize, gap: tune.moveGap },
    turn: { x: tune.turnX, y: tune.turnY, size: tune.turnSize, gap: tune.turnGap },
    fire: { x: tune.fireX, y: tune.fireY, size: tune.fireSize },
  };
}

/** The full geometry of one feel setting. */
export function padLayout(rawTune = DEFAULT_PAD_TUNE) {
  const tune = normalisePadTune(rawTune);
  return tune.shape === "square" ? squareLayout(tune) : circleLayout(tune);
}

/**
 * Sector index for an angle in compass degrees, keeping `previousSector` while
 * the angle stays inside it plus that sector's sticky band.
 */
export function sectorIndexAt(angleDegrees, layout, previousSector = null) {
  // Fold the angle into the chain's own domain, which starts at the left edge
  // of the forward sector rather than at compass zero.
  const shift = layout.sectors[0].start;
  let angle = (angleDegrees - shift) % 360;
  if (angle < 0) angle += 360;
  angle += shift;
  if (Number.isInteger(previousSector) && previousSector >= 0 && previousSector < layout.sectors.length) {
    const sector = layout.sectors[previousSector];
    if (angle >= sector.start - sector.sticky && angle <= sector.end + sector.sticky) {
      return previousSector;
    }
  }
  for (const sector of layout.sectors) {
    if (angle >= sector.start && angle < sector.end) return sector.index;
  }
  // Unreachable while the chain tiles the circle, which the layout guarantees.
  return 0;
}

function keepRing(ring, previousRing, distance, layout) {
  if (previousRing === undefined || previousRing === null || previousRing === ring) return ring;
  const bands = {
    hub: [0, layout.deadzone],
    ramp: [layout.deadzone, layout.full],
    full: [layout.full, 1],
  };
  const band = bands[previousRing];
  if (!band) return ring;
  const sticky = 0.02;
  return distance >= band[0] - sticky && distance <= band[1] + sticky ? previousRing : ring;
}

const STOPPED = Object.freeze({
  forward: 0, backup: 0, turnLeft: 0, turnRight: 0,
  targetRotation: null, sector: null, ring: "hub", magnitude: 0,
});

/**
 * Resolve a stick vector into strengths, plus the cell it landed in.
 *
 * `previous` is `{ sector, ring }` from the last pointer event, used only for
 * the sticky bands. Returns strengths in the shape `joystickButtons` uses so
 * the two schemes stay interchangeable, with `targetRotation` always null: this
 * scheme never asks for a heading, which is why the instant-turn assist has
 * nothing to do here.
 */
export function padCell(x, y, rawTune = DEFAULT_PAD_TUNE, previous = null) {
  const layout = padLayout(rawTune);
  return layout.shape === "square"
    ? squareCell(x, y, layout, previous)
    : circleCell(x, y, layout, previous);
}

function strengths(sector, ring, magnitude) {
  const buttons = SECTOR_BUTTONS[sector];
  return {
    forward: buttons.forward * magnitude,
    backup: buttons.backup * magnitude,
    turnLeft: buttons.turnLeft * magnitude,
    turnRight: buttons.turnRight * magnitude,
    targetRotation: null,
    sector,
    ring,
    magnitude,
  };
}

function circleCell(x, y, layout, previous) {
  const distance = Math.min(1, Math.hypot(x, y));
  if (distance === 0) return { ...STOPPED };
  let ring = distance <= layout.deadzone ? "hub"
    : (distance >= layout.full ? "full" : "ramp");
  if (layout.ramp <= 0 && ring === "ramp") ring = "full";
  ring = keepRing(ring, previous?.ring ?? null, distance, layout);
  if (ring === "hub" || (layout.ramp <= 0 && ring === "ramp")) return { ...STOPPED };
  const sector = sectorIndexAt(Math.atan2(x, -y) / DEG, layout, previous?.sector ?? null);
  const magnitude = layout.ramp <= 0
    ? 1
    : Math.min(1, (distance - layout.deadzone) / layout.ramp);
  return strengths(sector, ring, magnitude);
}

function squareCell(x, y, layout, previous) {
  const column = x < layout.x0 ? "left" : (x > -layout.x0 ? "right" : "center");
  const row = y < layout.y0 ? "top" : (y > layout.y1 ? "bottom" : "middle");
  const previousCell = Number.isInteger(previous?.sector)
    ? layout.cells.find((cell) => cell.direction === previous.sector) ?? null
    : null;
  // Keep the held cell while the finger is only just outside it: the same
  // anti-chatter rule the circle applies to its dividers.
  const held = previousCell !== null
    && x >= previousCell.sticky[0] && x <= previousCell.sticky[2]
    && y >= previousCell.sticky[1] && y <= previousCell.sticky[3];
  const cell = held ? previousCell : layout.cells.find(
    (candidate) => candidate.column === column && candidate.row === row,
  );
  if (!cell || cell.direction === null) return { ...STOPPED };
  return strengths(cell.direction, "full", 1);
}

// ------------------------------------------------------------------ artwork

function pointAt(radius, compassDegrees) {
  const a = compassDegrees * DEG;
  return [Math.sin(a) * radius, -Math.cos(a) * radius];
}

function round2(value) {
  return (Math.round(value * 100) / 100).toFixed(2);
}

function annularPath(inner, outer, from, to) {
  const [x0, y0] = pointAt(outer, from);
  const [x1, y1] = pointAt(outer, to);
  const [x2, y2] = pointAt(inner, to);
  const [x3, y3] = pointAt(inner, from);
  const large = Math.abs(to - from) > 180 ? 1 : 0;
  return `M${round2(x0)} ${round2(y0)} A${round2(outer)} ${round2(outer)} 0 ${large} 1 `
    + `${round2(x1)} ${round2(y1)} L${round2(x2)} ${round2(y2)} `
    + `A${round2(inner)} ${round2(inner)} 0 ${large} 0 ${round2(x3)} ${round2(y3)} Z`;
}

function rectAttrs([x, y, width, height]) {
  return `x="${round2(x * PAD_UNITS)}" y="${round2(y * PAD_UNITS)}" `
    + `width="${round2(width * PAD_UNITS)}" height="${round2(height * PAD_UNITS)}"`;
}

const GLYPH_SCALE = 1.18;
/** How far the glowing edges stop short of the rim, in pad units. */
const EDGE_INSET = 3;
/**
 * Where the angle pins sit, as a fraction of the pad radius. Neighbouring
 * boundaries alternate between the two, so two handles can never land on top
 * of each other even when the boundaries they move are a couple of degrees
 * apart — a narrow forward cone used to stack four dots on the nose.
 */
const PIN_RADII = [0.84, 0.46];
/** How far along its own grid line a square's handle sits, in percent. */
const PIN_OFFSET = 40;
/** Direction glyphs, drawn pointing outwards in a local frame. */
const GLYPHS = {
  // forward / reverse: a straight arrow
  arrow: '<path class="pad-stroke" d="M0 11 L0 -3 M-7 -2 L0 -11 L7 -2"/>',
  // forward + turn: leaves the hub, curves out and to the right
  bend: '<path class="pad-stroke" d="M-9.00 6.80 A14.00 14.00 0 0 1 4.02 -7.17"/>'
    + '<polygon class="pad-head" points="10.01,-7.58 3.66,-3.83 3.19,-10.42"/>',
  // turn in place: a circular arrow
  turn: '<path class="pad-stroke" d="M2.94 8.08 A8.60 8.60 0 0 1 8.34 -2.08"/>'
    + '<polygon class="pad-head" points="9.80,3.74 5.00,-1.86 11.40,-3.46"/>',
};
const MIRRORED = new Set([5, 6, 7]);

function circleArtwork(layout) {
  const { deadzone, full } = layout;
  const parts = ['    <g class="pad-cells">'];
  // The stop zone is one region with no direction of its own.
  parts.push(`      <circle class="pad-cell" data-band="hub" r="${round2(deadzone * PAD_UNITS)}"/>`);
  for (const sector of layout.sectors) {
    for (const [band, inner, outer] of [["ramp", deadzone, full], ["full", full, 1]]) {
      const d = annularPath(inner * PAD_UNITS, outer * PAD_UNITS, sector.start, sector.end);
      parts.push(`      <path class="pad-cell" data-dir="${sector.index}" data-band="${band}" d="${d}"/>`);
    }
  }
  parts.push("    </g>");

  for (const sector of layout.sectors) {
    const [x0, y0] = pointAt(deadzone * PAD_UNITS, sector.start);
    const [x1, y1] = pointAt(PAD_UNITS, sector.start);
    parts.push(`    <line class="pad-divider" x1="${round2(x0)}" y1="${round2(y0)}" `
      + `x2="${round2(x1)}" y2="${round2(y1)}"/>`);
  }

  // Two edges per direction — the glow that says which one is held. They stop
  // just short of the rim so the glow stays inside the pad.
  parts.push('    <g class="pad-edges">');
  for (const sector of layout.sectors) {
    for (const angle of [sector.start, sector.end]) {
      const [x0, y0] = pointAt(deadzone * PAD_UNITS, angle);
      const [x1, y1] = pointAt(PAD_UNITS - EDGE_INSET, angle);
      parts.push(`      <line class="pad-edge" data-dir="${sector.index}" x1="${round2(x0)}" `
        + `y1="${round2(y0)}" x2="${round2(x1)}" y2="${round2(y1)}"/>`);
    }
  }
  parts.push("    </g>");

  parts.push('    <g class="pad-rings">');
  parts.push(`      <circle class="pad-ring" data-ring="hub" r="${round2(deadzone * PAD_UNITS)}"/>`);
  parts.push(`      <circle class="pad-ring" data-ring="ramp" r="${round2(full * PAD_UNITS)}"/>`);
  // The rim itself is added by padArtwork(), which draws the border for both
  // shapes in one place.
  parts.push("    </g>");

  parts.push('    <g class="pad-glyphs">');
  for (const sector of layout.sectors) {
    const [x, y] = pointAt(layout.glyphRadius * PAD_UNITS, sector.center);
    const scale = MIRRORED.has(sector.index) ? -GLYPH_SCALE : GLYPH_SCALE;
    parts.push(`      <g class="pad-glyph" data-dir="${sector.index}" `
      + `transform="translate(${round2(x)} ${round2(y)}) rotate(${round2(sector.center)}) `
      + `scale(${scale} ${GLYPH_SCALE})">${GLYPHS[SECTOR_GLYPHS[sector.index]]}</g>`);
  }
  parts.push("    </g>");
  return parts;
}

function squareArtwork(layout) {
  const parts = ['    <g class="pad-cells">'];
  for (const cell of layout.cells) {
    const direction = cell.direction === null ? "" : ` data-dir="${cell.direction}"`;
    const band = cell.direction === null ? "hub" : "full";
    parts.push(`      <rect class="pad-cell"${direction} data-band="${band}" `
      + `${rectAttrs(cell.rect)}/>`);
  }
  parts.push("    </g>");

  parts.push('    <g class="pad-grid">');
  for (const x of [layout.x0, -layout.x0]) {
    parts.push(`      <line x1="${round2(x * PAD_UNITS)}" y1="${-PAD_UNITS}" `
      + `x2="${round2(x * PAD_UNITS)}" y2="${PAD_UNITS}"/>`);
  }
  for (const y of [layout.y0, layout.y1]) {
    parts.push(`      <line x1="${-PAD_UNITS}" y1="${round2(y * PAD_UNITS)}" `
      + `x2="${PAD_UNITS}" y2="${round2(y * PAD_UNITS)}"/>`);
  }
  parts.push("    </g>");

  // The glowing outline of whichever direction is held.
  parts.push('    <g class="pad-edges">');
  for (const cell of layout.cells) {
    if (cell.direction === null) continue;
    parts.push(`      <rect class="pad-edge" data-dir="${cell.direction}" ${rectAttrs(cell.rect)}/>`);
  }
  parts.push("    </g>");

  parts.push('    <g class="pad-glyphs">');
  for (const cell of layout.cells) {
    if (cell.direction === null) continue;
    const [x, y, width, height] = cell.rect;
    const tx = (x + width / 2) * PAD_UNITS;
    const ty = (y + height / 2) * PAD_UNITS;
    const scale = MIRRORED.has(cell.direction) ? -GLYPH_SCALE : GLYPH_SCALE;
    parts.push(`      <g class="pad-glyph" data-dir="${cell.direction}" `
      + `transform="translate(${round2(tx)} ${round2(ty)}) rotate(${round2(cell.direction * 45)}) `
      + `scale(${scale} ${GLYPH_SCALE})">${GLYPHS[SECTOR_GLYPHS[cell.direction]]}</g>`);
  }
  parts.push("    </g>");
  return parts;
}

/**
 * The whole pad picture for one feel setting, as markup.
 *
 * The cells the player drags across, the lines between them and the glow that
 * answers a press all come from the same layout the input path uses, so a
 * retuned stick redraws rather than leaving the picture behind.
 */
export function padArtwork(rawTune = DEFAULT_PAD_TUNE) {
  const layout = padLayout(rawTune);
  const square = layout.shape === "square";
  const parts = [
    '<svg class="joy-sectors" viewBox="0 0 200 200" aria-hidden="true" focusable="false">',
    '  <g transform="translate(100 100)">',
    ...(square ? squareArtwork(layout) : circleArtwork(layout)),
    square
      ? `    <rect class="pad-rim" x="${-PAD_UNITS}" y="${-PAD_UNITS}" `
        + `width="${2 * PAD_UNITS}" height="${2 * PAD_UNITS}" rx="16"/>`
      : `    <circle class="pad-rim" data-ring="full" r="${round2(PAD_UNITS)}"/>`,
    "  </g>",
    "</svg>",
  ];
  return parts.join("\n") + "\n";
}

/**
 * Where the draggable handles sit, as percentages of the pad — one per boundary
 * the player can move. Each handle names the field it edits, so a drag is just
 * "read the position, set that one field".
 */
export function padPins(rawTune = DEFAULT_PAD_TUNE) {
  const layout = padLayout(rawTune);
  const pins = [];
  if (layout.shape === "square") {
    // Each handle slides along its own grid line, so a pair of lines that are
    // only a few percent apart still gets two separate handles.
    const column = layout.tune.columnEdge;
    pins.push({ field: "columnEdge", side: "left", left: column, top: PIN_OFFSET });
    pins.push({ field: "columnEdge", side: "right", left: 100 - column, top: 100 - PIN_OFFSET });
    pins.push({ field: "rowTopEdge", side: "top", left: PIN_OFFSET, top: layout.tune.rowTopEdge });
    pins.push({
      field: "rowBottomEdge",
      side: "bottom",
      left: 100 - PIN_OFFSET,
      top: layout.tune.rowBottomEdge,
    });
    return pins;
  }
  ["forwardEdge", "frontDiagonalEdge", "turnEdge", "backEdge"].forEach((key, index) => {
    const angle = layout.tune[key];
    const [outer, inner] = index % 2 === 0 ? PIN_RADII : [...PIN_RADII].reverse();
    for (const [side, radius] of [["right", outer], ["left", inner]]) {
      const [x, y] = pointAt(radius, side === "right" ? angle : -angle);
      pins.push({ field: key, side, left: 50 + x * 50, top: 50 + y * 50 });
    }
  });
  return pins;
}

/** The value a pin drag implies: the angle, or the grid line, under the pointer. */
export function padPinValue(field, x, y) {
  const spec = fieldFor(field);
  if (spec === null) return null;
  if (spec.pin === "angle") {
    const angle = Math.abs(Math.atan2(x, -y) / DEG);
    return Math.min(180, angle);
  }
  if (spec.pin === "column") return ((x + 1) / 2) * 100;
  if (spec.pin === "row") return ((y + 1) / 2) * 100;
  return null;
}

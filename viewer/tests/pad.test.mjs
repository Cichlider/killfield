/**
 * The arrow-key stick: the mobile scheme that speaks arrow keys, in both of its
 * shapes, plus the feel settings that decide how it plays.
 *
 * Two things have to hold whatever the settings are, and most of this file is
 * those two: a full press is *exactly* the keyboard, and the picture drawn for
 * the player describes the geometry the input path actually reads. The rest
 * pins the tuning itself — one boundary per control, zones, sticky bands, and
 * the square grid.
 */

import assert from "node:assert/strict";
import {
  DEFAULT_FORWARD_ALIGNMENT_DEGREES,
  DEFAULT_TOUCH_SCHEME,
  Keyboard,
  TOUCH_SCHEMES,
  TouchControls,
  normaliseTouchScheme,
} from "../src/input.js";
import {
  DEFAULT_PAD_TUNE,
  PAD_SHAPES,
  PAD_TUNE_FIELDS,
  PAD_UNITS,
  padArtwork,
  padCell,
  padLayout,
  padPinValue,
  padPins,
  padTuneFields,
  padsLayout,
  normalisePadTune,
  setPadTuneField,
} from "../src/pad.js";

const AXES = ["forward", "backup", "turnLeft", "turnRight"];
const direct = (compass) => {
  const a = (compass * Math.PI) / 180;
  return [Math.sin(a), -Math.cos(a)];
};
const movementOf = (buttons) => Object.fromEntries(AXES.map((axis) => [axis, buttons[axis]]));
/** Compass degrees folded into (-180, 180], so 225 and -135 compare equal. */
const wrap = (degrees) => {
  let value = degrees % 360;
  if (value > 180) value -= 360;
  if (value <= -180) value += 360;
  return value;
};

/** Sector 0 is the nose; the rest follow clockwise. */
const DIRECTIONS = [
  { name: "up", button: "forward", compass: 0 },
  { name: "up-right", button: "forward+turnRight", compass: 45 },
  { name: "right", button: "turnRight", compass: 90 },
  { name: "down-right", button: "backup+turnRight", compass: 135 },
  { name: "down", button: "backup", compass: 180 },
  { name: "down-left", button: "backup+turnLeft", compass: 225 },
  { name: "left", button: "turnLeft", compass: 270 },
  { name: "up-left", button: "forward+turnLeft", compass: 315 },
];

function expected(button) {
  const want = { forward: 0, backup: 0, turnLeft: 0, turnRight: 0 };
  for (const part of button.split("+")) want[part] = 1;
  return want;
}

class FakeTarget {
  constructor() { this.listeners = new Map(); }
  addEventListener(type, listener) { this.listeners.set(type, listener); }
  dispatch(type, key) {
    this.listeners.get(type)({
      key, target: null, metaKey: false, ctrlKey: false, altKey: false, preventDefault() {},
    });
  }
}
const keysFor = { forward: "w", backup: "s", turnLeft: "a", turnRight: "d" };

// ---------------------------------------------------------------- tuning

// Anything at all can be handed to the model: a stored value from an older
// build, a hand-edited export, a field someone deleted. Unknown keys go away,
// missing ones fall back, out-of-range ones are clamped *and* snapped.
assert.deepEqual(normalisePadTune(null), { ...DEFAULT_PAD_TUNE });
assert.deepEqual(normalisePadTune("nonsense"), { ...DEFAULT_PAD_TUNE });
assert.deepEqual(normalisePadTune({ shape: "triangle" }).shape, "circle");
assert.deepEqual(normalisePadTune({ shape: "square" }).shape, "square");
assert.equal(normalisePadTune({ deadzone: -5 }).deadzone, 0);
assert.equal(normalisePadTune({ deadzone: 999 }).deadzone, 60);
assert.equal(normalisePadTune({ forwardEdge: 40 }).forwardEdge, 40, "2.5-degree grid");
assert.equal(normalisePadTune({ columnEdge: 999 }).columnEdge, 50);
assert.equal("extra" in normalisePadTune({ extra: 3 }), false);
assert.deepEqual(
  Object.keys(normalisePadTune({})).sort(),
  ["shape", ...PAD_TUNE_FIELDS.map((field) => field.key)].sort(),
);
assert.deepEqual(padTuneFields({ shape: "square" }, "sectors").map((f) => f.key),
  ["size", "columnEdge", "rowTopEdge", "rowBottomEdge"]);
assert.deepEqual(PAD_SHAPES, ["circle", "square"]);

// One control, one boundary. This is the whole point of tuning by boundary:
// moving a slider (or dragging a pin) must not rescale the neighbours, and a
// value that would cross its neighbour is clamped at the neighbour instead.
{
  const base = { ...DEFAULT_PAD_TUNE };
  const wide = setPadTuneField(base, "forwardEdge", 40);
  assert.equal(wide.forwardEdge, 40);
  assert.deepEqual(
    [wide.frontDiagonalEdge, wide.turnEdge, wide.backEdge], [67.5, 112.5, 157.5],
    "no other boundary moved",
  );
  assert.equal(setPadTuneField(base, "forwardEdge", 999).forwardEdge, 67.5, "clamped at its neighbour");
  assert.equal(setPadTuneField(base, "backEdge", 30).backEdge, 112.5, "clamped from the other side");
  assert.equal(setPadTuneField(base, "transition", 90).transition, 75, "the ramp shares the radius");
  assert.equal(setPadTuneField(base, "transition", 90).deadzone, 25, "and the centre is untouched");
  const square = setPadTuneField({ ...base, shape: "square" }, "rowTopEdge", 60);
  assert.equal(square.rowTopEdge, 60);
  assert.equal(square.rowBottomEdge, 67, "the other grid line stayed put");
  assert.equal(setPadTuneField({ ...base, shape: "square" }, "rowTopEdge", 90).rowTopEdge, 62,
    "rows keep a middle cell");
  // Nothing in a normalisation may produce a negative or inverted cell: a
  // hand-edited export with impossible boundaries comes back ordered, and the
  // widths still tile the circle.
  const squeezed = normalisePadTune({ forwardEdge: 10, frontDiagonalEdge: 0, turnEdge: 0, backEdge: 0 });
  const bounds = [squeezed.forwardEdge, squeezed.frontDiagonalEdge, squeezed.turnEdge, squeezed.backEdge];
  assert.ok(bounds.every((value, index) => index === 0 || value >= bounds[index - 1]), "ordered");
  assert.ok(bounds[0] >= 0 && bounds[3] <= 180, "inside the circle");
  const widths = padLayout(squeezed).sectors.map((sector) => sector.width);
  assert.ok(widths.every((width) => width >= 0));
  assert.equal(Math.round(widths.reduce((sum, width) => sum + width, 0)), 360);
}

// ---------------------------------------------------------------- circle

// The defaults reproduce the plain eight 45-degree wedges the scheme shipped
// with: the tuning surface is opt-in, not a new control.
{
  const layout = padLayout(DEFAULT_PAD_TUNE);
  assert.equal(layout.shape, "circle");
  assert.equal(layout.sectors.length, 8);
  for (const [index, sector] of layout.sectors.entries()) {
    assert.equal(sector.width, 45, `${sector.name} width`);
    assert.equal(wrap(sector.center), wrap(index * 45), `${sector.name} centre`);
  }
  assert.equal(layout.sectors.reduce((total, s) => total + s.width, 0), 360);
  assert.equal(layout.deadzone, 0.25);
  assert.equal(layout.ramp, 0.35);
  assert.equal(layout.full, 0.6);
}

// Dragging one boundary moves that boundary: the sector on each side of it
// changes width, and nothing else does.
{
  const base = padLayout(DEFAULT_PAD_TUNE);
  const moved = padLayout(setPadTuneField(DEFAULT_PAD_TUNE, "forwardEdge", 45));
  assert.equal(moved.sectors[0].width, 90, "the forward cone grew on both sides");
  assert.equal(moved.sectors[7].width, 22.5, "its neighbour gave up exactly that room");
  assert.equal(moved.sectors[1].width, 22.5);
  for (const index of [2, 3, 4, 5, 6]) {
    assert.equal(moved.sectors[index].width, base.sectors[index].width, `sector ${index} untouched`);
  }
  assert.equal(Math.round(moved.sectors.reduce((sum, s) => sum + s.width, 0)), 360);
}

// Left and right are mirror images for every setting; front and back are not.
{
  const layout = padLayout(setPadTuneField(DEFAULT_PAD_TUNE, "backEdge", 120));
  assert.equal(layout.sectors[0].width, 45);
  assert.equal(layout.sectors[4].width, 120, "a wide reverse cone");
  for (let index = 0; index < 8; index++) {
    const mirror = layout.sectors[(8 - index) % 8];
    assert.equal(layout.sectors[index].width, mirror.width, `mirror widths at ${index}`);
    assert.equal(wrap(layout.sectors[index].center + mirror.center), 0, `mirror centres at ${index}`);
  }
}

// The radial zones are fractions of the radius, and a ramp wider than the pad
// is simply full speed from the edge of the centre.
{
  const layout = padLayout({ deadzone: 30, transition: 90 });
  assert.equal(layout.deadzone, 0.3);
  assert.equal(layout.full, 1);
  assert.ok(Math.abs(layout.ramp - 0.7) < 1e-9);
  assert.equal(padLayout({ transition: 0 }).ramp, 0);
  assert.equal(padLayout({ transition: 0 }).full, padLayout({ transition: 0 }).deadzone);
}

// Dead centre is a real stop, and the ramp is linear from zero to full.
{
  const tune = { ...DEFAULT_PAD_TUNE, deadzone: 25, transition: 35 };
  for (const centre of [DEFAULT_PAD_TUNE, { deadzone: 0, transition: 0 }]) {
    const cell = padCell(0, 0, centre);
    assert.equal(cell.sector, null);
    assert.equal(cell.magnitude, 0, "an exactly-centred finger never pushes");
    assert.deepEqual(movementOf(cell), { forward: 0, backup: 0, turnLeft: 0, turnRight: 0 });
  }
  const half = padCell(0, -0.425, tune);
  assert.equal(half.sector, 0);
  assert.equal(half.ring, "ramp");
  assert.ok(Math.abs(half.magnitude - 0.5) < 1e-9);
  assert.equal(padCell(0, -0.6, tune).magnitude, 1);
  assert.equal(padCell(0, -0.6, tune).ring, "full");
  assert.equal(padCell(0, -1.8, tune).magnitude, 1, "dragging past the rim stays full");
  assert.equal(padCell(0, -0.25, tune).magnitude, 0);
  assert.equal(padCell(0, -0.24, tune).ring, "hub");
  const stepped = { ...DEFAULT_PAD_TUNE, deadzone: 25, transition: 0 };
  assert.equal(padCell(0, -0.3, stepped).magnitude, 1, "with no ramp, leaving the hub is full speed");
  assert.equal(padCell(0, -0.2, stepped).ring, "hub");
}

// The zone highlight may lag by a couple of percent so it does not flicker on
// the boundary, but the command itself never does.
{
  const tune = { ...DEFAULT_PAD_TUNE, deadzone: 25, transition: 35 };
  const held = padCell(0, -0.61, tune, { sector: 0, ring: "ramp" });
  assert.equal(held.ring, "ramp", "the picture stays where it was");
  assert.equal(held.magnitude, 1, "the command is already full strength");
  assert.equal(padCell(0, -0.7, tune, { sector: 0, ring: "ramp" }).ring, "full");
}

// Every direction presses its own key combination at full push — checked
// against the keyboard itself rather than restated.
for (const tune of [
  DEFAULT_PAD_TUNE,
  setPadTuneField(setPadTuneField(DEFAULT_PAD_TUNE, "forwardEdge", 40), "backEdge", 140),
]) {
  const layout = padLayout(tune);
  for (const direction of DIRECTIONS) {
    const sector = layout.sectors[DIRECTIONS.indexOf(direction)];
    assert.ok(sector.width > 0, `${direction.name} has an interior to push into`);
    const cell = padCell(...direct(sector.center), tune);
    assert.equal(cell.sector, sector.index, `${direction.name} sector`);
    assert.equal(cell.ring, "full");
    assert.equal(cell.magnitude, 1);
    assert.deepEqual(movementOf(cell), expected(direction.button), `${direction.name} buttons`);
    assert.equal(cell.targetRotation, null, "the sector scheme never asks for a heading");

    const target = new FakeTarget();
    const keyboard = new Keyboard(target, () => 0);
    for (const part of direction.button.split("+")) target.dispatch("keydown", keysFor[part]);
    assert.deepEqual(movementOf(keyboard.sampleStrengths()), movementOf(cell),
      `${direction.name} matches holding ${direction.button}`);
  }
}

// Which sector a finger is in follows the boundaries the player set, all the
// way around the circle and with no gap between neighbours.
{
  const tune = setPadTuneField(setPadTuneField(DEFAULT_PAD_TUNE, "forwardEdge", 45), "backEdge", 120);
  const layout = padLayout(tune);
  const marched = new Array(8).fill(0);
  for (let degrees = -180; degrees < 180; degrees += 0.5) {
    marched[padCell(...direct(degrees).map((v) => v * 0.8), tune).sector] += 0.5;
  }
  for (const sector of layout.sectors) {
    assert.ok(Math.abs(marched[sector.index] - sector.width) <= 0.51,
      `${sector.name}: swept ${marched[sector.index]}°, drawn ${sector.width}°`);
  }
}

// A held direction survives a finger resting on a divider, and is released once
// the divider is properly crossed — including on a sliver, where the sticky
// band shrinks with the sector.
{
  const tune = DEFAULT_PAD_TUNE;
  assert.equal(padCell(...direct(23), tune).sector, 1, "past the divider by a degree");
  assert.equal(padCell(...direct(23), tune, { sector: 0 }).sector, 0, "a held sector stays held");
  assert.equal(padCell(...direct(30), tune, { sector: 0 }).sector, 1, "released once crossed");
  assert.equal(padCell(...direct(25), tune, { sector: 1 }).sector, 1, "sticky on the way back");
  const sliver = padLayout(setPadTuneField(DEFAULT_PAD_TUNE, "forwardEdge", 2.5));
  assert.equal(sliver.sectors[0].sticky, 1.25, "sticky is capped by the sector width");
  assert.equal(padCell(...direct(1), sliver.tune).sector, 0);
  assert.equal(padCell(...direct(4), sliver.tune).sector, 1);
}

// ---------------------------------------------------------------- square

// Nine rectangles by default, mirrored left to right: a keyboard in a box, with
// no ramp at all.
{
  const tune = { shape: "square" };
  const layout = padLayout(tune);
  assert.equal(layout.shape, "square");
  assert.equal(layout.cells.length, 9);
  // Three (near) equal columns by default: the grid lines are whole percents,
  // so "a third" is 33%.
  assert.ok(Math.abs(layout.x0 - (0.33 * 2 - 1)) < 1e-9, "33% columns by default");
  const centre = padCell(0, 0, tune);
  assert.equal(centre.sector, null);
  assert.equal(centre.ring, "hub");
  assert.equal(centre.magnitude, 0);
  for (const direction of DIRECTIONS) {
    const cell = padCell(...direct(direction.compass).map((v) => v * 0.8), tune);
    assert.equal(cell.sector, DIRECTIONS.indexOf(direction), `${direction.name} cell`);
    assert.equal(cell.ring, "full");
    assert.equal(cell.magnitude, 1, "every cell is full strength");
    assert.deepEqual(movementOf(cell), expected(direction.button), `${direction.name} buttons`);
  }
}

// The grid lines are the parameters, and moving one does not move the rest.
{
  const wide = padLayout({ shape: "square", rowTopEdge: 10, rowBottomEdge: 20 });
  assert.equal(padCell(0, -0.9, wide.tune).sector, 0, "the top row is now thick");
  assert.equal(padCell(0, -0.7, wide.tune).sector, null, "with a shallow middle row");
  assert.equal(padCell(0, 0.8, wide.tune).sector, 4, "and a thick bottom row");
  const offCentre = setPadTuneField({ shape: "square" }, "rowTopEdge", 20);
  assert.equal(offCentre.rowTopEdge, 20);
  assert.equal(offCentre.rowBottomEdge, 67, "the other line stayed put");
  assert.equal(offCentre.columnEdge, 33, "and so did the columns");
  const narrow = setPadTuneField({ shape: "square" }, "columnEdge", 10);
  assert.deepEqual(
    [padCell(-0.9, 0, narrow).sector, padCell(0, 0, narrow).sector, padCell(0.9, 0, narrow).sector],
    [6, null, 2],
  );
}

// A finger resting on a grid line keeps the cell it had, like the circle.
{
  const tune = { shape: "square" };
  const rightEdge = -padLayout(tune).x0;
  assert.equal(padCell(0.9, 0, tune).sector, 2, "the middle of the right column");
  assert.equal(padCell(rightEdge + 0.01, 0, tune, { sector: 2 }).sector, 2, "still inside");
  assert.equal(padCell(rightEdge - 0.01, 0, tune, { sector: 2 }).sector, 2, "just outside, but held");
  assert.equal(padCell(rightEdge - 0.2, 0, tune, { sector: 2 }).sector, null, "well away: switches");
}

// ---------------------------------------------------------------- artwork

// The picture is generated from the same layout the input path reads, so it
// cannot describe a pad the player is not actually holding.
for (const shape of PAD_SHAPES) {
  const art = padArtwork({ shape });
  const count = (pattern) => (art.match(pattern) ?? []).length;
  assert.equal(count(/class="pad-cell"/g), shape === "square" ? 9 : 17,
    `${shape}: every region the input can resolve`);
  assert.equal(count(/NaN|undefined/g), 0, `${shape}: no unresolved numbers`);
  for (let index = 0; index < 8; index++) {
    assert.ok(art.includes(`class="pad-glyph" data-dir="${index}"`), `${shape} glyph ${index}`);
    assert.ok(art.includes(`data-dir="${index}"`), `${shape} direction ${index}`);
  }
  if (shape === "circle") {
    assert.equal(count(/class="pad-edge"/g), 16, "two edges per direction");
    assert.equal(count(/class="pad-divider"/g), 8, "eight boundaries");
    assert.equal(count(/class="pad-ring"/g), 2, "hub and ramp circles");
    assert.equal(count(/class="pad-rim"/g), 1);
  } else {
    assert.equal(count(/class="pad-edge"/g), 8, "one outline per direction");
    assert.equal(count(/class="pad-grid"|<line/g) > 0, true);
  }
}

// The drawn hub is the real dead centre, and it moves with the setting.
{
  const radiusOf = (markup) => Number(/data-band="hub" r="([\d.]+)"/.exec(markup)[1]);
  assert.ok(Math.abs(radiusOf(padArtwork({ deadzone: 50 })) - 0.5 * PAD_UNITS) < 0.02);
  assert.ok(Math.abs(radiusOf(padArtwork(DEFAULT_PAD_TUNE))
    - (DEFAULT_PAD_TUNE.deadzone / 100) * PAD_UNITS) < 0.02);
}

// Handles: one per boundary, sitting on that boundary, and a drag on one
// implies exactly that boundary's value.
{
  const pins = padPins(DEFAULT_PAD_TUNE);
  assert.deepEqual(pins.map((pin) => pin.field), [
    "forwardEdge", "forwardEdge",
    "frontDiagonalEdge", "frontDiagonalEdge",
    "turnEdge", "turnEdge",
    "backEdge", "backEdge",
  ]);
  // A handle still sits exactly on the boundary it moves: same angle, and its
  // distance from the middle is a radius along that same line.
  const onBoundary = (pin, angle) => {
    const dx = pin.left - 50;
    const dy = pin.top - 50;
    const bearing = (Math.atan2(dx, -dy) * 180) / Math.PI;
    const radius = Math.hypot(dx, dy) / 50;
    return Math.abs(wrap(bearing - angle)) < 1e-6 && radius > 0.4 && radius < 0.9;
  };
  assert.ok(onBoundary(pins[0], DEFAULT_PAD_TUNE.forwardEdge), "the right handle is on the boundary");
  assert.ok(onBoundary(pins[1], -DEFAULT_PAD_TUNE.forwardEdge), "and so is its mirror");
  assert.ok(onBoundary(pins[2], DEFAULT_PAD_TUNE.frontDiagonalEdge));
  assert.ok(onBoundary(pins[4], DEFAULT_PAD_TUNE.turnEdge));
  assert.ok(onBoundary(pins[6], DEFAULT_PAD_TUNE.backEdge));
  // Neighbouring boundaries are staggered, so two handles never sit on top of
  // each other — including the two ends of one narrow sector.
  // Handles must not overlap on a real pad: percentages of the pad, at the
  // default size, have to come out wider than the handle the CSS draws.
  const spacing = (a, b) => Math.hypot(a.left - b.left, a.top - b.top);
  const gapPx = (a, b) => (spacing(a, b) / 100) * DEFAULT_PAD_TUNE.size;
  const pinPx = Math.round(Math.max(14, Math.min(26, DEFAULT_PAD_TUNE.size * 0.15)));
  let closest = null;
  for (let index = 0; index < 8; index++) {
    for (let other = index + 1; other < 8; other++) {
      const gap = gapPx(pins[index], pins[other]);
      if (closest === null || gap < closest) closest = gap;
    }
  }
  assert.ok(closest > pinPx + 4, `handles stay apart: closest pair ${closest.toFixed(1)}px vs ${pinPx}px`);
  // The two ends of one boundary are staggered too, so even a two-degree cone
  // gets two separate handles instead of one smudge on the nose.
  const cone = padPins(setPadTuneField(DEFAULT_PAD_TUNE, "forwardEdge", 2.5));
  assert.ok(gapPx(cone[0], cone[1]) > pinPx + 2,
    `a narrow cone's handles stay apart: ${gapPx(cone[0], cone[1]).toFixed(1)}px`);
  const squarePins = padPins({ shape: "square" });
  assert.deepEqual(squarePins.map((pin) => pin.field), [
    "columnEdge", "columnEdge", "rowTopEdge", "rowBottomEdge",
  ]);
  assert.equal(squarePins[0].left, 33);
  assert.equal(squarePins[1].left, 67);
  assert.notEqual(squarePins[0].top, squarePins[1].top, "the two column handles are staggered");
  assert.notEqual(squarePins[2].left, squarePins[3].left, "and so are the row handles");

  // Dragging a handle: the value comes from where the pointer is, and either
  // side of a mirrored pair writes the same field.
  const angle = padPinValue("forwardEdge", ...direct(30));
  assert.ok(Math.abs(angle - 30) < 1e-9);
  assert.ok(Math.abs(padPinValue("forwardEdge", ...direct(-30)) - 30) < 1e-9);
  assert.ok(Math.abs(padPinValue("forwardEdge", ...direct(210)) - 150) < 1e-9);
  assert.equal(padPinValue("columnEdge", 0, 0), 50);
  assert.equal(padPinValue("rowTopEdge", 0, -1), 0);
  assert.equal(padPinValue("rowBottomEdge", 0, 0), 50);
  // ...and that it is the same field a slider would have written.
  const dragged = setPadTuneField({ shape: "square" }, "rowTopEdge", padPinValue("rowTopEdge", 0, -0.4));
  assert.equal(dragged.rowTopEdge, 30);
  assert.equal(dragged.rowBottomEdge, 67, "dragging one line leaves the others alone");
}

// ---------------------------------------------------------------- scheme

// resolveMovement is the seam the renderer and the physics tick share, so it
// gets exercised without a DOM: only the fields it reads are faked.
const idle = { forward: 0, backup: 0, turnLeft: 0, turnRight: 0, fire: 0 };
function resolve(overrides, keyboardStrengths = idle, rotation = 90) {
  const pad = {
    scheme: "sectors",
    joystickPointer: 1,
    joystickVector: { x: 0, y: -0.9 },
    tune: DEFAULT_PAD_TUNE,
    forwardAlignmentDegrees: DEFAULT_FORWARD_ALIGNMENT_DEGREES,
    sector: null,
    ring: null,
    ...overrides,
  };
  return TouchControls.prototype.resolveMovement.call(pad, keyboardStrengths, rotation);
}

// This is the whole point of the scheme: pushing "up" is forward, whatever
// direction the hull happens to face. The wheel, pointed the same way while
// facing east, pivots the hull to north and drives while it does — steering the
// player never had to time.
assert.deepEqual(movementOf(resolve({})), expected("forward"));
assert.deepEqual(
  movementOf(resolve({ scheme: "wheel" })),
  { forward: 1, backup: 0, turnLeft: 1, turnRight: 0 },
);
assert.deepEqual(movementOf(resolve({ joystickVector: { x: -0.9, y: 0 } })), expected("turnLeft"));
assert.deepEqual(movementOf(resolve({ joystickVector: { x: 0, y: 0 } })), {
  forward: 0, backup: 0, turnLeft: 0, turnRight: 0,
});

// Both shapes reach the input path, and so does the tuning.
assert.deepEqual(
  movementOf(resolve({ joystickVector: { x: -0.8, y: -0.8 }, tune: { shape: "square" } })),
  expected("forward+turnLeft"),
);
assert.deepEqual(
  movementOf(resolve({ joystickVector: { x: 0, y: -0.1 }, tune: { shape: "square" } })),
  { forward: 0, backup: 0, turnLeft: 0, turnRight: 0 },
  "the square's stop cell is a real stop",
);
assert.deepEqual(
  movementOf(resolve({ joystickVector: { x: 0, y: -0.3 }, tune: { deadzone: 50 } })),
  { forward: 0, backup: 0, turnLeft: 0, turnRight: 0 },
);
assert.deepEqual(
  movementOf(resolve({
    joystickVector: { x: 0.5, y: -0.5 },
    tune: setPadTuneField(DEFAULT_PAD_TUNE, "forwardEdge", 2.5),
  })),
  expected("forward+turnRight"),
  "a wide forward cone swallows the diagonal",
);

// Releasing the stick (or never touching it) hands control back to the keys.
assert.deepEqual(movementOf(resolve({ joystickPointer: null })), {
  forward: 0, backup: 0, turnLeft: 0, turnRight: 0,
});
assert.deepEqual(
  movementOf(resolve(
    { joystickPointer: null },
    { forward: 0.5, backup: 0, turnLeft: 0, turnRight: 0.25, fire: 0 },
  )),
  { forward: 0.5, backup: 0, turnLeft: 0, turnRight: 0.25 },
);
assert.deepEqual(
  movementOf(resolve(
    { joystickPointer: null },
    { forward: 0, backup: 0.4, turnLeft: 0.4, turnRight: 0, fire: 1 },
  )),
  { forward: 0, backup: 0.4, turnLeft: 0.4, turnRight: 0 },
  "fire is never part of movement",
);

// The pad resolves to one of the two schemes, and anything stored by an older
// build falls back to the new default instead of reaching the stick maths.
assert.deepEqual(TOUCH_SCHEMES, ["sectors", "pads", "wheel"]);
assert.equal(DEFAULT_TOUCH_SCHEME, "sectors");
assert.equal(normaliseTouchScheme("wheel"), "wheel");
assert.equal(normaliseTouchScheme("sectors"), "sectors");
assert.equal(normaliseTouchScheme(null), "sectors");
assert.equal(normaliseTouchScheme("pads"), "pads");
assert.equal(normaliseTouchScheme("dpad"), "sectors");


// ---------------------------------------------------------------- pads

// The three-finger scheme: two digital paddles and a trigger, each placed by
// its own pair of percentages, and a full press is still exactly the keyboard.
{
  const layout = padsLayout(DEFAULT_PAD_TUNE);
  assert.equal(layout.scheme, "pads");
  assert.deepEqual(
    [layout.move.x, layout.move.y, layout.turn.x, layout.turn.y, layout.fire.x, layout.fire.y],
    [22, 74, 78, 74, 78, 38],
    "forward/reverse on the left, turn on the right, trigger top-right",
  );
  assert.deepEqual(
    [layout.move.size, layout.turn.size, layout.fire.size],
    [DEFAULT_PAD_TUNE.moveSize, DEFAULT_PAD_TUNE.turnSize, DEFAULT_PAD_TUNE.fireSize],
  );
  assert.deepEqual(
    [layout.move.gap, layout.turn.gap],
    [DEFAULT_PAD_TUNE.moveGap, DEFAULT_PAD_TUNE.turnGap],
  );
  // The two paddles are separate: resizing one leaves the other alone, and so
  // does spacing one differently.
  const resized = setPadTuneField(DEFAULT_PAD_TUNE, "turnSize", 100);
  assert.deepEqual(
    [resized.turnSize, resized.moveSize, resized.turnGap, resized.moveGap], [100, 64, 8, 8],
  );
  const spaced = setPadTuneField(DEFAULT_PAD_TUNE, "moveGap", 0);
  assert.deepEqual([spaced.moveGap, spaced.turnGap, spaced.moveSize], [0, 8, 64]);
  assert.equal(setPadTuneField(DEFAULT_PAD_TUNE, "moveGap", 999).moveGap, 48);
  assert.equal(setPadTuneField(DEFAULT_PAD_TUNE, "turnSize", 4).turnSize, 40);
  assert.equal(setPadTuneField(DEFAULT_PAD_TUNE, "moveGap", 20).moveX, 22, "and nothing else moves");
  assert.deepEqual(
    padTuneFields(DEFAULT_PAD_TUNE, "pads").map((field) => field.key),
    ["moveSize", "moveGap", "turnSize", "turnGap", "fireSize",
      "moveX", "moveY", "turnX", "turnY", "fireX", "fireY"],
  );
  assert.deepEqual(
    padTuneFields(DEFAULT_PAD_TUNE, "sectors").map((field) => field.key),
    ["size", "deadzone", "transition", "forwardEdge", "frontDiagonalEdge", "turnEdge", "backEdge"],
    "the eight-way pad keeps its own list",
  );

  // Moving one placed pad leaves the others alone, and every position stays
  // inside the layer.
  const moved = setPadTuneField(DEFAULT_PAD_TUNE, "moveX", 40);
  assert.equal(moved.moveX, 40);
  assert.deepEqual([moved.turnX, moved.fireX, moved.moveSize], [78, 78, 64]);
  assert.equal(setPadTuneField(DEFAULT_PAD_TUNE, "moveX", 200).moveX, 95);
  assert.equal(setPadTuneField(DEFAULT_PAD_TUNE, "fireY", -10).fireY, 5);
  assert.equal(setPadTuneField(DEFAULT_PAD_TUNE, "moveSize", 4).moveSize, 40);
}

// The buttons resolve to the same four strengths the keyboard does: full while
// held, nothing when not, and each axis on its own.
{
  const held = (actions) => {
    const pad = {
      scheme: "pads",
      heldKeys: new Set(actions),
      keyPointers: new Map(),
      tune: DEFAULT_PAD_TUNE,
      joystickPointer: null,
      joystickVector: { x: 0, y: 0 },
      forwardAlignmentDegrees: DEFAULT_FORWARD_ALIGNMENT_DEGREES,
      sector: null,
      ring: null,
    };
    return TouchControls.prototype.resolveMovement.call(pad, idle, 90);
  };
  assert.deepEqual(movementOf(held(["forward"])), expected("forward"));
  assert.deepEqual(movementOf(held(["backup"])), expected("backup"));
  assert.deepEqual(movementOf(held(["turnLeft"])), expected("turnLeft"));
  assert.deepEqual(movementOf(held(["turnRight"])), expected("turnRight"));
  assert.deepEqual(movementOf(held(["forward", "turnLeft"])), expected("forward+turnLeft"));
  assert.deepEqual(movementOf(held([])), { forward: 0, backup: 0, turnLeft: 0, turnRight: 0 });
  // Both thumbs at once is the same as the two keys at once.
  const keyboard = new Keyboard(new FakeTarget(), () => 0);
  keyboard.pressed.add("w");
  keyboard.pressed.add("a");
  assert.deepEqual(movementOf(keyboard.sampleStrengths()), movementOf(held(["forward", "turnLeft"])));
  // The pads add to the keyboard rather than replacing it, and never double up.
  assert.deepEqual(
    movementOf(TouchControls.prototype.resolveMovement.call({
      scheme: "pads", heldKeys: new Set(["forward"]), keyPointers: new Map(),
      tune: DEFAULT_PAD_TUNE, joystickPointer: null, joystickVector: { x: 0, y: 0 },
      forwardAlignmentDegrees: DEFAULT_FORWARD_ALIGNMENT_DEGREES, sector: null, ring: null,
    }, { forward: 1, backup: 0, turnLeft: 0, turnRight: 0, fire: 0 }, 0)),
    expected("forward"),
  );
}

console.log("eight-direction stick, both shapes, three-finger pads OK");

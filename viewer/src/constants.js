/**
 * Render/input constants mirrored from the Rust engine (engine/src/*.rs) and
 * from killfield/src/constants.js. Only the subset the ported viewer logic
 * actually touches — the rest of the physics lives in the wasm module, not
 * here.
 */

// ---- frame rate ----
export const FPS = 25;

// ---- canvas logical footprint (fixed; the maze's scale is chosen so it
// always fits inside this box — see engine/src/game.rs's scale calc) ----
export const MOVIEWIDTH = 692;
export const MOVIEHEIGHT = 480;

// ---- ballistics (used only for the bullet-interpolation jump guard) ----
export const BULLETSPEED = 4.5; // px/frame at SCALE=50

// ---- tank steering (used by input.js's 128-direction joystick math) ----
export const TANK_TURN_SPEED = 10; // deg/frame

// ---- tank geometry, in local sprite units (render-only) ----
// Rotation 0 points UP (−y). The barrel fires along (rotation − 90)°.
export const TANK_BASE_WIDTH = 61.0;
export const TANK_BASE_HEIGHT = 81.0;
export const TANK_SHAPE_BARREL_HALF_WIDTH = 8.5;
export const TANK_SHAPE_BARREL_TIP_Y = -55.0;

export const DEG = Math.PI / 180.0;

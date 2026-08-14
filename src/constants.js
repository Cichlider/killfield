/**
 * Engine constants.
 *
 * The simulation is a fixed 25 FPS maze tank duel with ricocheting bullets.
 * Every physical quantity below is expressed at a reference cell size of
 * SCALE = 50 and multiplied by (scale / 50) at runtime, because the cell size
 * is re-derived from the maze dimensions every round.
 */

// ---- frame rate ----
export const FPS = 25;

// ---- playfield layout ----
export const MOVIEWIDTH = 692;
export const MOVIEHEIGHT = 480;
export const HEIGHTTOBOTTOM = 80;

// ---- bullets ----
export const STARTWEAPON = "bullet";
export const BULLETSPEED = 4.5; // px/frame at SCALE=50
export const BULLETLIFETIME = 250; // frames (10 s)
export const BULLETHITCHECKINTERVALS = 7; // substeps per frame
export const BULLETDEADLY = 0; // lethal from the muzzle, including to the shooter

// Referenced by the AI's dodge logic even though these weapons never spawn
// in the duel mode (the active-weapon list is empty).
export const FRAGSPEED = 4.5;
export const GATLINGSPEED = 5.5;

// ---- crates (never spawn in duel mode, but the timer still consumes RNG) ----
export const CRATESPAWNTIMEBASE = 350;
export const CRATESPAWNTIMERANDOM = 200;
export const CRATESPAWNMAZESIZESCALE = 2000;

// ---- round lifecycle ----
export const NUMBEROFFRAMESBEFOREEND = 125; // frames the world keeps running after a kill
export const NUMBEROFFRAMESFROZEN = 50; // freeze + score when endCount reaches this
export const NUMBEROFFRAMESBEFORERESET = 5; // gap between cleanup and the next round

// The gap between the two above is the residual-bullet settlement window:
// 125 - 50 = 75 frames (3 s) in which the apparent winner can still be killed
// by a bullet already in the air.
export const SETTLEMENT_FRAMES = NUMBEROFFRAMESBEFOREEND - NUMBEROFFRAMESFROZEN;

// ---- visual effects ----
export const MAXSHAKE = 8;

// ---- pathfinding ----
export const MAXDEADENDPENALTY = 5;

// ---- settings ----
export const SETTINGS_MAX_BULLETS = 5;
export const SETTINGS_MAX_CRATES = 3;
export const SETTINGS_CRATE_SPAWN_MODIFIER = 1;

// ---- tank physics ----
export const TANK_FORWARD_SPEED_BASE = 4.0; // × (scale/50) px/frame
export const TANK_BACKUP_SPEED_BASE = 2.5; // × (scale/50) px/frame
export const TANK_TURN_SPEED = 10; // deg/frame
export const TANK_MOVE_STEPS = 5; // substeps per frame
// Wall contact is resolved at 5-substep precision. The blocked normal is
// removed and the tangent is retained with more drag at steeper incidence.
export const TANK_WALL_SLIDE_MIN_RETENTION = 0.70;
export const TANK_WALL_SLIDE_MAX_RETENTION = 0.96;
export const TANK_WALL_SLIDE_INCIDENCE_DRAG = 0.30;
export const TANK_WALL_ALIGN_SPEED = 2.0; // maximum contact-induced deg/frame

// ---- tank geometry, in local sprite units ----
// Rotation 0 points UP (−y). The barrel fires along (rotation − 90)°.
export const TANK_BASE_WIDTH = 61.0;
export const TANK_BASE_HEIGHT = 81.0;
export const TANK_TURRET_WIDTH = 45.0;
export const TANK_TURRET_HEIGHT = 77.5;
export const TANK_DISPLAY_SCALE_FACTOR = 0.55 / 100.0; // × scale

// Union bounds of the whole tank, used for the cheap bounding-box pre-test.
export const TANK_BOUNDS_LOCAL = [-30.5, -55.0, 30.5, 40.5];

// Wall collision probe points at the barrel tip.
export const TANK_BARREL_HALF_WIDTH = TANK_TURRET_WIDTH / 6.0; // 7.5
export const TANK_BARREL_TIP_Y = (-TANK_TURRET_HEIGHT / 16.0) * 11.0; // -53.28125

// Bullet-vs-tank hit shape: the base rectangle union the barrel rectangle.
// The turret dome is entirely inside the base rectangle, so it adds nothing.
export const TANK_SHAPE_BARREL_HALF_WIDTH = 8.5;
export const TANK_SHAPE_BARREL_TIP_Y = -55.0;

// Render-only. Bullets are treated as dimensionless points by the hit test.
export const BULLET_VISUAL_RADIUS = 3.5;

export const DEG = Math.PI / 180.0;

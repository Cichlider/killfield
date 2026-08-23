//! Port of `killfield/src/constants.js`.
//!
//! The simulation is a fixed 25 FPS maze tank duel with ricocheting bullets.
//! Every physical quantity is expressed at a reference cell size of
//! SCALE = 50 and multiplied by (scale / 50) at runtime, because the cell
//! size is re-derived from the maze dimensions every round.

// ---- frame rate ----
pub const FPS: i32 = 25;

// ---- playfield layout ----
pub const MOVIEWIDTH: f64 = 692.0;
pub const MOVIEHEIGHT: f64 = 480.0;
pub const HEIGHTTOBOTTOM: f64 = 80.0;

// ---- bullets ----
pub const BULLETSPEED: f64 = 4.5; // px/frame at SCALE=50
pub const BULLETLIFETIME: i32 = 250; // frames (10 s)
pub const BULLETHITCHECKINTERVALS: i32 = 7; // substeps per frame
pub const BULLETDEADLY: i32 = 0;

// Referenced by the AI's dodge logic even though these weapons never spawn
// in duel mode (the active-weapon list is empty).
pub const FRAGSPEED: f64 = 4.5;
pub const GATLINGSPEED: f64 = 5.5;

// ---- crates (never spawn in duel mode, but the timer still consumes RNG) ----
pub const CRATESPAWNTIMEBASE: f64 = 350.0;
pub const CRATESPAWNTIMERANDOM: i32 = 200;
pub const CRATESPAWNMAZESIZESCALE: f64 = 2000.0;

// ---- round lifecycle ----
pub const NUMBEROFFRAMESBEFOREEND: i32 = 125; // world keeps running after a kill
pub const NUMBEROFFRAMESFROZEN: i32 = 50; // freeze + score at this endCount
pub const NUMBEROFFRAMESBEFORERESET: i32 = 5;

/// Residual-bullet settlement window: 125 - 50 = 75 frames (3 s) in which the
/// apparent winner can still be killed by a bullet already in the air.
pub const SETTLEMENT_FRAMES: i32 = NUMBEROFFRAMESBEFOREEND - NUMBEROFFRAMESFROZEN;

// ---- visual effects ----
pub const MAXSHAKE: f64 = 8.0;

// ---- pathfinding ----
pub const MAXDEADENDPENALTY: f64 = 5.0;

// ---- settings ----
pub const SETTINGS_MAX_BULLETS: usize = 5;
pub const SETTINGS_MAX_CRATES: usize = 3;
pub const SETTINGS_CRATE_SPAWN_MODIFIER: f64 = 1.0;

// ---- tank physics ----
pub const TANK_FORWARD_SPEED_BASE: f64 = 4.0; // x (scale/50) px/frame
pub const TANK_BACKUP_SPEED_BASE: f64 = 2.5; // x (scale/50) px/frame
pub const TANK_TURN_SPEED: f64 = 10.0; // deg/frame
pub const TANK_MOVE_STEPS: i32 = 5; // substeps per frame

// Wall contact is resolved at 5-substep precision. The blocked normal is
// removed and the tangent is retained with more drag at steeper incidence.
pub const TANK_WALL_SLIDE_MIN_RETENTION: f64 = 0.70;
pub const TANK_WALL_SLIDE_MAX_RETENTION: f64 = 0.96;
pub const TANK_WALL_SLIDE_INCIDENCE_DRAG: f64 = 0.30;
pub const TANK_WALL_ALIGN_SPEED: f64 = 2.0; // max contact-induced deg/frame

// A turn beside a wall can put one probe a fraction of a pixel inside the
// stroke even though shifting the hull slightly outward would make it valid.
pub const TANK_WALL_SEPARATION_BASE: f64 = 1.0; // px at reference scale
pub const TANK_WALL_SEPARATION_STEPS: i32 = 5;

// ---- tank geometry, in local sprite units ----
// Rotation 0 points UP (-y). The barrel fires along (rotation - 90) deg.
pub const TANK_BASE_WIDTH: f64 = 61.0;
pub const TANK_BASE_HEIGHT: f64 = 81.0;
pub const TANK_TURRET_WIDTH: f64 = 45.0;
pub const TANK_TURRET_HEIGHT: f64 = 77.5;
pub const TANK_DISPLAY_SCALE_FACTOR: f64 = 0.55 / 100.0; // x scale

/// Union bounds of the whole tank, for the cheap bounding-box pre-test.
pub const TANK_BOUNDS_LOCAL: [f64; 4] = [-30.5, -55.0, 30.5, 40.5];

// Wall collision probe points at the barrel tip.
pub const TANK_BARREL_HALF_WIDTH: f64 = TANK_TURRET_WIDTH / 6.0; // 7.5
pub const TANK_BARREL_TIP_Y: f64 = (-TANK_TURRET_HEIGHT / 16.0) * 11.0; // -53.28125

// Bullet-vs-tank hit shape: base rectangle union barrel rectangle.
// The turret dome is entirely inside the base rectangle, so it adds nothing.
pub const TANK_SHAPE_BARREL_HALF_WIDTH: f64 = 8.5;
pub const TANK_SHAPE_BARREL_TIP_Y: f64 = -55.0;

// Render-only. Bullets are dimensionless points to the hit test.
pub const BULLET_VISUAL_RADIUS: f64 = 3.5;

pub const DEG: f64 = std::f64::consts::PI / 180.0;

//! Deterministic world-direction controller shared by PPO and the web gamepad.

use crate::game::Game;

pub const DIRECTION_COUNT: usize = 128;
pub const MOVEMENT_HEAD_DIM: usize = DIRECTION_COUNT + 1;
pub const FIRE_HEAD_DIM: usize = 2;
pub const STOP: u16 = DIRECTION_COUNT as u16;
pub const NO_FIRE: u8 = 0;
pub const FIRE: u8 = 1;
pub const DIRECTION_STEP_DEGREES: f64 = 360.0 / DIRECTION_COUNT as f64;
const HUMAN_REVERSE_START_DEGREES: f64 = 135.0;

/// Movement indices are 128 clockwise world headings in 2.8125-degree
/// increments beginning at north, followed by STOP.
/// The tank never reverses: it turns along the shortest arc and only drives
/// forward once its hull is precisely aligned. The final turn uses a fractional
/// strength so all 128 headings are physically reachable despite 10° max turn.
pub fn apply_direction(game: &mut Game, tank: usize, movement: u16, fire: u8) {
    let movement = movement.min(STOP);
    let t = &mut game.tanks[tank];
    t.forward = false;
    t.backup = false;
    t.turn_left = false;
    t.turn_right = false;
    t.fire = fire == FIRE;
    t.forward_amount = None;
    t.backup_amount = None;
    t.turn_left_amount = None;
    t.turn_right_amount = None;

    if movement == STOP {
        return;
    }
    let desired = movement as f64 * DIRECTION_STEP_DEGREES;
    let error = (desired - t.rotation + 180.0).rem_euclid(360.0) - 180.0;
    if error.abs() <= 1e-6 {
        t.forward = true;
    } else if error > 0.0 {
        t.turn_right = true;
        t.turn_right_amount = Some((error.abs() / t.turn_speed).min(1.0));
    } else {
        t.turn_left = true;
        t.turn_left_amount = Some((error.abs() / t.turn_speed).min(1.0));
    }
}

/// Human world-direction control used by the browser wheel.
///
/// Unlike PPO's forward-only contract, the player moves immediately while
/// aligning the nose across a 270-degree sector. Only the 90-degree sector
/// centred directly behind the hull aligns the rear and reverses. The final
/// steering frame is proportional so the hull does not oscillate around a
/// requested heading.
pub fn apply_human_direction(game: &mut Game, tank: usize, movement: u16, fire: u8) {
    let movement = movement.min(STOP);
    let t = &mut game.tanks[tank];
    t.forward = false;
    t.backup = false;
    t.turn_left = false;
    t.turn_right = false;
    t.fire = fire == FIRE;
    t.forward_amount = Some(0.0);
    t.backup_amount = Some(0.0);
    t.turn_left_amount = Some(0.0);
    t.turn_right_amount = Some(0.0);

    if movement == STOP {
        return;
    }

    let desired = movement as f64 * DIRECTION_STEP_DEGREES;
    let nose_error = (desired - t.rotation + 180.0).rem_euclid(360.0) - 180.0;
    // Half-open [135°, 225°) rear sector: exactly 32 of the 128
    // quantised world headings select reverse.
    let backward = nose_error >= HUMAN_REVERSE_START_DEGREES
        || nose_error < -HUMAN_REVERSE_START_DEGREES;
    let forward = !backward;
    let alignment = if forward { desired } else { (desired + 180.0) % 360.0 };
    let error = (alignment - t.rotation + 180.0).rem_euclid(360.0) - 180.0;
    let turn_strength = (error.abs() / t.turn_speed).min(1.0);

    if forward {
        t.forward = true;
        t.forward_amount = Some(1.0);
    } else {
        t.backup = true;
        t.backup_amount = Some(1.0);
    }
    if error < 0.0 {
        t.turn_left = true;
        t.turn_left_amount = Some(turn_strength);
    } else if error > 0.0 {
        t.turn_right = true;
        t.turn_right_amount = Some(turn_strength);
    }
}

/// Compact ABI transport only. The policy still has two independent heads.
#[inline]
pub fn pack_action(movement: u16, fire: u8) -> u16 {
    movement.min(STOP) * FIRE_HEAD_DIM as u16 + fire.min(1) as u16
}

#[inline]
pub fn unpack_action(action: u16) -> (u16, u8) {
    let action = action.min((MOVEMENT_HEAD_DIM * FIRE_HEAD_DIM - 1) as u16);
    (action / FIRE_HEAD_DIM as u16, (action % FIRE_HEAD_DIM as u16) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_round_trips_all_head_pairs() {
        for movement in 0..MOVEMENT_HEAD_DIM as u16 {
            for fire in 0..FIRE_HEAD_DIM as u8 {
                assert_eq!(unpack_action(pack_action(movement, fire)), (movement, fire));
            }
        }
    }

    #[test]
    fn direction_controller_never_backs_up() {
        let mut game = Game::with_ai(123, 2, &[]);
        game.tanks[0].rotation = 180.0;
        apply_direction(&mut game, 0, 0, NO_FIRE);
        assert!(!game.tanks[0].backup);
        assert!(game.tanks[0].turn_left || game.tanks[0].turn_right);
        apply_direction(&mut game, 0, STOP, FIRE);
        assert!(!game.tanks[0].backup);
        assert!(game.tanks[0].fire);
    }

    #[test]
    fn direction_controller_can_reach_sub_degree_lattice() {
        let mut game = Game::with_ai(123, 2, &[]);
        game.tanks[0].rotation = 0.0;
        apply_direction(&mut game, 0, 1, NO_FIRE);
        assert!(game.tanks[0].turn_right);
        let strength = game.tanks[0].turn_right_amount.unwrap();
        assert!((strength * game.tanks[0].turn_speed - DIRECTION_STEP_DEGREES).abs() < 1e-9);
        assert_eq!(unpack_action(pack_action(STOP, FIRE)), (STOP, FIRE));
    }

    #[test]
    fn human_controller_uses_a_270_forward_and_90_reverse_split() {
        let mut game = Game::with_ai(123, 2, &[]);
        game.tanks[0].rotation = 0.0;

        apply_human_direction(&mut game, 0, 64, NO_FIRE);
        assert!(game.tanks[0].backup);
        assert!(!game.tanks[0].forward);
        assert!(!game.tanks[0].turn_left);
        assert!(!game.tanks[0].turn_right);

        apply_human_direction(&mut game, 0, 65, FIRE);
        assert!(game.tanks[0].backup);
        assert!(game.tanks[0].turn_right);
        assert!(game.tanks[0].fire);

        // 225 degrees is the first heading outside the half-open rear sector.
        apply_human_direction(&mut game, 0, 80, NO_FIRE);
        assert!(game.tanks[0].forward);
        assert!(game.tanks[0].turn_left);
    }
}

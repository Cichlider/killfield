//! Deterministic world-direction controller shared by PPO and the web gamepad.

use crate::game::Game;

pub const MOVEMENT_HEAD_DIM: usize = 17;
pub const FIRE_HEAD_DIM: usize = 2;
pub const STOP: u8 = 16;
pub const NO_FIRE: u8 = 0;
pub const FIRE: u8 = 1;

/// Movement indices are sixteen clockwise world headings in 22.5-degree
/// increments beginning at north, followed by STOP.
/// The tank never reverses: it turns along the shortest arc and only drives
/// forward once its hull is within ten degrees of the desired world heading.
pub fn apply_direction(game: &mut Game, tank: usize, movement: u8, fire: u8) {
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
    let desired = movement as f64 * 22.5;
    let error = (desired - t.rotation + 180.0).rem_euclid(360.0) - 180.0;
    if error.abs() <= 10.0 {
        t.forward = true;
    } else if error > 0.0 {
        t.turn_right = true;
    } else {
        t.turn_left = true;
    }
}

/// Human world-direction control used by the browser wheel.
///
/// Unlike PPO's forward-only contract, the player moves immediately while
/// aligning the nearer end of the hull: the nose drives toward targets in the
/// front hemisphere, while the rear aligns and reverses toward targets behind.
/// The final steering frame is proportional so the hull does not oscillate
/// around a 22.5-degree heading.
pub fn apply_human_direction(game: &mut Game, tank: usize, movement: u8, fire: u8) {
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

    let desired = movement as f64 * 22.5;
    let nose_error = (desired - t.rotation + 180.0).rem_euclid(360.0) - 180.0;
    let forward = nose_error.abs() <= 90.0;
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
pub fn pack_action(movement: u8, fire: u8) -> u8 {
    movement.min(STOP) * FIRE_HEAD_DIM as u8 + fire.min(1)
}

#[inline]
pub fn unpack_action(action: u8) -> (u8, u8) {
    let action = action.min((MOVEMENT_HEAD_DIM * FIRE_HEAD_DIM - 1) as u8);
    (action / FIRE_HEAD_DIM as u8, action % FIRE_HEAD_DIM as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_round_trips_all_head_pairs() {
        for movement in 0..MOVEMENT_HEAD_DIM as u8 {
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
    fn human_controller_aligns_the_nearest_end_while_moving() {
        let mut game = Game::with_ai(123, 2, &[]);
        game.tanks[0].rotation = 0.0;

        apply_human_direction(&mut game, 0, 8, NO_FIRE);
        assert!(game.tanks[0].backup);
        assert!(!game.tanks[0].forward);
        assert!(!game.tanks[0].turn_left);
        assert!(!game.tanks[0].turn_right);

        apply_human_direction(&mut game, 0, 10, FIRE);
        assert!(game.tanks[0].backup);
        assert!(game.tanks[0].turn_right);
        assert!(game.tanks[0].fire);

        apply_human_direction(&mut game, 0, 12, NO_FIRE);
        assert!(game.tanks[0].forward);
        assert!(game.tanks[0].turn_left);
    }
}

//! Discrete PPO action controller matching the human instant-turn wheel.

use crate::game::Game;

pub const DIRECTION_COUNT: usize = 360;
pub const REVERSE_OFFSET: u16 = DIRECTION_COUNT as u16;
pub const FIRE_ACTION: u16 = (DIRECTION_COUNT * 2) as u16;
pub const STOP_ACTION: u16 = FIRE_ACTION + 1;
pub const ACTION_COUNT: usize = DIRECTION_COUNT * 2 + 2;
pub const STOP: u16 = DIRECTION_COUNT as u16;
pub const NO_FIRE: u8 = 0;
pub const FIRE: u8 = 1;
pub const DIRECTION_STEP_DEGREES: f64 = 1.0;
const HUMAN_REVERSE_START_DEGREES: f64 = 135.0;

/// Human world-direction control used by the browser wheel.
///
/// Movement begins while the nose aligns across a 270-degree sector. Only the
/// 90-degree sector centred directly behind the hull aligns the rear and
/// reverses. The final steering frame is proportional so the hull does not
/// oscillate around a requested heading.
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
    let backward =
        nose_error >= HUMAN_REVERSE_START_DEGREES || nose_error < -HUMAN_REVERSE_START_DEGREES;
    let forward = !backward;
    let alignment = if forward {
        desired
    } else {
        (desired + 180.0) % 360.0
    };
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

/// Apply one `Discrete(722)` PPO action.
///
/// `0..=359` move forward in an absolute world direction, `360..=719` move
/// backward in an absolute world direction, `720` fires while stationary, and
/// `721` stops. Both direction wheels cover all 360 degrees and hull alignment
/// has no angular-speed limit.
pub fn apply_joystick_action(game: &mut Game, tank: usize, action: u16) {
    let action = action.min(STOP_ACTION);
    if action == FIRE_ACTION {
        apply_human_direction(game, tank, STOP, FIRE);
        return;
    }
    if action == STOP_ACTION {
        apply_human_direction(game, tank, STOP, NO_FIRE);
        return;
    }

    let backward = action >= REVERSE_OFFSET;
    let direction = action % REVERSE_OFFSET;
    let desired = direction as f64 * DIRECTION_STEP_DEGREES;
    let alignment = if backward {
        (desired + 180.0) % 360.0
    } else {
        desired
    };
    game.set_tank_rotation_if_clear(tank, alignment);
    let t = &mut game.tanks[tank];
    t.forward = !backward;
    t.backup = backward;
    t.turn_left = false;
    t.turn_right = false;
    t.fire = false;
    t.forward_amount = Some((!backward) as u8 as f64);
    t.backup_amount = Some(backward as u8 as f64);
    t.turn_left_amount = Some(0.0);
    t.turn_right_amount = Some(0.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppo_action_space_is_360_forward_360_reverse_fire_and_stop() {
        assert_eq!(ACTION_COUNT, 722);
        assert_eq!(FIRE_ACTION, 720);
        assert_eq!(STOP_ACTION, 721);
        let mut game = Game::with_ai(123, 2, &[]);
        game.tanks[0].rotation = 180.0;
        apply_joystick_action(&mut game, 0, 0);
        assert!(game.tanks[0].forward);
        assert_eq!(game.tanks[0].rotation, 0.0);
        apply_joystick_action(&mut game, 0, REVERSE_OFFSET);
        assert!(game.tanks[0].backup);
        assert_eq!(game.tanks[0].rotation, 180.0);
        apply_joystick_action(&mut game, 0, FIRE_ACTION);
        assert!(!game.tanks[0].backup);
        assert!(game.tanks[0].fire);
        apply_joystick_action(&mut game, 0, STOP_ACTION);
        assert!(!game.tanks[0].fire);
    }

    #[test]
    fn ppo_direction_snaps_without_turn_speed_limit() {
        let mut game = Game::with_ai(123, 2, &[]);
        game.tanks[0].rotation = 0.0;
        apply_joystick_action(&mut game, 0, 90);
        assert!((game.tanks[0].rotation - 90.0).abs() < 1e-9);
        assert!(game.tanks[0].forward);
        assert!(!game.tanks[0].turn_left);
        assert!(!game.tanks[0].turn_right);
    }

    #[test]
    fn human_controller_uses_a_270_forward_and_90_reverse_split() {
        let mut game = Game::with_ai(123, 2, &[]);
        game.tanks[0].rotation = 0.0;

        apply_human_direction(&mut game, 0, 180, NO_FIRE);
        assert!(game.tanks[0].backup);
        assert!(!game.tanks[0].forward);
        assert!(!game.tanks[0].turn_left);
        assert!(!game.tanks[0].turn_right);

        apply_human_direction(&mut game, 0, 181, FIRE);
        assert!(game.tanks[0].backup);
        assert!(game.tanks[0].turn_right);
        assert!(game.tanks[0].fire);

        // 225 degrees is the first heading outside the half-open rear sector.
        apply_human_direction(&mut game, 0, 225, NO_FIRE);
        assert!(game.tanks[0].forward);
        assert!(game.tanks[0].turn_left);
    }
}

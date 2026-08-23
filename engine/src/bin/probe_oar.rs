//! Small real O/A/R rollout used to measure the training-system budget.
//!
//! This is deliberately not a performance-only toy: it emits the frozen
//! schema-2 observations, chosen actions, rewards and episode boundaries that
//! the Python update probe consumes.

use kf_engine::game::Game;
use kf_engine::directional::{apply_direction, unpack_action};
use kf_engine::reward::RewardTracker;
use kf_engine::rng::Rng;
use kf_engine::semantic_obs::{
    encode, SemanticObsState, SemanticObservation, BULLET_SLOTS, OBS_DIM,
    OBS_SCHEMA_VERSION,
};
use std::collections::HashMap;
use std::fs;
use std::time::{Duration, Instant};

struct Env {
    game: Game,
    reward: RewardTracker,
    obs_state: SemanticObsState,
    frame_to_transition: HashMap<i64, usize>,
    round: i32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/tmp/kf-probe-oar".into());
    let steps: usize = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(8192);
    let env_count: usize = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(64);
    assert!(steps >= env_count && steps % env_count == 0);

    let mut envs: Vec<Env> = (0..env_count)
        .map(|i| {
            let game = Game::with_ai(700_000 + i as u32, 2, &[1]);
            let round = game.round_number;
            Env {
                game,
                reward: RewardTracker::new_r1(0),
                obs_state: SemanticObsState::default(),
                frame_to_transition: HashMap::new(),
                round,
            }
        })
        .collect();
    let mut rng = Rng::new(0x00c0_ffee);
    let mut observation = SemanticObservation::default();
    let mut obs = Vec::<f32>::with_capacity(steps * OBS_DIM);
    let mut masks = Vec::<u8>::with_capacity(steps * BULLET_SLOTS);
    let mut actions = Vec::<i64>::with_capacity(steps);
    let mut rewards = vec![0.0f32; steps];
    let mut dones = Vec::<u8>::with_capacity(steps);

    let mut t_obs = Duration::ZERO;
    let mut t_action = Duration::ZERO;
    let mut t_engine = Duration::ZERO;
    let mut t_reward = Duration::ZERO;
    let mut t_book = Duration::ZERO;
    let wall_start = Instant::now();

    for batch_step in 0..(steps / env_count) {
        for (env_index, env) in envs.iter_mut().enumerate() {
            let transition = batch_step * env_count + env_index;
            if env.round != env.game.round_number {
                env.round = env.game.round_number;
                env.obs_state.reset();
                env.frame_to_transition.clear();
            }

            let t = Instant::now();
            encode(&env.game, 0, &env.obs_state, &mut observation);
            obs.extend_from_slice(&observation.values);
            masks.extend(observation.bullet_mask.iter().map(|&x| x as u8));
            t_obs += t.elapsed();

            let t = Instant::now();
            let action = rng.randrange(34) as u8;
            let (movement, fire) = unpack_action(action);
            apply_direction(&mut env.game, 0, movement, fire);
            env.obs_state.push_action(movement, fire);
            actions.push(action as i64);
            t_action += t.elapsed();

            let mut decision_reward = 0.0f64;
            let mut done = false;
            for _ in 0..1 {
                let t = Instant::now();
                let events = env.game.step();
                t_engine += t.elapsed();

                let t = Instant::now();
                env.reward.process(&env.game, &events);
                let info = env.reward.info();
                let retro_sum: f64 = env
                    .reward
                    .retroactive_allocations()
                    .iter()
                    .map(|(_, value)| *value)
                    .sum();
                decision_reward += info[0] as f64 - retro_sum;
                let retro = env.reward.retroactive_allocations().to_vec();
                t_reward += t.elapsed();

                let t = Instant::now();
                for (frame, value) in retro {
                    if let Some(&old_transition) = env.frame_to_transition.get(&frame) {
                        rewards[old_transition] += value as f32;
                    } else {
                        // If the 8192-step probe began in the middle of a
                        // history window, keep conservation by returning the
                        // unplaceable part to the current transition.
                        decision_reward += value;
                    }
                }
                env.frame_to_transition.insert(env.game.frame, transition);
                if events
                    .iter()
                    .any(|event| matches!(event, kf_engine::game::Event::RoundEnd(_)))
                {
                    done = true;
                }
                t_book += t.elapsed();
            }
            rewards[transition] += decision_reward as f32;
            dones.push(done as u8);
        }
    }
    let rollout_seconds = wall_start.elapsed().as_secs_f64();

    let io_start = Instant::now();
    fs::create_dir_all(&out_dir).expect("create probe output");
    let mut obs_bytes = Vec::with_capacity(obs.len() * 4);
    for value in obs {
        obs_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let mut action_bytes = Vec::with_capacity(actions.len() * 8);
    for value in actions {
        action_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let mut reward_bytes = Vec::with_capacity(rewards.len() * 4);
    for value in rewards {
        reward_bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(format!("{out_dir}/obs.f32"), obs_bytes).expect("write obs");
    fs::write(format!("{out_dir}/bullet_mask.u8"), masks).expect("write masks");
    fs::write(format!("{out_dir}/action.i64"), action_bytes).expect("write actions");
    fs::write(format!("{out_dir}/reward.f32"), reward_bytes).expect("write rewards");
    fs::write(format!("{out_dir}/done.u8"), dones).expect("write dones");
    let io_seconds = io_start.elapsed().as_secs_f64();

    let phases = [
        ("observation", t_obs.as_secs_f64()),
        ("action", t_action.as_secs_f64()),
        ("engine", t_engine.as_secs_f64()),
        ("reward", t_reward.as_secs_f64()),
        ("bookkeeping", t_book.as_secs_f64()),
    ];
    let measured: f64 = phases.iter().map(|(_, seconds)| seconds).sum();
    let meta = format!(
        concat!(
            "{{\n",
            "  \"obs_schema_version\": {},\n",
            "  \"obs_dim\": {},\n",
            "  \"steps\": {},\n",
            "  \"envs\": {},\n",
            "  \"frames_per_step\": 1,\n",
            "  \"rollout_seconds\": {:.9},\n",
            "  \"steps_per_second\": {:.3},\n",
            "  \"io_seconds\": {:.9},\n",
            "  \"phase_seconds\": {{\n",
            "    \"observation\": {:.9},\n",
            "    \"action\": {:.9},\n",
            "    \"engine\": {:.9},\n",
            "    \"reward\": {:.9},\n",
            "    \"bookkeeping\": {:.9},\n",
            "    \"unattributed\": {:.9}\n",
            "  }}\n",
            "}}\n"
        ),
        OBS_SCHEMA_VERSION,
        OBS_DIM,
        steps,
        env_count,
        rollout_seconds,
        steps as f64 / rollout_seconds,
        io_seconds,
        phases[0].1,
        phases[1].1,
        phases[2].1,
        phases[3].1,
        phases[4].1,
        (rollout_seconds - measured).max(0.0),
    );
    fs::write(format!("{out_dir}/rollout.json"), meta).expect("write meta");
    println!("冻结 O/A/R 探针: {steps} 决策步, {env_count} 环境");
    println!(
        "rollout {:.3}s  = {:.0} 决策步/s  (每步 1 个引擎帧)",
        rollout_seconds,
        steps as f64 / rollout_seconds
    );
    for (name, seconds) in phases {
        println!(
            "  {:<12} {:>8.3}s  {:>5.1}%",
            name,
            seconds,
            100.0 * seconds / rollout_seconds
        );
    }
    println!("  {:<12} {:>8.3}s", "disk_io", io_seconds);
    println!("写入 {out_dir}");
}

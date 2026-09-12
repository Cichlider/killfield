/**
 * The opponent's per-frame state machine, and the recorder that captures a
 * ranked session.
 *
 * This file exists so the browser and the CI verifier drive the opponent from
 * one implementation rather than two. The delay queue, the opening pause and
 * the round reset all have to line up frame for frame; if they drift by a
 * single tick the replay diverges and every honest submission fails.
 */

import { HUMAN_SEAT, LIMITS, NO_ACTION, RejectedSubmission } from "./replay.js";

export const NEUTRAL_ACTION = 8; // stationary, no fire
/**
 * Ranked play is the default match with no handicap granted to the opponent.
 *
 * Both knobs exist to make the game approachable, and both make it easier: the
 * delay holds the opponent's actuation back by whole frames, and the opening
 * pause keeps it still while the round starts. A record is only comparable —
 * and only exercises the configuration the engine is actually tested in — if
 * the delay is zero and the pause is no longer than the default. A shorter
 * pause is allowed because it only makes the run harder.
 */
export const RANKED_DELAY_FRAMES = 0;
export const RANKED_OPENING_DELAY_SECONDS = 0.5;
export const FPS = 25;
export const KILLFIELD_RAYS = 512;
export const OPPONENT_SEAT = 0;
export const HYBRID_OBS_DIM = 1028;
export const HYBRID_BULLET_SLOTS = 10;
export const HYBRID_DODGE_OFFSET = 1018;
export const HYBRID_DODGE_DIM = 9;

/** Convert Discrete(18) into the exact full-strength human input recorded by
 * the browser. This is used by the opt-in policy pilot and mirrors
 * engine/src/score.rs CANDIDATES. */
export function policyActionToInput(action) {
  const throttle = Math.floor(action / 6);
  const turn = Math.floor((action % 6) / 2);
  return {
    forward: throttle === 2 ? 1 : 0,
    backup: throttle === 0 ? 1 : 0,
    turnLeft: turn === 0 ? 1 : 0,
    turnRight: turn === 2 ? 1 : 0,
    fire: action % 2,
  };
}

/**
 * How far below the best logit a submitted opponent action may sit before the
 * audit calls it forged.
 *
 * Measured over 3,000 frames of real play: the gap between the best and
 * second-best logit never fell below 9e-3, and reaches 5.0 at the median,
 * while replacing an action with "stand still" costs at least 7.8e-2. A
 * cross-engine Math.tanh difference moves a logit by ~1e-5. This sits two
 * orders above that noise and one order below the narrowest genuine tie.
 */
export const AUDIT_EPSILON = 1e-3;

/** Read one seat's Hybrid observation out of wasm memory. */
export function readObservation(wasm, handle, seat) {
  const ptr = wasm.kf_hybrid_observation(handle, seat);
  const view = new Float32Array(wasm.memory.buffer, ptr, wasm.kf_hybrid_observation_len());
  const mask = new Array(HYBRID_BULLET_SLOTS);
  for (let i = 0; i < HYBRID_BULLET_SLOTS; i += 1) {
    mask[i] = view[HYBRID_OBS_DIM + i] > 0.5;
  }
  return {
    observation: view,
    mask,
    dodge: view.subarray(HYBRID_DODGE_OFFSET, HYBRID_DODGE_OFFSET + HYBRID_DODGE_DIM),
  };
}

/**
 * Drives seat 0 through one ranked session.
 *
 * Laika and Killfield are driven inside `kf_step`, so for those this only
 * manages the opening pause; there is no action to record and nothing a
 * submission could forge. Hybrid is driven from JS, so its action is recorded
 * and re-derived by the verifier.
 */
export class OpponentDriver {
  constructor({ opponent, delayFrames, openingDelayFrames, policy = null }) {
    this.opponent = opponent;
    this.delayFrames = delayFrames;
    this.openingDelayFrames = opponent === "laika" ? 0 : openingDelayFrames;
    this.policy = policy;
    this.queue = [];
    this.pause = this.openingDelayFrames;
  }

  /** Wire the engine-side opponent up on a fresh handle. */
  attach(wasm, handle) {
    if (this.opponent !== "killfield") return;
    // A human is not Laika's script, so the planner gets the honest "assume
    // they hold their current buttons" model.
    wasm.kf_attach_mpc(handle, OPPONENT_SEAT, 7, KILLFIELD_RAYS, 1);
    wasm.kf_set_mpc_delay(handle, OPPONENT_SEAT, this.delayFrames);
    wasm.kf_set_mpc_enabled(handle, OPPONENT_SEAT, this.pause === 0 ? 1 : 0);
  }

  /**
   * Work out seat 0's action for this frame, advancing the delay queue.
   * Returns null when the engine drives the seat itself.
   *
   * Deciding is split from applying so the verifier can drive the engine with
   * the action the submission recorded — keeping the replayed trajectory
   * identical to the one the player saw — while still auditing that action
   * against the decision it derived independently.
   */
  decide(wasm, handle) {
    if (this.opponent !== "hybrid") return null;
    if (this.pause > 0) return { action: NEUTRAL_ACTION, logits: null };

    const { observation, mask, dodge } = readObservation(wasm, handle, OPPONENT_SEAT);
    const logits = this.policy.logits(observation, mask, dodge);
    let best = 0;
    for (let i = 1; i < logits.length; i += 1) if (logits[i] > logits[best]) best = i;
    this.queue.push({ action: best, logits });

    // Until the queue is deep enough the seat actuates nothing, which is what
    // the delay handicap means: it plans every frame but acts late.
    if (this.queue.length > this.delayFrames) return this.queue.shift();
    return { action: NEUTRAL_ACTION, logits: null };
  }

  apply(wasm, handle, action) {
    if (this.opponent !== "hybrid") return;
    wasm.kf_set_hybrid_action(handle, OPPONENT_SEAT, action);
  }

  /** Advance the opening pause and reset on a round boundary, after `kf_step`. */
  afterStep(wasm, handle, flags) {
    const newRound = (flags & 1) !== 0;
    if (newRound) {
      this.queue.length = 0;
      this.pause = this.openingDelayFrames;
      if (this.opponent === "killfield") {
        wasm.kf_set_mpc_enabled(handle, OPPONENT_SEAT, this.pause === 0 ? 1 : 0);
      }
      return;
    }
    if (this.pause > 0) {
      this.pause -= 1;
      if (this.pause === 0 && this.opponent === "killfield") {
        wasm.kf_set_mpc_enabled(handle, OPPONENT_SEAT, 1);
      }
    }
  }
}

/** Captures the frames and trigger edges a ranked session is replayed from. */
export class SessionRecorder {
  constructor() {
    this.frames = [];
    this.events = [];
  }

  get frameCount() { return this.frames.length; }

  /**
   * @param {{forward:number,backup:number,turnLeft:number,turnRight:number,fire:number}} input
   *   exactly the strengths handed to `kf_set_input` this frame
   * @param {number|null} action the opponent action applied, when JS drove it
   */
  frame(input, action) {
    this.frames.push({
      forward: input.forward,
      backup: input.backup,
      turnLeft: input.turnLeft,
      turnRight: input.turnRight,
      fire: input.fire ? 1 : 0,
      action: action === null ? NO_ACTION : action,
    });
  }

  /**
   * A trigger edge, which `kf_set_fire_immediate` applies between ticks rather
   * than on one. Anchored to the number of frames already stepped, which is
   * where the replay re-applies it.
   */
  fireEdge(pressed) {
    this.events.push({ frame: this.frames.length, kind: 0, value: pressed ? 1 : 0 });
  }

  session() {
    return { frames: this.frames, events: this.events };
  }
}

/** The winner the engine settled the round on: 0, 1, or 2 for a double kill. */
function lastWinner(wasm, handle) {
  const render = new Float32Array(
    wasm.memory.buffer, wasm.kf_render_ptr(handle), wasm.kf_render_len(handle),
  );
  return render[15];
}

/**
 * Replay a recorded session and audit it.
 *
 * The engine is driven with the actions the submission recorded, so the
 * trajectory is the one the player actually played; the opponent's decision is
 * re-derived alongside it purely to check that those recorded actions are what
 * the policy would have chosen. Because the trajectory never diverges, every
 * comparison happens on the identical observation the browser saw, and a
 * cross-engine float difference can only ever show up as a near-tie.
 *
 * @returns {{winners: number[], suspect: Array}}
 */
export function replaySession({ wasm, policy, config, session }) {
  const reject = (why) => { throw new RejectedSubmission(why); };
  const driver = new OpponentDriver({
    opponent: config.opponent,
    delayFrames: config.delayFrames,
    openingDelayFrames: Math.round(config.openingDelaySeconds * FPS),
    policy,
  });
  const handle = wasm.kf_new(config.seed, 0);
  driver.attach(wasm, handle);

  const winners = [];
  const suspect = [];
  let eventIndex = 0;
  try {
    for (let i = 0; i < session.frames.length; i += 1) {
      // Trigger edges land between ticks, which is where they are replayed.
      while (eventIndex < session.events.length && session.events[eventIndex].frame === i) {
        wasm.kf_set_fire_immediate(handle, HUMAN_SEAT, session.events[eventIndex].value);
        eventIndex += 1;
      }
      const frame = session.frames[i];
      wasm.kf_set_input(handle, HUMAN_SEAT, frame.forward, frame.backup,
        frame.turnLeft, frame.turnRight, frame.fire, 1);

      const decision = driver.decide(wasm, handle);
      if (decision === null) {
        if (frame.action !== NO_ACTION) {
          reject(`frame ${i} carries an opponent action, but ${config.opponent} is engine-driven`);
        }
      } else {
        if (frame.action === NO_ACTION) reject(`frame ${i} is missing its opponent action`);
        if (frame.action !== decision.action) {
          const gap = decision.logits === null
            ? Infinity
            : decision.logits[decision.action] - decision.logits[frame.action];
          if (!(gap <= AUDIT_EPSILON)) {
            suspect.push({ frame: i, recorded: frame.action, expected: decision.action, gap });
          }
        }
        driver.apply(wasm, handle, frame.action);
      }

      const flags = wasm.kf_step(handle);
      driver.afterStep(wasm, handle, flags);
      if (flags & 64) {
        winners.push(lastWinner(wasm, handle));
        if (winners.length > LIMITS.maxRounds) reject(`session runs past ${LIMITS.maxRounds} rounds`);
      }
    }
  } finally {
    wasm.kf_free(handle);
  }
  return { winners, suspect };
}

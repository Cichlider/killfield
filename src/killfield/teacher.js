/**
 * The search agent.
 *
 * A receding-horizon controller wrapped in a small amount of hand-written
 * machinery that exists because pure per-frame replanning plays badly:
 *
 *   - Commitment. A chosen move is held for a few frames, so the tank drives
 *     in a line instead of dithering between near-tied candidates.
 *   - Fire continuations. Firing competes with movement in the same score: the
 *     trigger is pressed for one simulated frame, then each of the nine safe
 *     no-fire controls is tried as its continuation.
 *   - Own-bullet guard. Plans can predate a bullet we just fired, so any
 *     movement that would drive into our own shot is replaced.
 *   - Stuck detection. If a commanded move produced no motion at all, the
 *     whole (throttle, turn) pair is penalised so we stop grinding a wall.
 *
 * The agent always drives tank 0.
 */

import * as C from "../constants.js";
import { LaikaAI } from "../laika.js";
import { HuntChainState, huntChainTimeMultiplier } from "./chain.js";
import {
  InverseDensityFieldBuilder, DEFAULT_RAYS, DEFAULT_BOUNCES, DEFAULT_FLIGHT_FRAMES,
} from "./field.js";
import {
  CANDIDATES, ROLLOUT_PLANS, STATIONARY_FIRE_ACTION, MPC_HORIZON, MPC_HOLD,
  COMMIT_MOVE_FRAMES, COMMIT_TURN_FRAMES, OWN_BULLET_GUARD_HORIZON,
  NO_EFFECT_REPEAT_PENALTY, densityRollout, postKillSurvivalScores,
  actionSelfHits, maskMovingFireScores, actionIndex, argmax,
} from "./score.js";
import { Rng } from "../rng.js";

function cellOf(game, tank) {
  return [Math.floor(tank.x / game.scale), Math.floor(tank.y / game.scale)];
}

export class KillFieldAgent {
  constructor({
    seed = 0,
    rayCount = DEFAULT_RAYS,
    maxBounces = DEFAULT_BOUNCES,
    maxFlightFrames = DEFAULT_FLIGHT_FRAMES,
    horizon = MPC_HORIZON,
    hold = MPC_HOLD,
    oppModel = "L2",
  } = {}) {
    this.rng = new Rng(seed);
    this.rayCount = rayCount;
    this.maxBounces = maxBounces;
    this.maxFlightFrames = maxFlightFrames;
    this.horizon = horizon;
    this.hold = hold;
    // Which controller the lookahead sandbox assumes tank 1 is running.
    // "L2" plays out the real Laika script — only sound when the opponent
    // actually is Laika. "L1" just freezes their current buttons, which is
    // the honest assumption against a human: we cannot script their play,
    // so pretending we can (and imagining them dying on schedule) is how the
    // agent ends up standing still against a live opponent who never died.
    this.oppModel = oppModel;
    this.reset();
  }

  reset() {
    this.game = null;
    this.roundNumber = null;
    this.builder = null;
    this.fieldCache = new Map();
    this.field = null;
    this.boxes = [];

    this.commitRemaining = 0;
    this.committedAction = [1, 1, 0];
    this.lastMotionAction = [1, 1, 0];
    this.lastAction = [1, 1, 0];
    this.lastDecisionKind = "none";
    this.bestFireContinuation = null;

    this.chain = new HuntChainState();
    this.chainTotal = 0.0;
    this.lastChainGain = 0.0;
    this.chainRound = null;
    this.chainTarget = null;
    this.chainCell = null;

    this.actionNoEffect = false;
    this.noEffectFrames = 0;
    this.observedPreviousAction = [1, 1, 0];
    this.effectRound = null;
    this.effectFrame = null;
    this.effectPose = null;
    this.effectAction = null;

    // Telemetry
    this.fieldBuilds = 0;
    this.fieldBuildMs = 0.0;
    this.ownBulletGuardEvents = 0;
    this.noEffectEvents = 0;
    this.planMs = [];
  }

  /** Did the last command actually move us? Detects grinding against a wall. */
  observeActionEffect(game) {
    const tank = game.tanks[0];
    if (this.effectRound !== game.roundNumber) {
      this.actionNoEffect = false;
      this.noEffectFrames = 0;
      this.effectRound = game.roundNumber;
      this.effectFrame = game.frame;
      this.effectPose = [tank.x, tank.y, tank.rotation];
      this.effectAction = null;
      return;
    }
    if (this.effectFrame === null || game.frame === this.effectFrame) return;
    const previous = this.effectPose;
    const action = this.effectAction;
    if (previous === null || action === null) return;

    const displacement = Math.hypot(tank.x - previous[0], tank.y - previous[1]);
    const rotationDelta = Math.abs(
      ((tank.rotation - previous[2] + 180.0) % 360.0 + 360.0) % 360.0 - 180.0,
    );
    const requestedTranslation = action[0] !== 1;
    const requestedTurn = action[1] !== 1;
    const moved = displacement > Math.max(1e-4, game.scale * 1e-4);
    const turned = rotationDelta > 1e-3;
    this.actionNoEffect = (requestedTranslation || requestedTurn) && !moved && !turned;
    this.noEffectFrames = this.actionNoEffect ? this.noEffectFrames + 1 : 0;
    if (this.actionNoEffect) this.noEffectEvents += 1;
  }

  emitAction(game, action, kind) {
    if (action[2] && !(action[0] === 1 && action[1] === 1)) action = [1, 1, 1];
    if (action[2] === 0 && !(action[0] === 1 && action[1] === 1)
        && actionSelfHits(game, action)) {
      const safety = postKillSurvivalScores(game, OWN_BULLET_GUARD_HORIZON);
      const picked = CANDIDATES[argmax(safety)];
      action = [picked[0], picked[1], 0];
      this.commitRemaining = 0;
      this.committedAction = action;
      this.ownBulletGuardEvents += 1;
      kind = `${kind}:own_bullet_guard`;
    }
    this.lastDecisionKind = kind;
    this.lastAction = action;
    if (action[0] !== 1 || action[1] !== 1) {
      this.lastMotionAction = [action[0], action[1], 0];
    }
    const tank = game.tanks[0];
    this.effectRound = game.roundNumber;
    this.effectFrame = game.frame;
    this.effectPose = [tank.x, tank.y, tank.rotation];
    this.effectAction = action;
    return action;
  }

  /** Fields are cached per enemy cell; a new round throws the cache away. */
  ensureField(game) {
    if (game !== this.game || game.roundNumber !== this.roundNumber) {
      this.game = game;
      this.roundNumber = game.roundNumber;
      this.builder = new InverseDensityFieldBuilder(
        game, this.rayCount, this.maxBounces, this.maxFlightFrames,
      );
      this.boxes = this.builder.boxes;
      this.fieldCache = new Map();
      this.commitRemaining = 0;
    }
    const target = cellOf(game, game.tanks[1]);
    const key = `${target[0]},${target[1]}`;
    if (!this.fieldCache.has(key)) {
      const started = performance.now();
      this.fieldCache.set(key, this.builder.build(target));
      this.fieldBuildMs += performance.now() - started;
      this.fieldBuilds += 1;
      this.commitRemaining = 0;
    }
    this.field = this.fieldCache.get(key);
    return this.field;
  }

  updateLiveChain(game, field) {
    const currentCell = cellOf(game, game.tanks[0]);
    const target = field.targetCell;
    if (this.chainRound !== game.roundNumber) {
      this.chain = new HuntChainState();
      this.chainRound = game.roundNumber;
      this.chainTarget = target;
      this.chainCell = currentCell;
      this.lastChainGain = 0.0;
      return;
    }
    this.chain.advance();
    const stable = this.chainTarget !== null
      && target[0] === this.chainTarget[0] && target[1] === this.chainTarget[1];
    const gain = this.chain.collectAscent(field, this.chainCell, currentCell, stable);
    this.lastChainGain = gain;
    this.chainTotal += gain;
    this.chainTarget = target;
    this.chainCell = currentCell;
  }

  /** A narrow firing window should trigger replanning, not force the shot. */
  static hasFireOpportunity(game) {
    const me = game.tanks[0];
    if (!(me.alive && game.tanks[1].alive && me.triggerReleased
        && game.weaponReady(me))) return false;
    return new LaikaAI(game, me).checkBulletPath(me.rotation).result === "HIT";
  }

  scores(game) {
    const field = this.ensureField(game);
    const seed = this.rng.randrange(1 << 30);
    const values = new Float64Array(CANDIDATES.length).fill(-1e9);
    this.bestFireContinuation = null;
    const me = game.tanks[0];
    const canFire = me.triggerReleased && game.weaponReady(me);

    // Nine persistent no-fire controls plus nine stationary-fire plans with a
    // different next-frame continuation. The latter collapse to the one real
    // fire action after their best continuation has been found.
    for (const plan of ROLLOUT_PLANS) {
      if (plan.kind === "fire_then_move" && !canFire) continue;
      let value = densityRollout(game, plan.firstAction, field, seed, {
        boxes: this.boxes,
        chainState: this.chain,
        horizon: this.horizon,
        hold: this.hold,
        oppModel: this.oppModel,
        continuationAction: plan.continuationAction,
      });

      // A fire plan's useful movement begins on its continuation frame, so a
      // recently failed wall-grinding control penalises that continuation.
      const effectAction = plan.continuationAction ?? plan.firstAction;
      if (this.actionNoEffect && this.observedPreviousAction !== null
          && effectAction[0] === this.observedPreviousAction[0]
          && effectAction[1] === this.observedPreviousAction[1]) {
        value -= NO_EFFECT_REPEAT_PENALTY;
      }

      const index = actionIndex(plan.firstAction);
      if (value > values[index]) {
        values[index] = value;
        if (plan.firstAction === STATIONARY_FIRE_ACTION) {
          this.bestFireContinuation = plan.continuationAction;
        }
      }
    }
    maskMovingFireScores(values);
    return values;
  }

  /**
   * Decide this frame's move.
   * @returns {number[]} a [throttle, turn, fire] triple
   */
  act(game) {
    const started = performance.now();
    this.lastDecisionKind = "none";
    if (!game.tanks[0].alive) return [1, 1, 0];
    this.observeActionEffect(game);
    this.observedPreviousAction = this.effectAction ?? [1, 1, 0];

    try {
      // Post-kill: the world is still live and our own bullets can still kill
      // us, so keep making explicit no-fire survival decisions.
      if (!game.tanks[1].alive) {
        if (this.actionNoEffect) this.commitRemaining = 0;
        if (this.commitRemaining > 0 && !game.tanks[0].hitSomething) {
          this.commitRemaining -= 1;
          const held = [this.committedAction[0], this.committedAction[1], 0];
          return this.emitAction(game, held, "post_kill_hold");
        }
        const values = postKillSurvivalScores(
          game, C.NUMBEROFFRAMESBEFOREEND - C.NUMBEROFFRAMESFROZEN,
        );
        const picked = CANDIDATES[argmax(values)];
        const action = [picked[0], picked[1], 0];
        this.committedAction = action;
        this.commitRemaining = Math.min(1,
          action[0] !== 1 ? COMMIT_MOVE_FRAMES : action[1] !== 1 ? COMMIT_TURN_FRAMES : 0);
        return this.emitAction(game, action, "post_kill_plan");
      }

      const field = this.ensureField(game);
      this.updateLiveChain(game, field);
      if (this.actionNoEffect) this.commitRemaining = 0;
      if (this.commitRemaining > 0 && KillFieldAgent.hasFireOpportunity(game)) {
        this.commitRemaining = 0;
      }

      if (this.commitRemaining > 0 && !game.tanks[0].hitSomething) {
        this.commitRemaining -= 1;
        return this.emitAction(game, this.committedAction, "hold");
      }

      const values = this.scores(game);
      const action = CANDIDATES[argmax(values)];
      if (action[2] === 0) {
        this.committedAction = action;
        this.commitRemaining = action[0] !== 1 ? COMMIT_MOVE_FRAMES
          : action[1] !== 1 ? COMMIT_TURN_FRAMES : 0;
      }
      const kind = action === STATIONARY_FIRE_ACTION ? "plan:fire_then_move" : "plan";
      return this.emitAction(game, action, kind);
    } finally {
      this.planMs.push(performance.now() - started);
      if (this.planMs.length > 600) this.planMs.shift();
    }
  }

  /** Decide and write the result onto tank 0. */
  drive(game) {
    const [throttle, turn, fire] = this.act(game);
    const me = game.tanks[0];
    me.forward = throttle === 2;
    me.backup = throttle === 0;
    me.turnLeft = turn === 0;
    me.turnRight = turn === 2;
    me.fire = fire === 1;
  }

  telemetry() {
    const sorted = this.planMs.slice().sort((a, b) => a - b);
    const at = (q) => (sorted.length ? sorted[Math.min(sorted.length - 1,
      Math.floor(q * sorted.length))] : 0);
    return {
      decision: this.lastDecisionKind,
      action: this.lastAction,
      fireContinuation: this.bestFireContinuation,
      fieldBuilds: this.fieldBuilds,
      meanFieldBuildMs: this.fieldBuildMs / Math.max(this.fieldBuilds, 1),
      cachedTargetCells: this.fieldCache.size,
      huntChain: this.chain.count,
      huntChainTotal: this.chainTotal,
      huntAgeFrames: this.chain.elapsedFrames,
      huntTimeMultiplier: huntChainTimeMultiplier(this.chain.elapsedFrames),
      ownBulletGuardEvents: this.ownBulletGuardEvents,
      noEffectEvents: this.noEffectEvents,
      planMedianMs: at(0.5),
      planP95Ms: at(0.95),
    };
  }
}

export { CANDIDATES, actionIndex };

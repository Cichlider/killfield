// Prism 2 (折光 2), a stateful game controller for schema 24.
import { HybridPolicy } from "./hybrid.js?v=e0283fc2";

const SEARCH_RANGE = 1.65;

export async function loadPrism2(hybrid, anchor, night) {
  const url = new URL("../assets/prism2.wasm?v=897b32fd", import.meta.url);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Prism 2 could not load its search engine (HTTP ${response.status})`);
  const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {});
  return createPrism2Factory(instance.exports, hybrid, anchor, night);
}

// Only the two occupied cells and the maze walls enter this encoder. The same
// embedding can be used by both seats and the simulated branches on a board.
class PrismModel extends HybridPolicy {
  constructor(manifest, weights) {
    super(manifest, weights);
    this.mapWalls = new Uint8Array(120);
    this.mapCells = new Map();
  }

  mapFeatures(observation) {
    let mine = 120, other = 120, changed = false;
    for (let cell = 0; cell < 120; cell++) {
      const start = cell * 7;
      const walls = observation[start] + 2 * observation[start + 1] + 4 * observation[start + 2]
        + 8 * observation[start + 3] + 16 * observation[start + 4];
      if (this.mapWalls[cell] !== walls) {
        this.mapWalls[cell] = walls;
        changed = true;
      }
      if (observation[start + 5]) mine = cell;
      if (observation[start + 6]) other = cell;
    }
    if (changed) this.mapCells.clear();
    const key = mine * 121 + other;
    let embedding = this.mapCells.get(key);
    if (!embedding) {
      if (this.mapCells.size >= 384) this.mapCells.clear();
      embedding = this.encodeMap(observation);
      this.mapCells.set(key, embedding);
    }
    return embedding;
  }
}

// Combine by tensor name. Anchor also supplies the derived-feature residual.
function blendedModel(hybrid, anchor, night) {
  const parts = [[night, 0.65], [anchor, 0.75], [hybrid, -0.4]];
  const weights = new Float32Array(anchor.weights.length);
  for (const [name, spec] of Object.entries(anchor.manifest.tensors)) {
    const tensors = parts.filter(([policy]) => Object.hasOwn(policy.manifest.tensors, name))
      .map(([policy, factor]) => [policy.tensor(name), factor]);
    for (let i = 0; i < spec.length; i++) {
      let sum = 0;
      for (const [tensor, factor] of tensors) sum += factor * tensor[i];
      weights[spec.offset + i] = halfValue(sum);
    }
  }
  // Match the released policy's signed eight-bit quantization, in blocks of 64.
  for (let start = 0; start < weights.length; start += 64) {
    const end = Math.min(start + 64, weights.length);
    let peak = 0;
    for (let i = start; i < end; i++) peak = Math.max(peak, Math.abs(weights[i]));
    if (!peak) continue;
    const scale = Math.fround(peak / 127);
    for (let i = start; i < end; i++) weights[i] = (Math.round(weights[i] / scale) || 0) * scale;
  }
  return new PrismModel(anchor.manifest, weights);
}

// The two seats share the immutable weights. Each match and each seat owns
// its own search world, opponent history, and settlement plan.
export function createPrism2Factory(api, hybrid, anchor, night) {
  const model = blendedModel(hybrid, anchor, night);
  return () => new PrismPolicy(api, model);
}

// Round published FP32 weights to binary16 once, then accumulate in FP32 as
// the viewer does. Round halfway values to even for identical browser weights.
function halfValue(value) {
  if (!value || !Number.isFinite(value)) return value;
  const magnitude = Math.abs(value);
  const unit = 2 ** Math.max(-24, Math.floor(Math.log2(magnitude)) - 10);
  const scaled = magnitude / unit, lower = Math.floor(scaled), fraction = scaled - lower;
  const up = fraction > 0.5 || (fraction === 0.5 && lower % 2 === 1);
  return Math.sign(value) * (lower + Number(up)) * unit;
}

// The private rollouts use the same named tensors and public features as the
// learned policy. Dense layers are evaluated in WebAssembly SIMD.
const tensorOrder = [
  'bullets.0.weight',
  'bullets.0.bias',
  'bullets.2.weight',
  'bullets.2.bias',
  'scalars.0.weight',
  'scalars.0.bias',
  'trunk.0.weight',
  'trunk.0.bias',
  'derived_up.weight',
  'derived_up.bias',
  'actor.weight',
  'actor.bias',
  'dodge_delta.0.weight',
  'dodge_delta.0.bias',
  'dodge_delta.2.weight',
  'dodge_delta.2.bias',
  'ammo_delta.0.weight',
  'ammo_delta.0.bias',
  'ammo_delta.2.weight',
  'ammo_delta.2.bias',
  'dodge_alpha_old',
  'ammo_alpha_old',
  'idle_logit_penalty'
];
class NativePredictor {
  constructor(api, model) {
    this.api = api;
    this.model = model;
    this.h = api.pn_new(model.weights.length);
    new Float32Array(api.memory.buffer, api.pn_weights(this.h), model.weights.length).set(
      model.weights
    );
    const offsets = new Uint32Array(api.memory.buffer, api.pn_offsets(this.h), 24);
    offsets.fill(0xffffffff);
    for (let i = 0; i < tensorOrder.length; i++) {
      const spec = model.manifest.tensors[tensorOrder[i]];
      if (spec) offsets[i] = spec.offset;
    }
  }
  close() {
    if (this.h) {
      this.api.pn_free(this.h);
      this.h = 0;
    }
  }
  logits(o) {
    const api = this.api,
      model = this.model;
    const inbox = new Float32Array(api.memory.buffer, api.pn_input(this.h), 1193);
    inbox.set(o.subarray(0, 1038));
    inbox.set(model.mapFeatures(o), 1038);
    if (model.manifest.tensors['derived_up.weight']) inbox.set(model.derivedFeatures(o), 1166);
    return new Float32Array(api.memory.buffer, api.pn_infer(this.h), 18);
  }
  act(o) {
    return argBest(this.logits(o));
  }
}

// A movement is throttle × turn. Its low bit requests the trigger.
function argBest(scores) {
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
  return best;
}

function observedControl(observation, mask, previous) {
  const forward = (observation[870] * observation[872] + observation[871] * observation[873]) * 0.5;
  let throttle = forward > 0.021 ? 2 : forward < -0.016 ? 0 : 1;
  if (throttle === 1 && observation[878] > 0.5 && previous >= 0) throttle = Math.floor(previous / 6);
  const turn = observation[874] > 0.32 ? 2 : observation[874] < -0.32 ? 0 : 1;
  let fired = 0;
  for (let slot = 0; slot < 10; slot++) {
    const offset = 900 + 10 * slot;
    if (mask[slot] && !observation[offset + 4] && observation[offset + 6] > 0.999) {
      fired = 1;
      break;
    }
  }
  return 6 * throttle + 2 * turn + fired;
}

class ImaginedDuel {
  constructor(api, model) {
    this.api = api;
    this.handle = api.ps_new();
    this.model = new NativePredictor(api, model);
  }
  close() {
    if (!this.handle) return;
    this.api.ps_free(this.handle);
    this.handle = 0;
    this.model.close();
  }
  sync(observation, mask, history) {
    const api = this.api;
    const buffer = new Float32Array(api.memory.buffer, api.ps_input(this.handle), 1044);
    buffer.set(observation.subarray(0, 1028));
    buffer.set(mask, 1028);
    buffer.set(history, 1038);
    api.ps_sync(this.handle);
  }
  predict(seat) {
    const api = this.api;
    const ptr = api.ps_observe(this.handle, seat, 1);
    return this.model.act(new Float32Array(api.memory.buffer, ptr, 1038));
  }
  probe(ours, theirs) {
    return this.api.ps_probe(this.handle, ours, theirs, 8);
  }
  rollout(ours, theirs, horizon, stride) {
    const api = this.api;
    api.ps_reset(this.handle);
    let alive = 3,
      firstDeath = -1;
    for (let tick = 0; tick < horizon; tick++) {
      if (tick >= 1 && (tick - 1) % stride === 0) {
        if (alive & 2) theirs = this.predict(1);
        if (alive & 1) ours = this.predict(0);
      }
      alive = api.ps_step(this.handle, ours, theirs) & 3;
      if (alive !== 3 && firstDeath < 0) firstDeath = tick;
      if (!alive || (firstDeath >= 0 && tick - firstDeath >= 2)) break;
    }
    const score = api.ps_value(this.handle);
    return firstDeath >= 0 ? score / (1 + 0.033 * firstDeath) : score;
  }
}

class PrismPolicy {
  constructor(api, base) {
    this.world = new ImaginedDuel(api, base);
    this.history = new Float32Array(6);
    this.needsReset = true;
  }
  close() {
    this.world.close();
  }
  startRound(observation) {
    this.needsReset = false;
    this.width = 0;
    this.height = 0;
    for (let cell = 0; cell < 120; cell++)
      if (observation[cell * 7]) {
        this.width = Math.max(this.width, Math.floor(cell / 10) + 1);
        this.height = Math.max(this.height, (cell % 10) + 1);
      }
    this.history.set([-1, -1, -1, 0, 0, 0]);
    this.escapePlan = null;
    this.settlementAge = 0;
    this.idleFrames = 0;
    this.repositionFrames = 0;
    this.quietOpponentFrames = 0;
  }
  record(observation, mask) {
    const history = this.history,
      previous = history[0];
    const action = observedControl(observation, mask, previous);
    if (observation[1006] > 0) {
      history[2] = history[1];
      history[1] = previous;
      history[0] = action;
      history[3] += Number(previous >= 0 && previous !== action);
      history[4]++;
      history[5] = action >> 1 === 4 ? history[5] + 1 : 0;
    }
    return action;
  }
  surviveSettlement(observation, mask) {
    const age = this.settlementAge++;
    if (!observation[1028]) return 8;
    if (!this.escapePlan || (!this.escapePlan.safe && age - this.escapePlan.start >= 2)) {
      this.world.sync(observation, mask, this.history);
      const code = this.world.api.ps_escape(this.world.handle, Math.max(1, 75 - age));
      this.escapePlan = {
        first: code & 31,
        next: (code >> 5) & 31,
        switchAt: (code >> 10) & 127,
        safe: Boolean(code & (1 << 17)),
        start: age
      };
    }
    const plan = this.escapePlan;
    return age - plan.start < plan.switchAt ? plan.first : plan.next;
  }
  breakStandoff(o, mask, range, observed) {
    const wallDetour = o[880] >= 0.032 && !o[890] && !o[895];
    const still = Math.hypot(o[860], o[861]) < 0.018 && Math.hypot(o[872], o[873]) < 0.018;
    this.idleFrames = still ? this.idleFrames + 1 : 0;
    if (this.idleFrames >= 65 && !this.repositionFrames && (range > 1.35 || wallDetour) && !o[1028])
      this.repositionFrames = 16;
    if (!this.repositionFrames) return null;
    this.repositionFrames--;
    if (!this.repositionFrames) this.idleFrames = 0;
    if ((range <= 1.25 && !wallDetour) || o[1028] || (o[895] && o[898] < 0.08)) return null;
    if (o[890] && !o[891] && o[864]) {
      if (o[1009]) return 8;
      this.repositionFrames = 0;
      this.idleFrames = 0;
      return 9;
    }
    this.world.sync(o, mask, this.history);
    return this.world.api.ps_reposition(this.world.handle, observed);
  }
  againstStationary(o, mask, observed, dodge) {
    const motionless = Math.hypot(o[872], o[873]) < 0.004 && Math.abs(o[874]) < 0.01 && !(observed & 1);
    this.quietOpponentFrames = motionless ? this.quietOpponentFrames + 1 : 0;
    if (this.quietOpponentFrames < 360 || (o[1000] > 0.35 && o[1028])) return null;
    const world = this.world;
    world.sync(o, mask, this.history);
    if (o[864] && !o[1009]) {
      let chosen = -1,
        best = 0.2;
      for (let m = 0; m < 9; m++) {
        if (dodge[m] < 0) continue;
        const action = m * 2 + 1;
        const value = world.api.ps_probe(world.handle, action, 8, 38);
        if (value > best) {
          best = value;
          chosen = action;
        }
      }
      if (chosen >= 0) return chosen;
    }
    const route = world.api.ps_reposition(world.handle, 8);
    return dodge[route >> 1] >= 0 ? route : null;
  }
  act(o, mask = o.subarray(1028), dodge = o.subarray(1018, 1027)) {
    if (!o[865] || o[1005]) {
      this.needsReset = true;
      return 8;
    }
    if (this.needsReset || o[1006] < this.lastClock || o[1006] === 0) this.startRound(o);
    this.lastClock = o[1006];
    const observed = this.record(o, mask);
    if (!o[877]) return this.surviveSettlement(o, mask);
    const stationary = this.againstStationary(o, mask, observed, dodge);
    if (stationary !== null) return stationary;

    const world = this.world;
    const logits = world.model.logits(o).slice();
    let baseline = argBest(logits);
    const range = o[885] * (this.width + this.height);
    const reposition = this.breakStandoff(o, mask, range, observed);
    if (reposition !== null) return reposition;

    const pointBlank = range < 1 || (o[895] && o[898] < 0.07);
    if (range > SEARCH_RANGE || (!o[864] && !o[876] && o[1010] < 0.05)) return baseline;

    world.sync(o, mask, this.history);
    const enemy = world.predict(1),
      mayFire = o[864] && !o[1009];
    if (pointBlank && !mayFire) {
      baseline &= ~1;
      for (let i = 0; i < 18; i += 2) logits[i] = Math.max(logits[i], logits[i + 1]);
    }
    const priorWeight = pointBlank ? 0.054 : 0.043;
    const probeWeight = pointBlank ? 0.08 : 0.06;
    const width = pointBlank ? 5 : 4;
    const horizon = pointBlank ? 8 : 10;
    const stride = pointBlank ? 2 : 3;
    const candidates = [];
    for (let movement = 0; movement < 9; movement++) {
      for (let fire = 0; fire <= (mayFire ? 1 : 0); fire++) {
        const action = 2 * movement + fire;
        const likely = world.probe(action, enemy);
        const returnFire = !(enemy % 2) && o[876] ? world.probe(action, enemy + 1) : likely;
        candidates.push({
          action,
          prior: (logits[action] - logits[baseline]) * priorWeight,
          probe: 0.72 * likely + 0.28 * returnFire
        });
      }
    }
    const first = candidates.find((candidate) => candidate.action === baseline) ?? {
      action: baseline,
      prior: 0,
      probe: world.probe(baseline, enemy)
    };
    candidates.sort((a, b) => b.prior + b.probe - (a.prior + a.probe));
    const shortlist = [
      first,
      ...candidates.filter((candidate) => candidate !== first).slice(0, width - 1)
    ];
    let chosen = baseline, best = -Infinity;
    for (const candidate of shortlist) {
      const outcome = world.rollout(candidate.action, enemy, horizon, stride);
      const score = outcome + candidate.prior + probeWeight * candidate.probe;
      if (score > best) {
        best = score;
        chosen = candidate.action;
      }
      if (candidate === first && outcome > 0.86) break;
    }
    return chosen;
  }
}

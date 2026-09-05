const MAP_W = 12;
const MAP_H = 10;
const MAP_C = 7;
const MAP_DIM = MAP_W * MAP_H * MAP_C;
const BULLET_OFFSET = 900;
const BULLET_SLOTS = 10;
const BULLET_DIM = 10;
const OBS_DIM = 1028;

function dense(input, weight, bias, outputs, activation = null) {
  const width = input.length;
  const out = new Float32Array(outputs);
  for (let o = 0; o < outputs; o += 1) {
    let sum = bias[o];
    const base = o * width;
    for (let i = 0; i < width; i += 1) sum += weight[base + i] * input[i];
    out[o] = activation === "relu" ? Math.max(0, sum)
      : activation === "tanh" ? Math.tanh(sum) : sum;
  }
  return out;
}

// PyTorch Conv2d cross-correlation over a CHW tensor, with OIHW weights.
function conv2d(input, inC, inH, inW, weight, bias, outC, stride = 1) {
  const outH = Math.floor((inH + 2 - 3) / stride) + 1;
  const outW = Math.floor((inW + 2 - 3) / stride) + 1;
  const out = new Float32Array(outC * outH * outW);
  for (let oc = 0; oc < outC; oc += 1) {
    for (let oy = 0; oy < outH; oy += 1) {
      for (let ox = 0; ox < outW; ox += 1) {
        let sum = bias[oc];
        for (let ic = 0; ic < inC; ic += 1) {
          for (let ky = 0; ky < 3; ky += 1) {
            const iy = oy * stride + ky - 1;
            if (iy < 0 || iy >= inH) continue;
            for (let kx = 0; kx < 3; kx += 1) {
              const ix = ox * stride + kx - 1;
              if (ix < 0 || ix >= inW) continue;
              const wi = (((oc * inC + ic) * 3 + ky) * 3 + kx);
              sum += weight[wi] * input[(ic * inH + iy) * inW + ix];
            }
          }
        }
        out[(oc * outH + oy) * outW + ox] = Math.max(0, sum);
      }
    }
  }
  return out;
}

function toChw(observation) {
  const grid = new Float32Array(MAP_DIM);
  for (let y = 0; y < MAP_W; y += 1) {
    for (let x = 0; x < MAP_H; x += 1) {
      for (let c = 0; c < MAP_C; c += 1) {
        grid[(c * MAP_W + y) * MAP_H + x] = observation[(y * MAP_H + x) * MAP_C + c];
      }
    }
  }
  return grid;
}

export class HybridPolicy {
  static async load(manifestUrl = "assets/hybrid.json", weightsUrl = "assets/hybrid.bin") {
    const manifest = await fetch(manifestUrl).then((response) => {
      if (!response.ok) throw new Error(`Hybrid manifest: HTTP ${response.status}`);
      return response.json();
    });
    const resolvedWeightsUrl = new URL(weightsUrl, location.href);
    const weights = new Float32Array(await fetch(resolvedWeightsUrl).then((response) => {
      if (!response.ok) throw new Error(`Hybrid weights: HTTP ${response.status}`);
      return response.arrayBuffer();
    }));
    return new HybridPolicy(manifest, weights);
  }

  constructor(manifest, weights) {
    if (manifest.schema !== 24 || manifest.observation !== OBS_DIM || manifest.actions !== 18) {
      throw new Error("Hybrid model does not match engine schema 24 / 1028 / 18");
    }
    if (weights.length !== manifest.floats) throw new Error("Hybrid weights are truncated");
    this.manifest = manifest;
    this.weights = weights;
  }

  tensor(name) {
    const spec = this.manifest.tensors[name];
    if (!spec) throw new Error(`Missing Hybrid tensor ${name}`);
    return this.weights.subarray(spec.offset, spec.offset + spec.length);
  }

  scalar(name) { return this.tensor(name)[0]; }

  logits(observation, mask, dodge) {
    const c1 = conv2d(toChw(observation), 7, 12, 10,
      this.tensor("map.0.weight"), this.tensor("map.0.bias"), 16, 1);
    const c2 = conv2d(c1, 16, 12, 10,
      this.tensor("map.2.weight"), this.tensor("map.2.bias"), 32, 2);
    const map = dense(c2, this.tensor("map.5.weight"), this.tensor("map.5.bias"), 128, "tanh");

    const mean = new Float32Array(32);
    const peak = new Float32Array(32);
    peak.fill(-Infinity);
    let count = 0;
    for (let slot = 0; slot < BULLET_SLOTS; slot += 1) {
      if (!mask[slot]) continue;
      const row = observation.subarray(BULLET_OFFSET + slot * BULLET_DIM,
        BULLET_OFFSET + (slot + 1) * BULLET_DIM);
      const b1 = dense(row, this.tensor("bullets.0.weight"), this.tensor("bullets.0.bias"), 32, "relu");
      const encoded = dense(b1, this.tensor("bullets.2.weight"), this.tensor("bullets.2.bias"), 32, "relu");
      for (let i = 0; i < 32; i += 1) {
        mean[i] += encoded[i];
        peak[i] = Math.max(peak[i], encoded[i]);
      }
      count += 1;
    }
    if (count) for (let i = 0; i < 32; i += 1) mean[i] /= count;
    else peak.fill(0);

    const scalarInput = new Float32Array(88);
    scalarInput.set(observation.subarray(MAP_DIM, BULLET_OFFSET), 0);
    scalarInput.set(observation.subarray(BULLET_OFFSET + 100, OBS_DIM), 60);
    const scalars = dense(scalarInput, this.tensor("scalars.0.weight"),
      this.tensor("scalars.0.bias"), 128, "tanh");
    const joined = new Float32Array(320);
    joined.set(map, 0); joined.set(scalars, 128); joined.set(mean, 256); joined.set(peak, 288);
    const features = dense(joined, this.tensor("trunk.0.weight"),
      this.tensor("trunk.0.bias"), 256, "tanh");
    const logits = dense(features, this.tensor("actor.weight"), this.tensor("actor.bias"), 18);

    const dodgeScale = this.scalar("dodge_scale");
    const ammo = observation[863];
    const hit = observation[890];
    const suicide = observation[891];
    const eta = Math.max(0, Math.min(1, observation[893] * 3));
    const hitSoon = hit * (1 - eta);
    const fireBias = this.scalar("ammo_scale") * ammo
      + this.scalar("shot_quality_scale") * hitSoon
      - this.scalar("ammo_lock_scale") * ((1 - ammo) ** 2) * (1 - hitSoon)
      - this.scalar("suicide_scale") * suicide;
    for (let action = 0; action < 18; action += 1) {
      logits[action] += dodgeScale * dodge[Math.floor(action / 2)];
      if (action % 2 === 1) logits[action] += fireBias;
    }
    const idlePressure = Math.max(0, Math.min(1, (observation[1027] * 25 - 8) / 17));
    logits[8] -= idlePressure * this.scalar("idle_logit_penalty");
    logits[9] -= idlePressure * this.scalar("idle_logit_penalty");
    return logits;
  }

  act(observation, mask, dodge) {
    const logits = this.logits(observation, mask, dodge);
    let best = 0;
    for (let i = 1; i < logits.length; i += 1) if (logits[i] > logits[best]) best = i;
    return best;
  }
}
